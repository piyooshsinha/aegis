//! Operator alerts: a client key nearing or exhausting its daily budget, and
//! a provider target going into cooldown.
//!
//! Every alert is kept in a short in-memory list for the dashboard. When a
//! webhook URL is configured it is also POSTed there as JSON; the `text` and
//! `content` fields make the same payload readable by Slack- and
//! Discord-style incoming webhooks.

use crate::config::AlertsCfg;
use crate::types::now_ms;
use serde_json::json;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::mpsc::{UnboundedSender, unbounded_channel};

const KEEP: usize = 100;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Info,
    Warning,
    Critical,
}

impl Severity {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Info => "info",
            Self::Warning => "warning",
            Self::Critical => "critical",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Alert {
    pub ts: i64,
    /// budget | cooldown | test
    pub kind: &'static str,
    pub severity: Severity,
    pub title: String,
    pub detail: String,
}

#[derive(Debug, Clone, Default)]
pub struct Delivery {
    pub sent: u64,
    pub failed: u64,
    pub last_error: Option<String>,
    pub last_ok_ms: Option<i64>,
}

pub struct Alerts {
    cfg: AlertsCfg,
    recent: Mutex<VecDeque<Alert>>,
    last_sent: Mutex<HashMap<String, i64>>,
    delivery: Arc<Mutex<Delivery>>,
    tx: Option<UnboundedSender<Alert>>,
}

impl Alerts {
    /// Must be called inside a Tokio runtime when a webhook is configured.
    pub fn new(cfg: &AlertsCfg, webhook_url: Option<String>) -> Self {
        let delivery = Arc::new(Mutex::new(Delivery::default()));
        let tx = webhook_url.filter(|u| u.starts_with("http://") || u.starts_with("https://")).map(|url| {
            let (tx, mut rx) = unbounded_channel::<Alert>();
            let delivery = delivery.clone();
            tokio::spawn(async move {
                let client = reqwest::Client::builder().timeout(Duration::from_secs(10)).build().unwrap_or_default();
                while let Some(a) = rx.recv().await {
                    let text = format!("[Aegis · {}] {} — {}", a.severity.as_str(), a.title, a.detail);
                    let body = json!({
                        "source": "aegis", "event": a.kind, "severity": a.severity.as_str(),
                        "title": a.title, "detail": a.detail, "ts": a.ts,
                        "text": text, "content": text,
                    });
                    let result = match client.post(&url).json(&body).send().await {
                        Ok(r) if r.status().is_success() => Ok(()),
                        Ok(r) => Err(format!("webhook answered {}", r.status())),
                        // `without_url`: the URL is a secret.
                        Err(e) => Err(format!("webhook unreachable: {}", e.without_url())),
                    };
                    let mut d = delivery.lock().unwrap_or_else(|e| e.into_inner());
                    match result {
                        Ok(()) => {
                            d.sent += 1;
                            d.last_ok_ms = Some(now_ms());
                            d.last_error = None;
                        }
                        Err(e) => {
                            tracing::warn!("alert not delivered: {e}");
                            d.failed += 1;
                            d.last_error = Some(e);
                        }
                    }
                }
            });
            tx
        });
        Self { cfg: cfg.clone(), recent: Mutex::new(VecDeque::new()), last_sent: Mutex::new(HashMap::new()), delivery, tx }
    }

    pub fn cfg(&self) -> &AlertsCfg {
        &self.cfg
    }

    pub fn webhook_configured(&self) -> bool {
        self.tx.is_some()
    }

    pub fn delivery(&self) -> Delivery {
        self.delivery.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Newest first.
    pub fn recent(&self) -> Vec<Alert> {
        self.recent.lock().unwrap_or_else(|e| e.into_inner()).iter().rev().cloned().collect()
    }

    /// Record an alert. `dedup` identifies "the same alert"; it is dropped if
    /// one with that id was raised within `min_interval_secs`.
    pub fn emit(&self, dedup: &str, kind: &'static str, severity: Severity, title: String, detail: String) -> bool {
        let now = now_ms();
        {
            let mut seen = self.last_sent.lock().unwrap_or_else(|e| e.into_inner());
            let gap = (self.cfg.min_interval_secs as i64) * 1000;
            if seen.get(dedup).is_some_and(|t| now - *t < gap) {
                return false;
            }
            seen.retain(|_, t| now - *t < 86_400_000);
            seen.insert(dedup.to_string(), now);
        }
        let alert = Alert { ts: now, kind, severity, title, detail };
        tracing::warn!(kind, severity = severity.as_str(), "alert: {} — {}", alert.title, alert.detail);
        {
            let mut r = self.recent.lock().unwrap_or_else(|e| e.into_inner());
            r.push_back(alert.clone());
            while r.len() > KEEP {
                r.pop_front();
            }
        }
        if let Some(tx) = &self.tx {
            let _ = tx.send(alert);
        }
        true
    }

    /// Called after a client key is charged. Alerts once when spend crosses
    /// the warning fraction and once when it reaches the budget.
    pub fn on_spend(&self, key_id: i64, key_name: &str, budget: f64, before: f64, after: f64) {
        if budget <= 0.0 {
            return;
        }
        let day = now_ms() / 86_400_000;
        let warn_at = budget * self.cfg.budget_warn_at;
        if before < budget && after >= budget {
            self.emit(
                &format!("budget:{key_id}:{day}:full"),
                "budget",
                Severity::Critical,
                format!("Key \"{key_name}\" has used its daily budget"),
                format!("${after:.4} of ${budget:.2} spent. Its requests are refused until 00:00 UTC."),
            );
        } else if self.cfg.budget_warn_at > 0.0 && before < warn_at && after >= warn_at {
            self.emit(
                &format!("budget:{key_id}:{day}:warn"),
                "budget",
                Severity::Warning,
                format!("Key \"{key_name}\" is at {:.0}% of its daily budget", after / budget * 100.0),
                format!("${after:.4} of ${budget:.2} spent today."),
            );
        }
    }

    pub fn on_cooldown(&self, target: &str, secs: u64, reason: &str) {
        if !self.cfg.cooldowns {
            return;
        }
        self.emit(
            &format!("cooldown:{target}"),
            "cooldown",
            Severity::Warning,
            format!("{target} is cooling down for {secs}s"),
            format!("{reason}. Requests skip this target until it recovers."),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_alerts_fire_once_per_level() {
        let a = Alerts::new(&AlertsCfg::default(), None);
        a.on_spend(1, "app", 1.0, 0.0, 0.5);
        assert!(a.recent().is_empty());
        a.on_spend(1, "app", 1.0, 0.5, 0.85);
        a.on_spend(1, "app", 1.0, 0.85, 0.9);
        assert_eq!(a.recent().len(), 1);
        assert_eq!(a.recent()[0].severity, Severity::Warning);
        a.on_spend(1, "app", 1.0, 0.9, 1.2);
        assert_eq!(a.recent()[0].severity, Severity::Critical);
        assert_eq!(a.recent().len(), 2);
        // A single charge that jumps past both levels reports the worse one.
        a.on_spend(2, "other", 1.0, 0.0, 3.0);
        assert_eq!(a.recent()[0].severity, Severity::Critical);
        assert_eq!(a.recent().len(), 3);
    }

    #[test]
    fn repeated_cooldowns_are_deduplicated() {
        let a = Alerts::new(&AlertsCfg::default(), None);
        a.on_cooldown("groq/m", 30, "rate limited by provider");
        a.on_cooldown("groq/m", 30, "rate limited by provider");
        a.on_cooldown("gemini/m", 30, "rate limited by provider");
        assert_eq!(a.recent().len(), 2);
        let off = Alerts::new(&AlertsCfg { cooldowns: false, ..Default::default() }, None);
        off.on_cooldown("groq/m", 30, "x");
        assert!(off.recent().is_empty());
    }
}

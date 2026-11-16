//! Per-target quota tracking and health.
//!
//! Two sources of truth are combined:
//! 1. Local sliding-window counters (requests and tokens per minute / day),
//!    checked against the limits declared in config before dispatching.
//! 2. What the provider reports: `429` responses with `retry-after`, and
//!    `x-ratelimit-remaining-*` headers on successful responses.
//!
//! Repeated failures open a short circuit-breaker so a dead provider is
//! skipped instead of adding latency to every request.

use crate::config::Limits;
use crate::providers::RateSnapshot;
use crate::types::now_secs;
use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;
use std::time::Duration;

const DEFAULT_COOLDOWN_SECS: u64 = 60;
const MAX_COOLDOWN_SECS: u64 = 3600;
const BREAKER_THRESHOLD: u32 = 3;

struct Sliding {
    span: u64,
    bucket: u64,
    /// (bucket index, requests, tokens)
    buckets: VecDeque<(u64, u64, u64)>,
}

impl Sliding {
    fn new(span: u64, bucket: u64) -> Self {
        Self { span, bucket, buckets: VecDeque::new() }
    }

    fn prune(&mut self, now: u64) {
        let oldest = now.saturating_sub(self.span) / self.bucket;
        while self.buckets.front().is_some_and(|b| b.0 <= oldest) {
            self.buckets.pop_front();
        }
    }

    fn add_at(&mut self, at: u64, reqs: u64, toks: u64) {
        let idx = at / self.bucket;
        match self.buckets.back_mut() {
            Some(b) if b.0 == idx => {
                b.1 += reqs;
                b.2 += toks;
            }
            _ => self.buckets.push_back((idx, reqs, toks)),
        }
    }

    fn totals(&mut self, now: u64) -> (u64, u64) {
        self.prune(now);
        self.buckets.iter().fold((0, 0), |(r, t), b| (r + b.1, t + b.2))
    }
}

struct TargetState {
    minute: Sliding,
    day: Sliding,
    cooldown_until: u64,
    cooldown_reason: String,
    failures: u32,
    last_error: Option<String>,
    last_ok: Option<u64>,
    remaining_requests: Option<u64>,
    remaining_tokens: Option<u64>,
}

impl Default for TargetState {
    fn default() -> Self {
        Self {
            minute: Sliding::new(60, 1),
            day: Sliding::new(86_400, 60),
            cooldown_until: 0,
            cooldown_reason: String::new(),
            failures: 0,
            last_error: None,
            last_ok: None,
            remaining_requests: None,
            remaining_tokens: None,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct TargetView {
    pub rpm_used: u64,
    pub tpm_used: u64,
    pub rpd_used: u64,
    pub tpd_used: u64,
    pub cooldown_secs: u64,
    pub cooldown_reason: String,
    pub failures: u32,
    pub last_error: Option<String>,
    pub last_ok: Option<u64>,
    pub remaining_requests: Option<u64>,
    pub remaining_tokens: Option<u64>,
}

#[derive(Default)]
pub struct QuotaTracker {
    inner: Mutex<HashMap<String, TargetState>>,
}

impl QuotaTracker {
    fn with<R>(&self, target: &str, f: impl FnOnce(&mut TargetState) -> R) -> R {
        let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        f(g.entry(target.to_string()).or_default())
    }

    /// Can this target take a request of roughly `est_tokens` right now?
    pub fn check(&self, target: &str, limits: &Limits, est_tokens: u64) -> Result<(), String> {
        let now = now_secs();
        self.with(target, |s| {
            if s.cooldown_until > now {
                return Err(format!("cooling down {}s ({})", s.cooldown_until - now, s.cooldown_reason));
            }
            let (rpm, tpm) = s.minute.totals(now);
            let (rpd, tpd) = s.day.totals(now);
            if let Some(l) = limits.rpm && rpm >= l {
                return Err(format!("requests/minute quota reached ({rpm}/{l})"));
            }
            if let Some(l) = limits.rpd && rpd >= l {
                return Err(format!("requests/day quota reached ({rpd}/{l})"));
            }
            if let Some(l) = limits.tpm && tpm + est_tokens > l {
                return Err(format!("tokens/minute quota reached ({tpm}/{l})"));
            }
            if let Some(l) = limits.tpd && tpd + est_tokens > l {
                return Err(format!("tokens/day quota reached ({tpd}/{l})"));
            }
            Ok(())
        })
    }

    /// Count a dispatched request.
    pub fn reserve(&self, target: &str) {
        let now = now_secs();
        self.with(target, |s| {
            s.minute.add_at(now, 1, 0);
            s.day.add_at(now, 1, 0);
        });
    }

    pub fn record_tokens(&self, target: &str, tokens: u64) {
        let now = now_secs();
        self.with(target, |s| {
            s.minute.add_at(now, 0, tokens);
            s.day.add_at(now, 0, tokens);
        });
    }

    /// Restore counters from the telemetry log after a restart.
    pub fn seed(&self, target: &str, at_secs: u64, reqs: u64, tokens: u64) {
        let now = now_secs();
        self.with(target, |s| {
            s.day.add_at(at_secs, reqs, tokens);
            if now.saturating_sub(at_secs) < 60 {
                s.minute.add_at(at_secs, reqs, tokens);
            }
        });
    }

    pub fn on_success(&self, target: &str, snap: &RateSnapshot) {
        let now = now_secs();
        self.with(target, |s| {
            s.failures = 0;
            s.last_error = None;
            s.last_ok = Some(now);
            s.remaining_requests = snap.remaining_requests;
            s.remaining_tokens = snap.remaining_tokens;
            // The provider says we are out: stop before earning a 429.
            let exhausted = [(snap.remaining_requests, snap.reset_requests), (snap.remaining_tokens, snap.reset_tokens)]
                .into_iter()
                .filter_map(|(rem, reset)| (rem == Some(0)).then_some(reset).flatten())
                .max();
            if let Some(reset) = exhausted {
                let secs = reset.as_secs().clamp(1, MAX_COOLDOWN_SECS);
                s.cooldown_until = now + secs;
                s.cooldown_reason = "provider reports quota exhausted".into();
            }
        });
    }

    pub fn on_rate_limited(&self, target: &str, retry_after: Option<Duration>, message: &str) {
        let now = now_secs();
        self.with(target, |s| {
            let secs = retry_after.map(|d| d.as_secs().max(1)).unwrap_or(DEFAULT_COOLDOWN_SECS).min(MAX_COOLDOWN_SECS);
            s.cooldown_until = now + secs;
            s.cooldown_reason = "rate limited by provider".into();
            s.last_error = Some(format!("429: {message}"));
            s.remaining_requests = Some(0);
        });
    }

    pub fn on_failure(&self, target: &str, message: &str) {
        let now = now_secs();
        self.with(target, |s| {
            s.failures += 1;
            s.last_error = Some(message.to_string());
            if s.failures >= BREAKER_THRESHOLD {
                let exp = (s.failures - BREAKER_THRESHOLD).min(4);
                s.cooldown_until = now + (30u64 << exp).min(300);
                s.cooldown_reason = format!("{} consecutive failures", s.failures);
            }
        });
    }

    pub fn view(&self, target: &str) -> TargetView {
        let now = now_secs();
        self.with(target, |s| {
            let (rpm_used, tpm_used) = s.minute.totals(now);
            let (rpd_used, tpd_used) = s.day.totals(now);
            TargetView {
                rpm_used,
                tpm_used,
                rpd_used,
                tpd_used,
                cooldown_secs: s.cooldown_until.saturating_sub(now),
                cooldown_reason: s.cooldown_reason.clone(),
                failures: s.failures,
                last_error: s.last_error.clone(),
                last_ok: s.last_ok,
                remaining_requests: s.remaining_requests,
                remaining_tokens: s.remaining_tokens,
            }
        })
    }

    /// Clear cooldown and failure state (admin action).
    pub fn reset(&self, target: &str) {
        self.with(target, |s| {
            s.cooldown_until = 0;
            s.cooldown_reason.clear();
            s.failures = 0;
            s.last_error = None;
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enforces_rpm_and_cooldown() {
        let q = QuotaTracker::default();
        let limits = Limits { rpm: Some(2), ..Default::default() };
        assert!(q.check("a/b", &limits, 10).is_ok());
        q.reserve("a/b");
        q.reserve("a/b");
        assert!(q.check("a/b", &limits, 10).unwrap_err().contains("requests/minute"));

        q.on_rate_limited("c/d", Some(Duration::from_secs(30)), "slow down");
        assert!(q.check("c/d", &Limits::default(), 0).unwrap_err().contains("cooling down"));
        q.reset("c/d");
        assert!(q.check("c/d", &Limits::default(), 0).is_ok());
    }

    #[test]
    fn token_budget_counts_estimate() {
        let q = QuotaTracker::default();
        let limits = Limits { tpm: Some(100), ..Default::default() };
        q.record_tokens("a/b", 90);
        assert!(q.check("a/b", &limits, 5).is_ok());
        assert!(q.check("a/b", &limits, 20).is_err());
    }

    #[test]
    fn breaker_opens_after_repeated_failures() {
        let q = QuotaTracker::default();
        for _ in 0..3 {
            q.on_failure("a/b", "boom");
        }
        assert!(q.check("a/b", &Limits::default(), 0).is_err());
        q.on_success("a/b", &RateSnapshot::default());
        assert_eq!(q.view("a/b").failures, 0);
    }
}

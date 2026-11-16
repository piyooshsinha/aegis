//! Client authentication and per-key limits.
//!
//! Client keys are shown once at creation; only their SHA-256 hash is stored.
//! Provider keys are a separate matter: they come from environment variables
//! and are never written to the database or to logs.

use crate::db::Db;
use crate::types::now_ms;
use axum::http::HeaderMap;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use subtle::ConstantTimeEq;

pub const KEY_PREFIX: &str = "aegis_sk_";

#[derive(Debug, Clone)]
pub struct Identity {
    pub key_id: Option<i64>,
    pub name: String,
    pub allow_paid: bool,
    pub rpm: Option<u64>,
    pub daily_budget_usd: Option<f64>,
    /// Empty means every model and alias is allowed.
    pub allowed_models: Vec<String>,
}

impl Identity {
    pub fn anonymous() -> Self {
        Self { key_id: None, name: "anonymous".into(), allow_paid: false, rpm: Some(30), daily_budget_usd: None, allowed_models: vec![] }
    }

    pub fn admin_playground() -> Self {
        Self { key_id: None, name: "playground".into(), allow_paid: true, rpm: None, daily_budget_usd: None, allowed_models: vec![] }
    }

    pub fn may_use(&self, model: &str) -> bool {
        self.allowed_models.is_empty() || self.allowed_models.iter().any(|m| m == model)
    }

    /// Cache partition for this client.
    pub fn scope(&self) -> String {
        match self.key_id {
            Some(id) => format!("key:{id}"),
            None => format!("anon:{}", self.name),
        }
    }
}

pub fn hash_key(key: &str) -> String {
    let out = Sha256::digest(key.as_bytes());
    hex::encode(&out[..])
}

/// 244 bits from the OS random source, via two v4 UUIDs.
pub fn generate_key() -> String {
    format!("{KEY_PREFIX}{}{}", uuid::Uuid::new_v4().simple(), uuid::Uuid::new_v4().simple())
}

pub fn random_token() -> String {
    format!("{}{}", uuid::Uuid::new_v4().simple(), uuid::Uuid::new_v4().simple())
}

pub fn constant_time_eq(a: &str, b: &str) -> bool {
    a.len() == b.len() && bool::from(a.as_bytes().ct_eq(b.as_bytes()))
}

/// `Authorization: Bearer <key>` or `x-api-key: <key>`.
pub fn presented_key(h: &HeaderMap) -> Option<String> {
    if let Some(v) = h.get("authorization").and_then(|v| v.to_str().ok()) {
        let v = v.trim();
        if let Some(rest) = v.strip_prefix("Bearer ").or_else(|| v.strip_prefix("bearer ")) {
            return Some(rest.trim().to_string());
        }
    }
    h.get("x-api-key").and_then(|v| v.to_str().ok()).map(|v| v.trim().to_string())
}

fn utc_day(ms: i64) -> i64 {
    ms / 86_400_000
}

#[derive(Default)]
pub struct KeyStore {
    by_hash: RwLock<HashMap<String, Arc<Identity>>>,
    windows: Mutex<HashMap<String, VecDeque<Instant>>>,
    spend: Mutex<HashMap<i64, (i64, f64)>>,
}

impl KeyStore {
    /// Load active keys and today's spend from the database.
    pub async fn reload(&self, db: &Db) -> anyhow::Result<()> {
        let rows = db.list_keys().await?;
        let map: HashMap<String, Arc<Identity>> = rows
            .into_iter()
            .filter(|k| k.revoked_at.is_none())
            .map(|k| {
                let ident = Identity {
                    key_id: Some(k.id),
                    name: k.name,
                    allow_paid: k.allow_paid,
                    rpm: k.rpm.and_then(|v| u64::try_from(v).ok()),
                    daily_budget_usd: k.daily_budget_usd,
                    allowed_models: k.allowed_models.split(',').map(str::trim).filter(|s| !s.is_empty()).map(str::to_string).collect(),
                };
                (k.key_hash, Arc::new(ident))
            })
            .collect();
        *self.by_hash.write().unwrap_or_else(|e| e.into_inner()) = map;

        let now = now_ms();
        let today = utc_day(now);
        let spent = db.spend_by_key(today * 86_400_000).await?;
        let mut g = self.spend.lock().unwrap_or_else(|e| e.into_inner());
        for (id, usd) in spent {
            g.insert(id, (today, usd));
        }
        Ok(())
    }

    pub fn authenticate(&self, presented: &str) -> Option<Arc<Identity>> {
        if !presented.starts_with(KEY_PREFIX) {
            return None;
        }
        // Lookup is by digest, so timing reveals nothing about stored keys.
        self.by_hash.read().unwrap_or_else(|e| e.into_inner()).get(&hash_key(presented)).cloned()
    }

    pub fn spent_today(&self, key_id: i64) -> f64 {
        let today = utc_day(now_ms());
        match self.spend.lock().unwrap_or_else(|e| e.into_inner()).get(&key_id) {
            Some((day, usd)) if *day == today => *usd,
            _ => 0.0,
        }
    }

    pub fn add_spend(&self, key_id: Option<i64>, usd: f64) {
        let Some(id) = key_id else { return };
        if usd <= 0.0 {
            return;
        }
        let today = utc_day(now_ms());
        let mut g = self.spend.lock().unwrap_or_else(|e| e.into_inner());
        let e = g.entry(id).or_insert((today, 0.0));
        if e.0 != today {
            *e = (today, 0.0);
        }
        e.1 += usd;
    }

    /// Enforce the client's own request rate and daily budget.
    pub fn check_limits(&self, ident: &Identity) -> Result<(), String> {
        if let (Some(id), Some(budget)) = (ident.key_id, ident.daily_budget_usd) {
            let spent = self.spent_today(id);
            if spent >= budget {
                return Err(format!("daily budget exhausted (${spent:.4} of ${budget:.2})"));
            }
        }
        if let Some(rpm) = ident.rpm {
            let now = Instant::now();
            let mut g = self.windows.lock().unwrap_or_else(|e| e.into_inner());
            let w = g.entry(ident.scope()).or_default();
            while w.front().is_some_and(|t| now.duration_since(*t) >= Duration::from_secs(60)) {
                w.pop_front();
            }
            if w.len() as u64 >= rpm {
                return Err(format!("key rate limit reached ({rpm} requests/minute)"));
            }
            w.push_back(now);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_are_unique_and_hash_stably() {
        let a = generate_key();
        let b = generate_key();
        assert_ne!(a, b);
        assert!(a.starts_with(KEY_PREFIX) && a.len() == KEY_PREFIX.len() + 64);
        assert_eq!(hash_key(&a), hash_key(&a));
        assert_ne!(hash_key(&a), hash_key(&b));
    }

    #[test]
    fn per_key_rate_limit() {
        let ks = KeyStore::default();
        let ident = Identity { rpm: Some(2), ..Identity::anonymous() };
        assert!(ks.check_limits(&ident).is_ok());
        assert!(ks.check_limits(&ident).is_ok());
        assert!(ks.check_limits(&ident).is_err());
    }

    #[test]
    fn budget_blocks_once_spent() {
        let ks = KeyStore::default();
        let ident = Identity { key_id: Some(7), rpm: None, daily_budget_usd: Some(0.01), ..Identity::anonymous() };
        assert!(ks.check_limits(&ident).is_ok());
        ks.add_spend(Some(7), 0.02);
        assert!(ks.check_limits(&ident).is_err());
    }
}

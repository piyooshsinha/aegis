//! Exact-match response cache.
//!
//! Caching a sampled answer makes a non-deterministic request deterministic,
//! which is rarely what a caller wants. So a request is only cached when it
//! is explicitly deterministic (temperature at or below the configured
//! ceiling, or a `seed`), or when the caller opts in with
//! `x-aegis-cache: force`. Entries expire after a TTL and can be purged.

use crate::config::{CacheCfg, CacheScope};
use crate::types::{ChatRequest, Usage};
use axum::http::HeaderMap;
use moka::future::Cache;
use serde_json::Value;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheMode {
    Default,
    Bypass,
    Force,
}

impl CacheMode {
    pub fn from_headers(h: &HeaderMap) -> Self {
        let get = |n: &str| h.get(n).and_then(|v| v.to_str().ok()).map(str::to_ascii_lowercase);
        match get("x-aegis-cache").as_deref() {
            Some("bypass") | Some("off") | Some("no-store") => return Self::Bypass,
            Some("force") => return Self::Force,
            _ => {}
        }
        match get("cache-control") {
            Some(cc) if cc.contains("no-cache") || cc.contains("no-store") => Self::Bypass,
            _ => Self::Default,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CachedEntry {
    pub body: Value,
    pub provider: String,
    pub model: String,
    pub tier: String,
    pub usage: Usage,
    pub cost_usd: f64,
    pub stored_at: i64,
}

#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub entries: u64,
    pub hits: u64,
    pub misses: u64,
    pub stores: u64,
    pub skipped: u64,
}

pub struct ResponseCache {
    cfg: CacheCfg,
    inner: Cache<String, Arc<CachedEntry>>,
    hits: AtomicU64,
    misses: AtomicU64,
    stores: AtomicU64,
    skipped: AtomicU64,
}

impl ResponseCache {
    pub fn new(cfg: &CacheCfg) -> Self {
        let inner = Cache::builder()
            .max_capacity(cfg.max_entries.max(1))
            .time_to_live(Duration::from_secs(cfg.ttl_secs.max(1)))
            .build();
        Self {
            cfg: cfg.clone(),
            inner,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            stores: AtomicU64::new(0),
            skipped: AtomicU64::new(0),
        }
    }

    pub fn cfg(&self) -> &CacheCfg {
        &self.cfg
    }

    /// Returns the cache key if this request may be served from / stored in
    /// the cache, or `None` if it must go upstream.
    pub fn key_for(&self, req: &ChatRequest, mode: CacheMode, key_scope: &str) -> Option<String> {
        if !self.cfg.enabled || mode == CacheMode::Bypass {
            self.skipped.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        let deterministic = match req.temperature {
            Some(t) => t <= self.cfg.max_temperature,
            None => self.cfg.cache_when_temperature_unset,
        } || req.seed.is_some();
        if !(deterministic || mode == CacheMode::Force) {
            self.skipped.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        let scope = match self.cfg.scope {
            CacheScope::Global => "global",
            CacheScope::PerKey => key_scope,
        };
        Some(req.fingerprint(scope))
    }

    pub async fn get(&self, key: &str) -> Option<Arc<CachedEntry>> {
        let hit = self.inner.get(key).await;
        if hit.is_some() {
            self.hits.fetch_add(1, Ordering::Relaxed);
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
        }
        hit
    }

    pub async fn put(&self, key: String, entry: CachedEntry) {
        self.stores.fetch_add(1, Ordering::Relaxed);
        self.inner.insert(key, Arc::new(entry)).await;
    }

    pub async fn purge(&self) {
        self.inner.invalidate_all();
        self.inner.run_pending_tasks().await;
    }

    pub async fn stats(&self) -> CacheStats {
        self.inner.run_pending_tasks().await;
        CacheStats {
            entries: self.inner.entry_count(),
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            stores: self.stores.load(Ordering::Relaxed),
            skipped: self.skipped.load(Ordering::Relaxed),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn req(v: Value) -> ChatRequest {
        serde_json::from_value(v).unwrap()
    }

    #[test]
    fn only_deterministic_requests_are_cacheable() {
        let c = ResponseCache::new(&CacheCfg::default());
        let base = json!({ "model": "m", "messages": [{ "role": "user", "content": "hi" }] });
        let mut hot = base.clone();
        hot["temperature"] = json!(0.9);
        let mut cold = base.clone();
        cold["temperature"] = json!(0.0);

        assert!(c.key_for(&req(base.clone()), CacheMode::Default, "k").is_none());
        assert!(c.key_for(&req(hot.clone()), CacheMode::Default, "k").is_none());
        assert!(c.key_for(&req(cold.clone()), CacheMode::Default, "k").is_some());
        assert!(c.key_for(&req(hot), CacheMode::Force, "k").is_some());
        assert!(c.key_for(&req(cold), CacheMode::Bypass, "k").is_none());
    }

    #[test]
    fn key_ignores_stream_flag_and_key_order_but_not_scope() {
        let c = ResponseCache::new(&CacheCfg::default());
        let a = req(json!({ "model": "m", "temperature": 0, "messages": [{ "role": "user", "content": "hi" }] }));
        let b = req(json!({ "stream": true, "messages": [{ "content": "hi", "role": "user" }], "temperature": 0, "model": "m" }));
        let ka = c.key_for(&a, CacheMode::Default, "k1").unwrap();
        assert_eq!(ka, c.key_for(&b, CacheMode::Default, "k1").unwrap());
        assert_ne!(ka, c.key_for(&a, CacheMode::Default, "k2").unwrap());
    }
}

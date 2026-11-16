//! Response cache: exact-match, with an optional semantic layer.
//!
//! Caching a sampled answer makes a non-deterministic request deterministic,
//! which is rarely what a caller wants. So a request is only cached when it
//! is explicitly deterministic (temperature at or below the configured
//! ceiling, or a `seed`), or when the caller opts in with
//! `x-aegis-cache: force`. Entries expire after a TTL and can be purged.

use crate::config::{CacheCfg, CacheScope, SemanticCfg};
use crate::types::{ChatRequest, Usage, now_ms};
use axum::http::HeaderMap;
use moka::future::Cache;
use serde_json::Value;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheMode {
    Default,
    Bypass,
    Force,
    /// Exact matches only: skip the semantic layer for this request.
    Exact,
}

impl CacheMode {
    pub fn from_headers(h: &HeaderMap) -> Self {
        let get = |n: &str| h.get(n).and_then(|v| v.to_str().ok()).map(str::to_ascii_lowercase);
        match get("x-aegis-cache").as_deref() {
            Some("bypass") | Some("off") | Some("no-store") => return Self::Bypass,
            Some("force") => return Self::Force,
            Some("exact") => return Self::Exact,
            _ => {}
        }
        match get("cache-control") {
            Some(cc) if cc.contains("no-cache") || cc.contains("no-store") => Self::Bypass,
            _ => Self::Default,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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
    /// Entry and the time it expires. The cache's own lifetime is an upper
    /// bound; entries restored after a restart keep their original expiry.
    inner: Cache<String, (Arc<CachedEntry>, i64)>,
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

    /// The partition label entries are filed under for this client.
    pub fn scope_for<'a>(&self, key_scope: &'a str) -> &'a str {
        match self.cfg.scope {
            CacheScope::Global => "global",
            CacheScope::PerKey => key_scope,
        }
    }

    pub async fn get(&self, key: &str) -> Option<Arc<CachedEntry>> {
        let hit = match self.inner.get(key).await {
            Some((entry, expires_ms)) if expires_ms > now_ms() => Some(entry),
            Some(_) => {
                self.inner.invalidate(key).await;
                None
            }
            None => None,
        };
        if hit.is_some() {
            self.hits.fetch_add(1, Ordering::Relaxed);
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
        }
        hit
    }

    /// Store an answer. Returns when it expires, for persisting.
    pub async fn put(&self, key: String, entry: CachedEntry) -> i64 {
        let expires_ms = now_ms() + (self.cfg.ttl_secs.max(1) as i64) * 1000;
        self.stores.fetch_add(1, Ordering::Relaxed);
        self.inner.insert(key, (Arc::new(entry), expires_ms)).await;
        expires_ms
    }

    /// Put back an entry saved by an earlier run. Expired ones are dropped.
    pub async fn restore(&self, key: String, entry: CachedEntry, expires_ms: i64) {
        if self.cfg.enabled && expires_ms > now_ms() {
            self.inner.insert(key, (Arc::new(entry), expires_ms)).await;
        }
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

// ---- semantic layer -------------------------------------------------------

struct SemEntry {
    partition: String,
    /// Unit-length embedding of the last user message.
    vector: Vec<f32>,
    entry: Arc<CachedEntry>,
    expires_ms: i64,
}

#[derive(Debug, Clone, Default)]
pub struct SemanticStats {
    pub entries: u64,
    pub hits: u64,
    pub misses: u64,
    pub stores: u64,
    /// Lookups skipped because the embedding call failed or timed out.
    pub embed_failures: u64,
}

/// Answers filed by the meaning of the last user message.
///
/// Two requests can only match if they are in the same partition: same
/// client scope, model, parameters, system prompt and earlier turns. Within a
/// partition, the stored answer whose message is most similar is returned if
/// its cosine similarity reaches the threshold. Lookup is a linear scan,
/// which is fast for the few thousand entries this is sized for.
pub struct SemanticIndex {
    cfg: SemanticCfg,
    ttl_ms: i64,
    entries: Mutex<VecDeque<SemEntry>>,
    hits: AtomicU64,
    misses: AtomicU64,
    stores: AtomicU64,
    embed_failures: AtomicU64,
}

fn normalize(mut v: Vec<f32>) -> Option<Vec<f32>> {
    let norm = v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
    if norm == 0.0 || !norm.is_finite() {
        return None;
    }
    for x in &mut v {
        *x = (*x as f64 / norm) as f32;
    }
    Some(v)
}

impl SemanticIndex {
    pub fn new(cache: &CacheCfg) -> Self {
        Self {
            cfg: cache.semantic.clone(),
            ttl_ms: (cache.ttl_secs.max(1) as i64) * 1000,
            entries: Mutex::new(VecDeque::new()),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            stores: AtomicU64::new(0),
            embed_failures: AtomicU64::new(0),
        }
    }

    pub fn cfg(&self) -> &SemanticCfg {
        &self.cfg
    }

    pub fn note_embed_failure(&self) {
        self.embed_failures.fetch_add(1, Ordering::Relaxed);
    }

    /// The closest stored answer in this partition, with its similarity, if
    /// it is close enough.
    pub fn lookup(&self, partition: &str, vector: &[f32]) -> Option<(Arc<CachedEntry>, f64)> {
        let query = normalize(vector.to_vec())?;
        let now = now_ms();
        let mut g = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        g.retain(|e| e.expires_ms > now);
        let best = g
            .iter()
            .filter(|e| e.partition == partition && e.vector.len() == query.len())
            .map(|e| (e, e.vector.iter().zip(&query).map(|(a, b)| (*a as f64) * (*b as f64)).sum::<f64>()))
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .filter(|(_, sim)| *sim >= self.cfg.threshold)
            .map(|(e, sim)| (e.entry.clone(), sim));
        match &best {
            Some(_) => self.hits.fetch_add(1, Ordering::Relaxed),
            None => self.misses.fetch_add(1, Ordering::Relaxed),
        };
        best
    }

    /// File an answer. Returns the unit-length vector and expiry time that
    /// were stored, so the caller can persist the same entry.
    pub fn insert(&self, partition: String, vector: Vec<f32>, entry: CachedEntry) -> Option<(Vec<f32>, i64)> {
        let vector = normalize(vector)?;
        let expires_ms = now_ms() + self.ttl_ms;
        self.stores.fetch_add(1, Ordering::Relaxed);
        self.push(partition, vector.clone(), entry, expires_ms);
        Some((vector, expires_ms))
    }

    /// Put back an entry saved by an earlier run. Expired ones are dropped.
    pub fn restore(&self, partition: String, vector: Vec<f32>, entry: CachedEntry, expires_ms: i64) {
        if expires_ms > now_ms() && !vector.is_empty() {
            self.push(partition, vector, entry, expires_ms);
        }
    }

    fn push(&self, partition: String, vector: Vec<f32>, entry: CachedEntry, expires_ms: i64) {
        let mut g = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        g.push_back(SemEntry { partition, vector, entry: Arc::new(entry), expires_ms });
        while g.len() > self.cfg.max_entries.max(1) {
            g.pop_front();
        }
    }

    pub fn purge(&self) {
        self.entries.lock().unwrap_or_else(|e| e.into_inner()).clear();
    }

    pub fn stats(&self) -> SemanticStats {
        SemanticStats {
            entries: self.entries.lock().unwrap_or_else(|e| e.into_inner()).len() as u64,
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            stores: self.stores.load(Ordering::Relaxed),
            embed_failures: self.embed_failures.load(Ordering::Relaxed),
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

    fn entry(tag: &str) -> CachedEntry {
        CachedEntry { body: json!({ "tag": tag }), provider: "p".into(), model: "m".into(), tier: "local".into(), usage: Usage::default(), cost_usd: 0.0, stored_at: 0 }
    }

    #[test]
    fn semantic_lookup_needs_same_partition_and_enough_similarity() {
        let idx = SemanticIndex::new(&CacheCfg::default());
        idx.insert("a".into(), vec![1.0, 0.0, 0.0], entry("x"));
        idx.insert("a".into(), vec![0.0, 1.0, 0.0], entry("y"));

        // Nearly the same direction as the first entry; length is irrelevant.
        let (hit, sim) = idx.lookup("a", &[5.0, 0.2, 0.0]).unwrap();
        assert_eq!(hit.body["tag"], "x");
        assert!(sim > 0.99);
        // Halfway between the two entries is not close enough to either.
        assert!(idx.lookup("a", &[1.0, 1.0, 0.0]).is_none());
        // Same vector, different conversation or client.
        assert!(idx.lookup("b", &[1.0, 0.0, 0.0]).is_none());
        // Vectors from a different embedding model never match.
        assert!(idx.lookup("a", &[1.0, 0.0]).is_none());
        assert!(idx.lookup("a", &[0.0, 0.0, 0.0]).is_none());

        idx.purge();
        assert!(idx.lookup("a", &[1.0, 0.0, 0.0]).is_none());
    }

    #[test]
    fn semantic_index_evicts_oldest() {
        let mut cfg = CacheCfg::default();
        cfg.semantic.max_entries = 2;
        let idx = SemanticIndex::new(&cfg);
        idx.insert("a".into(), vec![1.0, 0.0, 0.0], entry("1"));
        idx.insert("a".into(), vec![0.0, 1.0, 0.0], entry("2"));
        idx.insert("a".into(), vec![0.0, 0.0, 1.0], entry("3"));
        assert_eq!(idx.stats().entries, 2);
        assert!(idx.lookup("a", &[1.0, 0.0, 0.0]).is_none());
        assert!(idx.lookup("a", &[0.0, 0.0, 1.0]).is_some());
    }
}

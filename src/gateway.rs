//! The request pipeline: authenticate, check the cache, pick a provider,
//! dispatch (with fallback), account for usage, respond.

use crate::auth::{Identity, KeyStore, presented_key};
use crate::alerts::Alerts;
use crate::cache::{CacheMode, CachedEntry, ResponseCache, SemanticIndex};
use crate::config::{Config, ModelCfg, ProviderCfg, Tier};
use crate::db::{Db, RequestRecord};
use crate::providers::{Adapter, ChunkStream, ProviderError, RateSnapshot};
use crate::quota::QuotaTracker;
use crate::secrets::Sealer;
use crate::types::{ChatRequest, Usage, chunk_json, completion_json, est_tokens, new_completion_id, now_ms, usage_chunk_json};
use axum::Json;
use axum::extract::State;
use axum::http::{HeaderMap, HeaderName, HeaderValue, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use futures::StreamExt;
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::convert::Infallible;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::mpsc::UnboundedSender;

pub struct AppState {
    pub cfg: Config,
    pub adapters: HashMap<String, Adapter>,
    pub quota: QuotaTracker,
    pub cache: ResponseCache,
    pub semantic: SemanticIndex,
    pub alerts: Alerts,
    pub db: Db,
    pub keys: KeyStore,
    pub telemetry: UnboundedSender<RequestRecord>,
    pub admin_token: String,
    /// Present when `AEGIS_SECRET_KEY` is set; needed to store provider keys.
    pub sealer: Option<Sealer>,
    /// provider -> (hint, updated_at ms) for keys stored from the dashboard.
    pub stored_keys: Mutex<HashMap<String, (String, i64)>>,
    pub sessions: Mutex<HashSet<String>>,
    pub started: Instant,
}

pub type Shared = Arc<AppState>;

#[derive(Debug, Clone)]
pub struct Target {
    pub provider: usize,
    pub model: usize,
    pub key: String,
}

impl AppState {
    pub async fn build(cfg: Config, admin_token: String, sealer: Option<Sealer>) -> anyhow::Result<Shared> {
        let db = Db::open(&cfg.server.db_path).await?;
        let mut adapters = HashMap::new();
        for p in &cfg.providers {
            adapters.insert(p.name.clone(), Adapter::new(p)?);
        }
        // Keys saved from the dashboard override the environment.
        let mut stored_keys = HashMap::new();
        for row in db.list_provider_keys().await? {
            let Some(adapter) = adapters.get(&row.provider) else { continue };
            let Some(sealer) = &sealer else {
                tracing::warn!(provider = %row.provider, "stored provider key ignored: {} is not set", crate::secrets::SECRET_ENV);
                continue;
            };
            match sealer.open(&row.provider, &row.nonce, &row.ciphertext) {
                Ok(key) => {
                    adapter.set_stored_key(Some(key));
                    stored_keys.insert(row.provider, (row.hint, row.updated_at));
                }
                Err(e) => tracing::warn!("{e}"),
            }
        }
        let keys = KeyStore::default();
        keys.reload(&db).await?;

        // Quota counters survive restarts by replaying the last day of the log.
        let quota = QuotaTracker::default();
        for (target, minute, reqs, tokens) in db.usage_by_minute(now_ms() - 86_400_000).await? {
            quota.seed(&target, (minute.max(0) as u64) * 60, reqs.max(0) as u64, tokens.max(0) as u64);
        }

        let telemetry = db.spawn_writer(cfg.server.retention_days);
        let cache = ResponseCache::new(&cfg.cache);
        // Both cache layers are rebuilt from the database, so a restart or
        // deploy does not throw away what it has learned.
        if cfg.cache.enabled {
            let mut restored = 0usize;
            for (key, entry, expires_ms) in db.load_cached().await? {
                if let Ok(entry) = serde_json::from_str::<CachedEntry>(&entry) {
                    cache.restore(key, entry, expires_ms).await;
                    restored += 1;
                }
            }
            tracing::info!(restored, "response cache loaded");
        }
        let semantic = SemanticIndex::new(&cfg.cache);
        if cfg.cache.enabled && cfg.cache.semantic.enabled {
            let mut restored = 0usize;
            for (partition, vector, entry, expires_ms) in db.load_semantic(&cfg.cache.semantic.embedding_model).await? {
                if let Ok(entry) = serde_json::from_str::<CachedEntry>(&entry) {
                    semantic.restore(partition, vector, entry, expires_ms);
                    restored += 1;
                }
            }
            tracing::info!(restored, "semantic cache loaded");
        }
        let webhook = std::env::var(&cfg.alerts.webhook_url_env).ok().map(|v| v.trim().to_string()).filter(|v| !v.is_empty());
        let alerts = Alerts::new(&cfg.alerts, webhook);
        let state = Arc::new(Self {
            cfg,
            adapters,
            quota,
            cache,
            semantic,
            alerts,
            db,
            keys,
            telemetry,
            admin_token,
            sealer,
            stored_keys: Mutex::new(stored_keys),
            sessions: Mutex::new(HashSet::new()),
            started: Instant::now(),
        });

        // A cold embedding model can take longer to load than a lookup is
        // allowed to wait, so load it now rather than on the first request.
        if state.cfg.cache.enabled && state.cfg.cache.semantic.enabled {
            let warm = state.clone();
            tokio::spawn(async move {
                warm.warm_embedding_model().await;
                let every = warm.cfg.cache.semantic.keep_warm_secs;
                if every > 0 {
                    let mut tick = tokio::time::interval(Duration::from_secs(every.max(10)));
                    tick.tick().await;
                    loop {
                        tick.tick().await;
                        warm.warm_embedding_model().await;
                    }
                }
            });
        }
        Ok(state)
    }

    /// One throwaway embedding, with a generous timeout, to get the model
    /// loaded. Returns whether it answered.
    pub async fn warm_embedding_model(&self) -> bool {
        let Some(t) = self.target(&self.semantic.cfg().embedding_model) else { return false };
        let (p, m) = self.parts(&t);
        let Some(adapter) = self.adapters.get(&p.name) else { return false };
        if !p.api_key_env.is_empty() && adapter.api_key().is_none() {
            return false;
        }
        let inputs = ["warm up".to_string()];
        let started = Instant::now();
        match tokio::time::timeout(Duration::from_secs(120), adapter.embed(m, &inputs, None)).await {
            Ok(Ok(_)) => {
                tracing::debug!(target = %t.key, ms = started.elapsed().as_millis() as u64, "embedding model is warm");
                true
            }
            Ok(Err(e)) => {
                tracing::warn!(target = %t.key, "semantic cache: embedding model is not ready: {e}");
                false
            }
            Err(_) => {
                tracing::warn!(target = %t.key, "semantic cache: embedding model did not answer within 120s");
                false
            }
        }
    }

    pub fn target(&self, key: &str) -> Option<Target> {
        self.cfg.find_target(key).map(|(provider, model)| Target { provider, model, key: key.to_string() })
    }

    pub fn parts(&self, t: &Target) -> (&ProviderCfg, &ModelCfg) {
        let p = &self.cfg.providers[t.provider];
        (p, &p.models[t.model])
    }

    /// A model name is, in order: a route alias, an explicit `provider/model`,
    /// or a bare model id served by one or more providers.
    pub fn resolve(&self, model: &str) -> Vec<Target> {
        if let Some(route) = self.cfg.routes.iter().find(|r| r.alias == model) {
            return route.targets.iter().filter_map(|t| self.target(t)).collect();
        }
        if let Some(t) = self.target(model) {
            return vec![t];
        }
        self.cfg
            .providers
            .iter()
            .filter(|p| p.models.iter().any(|m| m.id == model))
            .filter_map(|p| self.target(&format!("{}/{}", p.name, model)))
            .collect()
    }

    /// Encrypt and save a provider key, and start using it immediately.
    pub async fn set_provider_key(&self, provider: &str, key: &str) -> anyhow::Result<()> {
        let sealer = self.sealer.as_ref().ok_or_else(|| anyhow::anyhow!("{} is not set", crate::secrets::SECRET_ENV))?;
        let adapter = self.adapters.get(provider).ok_or_else(|| anyhow::anyhow!("unknown provider {provider}"))?;
        let sealed = sealer.seal(provider, key)?;
        let hint = crate::secrets::hint(key);
        self.db.set_provider_key(provider, &sealed.nonce, &sealed.ciphertext, &hint).await?;
        adapter.set_stored_key(Some(key.to_string()));
        self.stored_keys.lock().unwrap_or_else(|e| e.into_inner()).insert(provider.to_string(), (hint, now_ms()));
        self.clear_provider_state(provider);
        Ok(())
    }

    /// Remove a stored key; the provider goes back to its environment variable.
    pub async fn remove_provider_key(&self, provider: &str) -> anyhow::Result<()> {
        let adapter = self.adapters.get(provider).ok_or_else(|| anyhow::anyhow!("unknown provider {provider}"))?;
        self.db.delete_provider_key(provider).await?;
        adapter.set_stored_key(None);
        self.stored_keys.lock().unwrap_or_else(|e| e.into_inner()).remove(provider);
        self.clear_provider_state(provider);
        Ok(())
    }

    /// A new key deserves a fresh start: drop cooldowns earned by the old one.
    fn clear_provider_state(&self, provider: &str) {
        if let Some(p) = self.cfg.providers.iter().find(|p| p.name == provider) {
            for m in &p.models {
                self.quota.reset(&format!("{}/{}", p.name, m.id));
            }
        }
    }

    // ---- quota outcomes, with alerts when a cooldown begins ----

    pub fn note_success(&self, target: &str, snap: &RateSnapshot) {
        if let Some(c) = self.quota.on_success(target, snap) {
            self.alerts.on_cooldown(target, c.secs, &c.reason);
        }
    }

    pub fn note_error(&self, target: &str, err: &ProviderError) {
        let cooldown = match err {
            ProviderError::RateLimited { retry_after, message } => Some(self.quota.on_rate_limited(target, *retry_after, message)),
            ProviderError::Rejected { .. } => None,
            other => self.quota.on_failure(target, &other.to_string()),
        };
        if let Some(c) = cooldown {
            self.alerts.on_cooldown(target, c.secs, &c.reason);
        }
    }

    /// Add to a client key's spend for today and alert on budget thresholds.
    pub fn charge(&self, key_id: Option<i64>, key_name: &str, budget: Option<f64>, usd: f64) {
        let Some(id) = key_id else { return };
        if usd <= 0.0 {
            return;
        }
        let before = self.keys.spent_today(id);
        self.keys.add_spend(key_id, usd);
        if let Some(budget) = budget {
            self.alerts.on_spend(id, key_name, budget, before, before + usd);
        }
    }

    /// Cache an answer under its exact key and save it for the next start.
    async fn cache_store(&self, key: String, entry: CachedEntry) {
        let json = serde_json::to_string(&entry).unwrap_or_default();
        let expires_ms = self.cache.put(key.clone(), entry).await;
        if let Err(e) = self.db.insert_cached(&key, &json, expires_ms, self.cache.cfg().max_entries).await {
            tracing::warn!("cached response not saved: {e}");
        }
    }

    /// File an answer in the semantic index and save it for the next start.
    async fn semantic_store(&self, partition: String, vector: Vec<f32>, entry: CachedEntry) {
        let json = serde_json::to_string(&entry).unwrap_or_default();
        let Some((unit, expires_ms)) = self.semantic.insert(partition.clone(), vector, entry) else { return };
        let cfg = self.semantic.cfg();
        if let Err(e) = self.db.insert_semantic(&cfg.embedding_model, &partition, &unit, &json, expires_ms, cfg.max_entries).await {
            tracing::warn!("semantic cache entry not saved: {e}");
        }
    }

    /// Empty both cache layers, in memory and on disk.
    pub async fn purge_caches(&self) {
        self.cache.purge().await;
        self.semantic.purge();
        if let Err(e) = self.db.purge_cached().await {
            tracing::warn!("response cache not purged from the database: {e}");
        }
        if let Err(e) = self.db.purge_semantic().await {
            tracing::warn!("semantic cache not purged from the database: {e}");
        }
    }

    /// Embed one text with the semantic cache's embedding model. Failure is
    /// never fatal: the request simply goes upstream.
    async fn embed_for_cache(&self, text: &str) -> Option<Vec<f32>> {
        let sem = self.semantic.cfg();
        let t = self.target(&sem.embedding_model)?;
        let (p, m) = self.parts(&t);
        let adapter = self.adapters.get(&p.name)?;
        let tokens = est_tokens(text);
        let usable = (p.api_key_env.is_empty() || adapter.api_key().is_some()) && self.quota.check(&t.key, &m.limits, tokens).is_ok();
        if !usable {
            self.semantic.note_embed_failure();
            return None;
        }
        self.quota.reserve(&t.key);
        let inputs = [text.to_string()];
        let call = adapter.embed(m, &inputs, None);
        match tokio::time::timeout(Duration::from_millis(sem.timeout_ms.max(100)), call).await {
            Ok(Ok((mut out, snap))) => {
                self.note_success(&t.key, &snap);
                self.quota.record_tokens(&t.key, out.prompt_tokens.unwrap_or(tokens));
                out.vectors.pop()
            }
            Ok(Err(e)) => {
                tracing::debug!(target = %t.key, error = %e, "semantic cache: embedding failed");
                self.note_error(&t.key, &e);
                self.semantic.note_embed_failure();
                None
            }
            Err(_) => {
                tracing::debug!(target = %t.key, "semantic cache: embedding timed out");
                self.semantic.note_embed_failure();
                None
            }
        }
    }

    /// Send one tiny request to a provider to check its key and reachability.
    /// Returns a sentence for the operator either way.
    pub async fn test_provider(&self, provider: &str) -> Result<String, String> {
        let p = self.cfg.providers.iter().find(|p| p.name == provider).ok_or_else(|| format!("Unknown provider {provider}."))?;
        let adapter = self.adapters.get(&p.name).ok_or_else(|| format!("Unknown provider {provider}."))?;
        if !p.api_key_env.is_empty() && adapter.api_key().is_none() {
            return Err(format!("{provider} has no API key yet."));
        }
        let m = p.models.iter().find(|m| !m.has("embeddings")).or(p.models.first()).ok_or_else(|| format!("{provider} has no models configured."))?;
        let target = format!("{}/{}", p.name, m.id);
        let started = Instant::now();
        self.quota.reserve(&target);
        let result = if m.has("embeddings") {
            let inputs = ["ping".to_string()];
            tokio::time::timeout(Duration::from_secs(30), adapter.embed(m, &inputs, None)).await.map(|r| r.map(|(_, snap)| snap))
        } else {
            let req: ChatRequest = serde_json::from_value(json!({
                "model": m.id, "max_tokens": 16,
                "messages": [{ "role": "user", "content": "Reply with the single word: ok" }],
            }))
            .map_err(|e| e.to_string())?;
            tokio::time::timeout(Duration::from_secs(30), adapter.complete(m, &req)).await.map(|r| r.map(|(_, snap)| snap))
        };
        let ms = started.elapsed().as_millis();
        match result {
            Ok(Ok(snap)) => {
                self.note_success(&target, &snap);
                Ok(format!("{provider} is working: {} answered in {ms} ms.", m.id))
            }
            Ok(Err(ProviderError::Auth(msg))) => Err(format!("{provider} refused the API key: {msg}")),
            Ok(Err(ProviderError::RateLimited { message, .. })) => {
                Err(format!("{provider} accepted the key but is rate limiting it right now: {message}"))
            }
            Ok(Err(ProviderError::Rejected { status, message })) => {
                Err(format!("{provider} accepted the key but rejected the test request for {} ({status}): {message}", m.id))
            }
            Ok(Err(ProviderError::Unavailable(msg))) => Err(format!("{provider} could not be reached or failed: {msg}")),
            Err(_) => Err(format!("{provider} did not answer within 30 seconds.")),
        }
    }

    /// Why this target cannot serve this request, if it cannot.
    fn ineligible(&self, t: &Target, needs: &Needs, ident: &Identity) -> Option<String> {
        let (p, m) = self.parts(t);
        if p.tier == Tier::Paid && !ident.allow_paid {
            return Some("client key may not use paid providers".into());
        }
        if !p.api_key_env.is_empty() && self.adapters.get(&p.name).is_none_or(|a| a.api_key().is_none()) {
            return Some(format!("not configured (no API key; {} is unset)", p.api_key_env));
        }
        if needs.embeddings != m.has("embeddings") {
            return Some(if needs.embeddings { "not an embedding model".into() } else { "embedding model cannot serve chat".into() });
        }
        for (needed, cap) in [(needs.tools, "tools"), (needs.vision, "vision"), (needs.json, "json")] {
            if needed && !m.has(cap) {
                return Some(format!("model lacks required capability: {cap}"));
            }
        }
        let wanted = needs.window_tokens;
        if wanted > m.context_window {
            return Some(format!("request (~{wanted} tokens) exceeds context window ({})", m.context_window));
        }
        self.quota.check(&t.key, &m.limits, needs.prompt_tokens).err()
    }
}

struct Needs {
    tools: bool,
    vision: bool,
    json: bool,
    embeddings: bool,
    /// Tokens counted against quota before dispatch.
    prompt_tokens: u64,
    /// Tokens that must fit in the model's context window.
    window_tokens: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct ChatOpts {
    pub cache: CacheMode,
    /// Only try the first target of the route.
    pub no_fallback: bool,
}

impl ChatOpts {
    pub fn from_headers(h: &HeaderMap) -> Self {
        let flag = |n: &str| h.get(n).and_then(|v| v.to_str().ok()).is_some_and(|v| matches!(v.trim(), "1" | "true" | "yes"));
        Self { cache: CacheMode::from_headers(h), no_fallback: flag("x-aegis-no-fallback") }
    }
}

pub fn api_error(status: StatusCode, kind: &str, message: impl Into<String>) -> Response {
    (status, Json(json!({ "error": { "message": message.into(), "type": kind, "code": status.as_u16() } }))).into_response()
}

fn set(h: &mut HeaderMap, name: &'static str, value: &str) {
    if let Ok(v) = HeaderValue::from_str(value) {
        h.insert(HeaderName::from_static(name), v);
    }
}

/// Headers that tell the caller exactly who answered, so fallback is never silent.
fn served_headers(rid: &str, provider: &str, model: &str, tier: &str, cache: &str, fallback: bool, attempts: usize) -> HeaderMap {
    let mut h = HeaderMap::new();
    set(&mut h, "x-aegis-request-id", rid);
    set(&mut h, "x-aegis-provider", provider);
    set(&mut h, "x-aegis-model", model);
    set(&mut h, "x-aegis-tier", tier);
    set(&mut h, "x-aegis-cache", cache);
    set(&mut h, "x-aegis-fallback", if fallback { "true" } else { "false" });
    set(&mut h, "x-aegis-attempts", &attempts.to_string());
    h
}

// ---- HTTP handlers --------------------------------------------------------

#[allow(clippy::result_large_err)]
fn identify(state: &AppState, headers: &HeaderMap) -> Result<Arc<Identity>, Response> {
    match presented_key(headers) {
        Some(k) => state
            .keys
            .authenticate(&k)
            .ok_or_else(|| api_error(StatusCode::UNAUTHORIZED, "invalid_api_key", "Invalid or revoked API key.")),
        None if !state.cfg.auth.required => Ok(Arc::new(Identity::anonymous())),
        None => Err(api_error(
            StatusCode::UNAUTHORIZED,
            "missing_api_key",
            "Missing API key. Send `Authorization: Bearer <aegis key>`.",
        )),
    }
}

pub async fn chat_completions(State(state): State<Shared>, headers: HeaderMap, Json(body): Json<Value>) -> Response {
    let ident = match identify(&state, &headers) {
        Ok(i) => i,
        Err(r) => return r,
    };
    let req: ChatRequest = match serde_json::from_value(body) {
        Ok(r) => r,
        Err(e) => return api_error(StatusCode::BAD_REQUEST, "invalid_request_error", format!("Invalid request body: {e}")),
    };
    run_chat(state, ident, ChatOpts::from_headers(&headers), req).await
}

pub async fn list_models(State(state): State<Shared>, headers: HeaderMap) -> Response {
    let ident = match identify(&state, &headers) {
        Ok(i) => i,
        Err(r) => return r,
    };
    let mut data = Vec::new();
    for r in &state.cfg.routes {
        if ident.may_use(&r.alias) {
            data.push(json!({ "id": r.alias, "object": "model", "owned_by": "aegis", "description": r.description, "targets": r.targets }));
        }
    }
    for p in &state.cfg.providers {
        for m in &p.models {
            let id = format!("{}/{}", p.name, m.id);
            if ident.may_use(&id) {
                data.push(json!({
                    "id": id, "object": "model", "owned_by": p.name, "tier": p.tier.as_str(),
                    "context_window": m.context_window, "capabilities": m.capabilities,
                }));
            }
        }
    }
    Json(json!({ "object": "list", "data": data })).into_response()
}

pub async fn healthz(State(state): State<Shared>) -> Response {
    Json(json!({ "status": "ok", "uptime_secs": state.started.elapsed().as_secs(), "providers": state.cfg.providers.len() })).into_response()
}

// ---- pipeline -------------------------------------------------------------

pub async fn run_chat(state: Shared, ident: Arc<Identity>, opts: ChatOpts, req: ChatRequest) -> Response {
    let rid = format!("req_{}", uuid::Uuid::new_v4().simple());
    let started = Instant::now();
    let mut rec = RequestRecord {
        request_id: rid.clone(),
        ts: now_ms(),
        key_id: ident.key_id,
        key_name: ident.name.clone(),
        requested_model: req.model.clone(),
        stream: req.stream,
        cache: "skip".into(),
        attempts: "[]".into(),
        ..Default::default()
    };
    let reject = |mut rec: RequestRecord, status: StatusCode, kind: &str, msg: String| {
        rec.status = "rejected".into();
        rec.http_status = status.as_u16() as i64;
        rec.error = msg.clone();
        rec.latency_ms = started.elapsed().as_millis() as i64;
        let _ = state.telemetry.send(rec);
        let mut resp = api_error(status, kind, msg);
        set(resp.headers_mut(), "x-aegis-request-id", &rid);
        resp
    };

    if req.messages.is_empty() {
        return reject(rec, StatusCode::BAD_REQUEST, "invalid_request_error", "`messages` must not be empty.".into());
    }
    if !ident.may_use(&req.model) {
        return reject(rec, StatusCode::FORBIDDEN, "model_not_allowed", format!("This key may not use model `{}`.", req.model));
    }
    let mut candidates = state.resolve(&req.model);
    if candidates.is_empty() {
        return reject(
            rec,
            StatusCode::NOT_FOUND,
            "model_not_found",
            format!("Unknown model `{}`. See GET /v1/models for aliases and provider/model ids.", req.model),
        );
    }
    if let Err(msg) = state.keys.check_limits(&ident) {
        return reject(rec, StatusCode::TOO_MANY_REQUESTS, "rate_limit_exceeded", msg);
    }
    if opts.no_fallback {
        candidates.truncate(1);
    }

    // ---- cache ----
    let cache_key = state.cache.key_for(&req, opts.cache, &ident.scope());
    if let Some(key) = &cache_key {
        if let Some(hit) = state.cache.get(key).await {
            rec.status = "ok".into();
            rec.http_status = 200;
            rec.cache = "hit".into();
            rec.provider = hit.provider.clone();
            rec.model = hit.model.clone();
            rec.tier = hit.tier.clone();
            rec.prompt_tokens = hit.usage.prompt_tokens as i64;
            rec.completion_tokens = hit.usage.completion_tokens as i64;
            rec.saved_usd = hit.cost_usd;
            rec.latency_ms = started.elapsed().as_millis() as i64;
            rec.ttfb_ms = rec.latency_ms;
            let _ = state.telemetry.send(rec);
            let headers = served_headers(&rid, &hit.provider, &hit.model, &hit.tier, "hit", false, 0);
            return if req.stream {
                (headers, replay_as_stream(&hit.body)).into_response()
            } else {
                (headers, Json(hit.body.clone())).into_response()
            };
        }
        rec.cache = "miss".into();
    }

    // ---- semantic cache: same question in other words ----
    let mut sem: Option<(String, Vec<f32>)> = None;
    if cache_key.is_some() && opts.cache != CacheMode::Exact && state.semantic.cfg().enabled {
        let scope = ident.scope();
        if let Some((partition, text)) = req.semantic_parts(state.cache.scope_for(&scope), state.semantic.cfg().min_chars)
            && let Some(vector) = state.embed_for_cache(&text).await
        {
            if let Some((hit, similarity)) = state.semantic.lookup(&partition, &vector) {
                rec.status = "ok".into();
                rec.http_status = 200;
                rec.cache = "semantic".into();
                rec.provider = hit.provider.clone();
                rec.model = hit.model.clone();
                rec.tier = hit.tier.clone();
                rec.prompt_tokens = hit.usage.prompt_tokens as i64;
                rec.completion_tokens = hit.usage.completion_tokens as i64;
                rec.saved_usd = hit.cost_usd;
                rec.latency_ms = started.elapsed().as_millis() as i64;
                rec.ttfb_ms = rec.latency_ms;
                let _ = state.telemetry.send(rec);
                let mut headers = served_headers(&rid, &hit.provider, &hit.model, &hit.tier, "semantic", false, 0);
                set(&mut headers, "x-aegis-cache-similarity", &format!("{similarity:.4}"));
                return if req.stream {
                    (headers, replay_as_stream(&hit.body)).into_response()
                } else {
                    (headers, Json(hit.body.clone())).into_response()
                };
            }
            sem = Some((partition, vector));
        }
    }

    // ---- route ----
    let needs = Needs {
        tools: req.needs_tools(),
        vision: req.needs_vision(),
        json: req.needs_json(),
        embeddings: false,
        prompt_tokens: req.est_prompt_tokens(),
        window_tokens: req.est_prompt_tokens() + req.max_out().unwrap_or(0),
    };
    let mut attempts: Vec<Value> = Vec::new();
    let mut tried = 0usize;
    let mut saw_rate_limit = false;

    for (idx, t) in candidates.iter().enumerate() {
        if let Some(reason) = state.ineligible(t, &needs, &ident) {
            saw_rate_limit |= reason.contains("quota") || reason.contains("cooling down");
            attempts.push(json!({ "target": t.key, "outcome": "skipped", "reason": reason }));
            continue;
        }
        let (p, m) = state.parts(t);
        let Some(adapter) = state.adapters.get(&p.name) else { continue };
        tried += 1;
        state.quota.reserve(&t.key);
        let fallback = idx > 0;

        let err = if req.stream {
            match adapter.stream(m, &req).await {
                // Wait for the first chunk before committing: if the provider
                // fails here, nothing has reached the client and we can still
                // fall back to the next target.
                Ok((mut upstream, snap)) => match upstream.next().await {
                    Some(Err(e)) => e,
                    first => {
                    let first = first.and_then(Result::ok);
                    state.note_success(&t.key, &snap);
                    attempts.push(json!({ "target": t.key, "outcome": "served" }));
                    rec.provider = p.name.clone();
                    rec.model = m.id.clone();
                    rec.tier = p.tier.as_str().into();
                    rec.fallback = fallback;
                    rec.attempts = Value::Array(attempts.clone()).to_string();
                    let headers = served_headers(&rid, &p.name, &m.id, p.tier.as_str(), &rec.cache, fallback, attempts.len());
                    let fin = Finalizer {
                        state: state.clone(),
                        rec,
                        target: t.key.clone(),
                        model: m.clone(),
                        cache_key,
                        sem,
                        budget: ident.daily_budget_usd,
                        started,
                        est_prompt: needs.prompt_tokens,
                        acc: Accumulator::default(),
                        usage: None,
                        first_chunk_ms: None,
                        error: None,
                        completed: false,
                        done: false,
                    };
                    return (headers, proxy_stream(first, upstream, fin)).into_response();
                    }
                },
                Err(e) => e,
            }
        } else {
            match adapter.complete(m, &req).await {
                Ok((completion, snap)) => {
                    state.note_success(&t.key, &snap);
                    attempts.push(json!({ "target": t.key, "outcome": "served" }));
                    let usage = completion.usage.unwrap_or_else(|| Usage {
                        prompt_tokens: needs.prompt_tokens,
                        completion_tokens: est_tokens(&completion.body.pointer("/choices/0/message").map(Value::to_string).unwrap_or_default()),
                        estimated: true,
                    });
                    let cost = m.cost(usage.prompt_tokens, usage.completion_tokens);
                    state.quota.record_tokens(&t.key, usage.total());
                    state.charge(ident.key_id, &ident.name, ident.daily_budget_usd, cost);
                    let finish = completion.body.pointer("/choices/0/finish_reason").and_then(Value::as_str).unwrap_or("stop");
                    if let Some(key) = cache_key
                        && finish != "content_filter"
                    {
                        let entry = CachedEntry {
                            body: completion.body.clone(),
                            provider: p.name.clone(),
                            model: m.id.clone(),
                            tier: p.tier.as_str().into(),
                            usage,
                            cost_usd: cost,
                            stored_at: now_ms(),
                        };
                        if let Some((partition, vector)) = sem {
                            state.semantic_store(partition, vector, entry.clone()).await;
                        }
                        state.cache_store(key, entry).await;
                    }
                    rec.status = "ok".into();
                    rec.http_status = 200;
                    rec.provider = p.name.clone();
                    rec.model = m.id.clone();
                    rec.tier = p.tier.as_str().into();
                    rec.prompt_tokens = usage.prompt_tokens as i64;
                    rec.completion_tokens = usage.completion_tokens as i64;
                    rec.usage_estimated = usage.estimated;
                    rec.cost_usd = cost;
                    rec.fallback = fallback;
                    rec.latency_ms = started.elapsed().as_millis() as i64;
                    rec.ttfb_ms = rec.latency_ms;
                    rec.attempts = Value::Array(attempts.clone()).to_string();
                    let headers = served_headers(&rid, &p.name, &m.id, p.tier.as_str(), &rec.cache, fallback, attempts.len());
                    let _ = state.telemetry.send(rec);
                    return (headers, Json(completion.body)).into_response();
                }
                Err(e) => e,
            }
        };

        // ---- this target failed ----
        saw_rate_limit |= matches!(err, ProviderError::RateLimited { .. });
        state.note_error(&t.key, &err);
        tracing::warn!(request_id = %rid, target = %t.key, error = %err, "provider attempt failed");
        attempts.push(json!({ "target": t.key, "outcome": "failed", "reason": err.to_string() }));

        if let ProviderError::Rejected { status, message } = &err {
            let status = StatusCode::from_u16(*status).unwrap_or(StatusCode::BAD_REQUEST);
            rec.provider = p.name.clone();
            rec.model = m.id.clone();
            rec.tier = p.tier.as_str().into();
            rec.status = "error".into();
            rec.http_status = status.as_u16() as i64;
            rec.error = message.clone();
            rec.fallback = fallback;
            rec.latency_ms = started.elapsed().as_millis() as i64;
            rec.attempts = Value::Array(attempts).to_string();
            let _ = state.telemetry.send(rec);
            let mut resp = api_error(status, "provider_rejected_request", format!("{}: {}", t.key, message));
            set(resp.headers_mut(), "x-aegis-request-id", &rid);
            set(resp.headers_mut(), "x-aegis-provider", &p.name);
            return resp;
        }
    }

    // ---- nothing could serve it ----
    let status = if saw_rate_limit && tried == 0 { StatusCode::TOO_MANY_REQUESTS } else { StatusCode::SERVICE_UNAVAILABLE };
    let summary = attempts
        .iter()
        .map(|a| format!("{} — {}", a["target"].as_str().unwrap_or("?"), a["reason"].as_str().unwrap_or("?")))
        .collect::<Vec<_>>()
        .join("; ");
    rec.status = "error".into();
    rec.http_status = status.as_u16() as i64;
    rec.error = format!("no provider could serve the request: {summary}");
    rec.latency_ms = started.elapsed().as_millis() as i64;
    rec.attempts = Value::Array(attempts.clone()).to_string();
    let _ = state.telemetry.send(rec);
    let mut resp = (
        status,
        Json(json!({ "error": {
            "message": format!("No provider could serve `{}`: {summary}", req.model),
            "type": "no_available_provider",
            "code": status.as_u16(),
            "attempts": attempts,
        } })),
    )
        .into_response();
    set(resp.headers_mut(), "x-aegis-request-id", &rid);
    resp
}

// ---- embeddings -----------------------------------------------------------

/// `input` as the OpenAI API accepts it: one string or a list of strings.
fn embedding_inputs(v: Option<&Value>) -> Result<Vec<String>, String> {
    let inputs = match v {
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) if !a.is_empty() && a.iter().all(Value::is_string) => {
            a.iter().filter_map(Value::as_str).map(str::to_string).collect()
        }
        Some(Value::Array(a)) if !a.is_empty() => {
            return Err("`input` must be text. Token-id arrays are not supported, because providers tokenize differently.".into());
        }
        _ => return Err("`input` must be a non-empty string or a non-empty array of strings.".into()),
    };
    if inputs.iter().any(String::is_empty) {
        return Err("`input` must not contain empty strings.".into());
    }
    if inputs.len() > 2048 {
        return Err("`input` may hold at most 2048 strings.".into());
    }
    Ok(inputs)
}

/// OpenAI-compatible `/v1/embeddings`.
///
/// Vectors from different embedding models cannot be compared, so unlike
/// chat, a route only falls back between targets serving the *same model id*
/// as its first embedding target. Any other target is skipped and reported.
pub async fn embeddings(State(state): State<Shared>, headers: HeaderMap, Json(body): Json<Value>) -> Response {
    let ident = match identify(&state, &headers) {
        Ok(i) => i,
        Err(r) => return r,
    };
    let opts = ChatOpts::from_headers(&headers);
    let rid = format!("req_{}", uuid::Uuid::new_v4().simple());
    let started = Instant::now();
    let model_name = body.get("model").and_then(Value::as_str).unwrap_or("").to_string();
    let mut rec = RequestRecord {
        request_id: rid.clone(),
        ts: now_ms(),
        key_id: ident.key_id,
        key_name: ident.name.clone(),
        requested_model: model_name.clone(),
        cache: "skip".into(),
        attempts: "[]".into(),
        ..Default::default()
    };
    let reject = |mut rec: RequestRecord, status: StatusCode, kind: &str, msg: String| {
        rec.status = "rejected".into();
        rec.http_status = status.as_u16() as i64;
        rec.error = msg.clone();
        rec.latency_ms = started.elapsed().as_millis() as i64;
        let _ = state.telemetry.send(rec);
        let mut resp = api_error(status, kind, msg);
        set(resp.headers_mut(), "x-aegis-request-id", &rid);
        resp
    };

    if model_name.is_empty() {
        return reject(rec, StatusCode::BAD_REQUEST, "invalid_request_error", "`model` is required.".into());
    }
    let inputs = match embedding_inputs(body.get("input")) {
        Ok(i) => i,
        Err(msg) => return reject(rec, StatusCode::BAD_REQUEST, "invalid_request_error", msg),
    };
    let base64_out = match body.get("encoding_format").and_then(Value::as_str) {
        None | Some("float") => false,
        Some("base64") => true,
        Some(other) => {
            return reject(rec, StatusCode::BAD_REQUEST, "invalid_request_error", format!("Unsupported encoding_format `{other}`."));
        }
    };
    let dimensions = body.get("dimensions").and_then(Value::as_u64);
    if !ident.may_use(&model_name) {
        return reject(rec, StatusCode::FORBIDDEN, "model_not_allowed", format!("This key may not use model `{model_name}`."));
    }
    let mut candidates = state.resolve(&model_name);
    if candidates.is_empty() {
        return reject(
            rec,
            StatusCode::NOT_FOUND,
            "model_not_found",
            format!("Unknown model `{model_name}`. See GET /v1/models for aliases and provider/model ids."),
        );
    }
    if let Err(msg) = state.keys.check_limits(&ident) {
        return reject(rec, StatusCode::TOO_MANY_REQUESTS, "rate_limit_exceeded", msg);
    }
    if opts.no_fallback {
        candidates.truncate(1);
    }

    let per_input: Vec<u64> = inputs.iter().map(|s| est_tokens(s)).collect();
    let needs = Needs {
        tools: false,
        vision: false,
        json: false,
        embeddings: true,
        prompt_tokens: per_input.iter().sum(),
        window_tokens: per_input.iter().copied().max().unwrap_or(0),
    };
    // The model every vector must come from: the route's first embedding model.
    let pinned = candidates.iter().map(|t| state.parts(t).1).find(|m| m.has("embeddings")).map(|m| m.id.clone());
    let mut attempts: Vec<Value> = Vec::new();
    let mut tried = 0usize;
    let mut saw_rate_limit = false;

    for (idx, t) in candidates.iter().enumerate() {
        let (p, m) = state.parts(t);
        let reason = state.ineligible(t, &needs, &ident).or_else(|| {
            (pinned.as_deref() != Some(m.id.as_str()))
                .then(|| format!("different embedding model than {}; its vectors would not be comparable", pinned.as_deref().unwrap_or("?")))
        });
        if let Some(reason) = reason {
            saw_rate_limit |= reason.contains("quota") || reason.contains("cooling down");
            attempts.push(json!({ "target": t.key, "outcome": "skipped", "reason": reason }));
            continue;
        }
        let Some(adapter) = state.adapters.get(&p.name) else { continue };
        tried += 1;
        state.quota.reserve(&t.key);
        let fallback = idx > 0;
        rec.provider = p.name.clone();
        rec.model = m.id.clone();
        rec.tier = p.tier.as_str().into();
        rec.fallback = fallback;

        match adapter.embed(m, &inputs, dimensions).await {
            Ok((out, snap)) => {
                state.note_success(&t.key, &snap);
                attempts.push(json!({ "target": t.key, "outcome": "served" }));
                let tokens = out.prompt_tokens.unwrap_or(needs.prompt_tokens);
                let cost = m.cost(tokens, 0);
                state.quota.record_tokens(&t.key, tokens);
                state.charge(ident.key_id, &ident.name, ident.daily_budget_usd, cost);
                let data: Vec<Value> = out
                    .vectors
                    .iter()
                    .enumerate()
                    .map(|(i, v)| {
                        let embedding = if base64_out {
                            use base64::Engine;
                            let bytes: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
                            json!(base64::engine::general_purpose::STANDARD.encode(bytes))
                        } else {
                            json!(v)
                        };
                        json!({ "object": "embedding", "index": i, "embedding": embedding })
                    })
                    .collect();
                rec.status = "ok".into();
                rec.http_status = 200;
                rec.prompt_tokens = tokens as i64;
                rec.usage_estimated = out.prompt_tokens.is_none();
                rec.cost_usd = cost;
                rec.latency_ms = started.elapsed().as_millis() as i64;
                rec.ttfb_ms = rec.latency_ms;
                rec.attempts = Value::Array(attempts.clone()).to_string();
                let headers = served_headers(&rid, &p.name, &m.id, p.tier.as_str(), "skip", fallback, attempts.len());
                let _ = state.telemetry.send(rec);
                let body = json!({
                    "object": "list", "data": data, "model": m.id,
                    "usage": { "prompt_tokens": tokens, "total_tokens": tokens },
                });
                return (headers, Json(body)).into_response();
            }
            Err(err) => {
                saw_rate_limit |= matches!(err, ProviderError::RateLimited { .. });
                state.note_error(&t.key, &err);
                tracing::warn!(request_id = %rid, target = %t.key, error = %err, "embedding attempt failed");
                attempts.push(json!({ "target": t.key, "outcome": "failed", "reason": err.to_string() }));
                if let ProviderError::Rejected { status, message } = &err {
                    let status = StatusCode::from_u16(*status).unwrap_or(StatusCode::BAD_REQUEST);
                    rec.status = "error".into();
                    rec.http_status = status.as_u16() as i64;
                    rec.error = message.clone();
                    rec.latency_ms = started.elapsed().as_millis() as i64;
                    rec.attempts = Value::Array(attempts).to_string();
                    let _ = state.telemetry.send(rec);
                    let mut resp = api_error(status, "provider_rejected_request", format!("{}: {}", t.key, message));
                    set(resp.headers_mut(), "x-aegis-request-id", &rid);
                    set(resp.headers_mut(), "x-aegis-provider", &p.name);
                    return resp;
                }
            }
        }
    }

    let status = if saw_rate_limit && tried == 0 { StatusCode::TOO_MANY_REQUESTS } else { StatusCode::SERVICE_UNAVAILABLE };
    let summary = attempts
        .iter()
        .map(|a| format!("{} — {}", a["target"].as_str().unwrap_or("?"), a["reason"].as_str().unwrap_or("?")))
        .collect::<Vec<_>>()
        .join("; ");
    if tried == 0 {
        rec.provider.clear();
        rec.model.clear();
        rec.tier.clear();
        rec.fallback = false;
    }
    rec.status = "error".into();
    rec.http_status = status.as_u16() as i64;
    rec.error = format!("no provider could serve the request: {summary}");
    rec.latency_ms = started.elapsed().as_millis() as i64;
    rec.attempts = Value::Array(attempts.clone()).to_string();
    let _ = state.telemetry.send(rec);
    let mut resp = (
        status,
        Json(json!({ "error": {
            "message": format!("No provider could serve embeddings for `{model_name}`: {summary}"),
            "type": "no_available_provider",
            "code": status.as_u16(),
            "attempts": attempts,
        } })),
    )
        .into_response();
    set(resp.headers_mut(), "x-aegis-request-id", &rid);
    resp
}

// ---- streaming ------------------------------------------------------------

type SseStream = Sse<std::pin::Pin<Box<dyn futures::Stream<Item = Result<Event, Infallible>> + Send>>>;

/// Serve a cached completion to a client that asked for a stream.
fn replay_as_stream(body: &Value) -> SseStream {
    let id = body.get("id").and_then(Value::as_str).map(str::to_string).unwrap_or_else(new_completion_id);
    let model = body.get("model").and_then(Value::as_str).unwrap_or("").to_string();
    let msg = body.pointer("/choices/0/message").cloned().unwrap_or(json!({}));
    let finish = body.pointer("/choices/0/finish_reason").and_then(Value::as_str).unwrap_or("stop").to_string();
    let mut delta = json!({ "role": "assistant", "content": msg.get("content").cloned().unwrap_or(Value::Null) });
    if let Some(calls) = msg.get("tool_calls").and_then(Value::as_array) {
        let indexed: Vec<Value> = calls
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let mut c = c.clone();
                c["index"] = json!(i);
                c
            })
            .collect();
        delta["tool_calls"] = json!(indexed);
    }
    let mut events = vec![chunk_json(&id, &model, delta, None).to_string(), chunk_json(&id, &model, json!({}), Some(&finish)).to_string()];
    if let Some(u) = body.get("usage").and_then(Usage::from_openai) {
        events.push(usage_chunk_json(&id, &model, &u).to_string());
    }
    events.push("[DONE]".into());
    let s = futures::stream::iter(events.into_iter().map(|d| Ok::<_, Infallible>(Event::default().data(d))));
    Sse::new(Box::pin(s))
}

/// Rebuilds the full message from stream deltas (for caching and accounting).
#[derive(Default)]
struct Accumulator {
    id: Option<String>,
    content: String,
    tool_calls: BTreeMap<u64, (String, String, String)>,
    finish: Option<String>,
}

impl Accumulator {
    fn observe(&mut self, chunk: &Value) {
        if self.id.is_none() {
            self.id = chunk.get("id").and_then(Value::as_str).map(str::to_string);
        }
        let Some(choice) = chunk.pointer("/choices/0") else { return };
        if let Some(t) = choice.pointer("/delta/content").and_then(Value::as_str) {
            self.content.push_str(t);
        }
        for tc in choice.pointer("/delta/tool_calls").and_then(Value::as_array).into_iter().flatten() {
            let idx = tc.get("index").and_then(Value::as_u64).unwrap_or(0);
            let e = self.tool_calls.entry(idx).or_default();
            if let Some(id) = tc.get("id").and_then(Value::as_str) {
                e.0 = id.to_string();
            }
            if let Some(n) = tc.pointer("/function/name").and_then(Value::as_str) {
                e.1.push_str(n);
            }
            if let Some(a) = tc.pointer("/function/arguments").and_then(Value::as_str) {
                e.2.push_str(a);
            }
        }
        if let Some(f) = choice.get("finish_reason").and_then(Value::as_str) {
            self.finish = Some(f.to_string());
        }
    }

    fn output_chars(&self) -> String {
        let mut s = self.content.clone();
        for (_, name, args) in self.tool_calls.values() {
            s.push_str(name);
            s.push_str(args);
        }
        s
    }

    fn to_completion(&self, model: &str, usage: &Usage) -> Value {
        let calls: Vec<Value> = self
            .tool_calls
            .values()
            .map(|(id, name, args)| json!({ "id": id, "type": "function", "function": { "name": name, "arguments": args } }))
            .collect();
        let content = if self.content.is_empty() && !calls.is_empty() { None } else { Some(self.content.clone()) };
        let id = self.id.clone().unwrap_or_else(new_completion_id);
        completion_json(&id, model, content, calls, self.finish.as_deref().unwrap_or("stop"), Some(usage))
    }
}

/// Owns the accounting for one streamed response. `Drop` covers the case
/// where the client disconnects and the stream is never polled to the end.
struct Finalizer {
    state: Shared,
    rec: RequestRecord,
    target: String,
    model: ModelCfg,
    cache_key: Option<String>,
    /// Partition and embedding to file the answer under in the semantic index.
    sem: Option<(String, Vec<f32>)>,
    budget: Option<f64>,
    started: Instant,
    est_prompt: u64,
    acc: Accumulator,
    usage: Option<Usage>,
    first_chunk_ms: Option<i64>,
    error: Option<String>,
    completed: bool,
    done: bool,
}

impl Finalizer {
    fn observe(&mut self, chunk: &Value) {
        if self.first_chunk_ms.is_none() {
            self.first_chunk_ms = Some(self.started.elapsed().as_millis() as i64);
        }
        if let Some(u) = chunk.get("usage").and_then(Usage::from_openai) {
            self.usage = Some(u);
        }
        self.acc.observe(chunk);
    }

    /// Settle quota, spend and telemetry. Returns a cache entry if the
    /// response is complete and cacheable.
    fn settle(&mut self) -> Option<(String, CachedEntry)> {
        if self.done {
            return None;
        }
        self.done = true;
        let usage = self.usage.unwrap_or_else(|| Usage {
            prompt_tokens: self.est_prompt,
            completion_tokens: est_tokens(&self.acc.output_chars()),
            estimated: true,
        });
        let cost = self.model.cost(usage.prompt_tokens, usage.completion_tokens);
        self.state.quota.record_tokens(&self.target, usage.total());
        self.state.charge(self.rec.key_id, &self.rec.key_name, self.budget, cost);

        let mut rec = std::mem::take(&mut self.rec);
        rec.prompt_tokens = usage.prompt_tokens as i64;
        rec.completion_tokens = usage.completion_tokens as i64;
        rec.usage_estimated = usage.estimated;
        rec.cost_usd = cost;
        rec.latency_ms = self.started.elapsed().as_millis() as i64;
        rec.ttfb_ms = self.first_chunk_ms.unwrap_or(rec.latency_ms);
        rec.http_status = 200;
        match (&self.error, self.completed) {
            (Some(e), _) => {
                rec.status = "error".into();
                rec.error = format!("stream failed mid-response: {e}");
                self.state.note_error(&self.target, &ProviderError::Unavailable(e.clone()));
            }
            (None, true) => rec.status = "ok".into(),
            (None, false) => {
                rec.status = "client_closed".into();
                rec.error = "client disconnected before the stream finished".into();
            }
        }
        let entry = (self.completed && self.error.is_none() && self.acc.finish.as_deref() != Some("content_filter"))
            .then(|| self.cache_key.take())
            .flatten()
            .map(|key| {
                (key, CachedEntry {
                    body: self.acc.to_completion(&rec.model, &usage),
                    provider: rec.provider.clone(),
                    model: rec.model.clone(),
                    tier: rec.tier.clone(),
                    usage,
                    cost_usd: cost,
                    stored_at: now_ms(),
                })
            });
        let _ = self.state.telemetry.send(rec);
        entry
    }
}

impl Drop for Finalizer {
    fn drop(&mut self) {
        self.settle();
    }
}

fn proxy_stream(
    first: Option<Value>,
    mut upstream: ChunkStream,
    mut fin: Finalizer,
) -> Sse<impl futures::Stream<Item = Result<Event, Infallible>> + Send> {
    let s = async_stream::stream! {
        if let Some(chunk) = first {
            fin.observe(&chunk);
            yield Ok(Event::default().data(chunk.to_string()));
        }
        loop {
            match upstream.next().await {
                Some(Ok(chunk)) => {
                    fin.observe(&chunk);
                    yield Ok(Event::default().data(chunk.to_string()));
                }
                Some(Err(e)) => {
                    // The client already has part of the answer, so we cannot
                    // switch providers; report the failure in-band.
                    fin.error = Some(e.to_string());
                    let payload = json!({ "error": { "message": e.to_string(), "type": "upstream_stream_error" } });
                    yield Ok(Event::default().data(payload.to_string()));
                    break;
                }
                None => {
                    fin.completed = true;
                    break;
                }
            }
        }
        yield Ok(Event::default().data("[DONE]"));
        if let Some((key, entry)) = fin.settle() {
            if let Some((partition, vector)) = fin.sem.take() {
                fin.state.semantic_store(partition, vector, entry.clone()).await;
            }
            fin.state.cache_store(key, entry).await;
        }
    };
    Sse::new(s).keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
}

//! The request pipeline: authenticate, check the cache, pick a provider,
//! dispatch (with fallback), account for usage, respond.

use crate::auth::{Identity, KeyStore, presented_key};
use crate::cache::{CacheMode, CachedEntry, ResponseCache};
use crate::config::{Config, ModelCfg, ProviderCfg, Tier};
use crate::db::{Db, RequestRecord};
use crate::providers::{Adapter, ChunkStream, ProviderError};
use crate::quota::QuotaTracker;
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
    pub db: Db,
    pub keys: KeyStore,
    pub telemetry: UnboundedSender<RequestRecord>,
    pub admin_token: String,
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
    pub async fn build(cfg: Config, admin_token: String) -> anyhow::Result<Shared> {
        let db = Db::open(&cfg.server.db_path).await?;
        let mut adapters = HashMap::new();
        for p in &cfg.providers {
            adapters.insert(p.name.clone(), Adapter::new(p)?);
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
        Ok(Arc::new(Self {
            cfg,
            adapters,
            quota,
            cache,
            db,
            keys,
            telemetry,
            admin_token,
            sessions: Mutex::new(HashSet::new()),
            started: Instant::now(),
        }))
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

    /// Why this target cannot serve this request, if it cannot.
    fn ineligible(&self, t: &Target, needs: &Needs, ident: &Identity) -> Option<String> {
        let (p, m) = self.parts(t);
        if p.tier == Tier::Paid && !ident.allow_paid {
            return Some("client key may not use paid providers".into());
        }
        if !p.api_key_env.is_empty() && self.adapters.get(&p.name).is_none_or(|a| a.api_key.is_none()) {
            return Some(format!("not configured ({} is unset)", p.api_key_env));
        }
        for (needed, cap) in [(needs.tools, "tools"), (needs.vision, "vision"), (needs.json, "json")] {
            if needed && !m.has(cap) {
                return Some(format!("model lacks required capability: {cap}"));
            }
        }
        let wanted = needs.prompt_tokens + needs.max_out.unwrap_or(0);
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
    prompt_tokens: u64,
    max_out: Option<u64>,
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

    // ---- route ----
    let needs = Needs {
        tools: req.needs_tools(),
        vision: req.needs_vision(),
        json: req.needs_json(),
        prompt_tokens: req.est_prompt_tokens(),
        max_out: req.max_out(),
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
                    state.quota.on_success(&t.key, &snap);
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
                    state.quota.on_success(&t.key, &snap);
                    attempts.push(json!({ "target": t.key, "outcome": "served" }));
                    let usage = completion.usage.unwrap_or_else(|| Usage {
                        prompt_tokens: needs.prompt_tokens,
                        completion_tokens: est_tokens(&completion.body.pointer("/choices/0/message").map(Value::to_string).unwrap_or_default()),
                        estimated: true,
                    });
                    let cost = m.cost(usage.prompt_tokens, usage.completion_tokens);
                    state.quota.record_tokens(&t.key, usage.total());
                    state.keys.add_spend(ident.key_id, cost);
                    let finish = completion.body.pointer("/choices/0/finish_reason").and_then(Value::as_str).unwrap_or("stop");
                    if let Some(key) = cache_key
                        && finish != "content_filter"
                    {
                        state
                            .cache
                            .put(key, CachedEntry {
                                body: completion.body.clone(),
                                provider: p.name.clone(),
                                model: m.id.clone(),
                                tier: p.tier.as_str().into(),
                                usage,
                                cost_usd: cost,
                                stored_at: now_ms(),
                            })
                            .await;
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
        match &err {
            ProviderError::RateLimited { retry_after, message } => {
                saw_rate_limit = true;
                state.quota.on_rate_limited(&t.key, *retry_after, message);
            }
            ProviderError::Rejected { .. } => {}
            other => state.quota.on_failure(&t.key, &other.to_string()),
        }
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
        self.state.keys.add_spend(self.rec.key_id, cost);

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
                self.state.quota.on_failure(&self.target, e);
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
            fin.state.cache.put(key, entry).await;
        }
    };
    Sse::new(s).keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
}

//! Provider adapters. Each one accepts the internal (OpenAI-shaped) request
//! and returns OpenAI-shaped completions or chunk streams.

pub mod anthropic;
pub mod gemini;
pub mod openai;

use crate::config::{ModelCfg, ProviderCfg, ProviderKind};
use crate::types::{ChatRequest, Completion};
use futures::Stream;
use reqwest::header::HeaderMap;
use serde_json::Value;
use std::pin::Pin;
use std::sync::RwLock;
use std::time::Duration;

pub type ChunkStream = Pin<Box<dyn Stream<Item = Result<Value, ProviderError>> + Send>>;

#[derive(Debug, Clone, thiserror::Error)]
pub enum ProviderError {
    #[error("rate limited: {message}")]
    RateLimited { retry_after: Option<Duration>, message: String },
    #[error("unavailable: {0}")]
    Unavailable(String),
    #[error("provider auth failed: {0}")]
    Auth(String),
    /// The provider understood the request and refused it (4xx). Another
    /// provider would likely refuse it too, so this is not retried elsewhere.
    #[error("rejected by provider ({status}): {message}")]
    Rejected { status: u16, message: String },
}

impl ProviderError {
    pub fn can_fallback(&self) -> bool {
        !matches!(self, Self::Rejected { .. })
    }
}

/// What a provider told us about remaining quota in its response headers.
#[derive(Debug, Clone, Default)]
pub struct RateSnapshot {
    pub remaining_requests: Option<u64>,
    pub remaining_tokens: Option<u64>,
    pub reset_requests: Option<Duration>,
    pub reset_tokens: Option<Duration>,
}

fn header_str<'a>(h: &'a HeaderMap, name: &str) -> Option<&'a str> {
    h.get(name).and_then(|v| v.to_str().ok()).map(str::trim)
}

/// Parses `12`, `7.66s`, `2m59.5s`, `1h2m`, `450ms`, or an RFC 3339 timestamp.
pub fn parse_reset(s: &str) -> Option<Duration> {
    if let Ok(secs) = s.parse::<f64>() {
        return (secs >= 0.0).then(|| Duration::from_secs_f64(secs));
    }
    if let Ok(ts) = chrono::DateTime::parse_from_rfc3339(s) {
        let delta = ts.timestamp_millis() - chrono::Utc::now().timestamp_millis();
        return Some(Duration::from_millis(delta.max(0) as u64));
    }
    let mut total = 0f64;
    let mut num = String::new();
    let mut chars = s.chars().peekable();
    let mut seen = false;
    while let Some(c) = chars.next() {
        if c.is_ascii_digit() || c == '.' {
            num.push(c);
            continue;
        }
        let n: f64 = num.parse().ok()?;
        num.clear();
        let unit = match c {
            'h' => 3600.0,
            'm' if chars.peek() == Some(&'s') => {
                chars.next();
                0.001
            }
            'm' => 60.0,
            's' => 1.0,
            _ => return None,
        };
        total += n * unit;
        seen = true;
    }
    (seen && num.is_empty()).then(|| Duration::from_secs_f64(total))
}

pub fn snapshot_from(h: &HeaderMap) -> RateSnapshot {
    let num = |names: &[&str]| names.iter().find_map(|n| header_str(h, n)).and_then(|v| v.parse::<u64>().ok());
    let dur = |names: &[&str]| names.iter().find_map(|n| header_str(h, n)).and_then(parse_reset);
    RateSnapshot {
        remaining_requests: num(&["x-ratelimit-remaining-requests", "anthropic-ratelimit-requests-remaining"]),
        remaining_tokens: num(&["x-ratelimit-remaining-tokens", "anthropic-ratelimit-tokens-remaining"]),
        reset_requests: dur(&["x-ratelimit-reset-requests", "anthropic-ratelimit-requests-reset"]),
        reset_tokens: dur(&["x-ratelimit-reset-tokens", "anthropic-ratelimit-tokens-reset"]),
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max { s.to_string() } else { format!("{}…", s.chars().take(max).collect::<String>()) }
}

pub fn transport_err(e: reqwest::Error) -> ProviderError {
    let kind = if e.is_timeout() {
        "timeout"
    } else if e.is_connect() {
        "connection failed"
    } else {
        "transport error"
    };
    // `without_url` keeps query strings (which may carry keys) out of logs.
    ProviderError::Unavailable(format!("{kind}: {}", e.without_url()))
}

/// Turn a non-2xx provider response into a classified error.
pub async fn error_from_response(resp: reqwest::Response) -> ProviderError {
    let status = resp.status().as_u16();
    let headers = resp.headers().clone();
    let body = resp.text().await.unwrap_or_default();
    let message = serde_json::from_str::<Value>(&body)
        .ok()
        .and_then(|v| {
            let v = if v.is_array() { v.get(0).cloned().unwrap_or(Value::Null) } else { v };
            let e = v.get("error")?;
            e.get("message").and_then(Value::as_str).map(str::to_string).or_else(|| e.as_str().map(str::to_string))
        })
        .unwrap_or_else(|| body.clone());
    let message = truncate(message.trim(), 400);
    match status {
        429 => {
            let snap = snapshot_from(&headers);
            let retry_after = header_str(&headers, "retry-after")
                .and_then(parse_reset)
                .or(snap.reset_requests)
                .or(snap.reset_tokens);
            ProviderError::RateLimited { retry_after, message }
        }
        401 | 403 => ProviderError::Auth(message),
        404 => ProviderError::Unavailable(format!("404 from provider: {message}")),
        408 | 409 | 425 | 500..=599 => ProviderError::Unavailable(format!("{status} from provider: {message}")),
        _ => ProviderError::Rejected { status, message },
    }
}

/// One entry of a provider's model list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemoteModel {
    pub id: String,
    pub context_window: Option<u64>,
    pub kind: RemoteKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemoteKind {
    Chat,
    Embedding,
    /// Speech, image, moderation and other models Aegis cannot route to.
    Other,
}

async fn fetch_json(rb: reqwest::RequestBuilder) -> Result<Value, ProviderError> {
    let resp = rb.timeout(Duration::from_secs(20)).send().await.map_err(transport_err)?;
    if !resp.status().is_success() {
        return Err(error_from_response(resp).await);
    }
    resp.json().await.map_err(transport_err)
}

/// `{ "data": [{ "id": ... }] }`, as OpenAI, Groq, Ollama and Anthropic return.
/// These lists do not say what a model is for, so that is read from its name.
pub fn parse_openai_models(v: &Value) -> Vec<RemoteModel> {
    const NOT_CHAT: &[&str] = &["whisper", "tts", "dall-e", "moderation", "image", "audio", "realtime", "transcribe", "guard", "sora", "speech"];
    v.get("data")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|m| {
            let id = m.get("id").and_then(Value::as_str)?.to_string();
            let lower = id.to_lowercase();
            let kind = if lower.contains("embed") {
                RemoteKind::Embedding
            } else if NOT_CHAT.iter().any(|w| lower.contains(w)) {
                RemoteKind::Other
            } else {
                RemoteKind::Chat
            };
            let context_window = ["context_window", "max_input_tokens", "context_length"].iter().find_map(|k| m.get(*k).and_then(Value::as_u64));
            Some(RemoteModel { id, context_window, kind })
        })
        .collect()
}

/// Gemini's list says which methods each model supports.
pub fn parse_gemini_models(v: &Value) -> Vec<RemoteModel> {
    v.get("models")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|m| {
            let name = m.get("name").and_then(Value::as_str)?;
            let methods: Vec<&str> = m.get("supportedGenerationMethods").and_then(Value::as_array).into_iter().flatten().filter_map(Value::as_str).collect();
            let kind = if methods.contains(&"generateContent") {
                RemoteKind::Chat
            } else if methods.iter().any(|x| x.contains("embed")) {
                RemoteKind::Embedding
            } else {
                RemoteKind::Other
            };
            Some(RemoteModel {
                id: name.strip_prefix("models/").unwrap_or(name).to_string(),
                context_window: m.get("inputTokenLimit").and_then(Value::as_u64),
                kind,
            })
        })
        .collect()
}

pub struct Embeddings {
    pub vectors: Vec<Vec<f32>>,
    /// Input tokens as reported by the provider, if it reports them.
    pub prompt_tokens: Option<u64>,
}

fn floats(v: Option<&Value>) -> Vec<f32> {
    v.and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_f64).map(|f| f as f32).collect()).unwrap_or_default()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeySource {
    Dashboard,
    Env,
    Unset,
}

pub struct Adapter {
    pub name: String,
    pub kind: ProviderKind,
    pub base_url: String,
    /// Key from the provider's environment variable, read at startup.
    env_key: Option<String>,
    /// Key set from the dashboard; takes precedence and can change at runtime.
    stored_key: RwLock<Option<String>>,
    pub timeout: Duration,
    /// Longest wait for the first streamed chunk before trying the next target.
    pub first_byte_timeout: Duration,
    pub client: reqwest::Client,
}

impl Adapter {
    pub fn new(p: &ProviderCfg) -> anyhow::Result<Self> {
        let env_key = if p.api_key_env.is_empty() {
            None
        } else {
            std::env::var(&p.api_key_env).ok().map(|v| v.trim().to_string()).filter(|v| !v.is_empty())
        };
        let timeout = Duration::from_secs(p.timeout_secs.max(1));
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(p.connect_timeout_secs.max(1)))
            .read_timeout(timeout)
            .user_agent(concat!("aegis-gateway/", env!("CARGO_PKG_VERSION")))
            .build()?;
        Ok(Self {
            name: p.name.clone(),
            kind: p.kind,
            base_url: p.base_url.trim_end_matches('/').to_string(),
            env_key,
            stored_key: RwLock::new(None),
            timeout,
            first_byte_timeout: Duration::from_secs(p.first_byte_timeout().max(1)),
            client,
        })
    }

    pub fn api_key(&self) -> Option<String> {
        self.stored_key.read().unwrap_or_else(|e| e.into_inner()).clone().or_else(|| self.env_key.clone())
    }

    pub fn key_source(&self) -> KeySource {
        if self.stored_key.read().unwrap_or_else(|e| e.into_inner()).is_some() {
            KeySource::Dashboard
        } else if self.env_key.is_some() {
            KeySource::Env
        } else {
            KeySource::Unset
        }
    }

    pub fn set_stored_key(&self, key: Option<String>) {
        *self.stored_key.write().unwrap_or_else(|e| e.into_inner()) = key;
    }

    pub async fn complete(&self, model: &ModelCfg, req: &ChatRequest) -> Result<(Completion, RateSnapshot), ProviderError> {
        match self.kind {
            ProviderKind::Openai => openai::complete(self, model, req).await,
            ProviderKind::Anthropic => anthropic::complete(self, model, req).await,
            ProviderKind::Gemini => gemini::complete(self, model, req).await,
        }
    }

    /// The models this key can use, as the provider's own API lists them.
    pub async fn list_models(&self) -> Result<Vec<RemoteModel>, ProviderError> {
        let mut models = match self.kind {
            ProviderKind::Openai => {
                let mut rb = self.client.get(format!("{}/models", self.base_url));
                if let Some(k) = self.api_key() {
                    rb = rb.bearer_auth(k);
                }
                let v = fetch_json(rb).await?;
                parse_openai_models(&v)
            }
            ProviderKind::Anthropic => {
                let rb = self
                    .client
                    .get(format!("{}/v1/models?limit=1000", self.base_url))
                    .header("x-api-key", self.api_key().unwrap_or_default())
                    .header("anthropic-version", anthropic::API_VERSION);
                let v = fetch_json(rb).await?;
                parse_openai_models(&v)
            }
            ProviderKind::Gemini => {
                let mut out = Vec::new();
                let mut page: Option<String> = None;
                // Follow pagination, but never loop forever on a misbehaving server.
                for _ in 0..10 {
                    let mut rb = self.client.get(format!("{}/models", self.base_url)).query(&[("pageSize", "1000")]).header("x-goog-api-key", self.api_key().unwrap_or_default());
                    if let Some(p) = &page {
                        rb = rb.query(&[("pageToken", p.as_str())]);
                    }
                    let v = fetch_json(rb).await?;
                    out.extend(parse_gemini_models(&v));
                    page = v.get("nextPageToken").and_then(Value::as_str).filter(|s| !s.is_empty()).map(str::to_string);
                    if page.is_none() {
                        break;
                    }
                }
                out
            }
        };
        models.sort_by(|a, b| a.id.cmp(&b.id));
        models.dedup_by(|a, b| a.id == b.id);
        Ok(models)
    }

    /// Embed each input as one vector, in order.
    pub async fn embed(&self, model: &ModelCfg, inputs: &[String], dimensions: Option<u64>) -> Result<(Embeddings, RateSnapshot), ProviderError> {
        let out = match self.kind {
            ProviderKind::Openai => openai::embed(self, model, inputs, dimensions).await,
            ProviderKind::Gemini => gemini::embed(self, model, inputs, dimensions).await,
            ProviderKind::Anthropic => Err(ProviderError::Rejected { status: 400, message: "Anthropic has no embeddings API".into() }),
        }?;
        if out.0.vectors.len() != inputs.len() || out.0.vectors.iter().any(Vec::is_empty) {
            return Err(ProviderError::Unavailable(format!(
                "provider returned {} embeddings for {} inputs",
                out.0.vectors.iter().filter(|v| !v.is_empty()).count(),
                inputs.len()
            )));
        }
        Ok(out)
    }

    /// Errors returned here happen before any byte reaches the client, so the
    /// router can still fall back. Errors inside the stream cannot be retried.
    pub async fn stream(&self, model: &ModelCfg, req: &ChatRequest) -> Result<(ChunkStream, RateSnapshot), ProviderError> {
        match self.kind {
            ProviderKind::Openai => openai::stream(self, model, req).await,
            ProviderKind::Anthropic => anthropic::stream(self, model, req).await,
            ProviderKind::Gemini => gemini::stream(self, model, req).await,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_model_lists() {
        let v = serde_json::json!({ "data": [
            { "id": "llama-3.3-70b-versatile", "context_window": 131072 },
            { "id": "text-embedding-3-small" },
            { "id": "whisper-large-v3" },
        ]});
        let m = parse_openai_models(&v);
        assert_eq!(m[0], RemoteModel { id: "llama-3.3-70b-versatile".into(), context_window: Some(131072), kind: RemoteKind::Chat });
        assert_eq!((m[1].kind, m[2].kind), (RemoteKind::Embedding, RemoteKind::Other));

        let g = serde_json::json!({ "models": [
            { "name": "models/gemini-x-flash", "inputTokenLimit": 1048576, "supportedGenerationMethods": ["generateContent", "countTokens"] },
            { "name": "models/embed-x", "supportedGenerationMethods": ["embedContent"] },
            { "name": "models/aqa", "supportedGenerationMethods": ["generateAnswer"] },
        ]});
        let m = parse_gemini_models(&g);
        assert_eq!(m[0], RemoteModel { id: "gemini-x-flash".into(), context_window: Some(1048576), kind: RemoteKind::Chat });
        assert_eq!((m[1].kind, m[2].kind), (RemoteKind::Embedding, RemoteKind::Other));
    }

    #[test]
    fn parses_reset_formats() {
        assert_eq!(parse_reset("12"), Some(Duration::from_secs(12)));
        assert_eq!(parse_reset("7.5s"), Some(Duration::from_secs_f64(7.5)));
        assert_eq!(parse_reset("2m30s"), Some(Duration::from_secs(150)));
        assert_eq!(parse_reset("450ms"), Some(Duration::from_millis(450)));
        assert_eq!(parse_reset("1h"), Some(Duration::from_secs(3600)));
        assert_eq!(parse_reset("soon"), None);
    }
}

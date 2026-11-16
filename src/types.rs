//! The gateway's internal request/response language is the OpenAI chat
//! completions format. Adapters translate to and from it.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::time::{SystemTime, UNIX_EPOCH};

fn is_false(b: &bool) -> bool {
    !*b
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Value>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub stream: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<Value>,
    /// Anything else the client sent is passed through to OpenAI-compatible
    /// providers untouched.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl ChatRequest {
    pub fn max_out(&self) -> Option<u64> {
        self.max_completion_tokens.or(self.max_tokens)
    }

    pub fn needs_tools(&self) -> bool {
        self.tools.as_ref().and_then(Value::as_array).is_some_and(|a| !a.is_empty())
    }

    pub fn needs_vision(&self) -> bool {
        self.messages.iter().any(|m| {
            m.get("content")
                .and_then(Value::as_array)
                .is_some_and(|parts| parts.iter().any(|p| p.get("type").and_then(Value::as_str) == Some("image_url")))
        })
    }

    pub fn needs_json(&self) -> bool {
        matches!(
            self.response_format.as_ref().and_then(|f| f.get("type")).and_then(Value::as_str),
            Some("json_object") | Some("json_schema")
        )
    }

    /// Rough token estimate (4 characters per token). Used for context-window
    /// checks, quota pre-checks and as a fallback when a provider omits usage.
    pub fn est_prompt_tokens(&self) -> u64 {
        let mut chars = serde_json::to_string(&self.messages).map(|s| s.len()).unwrap_or(0);
        if let Some(t) = &self.tools {
            chars += t.to_string().len();
        }
        (chars as u64).div_ceil(4)
    }

    /// For semantic caching: the text of the last user message, and a hash
    /// of everything else. Two requests may share an answer only if the
    /// hashes are equal and the texts mean the same. `None` when the request
    /// does not end in a plain-text user message, or uses tools or images,
    /// where a near-match is not safe to reuse.
    pub fn semantic_parts(&self, scope: &str, min_chars: usize) -> Option<(String, String)> {
        if self.needs_tools() || self.needs_vision() {
            return None;
        }
        let last = self.messages.last()?;
        if last.get("role").and_then(Value::as_str) != Some("user") {
            return None;
        }
        let text = text_of(last.get("content")?).trim().to_string();
        if text.chars().count() < min_chars.max(1) {
            return None;
        }
        let mut rest = self.clone();
        if let Some(m) = rest.messages.last_mut() {
            m["content"] = Value::Null;
        }
        Some((rest.fingerprint(&format!("semantic\u{0}{scope}")), text))
    }

    /// Stable hash of everything that influences the answer.
    pub fn fingerprint(&self, scope: &str) -> String {
        let mut v = serde_json::to_value(self).unwrap_or(Value::Null);
        if let Some(o) = v.as_object_mut() {
            for k in ["stream", "stream_options", "user", "metadata"] {
                o.remove(k);
            }
        }
        let mut canon = String::new();
        canonical(&v, &mut canon);
        let mut h = Sha256::new();
        h.update(scope.as_bytes());
        h.update([0u8]);
        h.update(canon.as_bytes());
        let out = h.finalize();
        hex::encode(&out[..])
    }
}

/// JSON with object keys sorted, so equal requests hash equally.
fn canonical(v: &Value, out: &mut String) {
    match v {
        Value::Object(m) => {
            let mut keys: Vec<&String> = m.keys().collect();
            keys.sort();
            out.push('{');
            for (i, k) in keys.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&Value::String((*k).clone()).to_string());
                out.push(':');
                canonical(&m[*k], out);
            }
            out.push('}');
        }
        Value::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                canonical(x, out);
            }
            out.push(']');
        }
        other => out.push_str(&other.to_string()),
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Usage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    /// True when the provider did not report usage and Aegis estimated it.
    pub estimated: bool,
}

impl Usage {
    pub fn from_openai(v: &Value) -> Option<Self> {
        let p = v.get("prompt_tokens")?.as_u64()?;
        let c = v.get("completion_tokens").and_then(Value::as_u64).unwrap_or(0);
        Some(Self { prompt_tokens: p, completion_tokens: c, estimated: false })
    }

    pub fn to_openai(&self) -> Value {
        json!({
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
        })
    }

    pub fn total(&self) -> u64 {
        self.prompt_tokens + self.completion_tokens
    }
}

pub fn est_tokens(text: &str) -> u64 {
    (text.len() as u64).div_ceil(4)
}

/// A full (non-streaming) answer in OpenAI `chat.completion` shape.
#[derive(Debug, Clone)]
pub struct Completion {
    pub body: Value,
    pub usage: Option<Usage>,
}

pub fn new_completion_id() -> String {
    format!("chatcmpl-{}", uuid::Uuid::new_v4().simple())
}

pub fn completion_json(
    id: &str,
    model: &str,
    content: Option<String>,
    tool_calls: Vec<Value>,
    finish: &str,
    usage: Option<&Usage>,
) -> Value {
    let mut message = json!({ "role": "assistant", "content": content });
    if !tool_calls.is_empty() {
        message["tool_calls"] = Value::Array(tool_calls);
    }
    let mut v = json!({
        "id": id,
        "object": "chat.completion",
        "created": now_secs(),
        "model": model,
        "choices": [{ "index": 0, "message": message, "finish_reason": finish }],
    });
    if let Some(u) = usage {
        v["usage"] = u.to_openai();
    }
    v
}

pub fn chunk_json(id: &str, model: &str, delta: Value, finish: Option<&str>) -> Value {
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": now_secs(),
        "model": model,
        "choices": [{ "index": 0, "delta": delta, "finish_reason": finish }],
    })
}

pub fn usage_chunk_json(id: &str, model: &str, usage: &Usage) -> Value {
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": now_secs(),
        "model": model,
        "choices": [],
        "usage": usage.to_openai(),
    })
}

/// Plain text of an OpenAI message `content` (string or array of parts).
pub fn text_of(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

pub fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

pub fn now_ms() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as i64).unwrap_or(0)
}

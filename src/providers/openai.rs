//! OpenAI-compatible providers: OpenAI, Groq, Ollama, vLLM, llama.cpp.
//! The request is already in the right shape, so this is a pass-through.

use super::{Adapter, ChunkStream, Embeddings, ProviderError, RateSnapshot, error_from_response, floats, snapshot_from, transport_err};
use crate::config::ModelCfg;
use crate::types::{ChatRequest, Completion, Usage};
use eventsource_stream::Eventsource;
use futures::StreamExt;
use serde_json::{Value, json};

fn body(model: &ModelCfg, req: &ChatRequest, stream: bool) -> Value {
    let mut b = serde_json::to_value(req).unwrap_or_else(|_| json!({}));
    b["model"] = json!(model.id);
    if stream {
        b["stream"] = json!(true);
        b["stream_options"] = json!({ "include_usage": true });
    } else if let Some(o) = b.as_object_mut() {
        o.remove("stream");
        o.remove("stream_options");
    }
    b
}

fn request(a: &Adapter, body: &Value) -> reqwest::RequestBuilder {
    let mut rb = a.client.post(format!("{}/chat/completions", a.base_url)).json(body);
    if let Some(k) = a.api_key() {
        rb = rb.bearer_auth(k);
    }
    rb
}

pub async fn complete(a: &Adapter, model: &ModelCfg, req: &ChatRequest) -> Result<(Completion, RateSnapshot), ProviderError> {
    let resp = request(a, &body(model, req, false)).timeout(a.timeout).send().await.map_err(transport_err)?;
    if !resp.status().is_success() {
        return Err(error_from_response(resp).await);
    }
    let snap = snapshot_from(resp.headers());
    let v: Value = resp.json().await.map_err(transport_err)?;
    if v.get("choices").and_then(Value::as_array).is_none() {
        return Err(ProviderError::Unavailable("provider returned a response without choices".into()));
    }
    let usage = v.get("usage").and_then(Usage::from_openai);
    Ok((Completion { body: v, usage }, snap))
}

pub async fn stream(a: &Adapter, model: &ModelCfg, req: &ChatRequest) -> Result<(ChunkStream, RateSnapshot), ProviderError> {
    let resp = request(a, &body(model, req, true)).send().await.map_err(transport_err)?;
    if !resp.status().is_success() {
        return Err(error_from_response(resp).await);
    }
    let snap = snapshot_from(resp.headers());
    let s = async_stream::stream! {
        let mut events = Box::pin(resp.bytes_stream().eventsource());
        while let Some(ev) = events.next().await {
            let ev = match ev {
                Ok(ev) => ev,
                Err(e) => {
                    yield Err(ProviderError::Unavailable(format!("stream interrupted: {e}")));
                    return;
                }
            };
            let data = ev.data.trim();
            if data == "[DONE]" {
                break;
            }
            let Ok(v) = serde_json::from_str::<Value>(data) else { continue };
            if let Some(err) = v.get("error") {
                let msg = err.get("message").and_then(Value::as_str).map(str::to_string).unwrap_or_else(|| err.to_string());
                yield Err(ProviderError::Unavailable(msg));
                return;
            }
            yield Ok(v);
        }
    };
    Ok((Box::pin(s), snap))
}

pub async fn embed(a: &Adapter, model: &ModelCfg, inputs: &[String], dimensions: Option<u64>) -> Result<(Embeddings, RateSnapshot), ProviderError> {
    let mut body = json!({ "model": model.id, "input": inputs, "encoding_format": "float" });
    if let Some(d) = dimensions {
        body["dimensions"] = json!(d);
    }
    let mut rb = a.client.post(format!("{}/embeddings", a.base_url)).json(&body);
    if let Some(k) = a.api_key() {
        rb = rb.bearer_auth(k);
    }
    let resp = rb.timeout(a.timeout).send().await.map_err(transport_err)?;
    if !resp.status().is_success() {
        return Err(error_from_response(resp).await);
    }
    let snap = snapshot_from(resp.headers());
    let v: Value = resp.json().await.map_err(transport_err)?;
    let mut rows: Vec<(u64, Vec<f32>)> = v
        .get("data")
        .and_then(Value::as_array)
        .ok_or_else(|| ProviderError::Unavailable("provider returned an embeddings response without data".into()))?
        .iter()
        .enumerate()
        .map(|(i, d)| (d.get("index").and_then(Value::as_u64).unwrap_or(i as u64), floats(d.get("embedding"))))
        .collect();
    rows.sort_by_key(|r| r.0);
    let prompt_tokens = v.pointer("/usage/prompt_tokens").and_then(Value::as_u64);
    Ok((Embeddings { vectors: rows.into_iter().map(|r| r.1).collect(), prompt_tokens }, snap))
}

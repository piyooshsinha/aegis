//! Anthropic Messages API adapter (raw HTTP; there is no official Rust SDK).
//!
//! Notes on translation:
//! - `temperature` / `top_p` are never forwarded: API version 2026-10-21
//!   rejects sampling parameters on every model.
//! - Forced tool choice (`required` or a named function) is rejected by
//!   current models, so it is downgraded to `auto`.

use super::{Adapter, ChunkStream, ProviderError, RateSnapshot, error_from_response, snapshot_from, transport_err};
use crate::config::ModelCfg;
use crate::types::{ChatRequest, Completion, Usage, chunk_json, completion_json, new_completion_id, text_of, usage_chunk_json};
use eventsource_stream::Eventsource;
use futures::StreamExt;
use serde_json::{Value, json};

const API_VERSION: &str = "2026-10-21";

fn image_block(url: &str) -> Value {
    if let Some(rest) = url.strip_prefix("data:")
        && let Some((meta, data)) = rest.split_once(',')
    {
        let media_type = meta.split(';').next().unwrap_or("image/png");
        return json!({ "type": "image", "source": { "type": "base64", "media_type": media_type, "data": data } });
    }
    json!({ "type": "image", "source": { "type": "url", "url": url } })
}

fn content_blocks(content: &Value) -> Vec<Value> {
    match content {
        Value::String(s) if !s.is_empty() => vec![json!({ "type": "text", "text": s })],
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| match p.get("type").and_then(Value::as_str) {
                Some("text") => {
                    let t = p.get("text").and_then(Value::as_str).unwrap_or("");
                    (!t.is_empty()).then(|| json!({ "type": "text", "text": t }))
                }
                Some("image_url") => p.pointer("/image_url/url").and_then(Value::as_str).map(image_block),
                _ => None,
            })
            .collect(),
        _ => vec![],
    }
}

fn push(out: &mut Vec<(String, Vec<Value>)>, role: &str, blocks: Vec<Value>) {
    if blocks.is_empty() {
        return;
    }
    match out.last_mut() {
        Some((r, existing)) if r == role => existing.extend(blocks),
        _ => out.push((role.to_string(), blocks)),
    }
}

fn body(model: &ModelCfg, req: &ChatRequest, stream: bool) -> Value {
    let mut system = String::new();
    let mut turns: Vec<(String, Vec<Value>)> = Vec::new();

    for m in &req.messages {
        let content = m.get("content").unwrap_or(&Value::Null);
        match m.get("role").and_then(Value::as_str).unwrap_or("user") {
            "system" | "developer" => {
                if !system.is_empty() {
                    system.push_str("\n\n");
                }
                system.push_str(&text_of(content));
            }
            "assistant" => {
                let mut blocks = content_blocks(content);
                for tc in m.get("tool_calls").and_then(Value::as_array).into_iter().flatten() {
                    let args = tc.pointer("/function/arguments").and_then(Value::as_str).unwrap_or("{}");
                    let input: Value = serde_json::from_str(args).unwrap_or_else(|_| json!({}));
                    blocks.push(json!({
                        "type": "tool_use",
                        "id": tc.get("id").and_then(Value::as_str).unwrap_or(""),
                        "name": tc.pointer("/function/name").and_then(Value::as_str).unwrap_or(""),
                        "input": input,
                    }));
                }
                push(&mut turns, "assistant", blocks);
            }
            "tool" => push(
                &mut turns,
                "user",
                vec![json!({
                    "type": "tool_result",
                    "tool_use_id": m.get("tool_call_id").and_then(Value::as_str).unwrap_or(""),
                    "content": text_of(content),
                })],
            ),
            _ => push(&mut turns, "user", content_blocks(content)),
        }
    }

    let cap = model.max_output.unwrap_or(if stream { 64_000 } else { 16_000 });
    let default_max = if stream { cap.min(64_000) } else { cap.min(16_000) };
    let max_tokens = req.max_out().map(|m| m.min(cap)).unwrap_or(default_max);

    let mut b = json!({
        "model": model.id,
        "max_tokens": max_tokens,
        "messages": turns.into_iter().map(|(role, content)| json!({ "role": role, "content": content })).collect::<Vec<_>>(),
    });
    if !system.is_empty() {
        b["system"] = json!(system);
    }
    if stream {
        b["stream"] = json!(true);
    }
    match &req.stop {
        Some(Value::String(s)) => b["stop_sequences"] = json!([s]),
        Some(Value::Array(a)) if !a.is_empty() => b["stop_sequences"] = json!(a),
        _ => {}
    }
    if let Some(tools) = req.tools.as_ref().and_then(Value::as_array).filter(|t| !t.is_empty()) {
        let converted: Vec<Value> = tools
            .iter()
            .filter_map(|t| {
                let f = t.get("function")?;
                Some(json!({
                    "name": f.get("name")?,
                    "description": f.get("description").cloned().unwrap_or(json!("")),
                    "input_schema": f.get("parameters").cloned().unwrap_or(json!({ "type": "object", "properties": {} })),
                }))
            })
            .collect();
        b["tools"] = json!(converted);
        b["tool_choice"] = match req.tool_choice.as_ref().and_then(Value::as_str) {
            Some("none") => json!({ "type": "none" }),
            _ => json!({ "type": "auto" }),
        };
    }
    b
}

fn request(a: &Adapter, body: &Value) -> reqwest::RequestBuilder {
    a.client
        .post(format!("{}/v1/messages", a.base_url))
        .header("x-api-key", a.api_key().unwrap_or_default())
        .header("anthropic-version", API_VERSION)
        .json(body)
}

fn finish_reason(stop_reason: &str) -> &'static str {
    match stop_reason {
        "max_tokens" => "length",
        "tool_use" => "tool_calls",
        "refusal" => "content_filter",
        _ => "stop",
    }
}

fn usage_of(u: &Value) -> (u64, u64) {
    let n = |k: &str| u.get(k).and_then(Value::as_u64).unwrap_or(0);
    (n("input_tokens") + n("cache_creation_input_tokens") + n("cache_read_input_tokens"), n("output_tokens"))
}

pub async fn complete(a: &Adapter, model: &ModelCfg, req: &ChatRequest) -> Result<(Completion, RateSnapshot), ProviderError> {
    let resp = request(a, &body(model, req, false)).timeout(a.timeout).send().await.map_err(transport_err)?;
    if !resp.status().is_success() {
        return Err(error_from_response(resp).await);
    }
    let snap = snapshot_from(resp.headers());
    let v: Value = resp.json().await.map_err(transport_err)?;

    let mut text = String::new();
    let mut tool_calls = Vec::new();
    for block in v.get("content").and_then(Value::as_array).into_iter().flatten() {
        match block.get("type").and_then(Value::as_str) {
            Some("text") => text.push_str(block.get("text").and_then(Value::as_str).unwrap_or("")),
            Some("tool_use") => tool_calls.push(json!({
                "id": block.get("id").cloned().unwrap_or(json!("")),
                "type": "function",
                "function": {
                    "name": block.get("name").cloned().unwrap_or(json!("")),
                    "arguments": block.get("input").cloned().unwrap_or(json!({})).to_string(),
                },
            })),
            _ => {}
        }
    }
    let (input, output) = usage_of(v.get("usage").unwrap_or(&Value::Null));
    let usage = Usage { prompt_tokens: input, completion_tokens: output, estimated: false };
    let finish = finish_reason(v.get("stop_reason").and_then(Value::as_str).unwrap_or("end_turn"));
    let content = if text.is_empty() && !tool_calls.is_empty() { None } else { Some(text) };
    let id = v.get("id").and_then(Value::as_str).map(str::to_string).unwrap_or_else(new_completion_id);
    let body = completion_json(&id, &model.id, content, tool_calls, finish, Some(&usage));
    Ok((Completion { body, usage: Some(usage) }, snap))
}

pub async fn stream(a: &Adapter, model: &ModelCfg, req: &ChatRequest) -> Result<(ChunkStream, RateSnapshot), ProviderError> {
    let resp = request(a, &body(model, req, true)).send().await.map_err(transport_err)?;
    if !resp.status().is_success() {
        return Err(error_from_response(resp).await);
    }
    let snap = snapshot_from(resp.headers());
    let model_id = model.id.clone();
    let s = async_stream::stream! {
        let mut events = Box::pin(resp.bytes_stream().eventsource());
        let mut id = new_completion_id();
        let mut usage = Usage::default();
        let mut tool_index: i64 = -1;
        let mut finish = "stop";
        while let Some(ev) = events.next().await {
            let ev = match ev {
                Ok(ev) => ev,
                Err(e) => {
                    yield Err(ProviderError::Unavailable(format!("stream interrupted: {e}")));
                    return;
                }
            };
            let Ok(v) = serde_json::from_str::<Value>(&ev.data) else { continue };
            match v.get("type").and_then(Value::as_str).unwrap_or("") {
                "message_start" => {
                    if let Some(mid) = v.pointer("/message/id").and_then(Value::as_str) {
                        id = mid.to_string();
                    }
                    let (input, output) = usage_of(v.pointer("/message/usage").unwrap_or(&Value::Null));
                    usage.prompt_tokens = input;
                    usage.completion_tokens = output;
                    yield Ok(chunk_json(&id, &model_id, json!({ "role": "assistant", "content": "" }), None));
                }
                "content_block_start" => {
                    if v.pointer("/content_block/type").and_then(Value::as_str) == Some("tool_use") {
                        tool_index += 1;
                        yield Ok(chunk_json(&id, &model_id, json!({ "tool_calls": [{
                            "index": tool_index,
                            "id": v.pointer("/content_block/id").cloned().unwrap_or(json!("")),
                            "type": "function",
                            "function": { "name": v.pointer("/content_block/name").cloned().unwrap_or(json!("")), "arguments": "" },
                        }] }), None));
                    }
                }
                "content_block_delta" => match v.pointer("/delta/type").and_then(Value::as_str) {
                    Some("text_delta") => {
                        let t = v.pointer("/delta/text").and_then(Value::as_str).unwrap_or("");
                        if !t.is_empty() {
                            yield Ok(chunk_json(&id, &model_id, json!({ "content": t }), None));
                        }
                    }
                    Some("input_json_delta") if tool_index >= 0 => {
                        let p = v.pointer("/delta/partial_json").and_then(Value::as_str).unwrap_or("");
                        yield Ok(chunk_json(&id, &model_id, json!({ "tool_calls": [{
                            "index": tool_index,
                            "function": { "arguments": p },
                        }] }), None));
                    }
                    _ => {}
                },
                "message_delta" => {
                    if let Some(sr) = v.pointer("/delta/stop_reason").and_then(Value::as_str) {
                        finish = finish_reason(sr);
                    }
                    if let Some(o) = v.pointer("/usage/output_tokens").and_then(Value::as_u64) {
                        usage.completion_tokens = o;
                    }
                }
                "message_stop" => break,
                "error" => {
                    let msg = v.pointer("/error/message").and_then(Value::as_str).unwrap_or("provider stream error").to_string();
                    yield Err(ProviderError::Unavailable(msg));
                    return;
                }
                _ => {}
            }
        }
        yield Ok(chunk_json(&id, &model_id, json!({}), Some(finish)));
        yield Ok(usage_chunk_json(&id, &model_id, &usage));
    };
    Ok((Box::pin(s), snap))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> ModelCfg {
        toml::from_str(r#"id = "claude-haiku-5-5""#).unwrap()
    }

    #[test]
    fn translates_messages_tools_and_drops_sampling() {
        let req: ChatRequest = serde_json::from_value(json!({
            "model": "x",
            "temperature": 0.2,
            "messages": [
                { "role": "system", "content": "be brief" },
                { "role": "user", "content": "weather in Paris?" },
                { "role": "assistant", "content": null, "tool_calls": [
                    { "id": "call_1", "type": "function", "function": { "name": "get_weather", "arguments": "{\"city\":\"Paris\"}" } }
                ] },
                { "role": "tool", "tool_call_id": "call_1", "content": "sunny" }
            ],
            "tools": [{ "type": "function", "function": { "name": "get_weather", "parameters": { "type": "object" } } }],
            "tool_choice": "required"
        }))
        .unwrap();
        let b = body(&model(), &req, false);
        assert_eq!(b["system"], "be brief");
        assert!(b.get("temperature").is_none());
        assert_eq!(b["messages"].as_array().unwrap().len(), 3);
        assert_eq!(b["messages"][1]["content"][0]["type"], "tool_use");
        assert_eq!(b["messages"][1]["content"][0]["input"]["city"], "Paris");
        assert_eq!(b["messages"][2]["content"][0]["type"], "tool_result");
        assert_eq!(b["tools"][0]["input_schema"]["type"], "object");
        assert_eq!(b["tool_choice"]["type"], "auto");
    }
}

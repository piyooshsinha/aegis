//! Google Gemini adapter using the native `generateContent` API.

use super::{Adapter, ChunkStream, Embeddings, ProviderError, RateSnapshot, error_from_response, floats, snapshot_from, transport_err};
use crate::config::ModelCfg;
use crate::types::{ChatRequest, Completion, Usage, chunk_json, completion_json, new_completion_id, text_of, usage_chunk_json};
use eventsource_stream::Eventsource;
use futures::StreamExt;
use serde_json::{Value, json};
use std::collections::HashMap;

/// Gemini accepts a subset of JSON Schema; drop the keywords it rejects.
fn clean_schema(v: &Value) -> Value {
    match v {
        Value::Object(m) => Value::Object(
            m.iter()
                .filter(|(k, _)| !matches!(k.as_str(), "additionalProperties" | "$schema" | "strict"))
                .map(|(k, v)| (k.clone(), clean_schema(v)))
                .collect(),
        ),
        Value::Array(a) => Value::Array(a.iter().map(clean_schema).collect()),
        other => other.clone(),
    }
}

fn parts(content: &Value) -> Vec<Value> {
    match content {
        Value::String(s) if !s.is_empty() => vec![json!({ "text": s })],
        Value::Array(items) => items
            .iter()
            .filter_map(|p| match p.get("type").and_then(Value::as_str) {
                Some("text") => Some(json!({ "text": p.get("text").and_then(Value::as_str).unwrap_or("") })),
                Some("image_url") => {
                    let url = p.pointer("/image_url/url").and_then(Value::as_str)?;
                    let (meta, data) = url.strip_prefix("data:")?.split_once(',')?;
                    let mime = meta.split(';').next().unwrap_or("image/png");
                    Some(json!({ "inlineData": { "mimeType": mime, "data": data } }))
                }
                _ => None,
            })
            .collect(),
        _ => vec![],
    }
}

fn push(out: &mut Vec<(String, Vec<Value>)>, role: &str, p: Vec<Value>) {
    if p.is_empty() {
        return;
    }
    match out.last_mut() {
        Some((r, existing)) if r == role => existing.extend(p),
        _ => out.push((role.to_string(), p)),
    }
}

fn body(model: &ModelCfg, req: &ChatRequest) -> Value {
    let mut system = String::new();
    let mut turns: Vec<(String, Vec<Value>)> = Vec::new();
    let mut call_names: HashMap<String, String> = HashMap::new();

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
                let mut p = parts(content);
                for tc in m.get("tool_calls").and_then(Value::as_array).into_iter().flatten() {
                    let name = tc.pointer("/function/name").and_then(Value::as_str).unwrap_or("").to_string();
                    if let Some(id) = tc.get("id").and_then(Value::as_str) {
                        call_names.insert(id.to_string(), name.clone());
                    }
                    let args = tc.pointer("/function/arguments").and_then(Value::as_str).unwrap_or("{}");
                    let args: Value = serde_json::from_str(args).unwrap_or_else(|_| json!({}));
                    p.push(json!({ "functionCall": { "name": name, "args": args } }));
                }
                push(&mut turns, "model", p);
            }
            "tool" => {
                let id = m.get("tool_call_id").and_then(Value::as_str).unwrap_or("");
                let name = call_names.get(id).cloned().unwrap_or_else(|| id.to_string());
                push(
                    &mut turns,
                    "user",
                    vec![json!({ "functionResponse": { "name": name, "response": { "result": text_of(content) } } })],
                );
            }
            _ => push(&mut turns, "user", parts(content)),
        }
    }

    let mut gen_cfg = json!({});
    if let Some(t) = req.temperature {
        gen_cfg["temperature"] = json!(t);
    }
    if let Some(t) = req.top_p {
        gen_cfg["topP"] = json!(t);
    }
    // Gemini counts thinking tokens against maxOutputTokens, so a small
    // limit can be spent entirely on thinking and return no text. OpenAI
    // callers mean "visible output" by max_tokens; leave headroom for the
    // thinking on top, up to the model's own ceiling.
    if let Some(requested) = req.max_out() {
        let with_headroom = requested.saturating_mul(4).max(THINKING_HEADROOM_FLOOR);
        gen_cfg["maxOutputTokens"] = json!(model.max_output.map_or(with_headroom, |cap| with_headroom.min(cap)));
    } else if let Some(m) = model.max_output {
        gen_cfg["maxOutputTokens"] = json!(m);
    }
    match &req.stop {
        Some(Value::String(s)) => gen_cfg["stopSequences"] = json!([s]),
        Some(Value::Array(a)) if !a.is_empty() => gen_cfg["stopSequences"] = json!(a),
        _ => {}
    }
    if req.needs_json() {
        gen_cfg["responseMimeType"] = json!("application/json");
        if let Some(schema) = req.response_format.as_ref().and_then(|f| f.pointer("/json_schema/schema")) {
            gen_cfg["responseSchema"] = clean_schema(schema);
        }
    }

    let mut b = json!({
        "contents": turns.into_iter().map(|(role, parts)| json!({ "role": role, "parts": parts })).collect::<Vec<_>>(),
        "generationConfig": gen_cfg,
    });
    if !system.is_empty() {
        b["systemInstruction"] = json!({ "parts": [{ "text": system }] });
    }
    if let Some(tools) = req.tools.as_ref().and_then(Value::as_array).filter(|t| !t.is_empty()) {
        let decls: Vec<Value> = tools
            .iter()
            .filter_map(|t| {
                let f = t.get("function")?;
                let mut d = json!({ "name": f.get("name")?, "description": f.get("description").cloned().unwrap_or(json!("")) });
                if let Some(p) = f.get("parameters") {
                    d["parameters"] = clean_schema(p);
                }
                Some(d)
            })
            .collect();
        b["tools"] = json!([{ "functionDeclarations": decls }]);
    }
    b
}

fn request(a: &Adapter, model: &ModelCfg, method: &str, body: &Value) -> reqwest::RequestBuilder {
    a.client
        .post(format!("{}/models/{}:{}", a.base_url, model.id, method))
        .header("x-goog-api-key", a.api_key().unwrap_or_default())
        .json(body)
}

/// Smallest output limit sent to Gemini when the caller sets one.
const THINKING_HEADROOM_FLOOR: u64 = 16_384;

fn finish_reason(reason: &str, has_calls: bool) -> &'static str {
    match reason {
        _ if has_calls => "tool_calls",
        "MAX_TOKENS" => "length",
        "STOP" | "" | "FINISH_REASON_UNSPECIFIED" => "stop",
        _ => "content_filter",
    }
}

fn usage_of(v: &Value) -> Option<Usage> {
    let u = v.get("usageMetadata")?;
    let prompt = u.get("promptTokenCount").and_then(Value::as_u64)?;
    let total = u.get("totalTokenCount").and_then(Value::as_u64).unwrap_or(prompt);
    // Thinking tokens are billed as output but excluded from candidatesTokenCount.
    Some(Usage { prompt_tokens: prompt, completion_tokens: total.saturating_sub(prompt), estimated: false })
}

/// Text and tool calls from one response (or one stream chunk).
fn extract(v: &Value) -> (String, Vec<(String, String)>, String) {
    let cand = v.pointer("/candidates/0").unwrap_or(&Value::Null);
    let mut text = String::new();
    let mut calls = Vec::new();
    for p in cand.pointer("/content/parts").and_then(Value::as_array).into_iter().flatten() {
        if p.get("thought").and_then(Value::as_bool) == Some(true) {
            continue;
        }
        if let Some(t) = p.get("text").and_then(Value::as_str) {
            text.push_str(t);
        }
        if let Some(fc) = p.get("functionCall") {
            calls.push((
                fc.get("name").and_then(Value::as_str).unwrap_or("").to_string(),
                fc.get("args").cloned().unwrap_or(json!({})).to_string(),
            ));
        }
    }
    (text, calls, cand.get("finishReason").and_then(Value::as_str).unwrap_or("").to_string())
}

fn tool_call(name: &str, args: &str) -> Value {
    json!({
        "id": format!("call_{}", uuid::Uuid::new_v4().simple()),
        "type": "function",
        "function": { "name": name, "arguments": args },
    })
}

pub async fn complete(a: &Adapter, model: &ModelCfg, req: &ChatRequest) -> Result<(Completion, RateSnapshot), ProviderError> {
    let resp = request(a, model, "generateContent", &body(model, req)).timeout(a.timeout).send().await.map_err(transport_err)?;
    if !resp.status().is_success() {
        return Err(error_from_response(resp).await);
    }
    let snap = snapshot_from(resp.headers());
    let v: Value = resp.json().await.map_err(transport_err)?;
    let (text, calls, reason) = extract(&v);
    let usage = usage_of(&v);
    let tool_calls: Vec<Value> = calls.iter().map(|(n, a)| tool_call(n, a)).collect();
    let finish = finish_reason(&reason, !tool_calls.is_empty());
    let content = if text.is_empty() && !tool_calls.is_empty() { None } else { Some(text) };
    let body = completion_json(&new_completion_id(), &model.id, content, tool_calls, finish, usage.as_ref());
    Ok((Completion { body, usage }, snap))
}

pub async fn stream(a: &Adapter, model: &ModelCfg, req: &ChatRequest) -> Result<(ChunkStream, RateSnapshot), ProviderError> {
    let resp = request(a, model, "streamGenerateContent?alt=sse", &body(model, req)).send().await.map_err(transport_err)?;
    if !resp.status().is_success() {
        return Err(error_from_response(resp).await);
    }
    let snap = snapshot_from(resp.headers());
    let model_id = model.id.clone();
    let s = async_stream::stream! {
        let mut events = Box::pin(resp.bytes_stream().eventsource());
        let id = new_completion_id();
        let mut usage: Option<Usage> = None;
        let mut tool_index = 0u64;
        let mut reason = String::new();
        yield Ok(chunk_json(&id, &model_id, json!({ "role": "assistant", "content": "" }), None));
        while let Some(ev) = events.next().await {
            let ev = match ev {
                Ok(ev) => ev,
                Err(e) => {
                    yield Err(ProviderError::Unavailable(format!("stream interrupted: {e}")));
                    return;
                }
            };
            let Ok(v) = serde_json::from_str::<Value>(&ev.data) else { continue };
            if let Some(err) = v.get("error") {
                let msg = err.get("message").and_then(Value::as_str).unwrap_or("provider stream error").to_string();
                yield Err(ProviderError::Unavailable(msg));
                return;
            }
            let (text, calls, r) = extract(&v);
            if !text.is_empty() {
                yield Ok(chunk_json(&id, &model_id, json!({ "content": text }), None));
            }
            for (name, args) in calls {
                let mut tc = tool_call(&name, &args);
                tc["index"] = json!(tool_index);
                tool_index += 1;
                yield Ok(chunk_json(&id, &model_id, json!({ "tool_calls": [tc] }), None));
            }
            if !r.is_empty() {
                reason = r;
            }
            if let Some(u) = usage_of(&v) {
                usage = Some(u);
            }
        }
        yield Ok(chunk_json(&id, &model_id, json!({}), Some(finish_reason(&reason, tool_index > 0))));
        if let Some(u) = usage {
            yield Ok(usage_chunk_json(&id, &model_id, &u));
        }
    };
    Ok((Box::pin(s), snap))
}

pub async fn embed(a: &Adapter, model: &ModelCfg, inputs: &[String], dimensions: Option<u64>) -> Result<(Embeddings, RateSnapshot), ProviderError> {
    let requests: Vec<Value> = inputs
        .iter()
        .map(|text| {
            let mut r = json!({ "model": format!("models/{}", model.id), "content": { "parts": [{ "text": text }] } });
            if let Some(d) = dimensions {
                r["outputDimensionality"] = json!(d);
            }
            r
        })
        .collect();
    let resp = request(a, model, "batchEmbedContents", &json!({ "requests": requests })).timeout(a.timeout).send().await.map_err(transport_err)?;
    if !resp.status().is_success() {
        return Err(error_from_response(resp).await);
    }
    let snap = snapshot_from(resp.headers());
    let v: Value = resp.json().await.map_err(transport_err)?;
    let vectors = v
        .get("embeddings")
        .and_then(Value::as_array)
        .ok_or_else(|| ProviderError::Unavailable("provider returned an embeddings response without embeddings".into()))?
        .iter()
        .map(|e| floats(e.get("values")))
        .collect();
    // Gemini does not report token usage for embeddings.
    Ok((Embeddings { vectors, prompt_tokens: None }, snap))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn translates_request() {
        let model: ModelCfg = toml::from_str(r#"id = "gemini-flash""#).unwrap();
        let req: ChatRequest = serde_json::from_value(json!({
            "model": "x",
            "temperature": 0.0,
            "max_tokens": 50,
            "response_format": { "type": "json_object" },
            "messages": [
                { "role": "system", "content": "be brief" },
                { "role": "user", "content": "hi" },
                { "role": "assistant", "content": "hello" },
                { "role": "user", "content": "again" }
            ]
        }))
        .unwrap();
        let b = body(&model, &req);
        assert_eq!(b["systemInstruction"]["parts"][0]["text"], "be brief");
        assert_eq!(b["contents"][1]["role"], "model");
        // max_tokens 50 is sent with headroom for thinking tokens.
        assert_eq!(b["generationConfig"]["maxOutputTokens"], 16_384);
        let capped = ModelCfg { max_output: Some(8_000), ..model.clone() };
        assert_eq!(body(&capped, &req)["generationConfig"]["maxOutputTokens"], 8_000);
        let mut big = req.clone();
        big.max_tokens = Some(10_000);
        assert_eq!(body(&model, &big)["generationConfig"]["maxOutputTokens"], 40_000);
        assert_eq!(b["generationConfig"]["responseMimeType"], "application/json");
    }
}

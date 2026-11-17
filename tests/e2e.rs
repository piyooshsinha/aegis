//! End-to-end tests: a real gateway on a local port, talking to a mock
//! OpenAI-compatible upstream that can succeed, rate-limit, fail or reject.

use aegis::auth::{generate_key, hash_key};
use aegis::config::Config;
use aegis::gateway::{AppState, Shared};
use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use serde_json::{Value, json};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

#[derive(Default)]
struct Mock {
    good_calls: AtomicUsize,
    limited_calls: AtomicUsize,
    embed_calls: AtomicUsize,
}

/// A toy embedding: a bag of lower-cased words hashed into 32 buckets. Texts
/// with the same words (any case or punctuation) get the same vector.
fn toy_embedding(text: &str) -> Vec<f32> {
    let mut v = vec![0f32; 32];
    for word in text.to_lowercase().split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()) {
        let bucket = word.bytes().fold(7usize, |h, b| h.wrapping_mul(31).wrapping_add(b as usize)) % 32;
        v[bucket] += 1.0;
    }
    v
}

async fn upstream_embeddings(State(m): State<Arc<Mock>>, Path(kind): Path<String>, Json(body): Json<Value>) -> Response {
    if kind == "down" {
        return (StatusCode::INTERNAL_SERVER_ERROR, "boom").into_response();
    }
    m.embed_calls.fetch_add(1, Ordering::SeqCst);
    let inputs = body["input"].as_array().cloned().unwrap_or_default();
    // Returned out of order on purpose: clients must sort by `index`.
    let data: Vec<Value> = inputs
        .iter()
        .enumerate()
        .rev()
        .map(|(i, t)| json!({ "object": "embedding", "index": i, "embedding": toy_embedding(t.as_str().unwrap_or("")) }))
        .collect();
    Json(json!({ "object": "list", "model": body["model"], "data": data, "usage": { "prompt_tokens": 5 * inputs.len(), "total_tokens": 5 * inputs.len() } })).into_response()
}

async fn upstream(State(m): State<Arc<Mock>>, Path(kind): Path<String>, Json(body): Json<Value>) -> Response {
    match kind.as_str() {
        "limited" => {
            m.limited_calls.fetch_add(1, Ordering::SeqCst);
            let mut h = HeaderMap::new();
            h.insert("retry-after", "30".parse().unwrap());
            (StatusCode::TOO_MANY_REQUESTS, h, Json(json!({ "error": { "message": "slow down" } }))).into_response()
        }
        "down" => (StatusCode::INTERNAL_SERVER_ERROR, "boom").into_response(),
        "bad" => (StatusCode::BAD_REQUEST, Json(json!({ "error": { "message": "bad parameter" } }))).into_response(),
        "midfail" => {
            let sse = "data: {\"error\":{\"message\":\"model crashed\"}}\n\n";
            ([("content-type", "text/event-stream")], sse).into_response()
        }
        _ => {
            m.good_calls.fetch_add(1, Ordering::SeqCst);
            let model = body["model"].as_str().unwrap_or("m").to_string();
            if body["stream"] == json!(true) {
                assert_eq!(body["stream_options"]["include_usage"], json!(true));
                let chunk = |delta: Value, finish: Value| {
                    format!(
                        "data: {}\n\n",
                        json!({ "id": "c1", "object": "chat.completion.chunk", "model": model,
                                "choices": [{ "index": 0, "delta": delta, "finish_reason": finish }] })
                    )
                };
                let mut sse = String::new();
                sse.push_str(&chunk(json!({ "role": "assistant", "content": "Hel" }), Value::Null));
                sse.push_str(&chunk(json!({ "content": "lo" }), Value::Null));
                sse.push_str(&chunk(json!({}), json!("stop")));
                sse.push_str(&format!(
                    "data: {}\n\n",
                    json!({ "id": "c1", "model": model, "choices": [], "usage": { "prompt_tokens": 7, "completion_tokens": 2 } })
                ));
                sse.push_str("data: [DONE]\n\n");
                ([("content-type", "text/event-stream")], sse).into_response()
            } else {
                let mut h = HeaderMap::new();
                h.insert("x-ratelimit-remaining-requests", "41".parse().unwrap());
                (
                    h,
                    Json(json!({
                        "id": "c1", "object": "chat.completion", "model": model,
                        "choices": [{ "index": 0, "message": { "role": "assistant", "content": "Hello" }, "finish_reason": "stop" }],
                        "usage": { "prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9 }
                    })),
                )
                    .into_response()
            }
        }
    }
}

/// The provider's own model list, as `GET /models` returns it.
async fn upstream_models(Path(kind): Path<String>) -> Response {
    if kind == "down" {
        return (StatusCode::INTERNAL_SERVER_ERROR, "boom").into_response();
    }
    Json(json!({ "object": "list", "data": [
        { "id": "m", "context_window": 4000 },
        { "id": "m2", "context_window": 9000 },
        { "id": "text-embed-x" },
        { "id": "whisper-1" },
    ]}))
    .into_response()
}

struct Harness {
    base: String,
    state: Shared,
    mock: Arc<Mock>,
    http: reqwest::Client,
    key: String,
    free_key: String,
}

async fn spawn(app: Router) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    format!("http://{addr}")
}

async fn harness() -> Harness {
    harness_with("").await
}

/// `extra` is appended to the config, for tests that turn on optional features.
async fn harness_with(extra: &str) -> Harness {
    let mock = Arc::new(Mock::default());
    let up = spawn(
        Router::new()
            .route("/{kind}/chat/completions", post(upstream))
            .route("/{kind}/embeddings", post(upstream_embeddings))
            .route("/{kind}/models", axum::routing::get(upstream_models))
            .with_state(mock.clone()),
    )
    .await;
    let db = std::env::temp_dir().join(format!("aegis-test-{}.db", uuid::Uuid::new_v4().simple()));
    let cfg = Config::parse(&format!(
        r#"
[server]
db_path = "{db}"

[[providers]]
name = "limited"
kind = "openai"
base_url = "{up}/limited"
tier = "free"
models = [{{ id = "m", context_window = 4000, capabilities = ["tools"] }}]

[[providers]]
name = "down"
kind = "openai"
base_url = "{up}/down"
tier = "free"
models = [{{ id = "m" }}]

[[providers]]
name = "bad"
kind = "openai"
base_url = "{up}/bad"
tier = "free"
models = [{{ id = "m" }}]

[[providers]]
name = "midfail"
kind = "openai"
base_url = "{up}/midfail"
tier = "free"
models = [{{ id = "m" }}]

[[providers]]
name = "notools"
kind = "openai"
base_url = "{up}/good"
tier = "local"
models = [{{ id = "small", context_window = 200 }}]

[[providers]]
name = "good"
kind = "openai"
base_url = "{up}/good"
tier = "local"
models = [{{ id = "m", context_window = 4000, capabilities = ["tools"], limits = {{ rpm = 1000 }} }}]

[[providers]]
name = "pricey"
kind = "openai"
base_url = "{up}/good"
tier = "paid"
models = [{{ id = "m", input_per_mtok = 1000000.0, output_per_mtok = 1000000.0 }}]

[[providers]]
name = "nokey"
kind = "openai"
base_url = "{up}/good"
api_key_env = "AEGIS_TEST_UNSET_KEY_VAR"
tier = "free"
models = [{{ id = "m" }}]

[[providers]]
name = "embdown"
kind = "openai"
base_url = "{up}/down"
tier = "local"
models = [{{ id = "emb", capabilities = ["embeddings"] }}]

[[providers]]
name = "embother"
kind = "openai"
base_url = "{up}/good"
tier = "local"
models = [{{ id = "other-emb", capabilities = ["embeddings"] }}]

[[providers]]
name = "embed"
kind = "openai"
base_url = "{up}/good"
tier = "local"
models = [{{ id = "emb", context_window = 50, capabilities = ["embeddings"], input_per_mtok = 2.0 }}]

[[routes]]
alias = "embed"
targets = ["embdown/emb", "embother/other-emb", "embed/emb"]

[[routes]]
alias = "fast"
targets = ["limited/m", "down/m", "good/m"]

[[routes]]
alias = "capable"
targets = ["notools/small", "good/m"]

[[routes]]
alias = "paid-only"
targets = ["nokey/m", "pricey/m"]

[[routes]]
alias = "reject"
targets = ["bad/m", "good/m"]

[[routes]]
alias = "stream-recover"
targets = ["midfail/m", "good/m"]

{extra}
"#,
        db = db.display(),
    ))
    .unwrap();
    let state = AppState::build(cfg, "admin-secret".into(), Some(aegis::secrets::Sealer::new("e2e-test-secret-0123456789abcdef0123").unwrap())).await.unwrap();

    let key = generate_key();
    state.db.create_key("full", "aegis_sk_full", &hash_key(&key), None, Some(5.0), true, "").await.unwrap();
    let free_key = generate_key();
    state.db.create_key("free", "aegis_sk_free", &hash_key(&free_key), Some(3), None, false, "").await.unwrap();
    state.keys.reload(&state.db).await.unwrap();

    let base = spawn(aegis::router(state.clone())).await;
    Harness { base, state, mock, http: reqwest::Client::new(), key, free_key }
}

impl Harness {
    async fn chat(&self, key: &str, body: Value) -> reqwest::Response {
        self.http.post(format!("{}/v1/chat/completions", self.base)).bearer_auth(key).json(&body).send().await.unwrap()
    }
}

fn hdr<'a>(r: &'a reqwest::Response, name: &str) -> &'a str {
    r.headers().get(name).and_then(|v| v.to_str().ok()).unwrap_or("")
}

fn msg(model: &str, text: &str) -> Value {
    json!({ "model": model, "messages": [{ "role": "user", "content": text }] })
}

#[tokio::test]
async fn rejects_missing_and_bad_keys() {
    let h = harness().await;
    let r = h.http.post(format!("{}/v1/chat/completions", h.base)).json(&msg("fast", "hi")).send().await.unwrap();
    assert_eq!(r.status(), 401);
    let r = h.chat("aegis_sk_nope", msg("fast", "hi")).await;
    assert_eq!(r.status(), 401);
    let r = h.chat(&h.key, msg("no-such-model", "hi")).await;
    assert_eq!(r.status(), 404);
}

#[tokio::test]
async fn falls_back_past_rate_limited_and_dead_providers() {
    let h = harness().await;
    let r = h.chat(&h.key, msg("fast", "hi")).await;
    assert_eq!(r.status(), 200);
    assert_eq!(hdr(&r, "x-aegis-provider"), "good");
    assert_eq!(hdr(&r, "x-aegis-fallback"), "true");
    assert_eq!(hdr(&r, "x-aegis-attempts"), "3");
    let body: Value = r.json().await.unwrap();
    assert_eq!(body["choices"][0]["message"]["content"], "Hello");

    // The 429 put `limited` in cooldown for its retry-after, so it is not called again.
    let r = h.chat(&h.key, msg("fast", "hi again")).await;
    assert_eq!(r.status(), 200);
    assert_eq!(h.mock.limited_calls.load(Ordering::SeqCst), 1);
    assert!(h.state.quota.view("limited/m").cooldown_secs > 20);
    // Remaining-quota headers from the provider are tracked too.
    assert_eq!(h.state.quota.view("good/m").remaining_requests, Some(41));

    // With fallback disabled the caller sees the limit instead.
    let r = h
        .http
        .post(format!("{}/v1/chat/completions", h.base))
        .bearer_auth(&h.key)
        .header("x-aegis-no-fallback", "true")
        .json(&msg("fast", "hi"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 429);
    let body: Value = r.json().await.unwrap();
    assert_eq!(body["error"]["type"], "no_available_provider");
}

#[tokio::test]
async fn caches_only_deterministic_requests() {
    let h = harness().await;
    let mut cold = msg("good/m", "what is 2+2?");
    cold["temperature"] = json!(0);

    let r = h.chat(&h.key, cold.clone()).await;
    assert_eq!(hdr(&r, "x-aegis-cache"), "miss");
    let r = h.chat(&h.key, cold.clone()).await;
    assert_eq!(hdr(&r, "x-aegis-cache"), "hit");
    assert_eq!(h.mock.good_calls.load(Ordering::SeqCst), 1);

    // A different client key does not see that entry (per-key scope).
    let r = h.chat(&h.free_key, cold.clone()).await;
    assert_eq!(hdr(&r, "x-aegis-cache"), "miss");

    // Sampled requests always go upstream.
    let mut hot = msg("good/m", "tell me a joke");
    hot["temperature"] = json!(0.9);
    for _ in 0..2 {
        let r = h.chat(&h.key, hot.clone()).await;
        assert_eq!(hdr(&r, "x-aegis-cache"), "skip");
    }
    assert_eq!(h.mock.good_calls.load(Ordering::SeqCst), 4);

    // Bypass header and purge both force a fresh call.
    let r = h
        .http
        .post(format!("{}/v1/chat/completions", h.base))
        .bearer_auth(&h.key)
        .header("x-aegis-cache", "bypass")
        .json(&cold)
        .send()
        .await
        .unwrap();
    assert_eq!(hdr(&r, "x-aegis-cache"), "skip");
    h.state.cache.purge().await;
    let r = h.chat(&h.key, cold).await;
    assert_eq!(hdr(&r, "x-aegis-cache"), "miss");
}

#[tokio::test]
async fn streams_and_replays_cached_streams() {
    let h = harness().await;
    let mut req = msg("good/m", "stream please");
    req["stream"] = json!(true);
    req["temperature"] = json!(0);

    let r = h.chat(&h.key, req.clone()).await;
    assert_eq!(r.status(), 200);
    assert!(hdr(&r, "content-type").starts_with("text/event-stream"));
    let text = r.text().await.unwrap();
    assert!(text.contains("\"content\":\"Hel\"") && text.contains("\"content\":\"lo\""));
    assert!(text.trim_end().ends_with("data: [DONE]"));

    // The assembled stream was cached once it completed; replay it as SSE.
    tokio::time::sleep(Duration::from_millis(100)).await;
    let r = h.chat(&h.key, req.clone()).await;
    assert_eq!(hdr(&r, "x-aegis-cache"), "hit");
    let text = r.text().await.unwrap();
    assert!(text.contains("\"content\":\"Hello\"") && text.contains("[DONE]"));
    assert_eq!(h.mock.good_calls.load(Ordering::SeqCst), 1);

    // ...and the same entry answers a non-streaming request.
    req["stream"] = json!(false);
    let r = h.chat(&h.key, req).await;
    assert_eq!(hdr(&r, "x-aegis-cache"), "hit");
    let body: Value = r.json().await.unwrap();
    assert_eq!(body["choices"][0]["message"]["content"], "Hello");
    assert_eq!(body["usage"]["prompt_tokens"], 7);
}

#[tokio::test]
async fn stream_failing_before_first_chunk_falls_back() {
    let h = harness().await;
    let mut req = msg("stream-recover", "hi");
    req["stream"] = json!(true);
    let r = h.chat(&h.key, req).await;
    assert_eq!(r.status(), 200);
    assert_eq!(hdr(&r, "x-aegis-provider"), "good");
    assert_eq!(hdr(&r, "x-aegis-fallback"), "true");
    assert!(r.text().await.unwrap().contains("Hel"));
}

#[tokio::test]
async fn routes_by_capability_and_context_window() {
    let h = harness().await;
    // Plain request: the small model is first and can take it.
    let r = h.chat(&h.key, msg("capable", "hi")).await;
    assert_eq!(hdr(&r, "x-aegis-model"), "small");

    // Tools needed: the small model cannot, so it is skipped.
    let mut with_tools = msg("capable", "hi");
    with_tools["tools"] = json!([{ "type": "function", "function": { "name": "f", "parameters": { "type": "object" } } }]);
    let r = h.chat(&h.key, with_tools).await;
    assert_eq!(hdr(&r, "x-aegis-provider"), "good");
    assert_eq!(hdr(&r, "x-aegis-fallback"), "true");

    // Too long for the 200-token window: skipped as well.
    let r = h.chat(&h.key, msg("capable", &"word ".repeat(400))).await;
    assert_eq!(hdr(&r, "x-aegis-provider"), "good");
}

#[tokio::test]
async fn paid_tier_needs_permission_and_budget() {
    let h = harness().await;
    // Free-only key: unconfigured provider skipped, paid provider refused.
    let r = h.chat(&h.free_key, msg("paid-only", "hi")).await;
    assert_eq!(r.status(), 503);
    let body: Value = r.json().await.unwrap();
    let attempts = body["error"]["attempts"].as_array().unwrap();
    assert!(attempts[0]["reason"].as_str().unwrap().contains("not configured"));
    assert!(attempts[1]["reason"].as_str().unwrap().contains("paid"));

    // Permitted key is served, and the (absurd) price exhausts its $5 budget.
    let r = h.chat(&h.key, msg("paid-only", "hi")).await;
    assert_eq!(r.status(), 200);
    assert_eq!(hdr(&r, "x-aegis-tier"), "paid");
    let r = h.chat(&h.key, msg("paid-only", "hi")).await;
    assert_eq!(r.status(), 429);
    assert!(r.json::<Value>().await.unwrap()["error"]["message"].as_str().unwrap().contains("budget"));
}

#[tokio::test]
async fn provider_rejections_are_passed_through_without_fallback() {
    let h = harness().await;
    let r = h.chat(&h.key, msg("reject", "hi")).await;
    assert_eq!(r.status(), 400);
    assert_eq!(h.mock.good_calls.load(Ordering::SeqCst), 0);
    assert!(r.json::<Value>().await.unwrap()["error"]["message"].as_str().unwrap().contains("bad parameter"));
}

#[tokio::test]
async fn enforces_key_rate_limit_and_logs_requests() {
    let h = harness().await;
    for _ in 0..3 {
        assert_eq!(h.chat(&h.free_key, msg("good/m", "hi")).await.status(), 200);
    }
    let r = h.chat(&h.free_key, msg("good/m", "hi")).await;
    assert_eq!(r.status(), 429);

    tokio::time::sleep(Duration::from_millis(200)).await;
    let rows = h.state.db.recent_requests(10).await.unwrap();
    assert_eq!(rows.len(), 4);
    assert_eq!(rows[0].status, "rejected");
    assert_eq!(rows[1].status, "ok");
    assert_eq!((rows[1].prompt_tokens, rows[1].completion_tokens), (7, 2));
    let s = h.state.db.summary(0).await.unwrap();
    assert_eq!(s.requests, 4);
}

#[tokio::test]
async fn dashboard_requires_sign_in() {
    let h = harness().await;
    let client = reqwest::Client::builder().redirect(reqwest::redirect::Policy::none()).build().unwrap();
    let r = client.get(format!("{}/ui", h.base)).send().await.unwrap();
    assert_eq!(r.status(), 303);

    let r = client.post(format!("{}/ui/login", h.base)).body("token=wrong").header("content-type", "application/x-www-form-urlencoded").send().await.unwrap();
    assert_eq!(r.status(), 401);

    let r = client
        .post(format!("{}/ui/login", h.base))
        .body("token=admin-secret")
        .header("content-type", "application/x-www-form-urlencoded")
        .send()
        .await
        .unwrap();
    let cookie = r.headers().get("set-cookie").unwrap().to_str().unwrap().split(';').next().unwrap().to_string();
    for page in ["/ui", "/ui/requests", "/ui/providers", "/ui/keys", "/ui/cache", "/ui/playground"] {
        let r = client.get(format!("{}{page}", h.base)).header("cookie", &cookie).send().await.unwrap();
        assert_eq!(r.status(), 200, "{page}");
        assert!(r.text().await.unwrap().contains("Aegis"));
    }
    // The playground endpoint refuses unauthenticated callers.
    let r = client.post(format!("{}/ui/playground/chat", h.base)).json(&msg("good/m", "hi")).send().await.unwrap();
    assert_eq!(r.status(), 401);
}

#[tokio::test]
async fn provider_key_set_from_dashboard_is_encrypted_and_used_at_once() {
    let h = harness().await;
    let client = reqwest::Client::builder().redirect(reqwest::redirect::Policy::none()).build().unwrap();
    let form = |path: &str, body: &str| {
        client.post(format!("{}{path}", h.base)).body(body.to_string()).header("content-type", "application/x-www-form-urlencoded")
    };

    // No key yet: the target is skipped.
    assert_eq!(h.chat(&h.key, msg("nokey/m", "hi")).await.status(), 503);

    // Saving a key needs a signed-in admin.
    let r = form("/ui/providers/key", "provider=nokey&key=sk-test-provider-key-1234").send().await.unwrap();
    assert_eq!(r.status(), 303);
    assert!(h.state.adapters["nokey"].api_key().is_none());

    let r = form("/ui/login", "token=admin-secret").send().await.unwrap();
    let cookie = r.headers().get("set-cookie").unwrap().to_str().unwrap().split(';').next().unwrap().to_string();

    let r = form("/ui/providers/key", "provider=nokey&key=has+spaces").header("cookie", &cookie).send().await.unwrap();
    assert_eq!(r.status(), 400);
    let r = form("/ui/providers/key", "provider=good&key=sk-test-provider-key-1234").header("cookie", &cookie).send().await.unwrap();
    assert_eq!(r.status(), 400, "providers without api_key_env take no key");

    let r = form("/ui/providers/key", "provider=nokey&key=sk-test-provider-key-1234").header("cookie", &cookie).send().await.unwrap();
    assert_eq!(r.status(), 303);

    // Used without a restart, and stored only as ciphertext.
    let r = h.chat(&h.key, msg("nokey/m", "hi")).await;
    assert_eq!(r.status(), 200);
    assert_eq!(hdr(&r, "x-aegis-provider"), "nokey");
    let rows = h.state.db.list_provider_keys().await.unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].hint, "1234");
    assert!(!String::from_utf8_lossy(&rows[0].ciphertext).contains("sk-test"));
    let opened = h.state.sealer.as_ref().unwrap().open("nokey", &rows[0].nonce, &rows[0].ciphertext).unwrap();
    assert_eq!(opened, "sk-test-provider-key-1234");

    // The page shows the hint, never the key.
    let page = client.get(format!("{}/ui/providers", h.base)).header("cookie", &cookie).send().await.unwrap().text().await.unwrap();
    assert!(page.contains("1234") && !page.contains("sk-test-provider-key"));

    let r = form("/ui/providers/key/remove", "provider=nokey").header("cookie", &cookie).send().await.unwrap();
    assert_eq!(r.status(), 303);
    assert_eq!(h.chat(&h.key, msg("nokey/m", "hi")).await.status(), 503);
    assert!(h.state.db.list_provider_keys().await.unwrap().is_empty());
}

impl Harness {
    async fn post(&self, path: &str, key: &str, headers: &[(&str, &str)], body: &Value) -> reqwest::Response {
        let mut rb = self.http.post(format!("{}{path}", self.base)).bearer_auth(key).json(body);
        for (k, v) in headers {
            rb = rb.header(*k, *v);
        }
        rb.send().await.unwrap()
    }
}

#[tokio::test]
async fn embeddings_fall_back_only_to_the_same_model() {
    let h = harness().await;
    let body = json!({ "model": "embed", "input": ["first text", "second text"] });
    let r = h.post("/v1/embeddings", &h.key, &[], &body).await;
    assert_eq!(r.status(), 200);
    assert_eq!(hdr(&r, "x-aegis-provider"), "embed");
    assert_eq!(hdr(&r, "x-aegis-fallback"), "true");
    let v: Value = r.json().await.unwrap();
    assert_eq!(v["model"], "emb");
    assert_eq!(v["usage"]["prompt_tokens"], 10);
    // Order follows the input even though the provider answered out of order.
    let want: Vec<f64> = toy_embedding("second text").iter().map(|f| *f as f64).collect();
    let got: Vec<f64> = v["data"][1]["embedding"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
    assert_eq!(v["data"][1]["index"], 1);
    assert_eq!(got, want);

    // The log shows the dead provider and the skipped different model.
    tokio::time::sleep(Duration::from_millis(300)).await;
    let rows = h.state.db.recent_requests(10).await.unwrap();
    let attempts: Value = serde_json::from_str(&rows[0].attempts).unwrap();
    assert_eq!(attempts[0]["outcome"], "failed");
    assert!(attempts[1]["reason"].as_str().unwrap().contains("different embedding model"));
    assert_eq!(attempts[2]["outcome"], "served");
    assert!(rows[0].cost_usd > 0.0);

    // base64 output, as the OpenAI Python SDK asks for by default.
    let r = h.post("/v1/embeddings", &h.key, &[], &json!({ "model": "embed/emb", "input": "first text", "encoding_format": "base64" })).await;
    let v: Value = r.json().await.unwrap();
    use base64::Engine;
    let bytes = base64::engine::general_purpose::STANDARD.decode(v["data"][0]["embedding"].as_str().unwrap()).unwrap();
    let decoded: Vec<f32> = bytes.chunks(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    assert_eq!(decoded, toy_embedding("first text"));

    // Bad input, a chat model, an oversized input, and no key.
    let r = h.post("/v1/embeddings", &h.key, &[], &json!({ "model": "embed", "input": [[1, 2, 3]] })).await;
    assert_eq!(r.status(), 400);
    let r = h.post("/v1/embeddings", &h.key, &[], &json!({ "model": "good/m", "input": "hi" })).await;
    assert_eq!(r.status(), 503);
    assert!(r.text().await.unwrap().contains("not an embedding model"));
    let r = h.post("/v1/embeddings", &h.key, &[], &json!({ "model": "embed/emb", "input": "x".repeat(400) })).await;
    assert_eq!(r.status(), 503);
    let r = h.http.post(format!("{}/v1/embeddings", h.base)).json(&body).send().await.unwrap();
    assert_eq!(r.status(), 401);

    // And an embedding model never answers chat.
    let r = h.chat(&h.key, msg("embed/emb", "hi")).await;
    assert_eq!(r.status(), 503);
    assert!(r.text().await.unwrap().contains("embedding model cannot serve chat"));
}

const SEMANTIC: &str = r#"
[cache.semantic]
enabled = true
embedding_model = "embed/emb"
threshold = 0.95
min_chars = 10
"#;

#[tokio::test]
async fn semantic_cache_matches_rewordings_but_not_other_questions() {
    let h = harness_with(SEMANTIC).await;
    let ask = |text: &str| {
        let mut m = msg("good/m", text);
        m["temperature"] = json!(0);
        m
    };

    let r = h.chat(&h.key, ask("What is the capital of France?")).await;
    assert_eq!(hdr(&r, "x-aegis-cache"), "miss");
    assert_eq!(h.mock.good_calls.load(Ordering::SeqCst), 1);

    // Same words, different case and punctuation: served without an upstream call.
    let r = h.chat(&h.key, ask("what is the capital of france")).await;
    assert_eq!(r.status(), 200);
    assert_eq!(hdr(&r, "x-aegis-cache"), "semantic");
    assert!(hdr(&r, "x-aegis-cache-similarity").parse::<f64>().unwrap() >= 0.95);
    assert_eq!(r.json::<Value>().await.unwrap()["choices"][0]["message"]["content"], "Hello");
    assert_eq!(h.mock.good_calls.load(Ordering::SeqCst), 1);

    // Streaming clients get the stored answer replayed.
    let mut s = ask("WHAT IS THE CAPITAL OF FRANCE");
    s["stream"] = json!(true);
    let r = h.chat(&h.key, s).await;
    assert_eq!(hdr(&r, "x-aegis-cache"), "semantic");
    assert!(r.text().await.unwrap().contains("\"content\":\"Hello\""));

    // A different question goes upstream.
    let r = h.chat(&h.key, ask("How do I bake sourdough bread at home?")).await;
    assert_eq!(hdr(&r, "x-aegis-cache"), "miss");
    assert_eq!(h.mock.good_calls.load(Ordering::SeqCst), 2);

    // Same question under a different system prompt is a different conversation.
    let mut sys = ask("what is the capital of france");
    sys["messages"] = json!([{ "role": "system", "content": "Answer in French." }, { "role": "user", "content": "what is the capital of france" }]);
    assert_eq!(hdr(&h.chat(&h.key, sys).await, "x-aegis-cache"), "miss");

    // Another client key, the exact-only header, and sampled requests never match.
    assert_eq!(hdr(&h.chat(&h.free_key, ask("what is the capital of france")).await, "x-aegis-cache"), "miss");
    let r = h.post("/v1/chat/completions", &h.key, &[("x-aegis-cache", "exact")], &ask("What is the capital of France")).await;
    assert_eq!(hdr(&r, "x-aegis-cache"), "miss");
    let mut hot = ask("what is the capital of france");
    hot["temperature"] = json!(0.9);
    assert_eq!(hdr(&h.chat(&h.key, hot).await, "x-aegis-cache"), "skip");

    // Too short to match by meaning.
    h.chat(&h.key, ask("Yes ok")).await;
    assert_eq!(hdr(&h.chat(&h.key, ask("yes ok")).await, "x-aegis-cache"), "miss");

    let stats = h.state.semantic.stats();
    assert_eq!(stats.hits, 2);
    assert!(stats.stores >= 2);

    // Logged as a semantic hit, and counted as a cache hit in the summary.
    tokio::time::sleep(Duration::from_millis(300)).await;
    let rows = h.state.db.recent_requests(50).await.unwrap();
    assert_eq!(rows.iter().filter(|r| r.cache == "semantic").count(), 2);
    assert_eq!(h.state.db.summary(0).await.unwrap().cache_hits, 2);

    // Purging from the dashboard empties the semantic index too.
    let client = reqwest::Client::builder().redirect(reqwest::redirect::Policy::none()).build().unwrap();
    let r = client.post(format!("{}/ui/login", h.base)).body("token=admin-secret").header("content-type", "application/x-www-form-urlencoded").send().await.unwrap();
    let cookie = r.headers().get("set-cookie").unwrap().to_str().unwrap().split(';').next().unwrap().to_string();
    client.post(format!("{}/ui/cache/purge", h.base)).header("cookie", &cookie).send().await.unwrap();
    assert_eq!(h.state.semantic.stats().entries, 0);
    assert_eq!(hdr(&h.chat(&h.key, ask("what is the capital of france")).await, "x-aegis-cache"), "miss");
}

#[tokio::test]
async fn semantic_cache_survives_a_dead_embedding_model() {
    let h = harness_with(&SEMANTIC.replace("embed/emb", "embdown/emb")).await;
    let mut m = msg("good/m", "What is the capital of France?");
    m["temperature"] = json!(0);
    for expected in ["miss", "hit"] {
        let r = h.chat(&h.key, m.clone()).await;
        assert_eq!(r.status(), 200);
        assert_eq!(hdr(&r, "x-aegis-cache"), expected);
    }
    assert!(h.state.semantic.stats().embed_failures >= 1);
}

#[tokio::test]
async fn alerts_fire_for_budget_and_cooldown_and_reach_the_webhook() {
    let h = harness().await;
    assert!(h.state.alerts.recent().is_empty());

    // limited/m answers 429: one cooldown alert, not one per request.
    h.chat(&h.key, msg("fast", "hi")).await;
    h.chat(&h.key, msg("fast", "hi")).await;
    let cool: Vec<_> = h.state.alerts.recent().into_iter().filter(|a| a.kind == "cooldown").collect();
    assert_eq!(cool.iter().filter(|a| a.title.contains("limited/m")).count(), 1);
    assert!(cool.iter().any(|a| a.detail.contains("slow down")));

    // The absurdly priced model blows the $5 budget in one request.
    h.chat(&h.key, msg("paid-only", "hi")).await;
    let budget: Vec<_> = h.state.alerts.recent().into_iter().filter(|a| a.kind == "budget").collect();
    assert_eq!(budget.len(), 1);
    assert_eq!(budget[0].severity, aegis::alerts::Severity::Critical);
    assert!(budget[0].title.contains("full"));

    // The Alerts page lists them, and its test button raises one more.
    let client = reqwest::Client::builder().redirect(reqwest::redirect::Policy::none()).build().unwrap();
    let r = client.post(format!("{}/ui/login", h.base)).body("token=admin-secret").header("content-type", "application/x-www-form-urlencoded").send().await.unwrap();
    let cookie = r.headers().get("set-cookie").unwrap().to_str().unwrap().split(';').next().unwrap().to_string();
    let page = client.get(format!("{}/ui/alerts", h.base)).header("cookie", &cookie).send().await.unwrap().text().await.unwrap();
    assert!(page.contains("has used its daily budget") && page.contains("limited/m is cooling down"));
    assert_eq!(client.post(format!("{}/ui/alerts/test", h.base)).send().await.unwrap().status(), 303);
    let before = h.state.alerts.recent().len();
    let r = client.post(format!("{}/ui/alerts/test", h.base)).header("cookie", &cookie).send().await.unwrap();
    assert!(r.text().await.unwrap().contains("Test alert"));
    assert_eq!(h.state.alerts.recent().len(), before + 1);

    // Delivery: a configured webhook receives the alert as JSON.
    let got: Arc<std::sync::Mutex<Vec<Value>>> = Arc::default();
    let sink = got.clone();
    let hook = spawn(Router::new().route(
        "/hook",
        post(move |Json(v): Json<Value>| {
            let sink = sink.clone();
            async move {
                sink.lock().unwrap().push(v);
                StatusCode::NO_CONTENT
            }
        }),
    ))
    .await;
    let alerts = aegis::alerts::Alerts::new(&aegis::config::AlertsCfg::default(), Some(format!("{hook}/hook")));
    assert!(alerts.webhook_configured());
    alerts.on_cooldown("groq/m", 30, "rate limited by provider");
    for _ in 0..50 {
        if alerts.delivery().sent == 1 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let got = got.lock().unwrap().clone();
    assert_eq!(got.len(), 1);
    assert_eq!(got[0]["event"], "cooldown");
    assert!(got[0]["text"].as_str().unwrap().contains("groq/m is cooling down"));

    // A dead webhook is reported, not fatal.
    let dead = aegis::alerts::Alerts::new(&aegis::config::AlertsCfg::default(), Some("http://127.0.0.1:1/hook".into()));
    dead.on_cooldown("groq/m", 30, "x");
    for _ in 0..100 {
        if dead.delivery().failed == 1 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(dead.delivery().last_error.unwrap().contains("unreachable"));
}

#[tokio::test]
async fn provider_test_button_reports_working_and_broken_providers() {
    let h = harness().await;
    let client = reqwest::Client::builder().redirect(reqwest::redirect::Policy::none()).build().unwrap();
    let form = |body: &str| client.post(format!("{}/ui/providers/test", h.base)).body(body.to_string()).header("content-type", "application/x-www-form-urlencoded");
    assert_eq!(form("provider=good").send().await.unwrap().status(), 303);

    let r = client.post(format!("{}/ui/login", h.base)).body("token=admin-secret").header("content-type", "application/x-www-form-urlencoded").send().await.unwrap();
    let cookie = r.headers().get("set-cookie").unwrap().to_str().unwrap().split(';').next().unwrap().to_string();

    let page = form("provider=good").header("cookie", &cookie).send().await.unwrap().text().await.unwrap();
    assert!(page.contains("good is working: m answered in"));
    let page = form("provider=down").header("cookie", &cookie).send().await.unwrap().text().await.unwrap();
    assert!(page.contains("down could not be reached or failed"));
    let page = form("provider=limited").header("cookie", &cookie).send().await.unwrap().text().await.unwrap();
    assert!(page.contains("rate limiting it right now"));
    let page = form("provider=nokey").header("cookie", &cookie).send().await.unwrap().text().await.unwrap();
    assert!(page.contains("nokey has no API key yet"));
    // Embedding-only providers are tested with an embedding.
    let page = form("provider=embed").header("cookie", &cookie).send().await.unwrap().text().await.unwrap();
    assert!(page.contains("embed is working"));
    // A failed test does not trip the circuit breaker.
    assert_eq!(h.state.quota.view("down/m").failures, 0);
}

#[tokio::test]
async fn semantic_cache_survives_a_restart_until_purged() {
    let h = harness_with(SEMANTIC).await;
    let ask = |text: &str| {
        let mut m = msg("good/m", text);
        m["temperature"] = json!(0);
        m
    };
    let restart = || async {
        let state = AppState::build(h.state.base_cfg.clone(), "admin-secret".into(), None).await.unwrap();
        state.keys.reload(&state.db).await.unwrap();
        let base = spawn(aegis::router(state.clone())).await;
        (state, base)
    };
    let chat = |base: String, body: Value| {
        let (http, key) = (h.http.clone(), h.key.clone());
        async move { http.post(format!("{base}/v1/chat/completions")).bearer_auth(key).json(&body).send().await.unwrap() }
    };

    assert_eq!(hdr(&h.chat(&h.key, ask("What is the capital of France?")).await, "x-aegis-cache"), "miss");
    assert_eq!(h.mock.good_calls.load(Ordering::SeqCst), 1);

    // A second gateway on the same database stands in for a restart.
    let (state2, base2) = restart().await;
    assert_eq!(state2.semantic.stats().entries, 1);
    let r = chat(base2.clone(), ask("what is the capital of france")).await;
    assert_eq!(hdr(&r, "x-aegis-cache"), "semantic");
    assert_eq!(r.json::<Value>().await.unwrap()["choices"][0]["message"]["content"], "Hello");
    assert_eq!(h.mock.good_calls.load(Ordering::SeqCst), 1);

    // A purge empties the saved copy too, so nothing comes back afterwards.
    state2.purge_caches().await;
    let (state3, base3) = restart().await;
    assert_eq!(state3.semantic.stats().entries, 0);
    assert_eq!(hdr(&chat(base3, ask("what is the capital of france")).await, "x-aegis-cache"), "miss");

    // Entries made with another embedding model, and expired ones, are not loaded.
    let db = &h.state.db;
    db.insert_semantic("some/other-model", "p", &[1.0, 0.0], "{}", i64::MAX, 100).await.unwrap();
    assert!(db.load_semantic("embed/emb").await.unwrap().iter().all(|r| r.0 != "p"));
    db.insert_semantic("embed/emb", "old", &[1.0, 0.0], "{}", 1, 100).await.unwrap();
    assert!(db.load_semantic("embed/emb").await.unwrap().iter().all(|r| r.0 != "old"));
    // The table is capped at the newest `max` rows.
    for i in 0..5 {
        db.insert_semantic("embed/emb", &format!("p{i}"), &[1.0, 0.0], "{}", i64::MAX, 3).await.unwrap();
    }
    let kept: Vec<String> = db.load_semantic("embed/emb").await.unwrap().into_iter().map(|r| r.0).collect();
    assert_eq!(kept, ["p2", "p3", "p4"]);
}

#[tokio::test]
async fn exact_cache_survives_a_restart_and_embedding_model_is_warmed() {
    let h = harness_with(SEMANTIC).await;
    // Startup sends one warm-up embedding before any request arrives.
    for _ in 0..50 {
        if h.mock.embed_calls.load(Ordering::SeqCst) >= 1 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert_eq!(h.mock.embed_calls.load(Ordering::SeqCst), 1);
    assert!(h.state.warm_embedding_model().await);

    // "Hi" is too short for the semantic layer, so only the exact cache can serve it.
    let mut m = msg("good/m", "Hi");
    m["temperature"] = json!(0);
    assert_eq!(hdr(&h.chat(&h.key, m.clone()).await, "x-aegis-cache"), "miss");
    let mut s = m.clone();
    s["stream"] = json!(true);
    s["messages"][0]["content"] = json!("Yo");
    h.chat(&h.key, s.clone()).await.text().await.unwrap();
    tokio::time::sleep(Duration::from_millis(200)).await;

    let restart = || async {
        let state = AppState::build(h.state.base_cfg.clone(), "admin-secret".into(), None).await.unwrap();
        state.keys.reload(&state.db).await.unwrap();
        let base = spawn(aegis::router(state.clone())).await;
        (state, base)
    };
    let (state2, base2) = restart().await;
    let before = h.mock.good_calls.load(Ordering::SeqCst);
    for body in [&m, &s] {
        let r = h.http.post(format!("{base2}/v1/chat/completions")).bearer_auth(&h.key).json(body).send().await.unwrap();
        assert_eq!(hdr(&r, "x-aegis-cache"), "hit");
        assert!(r.text().await.unwrap().contains("Hel"));
    }
    assert_eq!(h.mock.good_calls.load(Ordering::SeqCst), before);

    // Purging removes the saved copy; an expired row is never loaded.
    state2.purge_caches().await;
    assert!(h.state.db.load_cached().await.unwrap().is_empty());
    h.state.db.insert_cached("stale", "{}", 1, 10).await.unwrap();
    assert!(h.state.db.load_cached().await.unwrap().is_empty());
    let (_, base3) = restart().await;
    let r = h.http.post(format!("{base3}/v1/chat/completions")).bearer_auth(&h.key).json(&m).send().await.unwrap();
    assert_eq!(hdr(&r, "x-aegis-cache"), "miss");
}

#[tokio::test]
async fn usage_is_broken_down_by_client_key() {
    let h = harness().await;
    let mut cold = msg("good/m", "same question");
    cold["temperature"] = json!(0);
    h.chat(&h.key, cold.clone()).await; // upstream
    h.chat(&h.key, cold.clone()).await; // cache hit
    h.chat(&h.key, msg("paid-only", "hi")).await; // costs money
    h.chat(&h.free_key, msg("good/m", "hi")).await;
    h.chat(&h.free_key, msg("reject", "hi")).await; // provider error
    tokio::time::sleep(Duration::from_millis(300)).await;

    let rows = h.state.db.by_key(0).await.unwrap();
    assert_eq!(rows.len(), 2);
    // Biggest spender first.
    let (full, free) = (&rows[0], &rows[1]);
    assert_eq!((full.key_name.as_str(), full.requests, full.cache_hits, full.errors), ("full", 3, 1, 0));
    assert!(full.cost_usd > 0.0);
    // 9 tokens for each of the two upstream calls; the cache hit adds none.
    assert_eq!(full.tokens, 18);
    assert_eq!((free.key_name.as_str(), free.requests, free.cache_hits, free.errors), ("free", 2, 0, 1));
    assert_eq!(free.cost_usd, 0.0);
    assert!(h.state.db.by_key(aegis::types::now_ms() + 60_000).await.unwrap().is_empty());

    let client = reqwest::Client::builder().redirect(reqwest::redirect::Policy::none()).build().unwrap();
    let r = client.post(format!("{}/ui/login", h.base)).body("token=admin-secret").header("content-type", "application/x-www-form-urlencoded").send().await.unwrap();
    let cookie = r.headers().get("set-cookie").unwrap().to_str().unwrap().split(';').next().unwrap().to_string();
    for days in ["", "?days=7", "?days=30", "?days=999"] {
        let page = client.get(format!("{}/ui/keys{days}", h.base)).header("cookie", &cookie).send().await.unwrap().text().await.unwrap();
        assert!(page.contains("Usage by key") && page.contains("100%"), "{days}");
    }
}

#[tokio::test]
async fn models_can_be_chosen_from_the_providers_list() {
    let h = harness().await;
    let client = reqwest::Client::builder().redirect(reqwest::redirect::Policy::none()).build().unwrap();
    let r = client.post(format!("{}/ui/login", h.base)).body("token=admin-secret").header("content-type", "application/x-www-form-urlencoded").send().await.unwrap();
    let cookie = r.headers().get("set-cookie").unwrap().to_str().unwrap().split(';').next().unwrap().to_string();
    let post = |path: &str, body: &str, signed_in: bool| {
        let mut rb = client.post(format!("{}{path}", h.base)).body(body.to_string()).header("content-type", "application/x-www-form-urlencoded");
        if signed_in {
            rb = rb.header("cookie", &cookie);
        }
        rb
    };
    let text = |r: reqwest::Response| async move { r.text().await.unwrap() };
    let model_of = |r: reqwest::Response| async move { r.json::<Value>().await.unwrap()["model"].as_str().unwrap_or("").to_string() };

    // The page lists what the provider offers.
    let page = text(client.get(format!("{}/ui/providers/good/models", h.base)).header("cookie", &cookie).send().await.unwrap()).await;
    assert!(page.contains("Models in use") && page.contains("m2") && page.contains("not usable here"));
    assert!(page.contains("lists 2 chat and 1 embedding models"));

    // Not signed in: nothing changes.
    assert_eq!(post("/ui/providers/good/models/replace", "from=m&to=m2", false).send().await.unwrap().status(), 303);
    assert_eq!(model_of(h.chat(&h.key, msg("fast", "hi")).await).await, "m");

    // Replace m with m2: routes follow, the old id is gone, the context window comes from the list.
    let page = text(post("/ui/providers/good/models/replace", "from=m&to=m2", true).send().await.unwrap()).await;
    assert!(page.contains("good now uses m2 in place of m"));
    let r = h.chat(&h.key, msg("fast", "hi")).await;
    assert_eq!(hdr(&r, "x-aegis-model"), "m2");
    assert_eq!(model_of(r).await, "m2");
    assert_eq!(h.chat(&h.key, msg("good/m", "hi")).await.status(), 404);
    let cfg = h.state.cfg();
    let good = cfg.providers.iter().find(|p| p.name == "good").unwrap();
    assert_eq!((good.models[0].id.as_str(), good.models[0].context_window), ("m2", 9000));
    assert!(good.models[0].has("tools"), "capabilities carry over");
    assert!(cfg.routes.iter().find(|r| r.alias == "fast").unwrap().targets.contains(&"good/m2".to_string()));

    // It survives a restart.
    let state2 = AppState::build(h.state.base_cfg.clone(), "admin-secret".into(), None).await.unwrap();
    assert_eq!(state2.cfg().providers.iter().find(|p| p.name == "good").unwrap().models[0].id, "m2");

    // Things that must be refused.
    let page = text(post("/ui/providers/good/models/replace", "from=m&to=whisper-1", true).send().await.unwrap()).await;
    assert!(page.contains("not a chat or embedding model"));
    let page = text(post("/ui/providers/good/models/replace", "from=nope&to=m2", true).send().await.unwrap()).await;
    assert!(page.contains("has no model nope in the config file"));
    let page = text(post("/ui/providers/good/models/add", "model=m2", true).send().await.unwrap()).await;
    assert!(page.contains("good already has m2"));
    let page = text(post("/ui/providers/good/models/replace", "from=m&to=has+space", true).send().await.unwrap()).await;
    assert!(page.contains("no spaces"));

    // Back to the config file's model.
    let page = text(post("/ui/providers/good/models/restore", "from=m", true).send().await.unwrap()).await;
    assert!(page.contains("good is back to m"));
    assert_eq!(model_of(h.chat(&h.key, msg("fast", "hi")).await).await, "m");

    // Add an extra model: callable by its id, not part of any route.
    let page = text(post("/ui/providers/good/models/add", "model=m2&tools=on", true).send().await.unwrap()).await;
    assert!(page.contains("Apps can now call good/m2"));
    assert_eq!(model_of(h.chat(&h.key, msg("good/m2", "hi")).await).await, "m2");
    assert_eq!(model_of(h.chat(&h.key, msg("fast", "hi")).await).await, "m");
    let cfg = h.state.cfg();
    let added = cfg.providers.iter().find(|p| p.name == "good").unwrap().models.iter().find(|m| m.id == "m2").unwrap().clone();
    assert!(added.has("tools") && !added.has("vision") && added.context_window == 9000);
    let listed: Value = h.http.get(format!("{}/v1/models", h.base)).bearer_auth(&h.key).send().await.unwrap().json().await.unwrap();
    assert!(listed["data"].as_array().unwrap().iter().any(|m| m["id"] == "good/m2"));

    // An embedding model is recognised from the list and serves embeddings.
    post("/ui/providers/good/models/add", "model=text-embed-x&tools=on", true).send().await.unwrap();
    let r = h.post("/v1/embeddings", &h.key, &[], &json!({ "model": "good/text-embed-x", "input": "hello" })).await;
    assert_eq!(r.status(), 200);

    // Only dashboard-added models can be removed here.
    let page = text(post("/ui/providers/good/models/remove", "model=m", true).send().await.unwrap()).await;
    assert!(page.contains("can only be changed in the config file"));
    let page = text(post("/ui/providers/good/models/remove", "model=m2", true).send().await.unwrap()).await;
    assert!(page.contains("Removed m2"));
    assert_eq!(h.chat(&h.key, msg("good/m2", "hi")).await.status(), 404);

    // When the provider's list cannot be fetched, a model id can still be typed.
    let page = text(client.get(format!("{}/ui/providers/down/models", h.base)).header("cookie", &cookie).send().await.unwrap()).await;
    assert!(page.contains("Could not fetch the model list"));
    let page = text(post("/ui/providers/down/models/replace", "from=m&to=typed-by-hand", true).send().await.unwrap()).await;
    assert!(page.contains("down now uses typed-by-hand"));
    assert_eq!(client.get(format!("{}/ui/providers/nope/models", h.base)).header("cookie", &cookie).send().await.unwrap().status(), 404);

    // Saving a provider key lands on that provider's model list.
    let r = post("/ui/providers/key", "provider=nokey&key=sk-test-provider-key-1234", true).send().await.unwrap();
    assert_eq!(r.headers().get("location").unwrap(), "/ui/providers/nokey/models");
}

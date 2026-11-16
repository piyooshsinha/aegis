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
    let mock = Arc::new(Mock::default());
    let up = spawn(Router::new().route("/{kind}/chat/completions", post(upstream)).with_state(mock.clone())).await;
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

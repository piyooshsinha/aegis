//! Admin dashboard, rendered server-side in Rust with `maud`.
//!
//! Signed-in admins get an overview, the request log, provider health and
//! quotas, client key management, cache controls and a playground.

use crate::auth::{Identity, constant_time_eq, generate_key, hash_key, random_token};
use crate::db::{HourBucket, RequestRecord};
use crate::gateway::{ChatOpts, Shared, run_chat};
use crate::providers::KeySource;
use crate::types::{ChatRequest, now_ms};
use axum::extract::{Form, Path, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Redirect, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use maud::{DOCTYPE, Markup, PreEscaped, html};
use serde::Deserialize;
use serde_json::Value;
use std::sync::Arc;

const COOKIE: &str = "aegis_session";

pub fn routes() -> Router<Shared> {
    Router::new()
        .route("/", get(|| async { Redirect::to("/ui") }))
        .route("/ui/login", get(login_page).post(login))
        .route("/ui/logout", post(logout))
        .route("/ui", get(overview))
        .route("/ui/requests", get(requests))
        .route("/ui/providers", get(providers))
        .route("/ui/providers/reset", post(reset_target))
        .route("/ui/providers/key", post(set_provider_key))
        .route("/ui/providers/key/remove", post(remove_provider_key))
        .route("/ui/keys", get(keys).post(create_key))
        .route("/ui/keys/{id}/revoke", post(revoke_key))
        .route("/ui/cache", get(cache))
        .route("/ui/cache/purge", post(purge_cache))
        .route("/ui/playground", get(playground))
        .route("/ui/playground/chat", post(playground_chat))
}

// ---- session --------------------------------------------------------------

fn session_token(h: &HeaderMap) -> Option<String> {
    h.get(header::COOKIE)?.to_str().ok()?.split(';').find_map(|c| {
        let (k, v) = c.trim().split_once('=')?;
        (k == COOKIE).then(|| v.to_string())
    })
}

fn signed_in(state: &Shared, h: &HeaderMap) -> bool {
    session_token(h).is_some_and(|t| state.sessions.lock().unwrap_or_else(|e| e.into_inner()).contains(&t))
}

macro_rules! require_admin {
    ($state:expr, $headers:expr) => {
        if !signed_in(&$state, &$headers) {
            return Redirect::to("/ui/login").into_response();
        }
    };
}

#[derive(Deserialize)]
struct LoginForm {
    token: String,
}

async fn login_page(State(state): State<Shared>, headers: HeaderMap) -> Response {
    if signed_in(&state, &headers) {
        return Redirect::to("/ui").into_response();
    }
    login_view(None).into_response()
}

fn login_view(error: Option<&str>) -> Markup {
    html! {
        (DOCTYPE)
        html lang="en" {
            head { (head("Sign in")) }
            body.login {
                form.card.login-card method="post" action="/ui/login" {
                    div.brand { (shield()) span { "Aegis" } }
                    p.muted { "AI gateway admin console" }
                    @if let Some(e) = error { p.alert { (e) } }
                    label for="token" { "Admin token" }
                    input #token type="password" name="token" autocomplete="current-password" autofocus required;
                    button.primary type="submit" { "Sign in" }
                }
            }
        }
    }
}

async fn login(State(state): State<Shared>, Form(f): Form<LoginForm>) -> Response {
    if !constant_time_eq(f.token.trim(), &state.admin_token) {
        return (StatusCode::UNAUTHORIZED, login_view(Some("That token is not correct."))).into_response();
    }
    let token = random_token();
    state.sessions.lock().unwrap_or_else(|e| e.into_inner()).insert(token.clone());
    let cookie = format!("{COOKIE}={token}; Path=/; HttpOnly; SameSite=Strict; Max-Age=604800");
    ([(header::SET_COOKIE, cookie)], Redirect::to("/ui")).into_response()
}

async fn logout(State(state): State<Shared>, headers: HeaderMap) -> Response {
    if let Some(t) = session_token(&headers) {
        state.sessions.lock().unwrap_or_else(|e| e.into_inner()).remove(&t);
    }
    let cookie = format!("{COOKIE}=; Path=/; HttpOnly; SameSite=Strict; Max-Age=0");
    ([(header::SET_COOKIE, cookie)], Redirect::to("/ui/login")).into_response()
}

// ---- formatting -----------------------------------------------------------

fn int(n: i64) -> String {
    let s = n.abs().to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    if n < 0 { format!("-{out}") } else { out }
}

fn usd(v: f64) -> String {
    if v == 0.0 {
        "$0".into()
    } else if v < 0.01 {
        format!("${v:.5}")
    } else {
        format!("${v:.2}")
    }
}

fn ms(v: i64) -> String {
    if v >= 1000 { format!("{:.2} s", v as f64 / 1000.0) } else { format!("{v} ms") }
}

fn uptime(secs: u64) -> String {
    match secs {
        0..=59 => format!("{secs}s"),
        60..=3599 => format!("{}m {}s", secs / 60, secs % 60),
        3600..=86_399 => format!("{}h {}m", secs / 3600, secs % 3600 / 60),
        _ => format!("{}d {}h", secs / 86_400, secs % 86_400 / 3600),
    }
}

fn price(m: &crate::config::ModelCfg) -> String {
    if m.input_per_mtok == 0.0 && m.output_per_mtok == 0.0 {
        "no charge".into()
    } else {
        format!("{} in / {} out per MTok", usd(m.input_per_mtok), usd(m.output_per_mtok))
    }
}

fn pct(part: i64, whole: i64) -> String {
    if whole == 0 { "—".into() } else { format!("{:.1}%", part as f64 * 100.0 / whole as f64) }
}

fn compact(n: u64) -> String {
    match n {
        0..=9_999 => int(n as i64),
        10_000..=999_999 => format!("{:.1}k", n as f64 / 1e3),
        _ => format!("{:.2}M", n as f64 / 1e6),
    }
}

fn time(ts_ms: i64) -> Markup {
    let fallback = chrono::DateTime::from_timestamp_millis(ts_ms).map(|d| d.format("%Y-%m-%d %H:%M:%S UTC").to_string()).unwrap_or_default();
    html! { time data-ts=(ts_ms) { (fallback) } }
}

// ---- layout ---------------------------------------------------------------

fn shield() -> Markup {
    html! {
        svg.shield viewBox="0 0 24 24" aria-hidden="true" {
            path d="M12 2 4 5v6c0 5 3.4 9.3 8 11 4.6-1.7 8-6 8-11V5l-8-3Z" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round" {}
            path d="m8.5 12 2.5 2.5 4.5-5" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" {}
        }
    }
}

fn head(title: &str) -> Markup {
    html! {
        meta charset="utf-8";
        meta name="viewport" content="width=device-width, initial-scale=1";
        title { (title) " · Aegis" }
        style { (PreEscaped(CSS)) }
    }
}

fn page(title: &str, active: &str, live: bool, body: Markup) -> Markup {
    let nav = [
        ("/ui", "Overview"),
        ("/ui/requests", "Requests"),
        ("/ui/providers", "Providers"),
        ("/ui/keys", "Keys"),
        ("/ui/cache", "Cache"),
        ("/ui/playground", "Playground"),
    ];
    html! {
        (DOCTYPE)
        html lang="en" {
            head { (head(title)) }
            body {
                header.top {
                    a.brand href="/ui" { (shield()) span { "Aegis" } }
                    nav {
                        @for (href, label) in nav {
                            a.active[href == active] href=(href) { (label) }
                        }
                    }
                    form method="post" action="/ui/logout" { button.ghost type="submit" { "Sign out" } }
                }
                main data-live[live] {
                    div.title-row {
                        h1 { (title) }
                        @if live { span.live-dot title="Refreshes every 5 seconds" { "live" } }
                    }
                    div #live { (body) }
                }
                script { (PreEscaped(JS)) }
            }
        }
    }
}

fn badge(kind: &str, text: &str) -> Markup {
    html! { span class={ "badge " (kind) } { (text) } }
}

fn tier_badge(tier: &str) -> Markup {
    match tier {
        "" => html! {},
        t => badge(&format!("tier-{t}"), t),
    }
}

// ---- overview -------------------------------------------------------------

fn chart(buckets: &[HourBucket]) -> Markup {
    let now_hour = now_ms() / 3_600_000;
    let hours: Vec<HourBucket> = (0..24)
        .map(|i| {
            let h = now_hour - 23 + i;
            buckets.iter().find(|b| b.hour == h).cloned().unwrap_or(HourBucket { hour: h, ..Default::default() })
        })
        .collect();
    let max = hours.iter().map(|b| b.upstream + b.cache_hits + b.errors).max().unwrap_or(0).max(1) as f64;
    let (w, h, pad) = (720.0, 150.0, 4.0);
    let bw = w / 24.0;
    html! {
        svg.chart viewBox="0 0 720 172" role="img" aria-label="Requests per hour over the last 24 hours" {
            @for (i, b) in hours.iter().enumerate() {
                @let x = i as f64 * bw + pad / 2.0;
                @let mut y = h;
                @for (count, class) in [(b.upstream, "bar-up"), (b.cache_hits, "bar-hit"), (b.errors, "bar-err")] {
                    @let bh = count as f64 / max * (h - 8.0);
                    @if count > 0 {
                        rect class=(class) x=(format!("{x:.1}")) y=(format!("{:.1}", { y -= bh; y })) width=(format!("{:.1}", bw - pad)) height=(format!("{bh:.1}")) rx="2" {
                            title { (int(count)) " " (match class { "bar-up" => "from providers", "bar-hit" => "from cache", _ => "errors" }) }
                        }
                    }
                }
                @if i % 4 == 3 || i == 0 {
                    text.tick x=(format!("{:.1}", x + (bw - pad) / 2.0)) y="168" text-anchor="middle" data-hour=(b.hour * 3_600_000) { ((b.hour % 24)) ":00" }
                }
            }
            line.axis x1="0" y1="150.5" x2="720" y2="150.5" {}
        }
        div.legend {
            span { i.sw.bar-up {} "From providers" }
            span { i.sw.bar-hit {} "From cache" }
            span { i.sw.bar-err {} "Errors" }
        }
    }
}

async fn overview(State(state): State<Shared>, headers: HeaderMap) -> Response {
    require_admin!(state, headers);
    let since = now_ms() - 86_400_000;
    let s = state.db.summary(since).await.unwrap_or_default();
    let hourly = state.db.hourly(since).await.unwrap_or_default();
    let by = state.db.by_provider(since).await.unwrap_or_default();
    let upstream = s.requests - s.cache_hits;
    let body = html! {
        p.muted { "Last 24 hours · running for " (uptime(state.started.elapsed().as_secs())) }
        section.tiles {
            div.tile { span.k { "Requests" } span.v { (int(s.requests)) } span.sub { (int(upstream)) " reached a provider" } }
            div.tile { span.k { "Cache hit rate" } span.v { (pct(s.cache_hits, s.requests)) } span.sub { (int(s.cache_hits)) " served from cache" } }
            div.tile { span.k { "Spend" } span.v { (usd(s.cost_usd)) } span.sub { (usd(s.saved_usd)) " avoided by cache" } }
            div.tile { span.k { "Tokens" } span.v { (compact((s.prompt_tokens + s.completion_tokens) as u64)) } span.sub { (compact(s.prompt_tokens as u64)) " in · " (compact(s.completion_tokens as u64)) " out" } }
            div.tile { span.k { "Latency p50 / p95" } span.v { (ms(s.p50_ms)) } span.sub { "p95 " (ms(s.p95_ms)) " · provider calls" } }
            div.tile { span.k { "Fallbacks" } span.v { (int(s.fallbacks)) } span.sub { (pct(s.errors, s.requests)) " error rate" } }
        }
        section.card {
            h2 { "Requests per hour" }
            (chart(&hourly))
        }
        section.card {
            h2 { "Where requests went" }
            @if by.is_empty() {
                p.empty { "No provider calls yet. Send a request to " code { "/v1/chat/completions" } " or try the Playground." }
            } @else {
                div.table-wrap { table {
                    thead { tr { th { "Provider" } th { "Model" } th { "Tier" } th.n { "Requests" } th.n { "Errors" } th.n { "Tokens" } th.n { "Spend" } th.n { "Avg latency" } } }
                    tbody {
                        @for p in &by {
                            tr {
                                td { (p.provider) } td.mono { (p.model) } td { (tier_badge(&p.tier)) }
                                td.n { (int(p.requests)) } td.n { (int(p.errors)) } td.n { (compact(p.tokens as u64)) }
                                td.n { (usd(p.cost_usd)) } td.n { (ms(p.avg_latency_ms as i64)) }
                            }
                        }
                    }
                } }
            }
        }
    };
    page("Overview", "/ui", true, body).into_response()
}

// ---- requests -------------------------------------------------------------

fn attempts_view(r: &RequestRecord) -> Markup {
    let attempts: Vec<Value> = serde_json::from_str(&r.attempts).unwrap_or_default();
    html! {
        @if attempts.len() > 1 || (!r.error.is_empty() && !attempts.is_empty()) {
            details {
                summary { (attempts.len()) " targets considered" }
                ol.attempts {
                    @for a in &attempts {
                        li {
                            span.mono { (a["target"].as_str().unwrap_or("?")) }
                            " — " (a["outcome"].as_str().unwrap_or("?"))
                            @if let Some(reason) = a["reason"].as_str() { span.muted { ": " (reason) } }
                        }
                    }
                }
            }
        }
        @if !r.error.is_empty() { div.err-text { (r.error) } }
    }
}

async fn requests(State(state): State<Shared>, headers: HeaderMap) -> Response {
    require_admin!(state, headers);
    let rows = state.db.recent_requests(200).await.unwrap_or_default();
    let body = html! {
        p.muted { "Most recent 200 requests." }
        section.card {
            @if rows.is_empty() {
                p.empty { "Nothing logged yet." }
            } @else {
                div.table-wrap { table.requests {
                    thead { tr { th { "Time" } th { "Client" } th { "Asked for" } th { "Served by" } th { "Cache" } th.n { "Tokens" } th.n { "Cost" } th.n { "Latency" } th { "Status" } } }
                    tbody {
                        @for r in &rows {
                            tr {
                                td.nowrap { (time(r.ts)) }
                                td { (r.key_name) }
                                td.mono { (r.requested_model) @if r.stream { " " (badge("plain", "stream")) } }
                                td {
                                    @if r.provider.is_empty() { span.muted { "—" } } @else {
                                        span.mono { (r.provider) "/" (r.model) } " " (tier_badge(&r.tier))
                                        @if r.fallback { " " (badge("warn", "fallback")) }
                                    }
                                    (attempts_view(r))
                                }
                                td { (badge(match r.cache.as_str() { "hit" => "ok", "miss" => "plain", _ => "dim" }, &r.cache)) }
                                td.n { (int(r.prompt_tokens)) " / " (int(r.completion_tokens)) @if r.usage_estimated { span.muted title="Provider did not report usage; estimated" { " ~" } } }
                                td.n { @if r.cache == "hit" { span.muted { "saved " (usd(r.saved_usd)) } } @else { (usd(r.cost_usd)) } }
                                td.n { (ms(r.latency_ms)) }
                                td { (badge(match r.status.as_str() { "ok" => "ok", "error" => "bad", _ => "warn" }, &r.status)) }
                            }
                        }
                    }
                } }
            }
        }
    };
    page("Requests", "/ui/requests", true, body).into_response()
}

// ---- providers ------------------------------------------------------------

fn meter(label: &str, used: u64, limit: Option<u64>) -> Markup {
    html! {
        div.meter {
            div.meter-head { span { (label) } span.mono { (compact(used)) " / " (limit.map(compact).unwrap_or_else(|| "∞".into())) } }
            @if let Some(l) = limit {
                @let p = if l == 0 { 100.0 } else { (used as f64 / l as f64 * 100.0).min(100.0) };
                div.track { div class={ "fill" @if p >= 90.0 { " hot" } @else if p >= 70.0 { " warm" } } style=(format!("width:{p:.1}%")) {} }
            }
        }
    }
}

async fn providers(State(state): State<Shared>, headers: HeaderMap) -> Response {
    require_admin!(state, headers);
    providers_view(&state, None).into_response()
}

fn providers_view(state: &Shared, error: Option<String>) -> Markup {
    let stored = state.stored_keys.lock().unwrap_or_else(|e| e.into_inner()).clone();
    let body = html! {
        @if let Some(e) = &error { p.alert { (e) } }
        section.card {
            h2 { "Routes" }
            p.muted { "A request for an alias goes to the first target that is configured, capable of the request, and within quota." }
            div.table-wrap { table {
                thead { tr { th { "Alias" } th { "Order" } th { "Purpose" } } }
                tbody {
                    @for r in &state.cfg.routes {
                        tr {
                            td.mono { (r.alias) }
                            td { @for (i, t) in r.targets.iter().enumerate() { @if i > 0 { span.arrow { " → " } } span.mono { (t) } } }
                            td.muted { (r.description) }
                        }
                    }
                }
            } }
        }
        @for p in &state.cfg.providers {
            @let needs_key = !p.api_key_env.is_empty();
            @let source = state.adapters.get(&p.name).map_or(KeySource::Unset, |a| a.key_source());
            @let configured = !needs_key || source != KeySource::Unset;
            section.card {
                div.card-head {
                    h2 { (p.name) " " (tier_badge(p.tier.as_str())) }
                    @if !needs_key { (badge("ok", "no key needed")) }
                    @else if source == KeySource::Dashboard {
                        @let hint = stored.get(&p.name).map(|s| s.0.as_str()).unwrap_or("");
                        (badge("ok", &if hint.is_empty() { "key saved here".to_string() } else { format!("key saved here ····{hint}") }))
                    }
                    @else if source == KeySource::Env { (badge("ok", &format!("key from {}", p.api_key_env))) }
                    @else { (badge("bad", "no API key")) }
                }
                p.muted { (p.kind.as_str()) " · " span.mono { (p.base_url) } }
                @if needs_key {
                    @if state.sealer.is_some() {
                        details.keybox {
                            summary { @if source == KeySource::Dashboard { "Replace API key" } @else { "Set API key" } }
                            p.muted {
                                "The key is encrypted before it is saved, takes effect immediately, and cannot be shown again."
                                @if source == KeySource::Env { " It will be used instead of " span.mono { (p.api_key_env) } "." }
                            }
                            form.key-row method="post" action="/ui/providers/key" {
                                input type="hidden" name="provider" value=(p.name);
                                input type="password" name="key" placeholder="Paste the provider API key" autocomplete="off" spellcheck="false" required maxlength="500";
                                button.primary type="submit" { "Save key" }
                            }
                            @if source == KeySource::Dashboard {
                                form.inline method="post" action="/ui/providers/key/remove" data-confirm=(format!("Remove the saved key for {}?", p.name)) {
                                    input type="hidden" name="provider" value=(p.name);
                                    button.ghost.small.danger type="submit" { "Remove saved key" }
                                }
                            }
                        }
                    } @else {
                        p.muted { "To set this key from the dashboard, start Aegis with " span.mono { (crate::secrets::SECRET_ENV) } ". Until then it is read from " span.mono { (p.api_key_env) } "." }
                    }
                }
                @for m in &p.models {
                    @let key = format!("{}/{}", p.name, m.id);
                    @let v = state.quota.view(&key);
                    div.model {
                        div.model-head {
                            span.mono.strong { (m.id) }
                            @for c in &m.capabilities { (badge("plain", c)) }
                            span.muted { (compact(m.context_window)) " token context · " (price(m)) }
                            @if v.cooldown_secs > 0 {
                                (badge("bad", &format!("cooling down {}s", v.cooldown_secs)))
                                form.inline method="post" action="/ui/providers/reset" {
                                    input type="hidden" name="target" value=(key);
                                    button.ghost.small type="submit" { "Clear" }
                                }
                            } @else if configured { (badge("ok", "available")) }
                        }
                        div.meters {
                            (meter("Requests / min", v.rpm_used, m.limits.rpm))
                            (meter("Requests / day", v.rpd_used, m.limits.rpd))
                            (meter("Tokens / min", v.tpm_used, m.limits.tpm))
                            (meter("Tokens / day", v.tpd_used, m.limits.tpd))
                        }
                        @if v.cooldown_secs > 0 { p.muted { "Reason: " (v.cooldown_reason) } }
                        @if let Some(rem) = v.remaining_requests { p.muted { "Provider reports " (int(rem as i64)) " requests remaining" @if let Some(t) = v.remaining_tokens { " and " (compact(t)) " tokens" } "." } }
                        @if let Some(e) = &v.last_error { p.err-text { "Last error: " (e) } }
                    }
                }
            }
        }
    };
    page("Providers", "/ui/providers", true, body)
}

#[derive(Deserialize)]
struct ProviderKeyForm {
    provider: String,
    #[serde(default)]
    key: String,
}

async fn set_provider_key(State(state): State<Shared>, headers: HeaderMap, Form(f): Form<ProviderKeyForm>) -> Response {
    require_admin!(state, headers);
    let key = f.key.trim();
    let needs_key = state.cfg.providers.iter().any(|p| p.name == f.provider && !p.api_key_env.is_empty());
    let problem = if !needs_key {
        Some("That provider does not take an API key.".to_string())
    } else if key.is_empty() {
        Some("Paste a key before saving.".to_string())
    } else if key.len() > 500 || !key.chars().all(|c| c.is_ascii_graphic()) {
        Some("That does not look like an API key: it has spaces or unusual characters.".to_string())
    } else {
        state.set_provider_key(&f.provider, key).await.err().map(|e| format!("Could not save the key: {e}"))
    };
    match problem {
        Some(e) => (StatusCode::BAD_REQUEST, providers_view(&state, Some(e))).into_response(),
        None => {
            tracing::info!(provider = %f.provider, "provider key set from dashboard");
            Redirect::to("/ui/providers").into_response()
        }
    }
}

async fn remove_provider_key(State(state): State<Shared>, headers: HeaderMap, Form(f): Form<ProviderKeyForm>) -> Response {
    require_admin!(state, headers);
    match state.remove_provider_key(&f.provider).await {
        Ok(()) => {
            tracing::info!(provider = %f.provider, "provider key removed from dashboard");
            Redirect::to("/ui/providers").into_response()
        }
        Err(e) => (StatusCode::BAD_REQUEST, providers_view(&state, Some(format!("Could not remove the key: {e}")))).into_response(),
    }
}

#[derive(Deserialize)]
struct ResetForm {
    target: String,
}

async fn reset_target(State(state): State<Shared>, headers: HeaderMap, Form(f): Form<ResetForm>) -> Response {
    require_admin!(state, headers);
    if state.cfg.find_target(&f.target).is_some() {
        state.quota.reset(&f.target);
    }
    Redirect::to("/ui/providers").into_response()
}

// ---- keys -----------------------------------------------------------------

#[derive(Deserialize)]
struct KeyForm {
    name: String,
    #[serde(default)]
    rpm: String,
    #[serde(default)]
    daily_budget: String,
    #[serde(default)]
    models: String,
    #[serde(default)]
    allow_paid: Option<String>,
}

async fn keys_view(state: &Shared, created: Option<(String, String)>, error: Option<String>) -> Markup {
    let rows = state.db.list_keys().await.unwrap_or_default();
    let body = html! {
        @if let Some((name, key)) = &created {
            section.card.notice {
                h2 { "Key created for " (name) }
                p { "Copy it now. Only a hash is stored, so it cannot be shown again." }
                div.copy-row { code.secret #new-key { (key) } button.primary type="button" data-copy="new-key" { "Copy" } }
            }
        }
        @if let Some(e) = &error { p.alert { (e) } }
        section.card {
            h2 { "Create a client key" }
            form.grid-form method="post" action="/ui/keys" {
                label { span { "Name" } input type="text" name="name" placeholder="e.g. notes-app" required maxlength="60"; }
                label { span { "Requests per minute" } input type="number" name="rpm" min="1" placeholder="unlimited"; }
                label { span { "Daily budget (USD)" } input type="number" name="daily_budget" min="0" step="0.01" placeholder="unlimited"; }
                label { span { "Allowed models" } input type="text" name="models" placeholder="all (comma-separated to restrict)"; }
                label.check { input type="checkbox" name="allow_paid" value="1"; span { "May fall back to paid providers" } }
                div { button.primary type="submit" { "Create key" } }
            }
        }
        section.card {
            h2 { "Client keys" }
            @if rows.is_empty() {
                p.empty { "No keys yet. Create one above, then send it as " code { "Authorization: Bearer …" } "." }
            } @else {
                div.table-wrap { table {
                    thead { tr { th { "Name" } th { "Key" } th { "Created" } th.n { "Req / min" } th.n { "Daily budget" } th.n { "Spent today" } th { "Paid tier" } th { "Models" } th {} } }
                    tbody {
                        @for k in &rows {
                            tr.revoked[k.revoked_at.is_some()] {
                                td { (k.name) }
                                td.mono { (k.prefix) "…" }
                                td.nowrap { (time(k.created_at)) }
                                td.n { (k.rpm.map(int).unwrap_or_else(|| "∞".into())) }
                                td.n { (k.daily_budget_usd.map(usd).unwrap_or_else(|| "∞".into())) }
                                td.n { (usd(state.keys.spent_today(k.id))) }
                                td { @if k.allow_paid { (badge("warn", "allowed")) } @else { (badge("dim", "free only")) } }
                                td.mono { @if k.allowed_models.is_empty() { "all" } @else { (k.allowed_models) } }
                                td {
                                    @if k.revoked_at.is_some() { (badge("bad", "revoked")) } @else {
                                        form.inline method="post" action=(format!("/ui/keys/{}/revoke", k.id)) data-confirm=(format!("Revoke the key for {}? Apps using it will stop working.", k.name)) {
                                            button.ghost.small.danger type="submit" { "Revoke" }
                                        }
                                    }
                                }
                            }
                        }
                    }
                } }
            }
        }
    };
    page("Keys", "/ui/keys", false, body)
}

async fn keys(State(state): State<Shared>, headers: HeaderMap) -> Response {
    require_admin!(state, headers);
    keys_view(&state, None, None).await.into_response()
}

async fn create_key(State(state): State<Shared>, headers: HeaderMap, Form(f): Form<KeyForm>) -> Response {
    require_admin!(state, headers);
    let name = f.name.trim();
    if name.is_empty() {
        return keys_view(&state, None, Some("Give the key a name.".into())).await.into_response();
    }
    let rpm = f.rpm.trim().parse::<i64>().ok().filter(|v| *v > 0);
    let budget = f.daily_budget.trim().parse::<f64>().ok().filter(|v| *v >= 0.0);
    let models: Vec<&str> = f.models.split(',').map(str::trim).filter(|m| !m.is_empty()).collect();
    if let Some(bad) = models.iter().find(|m| state.resolve(m).is_empty()) {
        return keys_view(&state, None, Some(format!("Unknown model or alias: {bad}"))).await.into_response();
    }
    let key = generate_key();
    let prefix: String = key.chars().take(crate::auth::KEY_PREFIX.len() + 6).collect();
    match state.db.create_key(name, &prefix, &hash_key(&key), rpm, budget, f.allow_paid.is_some(), &models.join(",")).await {
        Ok(_) => {
            let _ = state.keys.reload(&state.db).await;
            keys_view(&state, Some((name.to_string(), key)), None).await.into_response()
        }
        Err(e) => keys_view(&state, None, Some(format!("Could not create the key: {e}"))).await.into_response(),
    }
}

async fn revoke_key(State(state): State<Shared>, headers: HeaderMap, Path(id): Path<i64>) -> Response {
    require_admin!(state, headers);
    let _ = state.db.revoke_key(id).await;
    let _ = state.keys.reload(&state.db).await;
    Redirect::to("/ui/keys").into_response()
}

// ---- cache ----------------------------------------------------------------

async fn cache(State(state): State<Shared>, headers: HeaderMap) -> Response {
    require_admin!(state, headers);
    let s = state.cache.stats().await;
    let c = state.cache.cfg();
    let lookups = s.hits + s.misses;
    let body = html! {
        p.muted { "Counters since the gateway started." }
        section.tiles {
            div.tile { span.k { "Entries" } span.v { (int(s.entries as i64)) } span.sub { "of " (int(c.max_entries as i64)) " max" } }
            div.tile { span.k { "Hit rate" } span.v { (pct(s.hits as i64, lookups as i64)) } span.sub { (int(s.hits as i64)) " hits · " (int(s.misses as i64)) " misses" } }
            div.tile { span.k { "Stored" } span.v { (int(s.stores as i64)) } span.sub { "responses written" } }
            div.tile { span.k { "Not cacheable" } span.v { (int(s.skipped as i64)) } span.sub { "sampled or bypassed requests" } }
        }
        section.card {
            h2 { "Policy" }
            dl.kv {
                dt { "Status" } dd { @if c.enabled { (badge("ok", "enabled")) } @else { (badge("dim", "disabled")) } }
                dt { "Entry lifetime" } dd { (int(c.ttl_secs as i64)) " seconds" }
                dt { "Cached when" } dd {
                    "temperature ≤ " (c.max_temperature) ", or a " code { "seed" } " is set"
                    @if c.cache_when_temperature_unset { ", or temperature is not set" }
                }
                dt { "Sharing" } dd { @if c.scope == crate::config::CacheScope::Global { "Shared by all clients" } @else { "Separate per client key" } }
                dt { "Per request" } dd { code { "x-aegis-cache: bypass" } " skips the cache · " code { "x-aegis-cache: force" } " caches a sampled request" }
            }
        }
        section.card {
            h2 { "Purge" }
            p.muted { "Removes every cached response. The next identical request goes to a provider." }
            form method="post" action="/ui/cache/purge" data-confirm="Remove all cached responses?" { button.danger type="submit" { "Purge cache" } }
        }
    };
    page("Cache", "/ui/cache", true, body).into_response()
}

async fn purge_cache(State(state): State<Shared>, headers: HeaderMap) -> Response {
    require_admin!(state, headers);
    state.cache.purge().await;
    Redirect::to("/ui/cache").into_response()
}

// ---- playground -----------------------------------------------------------

async fn playground(State(state): State<Shared>, headers: HeaderMap) -> Response {
    require_admin!(state, headers);
    let body = html! {
        p.muted { "Requests here run through the full pipeline as the " code { "playground" } " client, which may use paid providers." }
        div.play {
            section.card {
                form #play-form {
                    label { span { "Model" }
                        select name="model" {
                            optgroup label="Routes" { @for r in &state.cfg.routes { option value=(r.alias) { (r.alias) } } }
                            optgroup label="Direct" {
                                @for p in &state.cfg.providers { @for m in &p.models { option value=(format!("{}/{}", p.name, m.id)) { (p.name) "/" (m.id) } } }
                            }
                        }
                    }
                    label { span { "System prompt" } textarea name="system" rows="2" placeholder="Optional" {} }
                    label { span { "Message" } textarea name="prompt" rows="6" required placeholder="Ask something…" {} }
                    div.row {
                        label { span { "Temperature" } input type="number" name="temperature" min="0" max="2" step="0.1" value="0"; }
                        label { span { "Cache" } select name="cache" { option value="" { "Default policy" } option value="bypass" { "Bypass" } option value="force" { "Force" } } }
                        label.check { input type="checkbox" name="stream" checked; span { "Stream" } }
                    }
                    button.primary type="submit" #play-send { "Send" }
                }
            }
            section.card {
                h2 { "Response" }
                div #play-meta .meta {}
                pre #play-out .output { span.muted { "The answer appears here." } }
            }
        }
    };
    page("Playground", "/ui/playground", false, body).into_response()
}

async fn playground_chat(State(state): State<Shared>, headers: HeaderMap, Json(body): Json<Value>) -> Response {
    if !signed_in(&state, &headers) {
        return crate::gateway::api_error(StatusCode::UNAUTHORIZED, "not_signed_in", "Sign in to the admin console first.");
    }
    let req: ChatRequest = match serde_json::from_value(body) {
        Ok(r) => r,
        Err(e) => return crate::gateway::api_error(StatusCode::BAD_REQUEST, "invalid_request_error", e.to_string()),
    };
    run_chat(state, Arc::new(Identity::admin_playground()), ChatOpts::from_headers(&headers), req).await
}

// ---- assets ---------------------------------------------------------------

const CSS: &str = r#"
:root{
  --bg:#f4f5f2;--panel:#fff;--ink:#16211f;--muted:#5d6b67;--line:#dfe3dd;--soft:#eef1ec;
  --accent:#0f766e;--accent-ink:#fff;--ok:#1a7f4b;--ok-bg:#dff3e7;--warn:#9a5b00;--warn-bg:#fdeccb;
  --bad:#b3261e;--bad-bg:#fbe0dd;--hit:#3f8f5a;--up:#0f766e;--err:#c2410c;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  --sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
}
@media (prefers-color-scheme:dark){:root{
  --bg:#0e1514;--panel:#151e1d;--ink:#e6efec;--muted:#93a39e;--line:#263331;--soft:#1b2625;
  --accent:#2dd4bf;--accent-ink:#04201c;--ok:#5fd38d;--ok-bg:#12331f;--warn:#f2b84b;--warn-bg:#3a2a08;
  --bad:#ff8a7a;--bad-bg:#3d1613;--hit:#5fd38d;--up:#2dd4bf;--err:#fb923c;
}}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 var(--sans)}
a{color:inherit}
code,.mono{font-family:var(--mono);font-size:.92em}
code{background:var(--soft);padding:.1em .35em;border-radius:4px}
.muted{color:var(--muted)}.strong{font-weight:600}.nowrap{white-space:nowrap}
.top{display:flex;align-items:center;gap:20px;padding:10px 24px;background:var(--panel);border-bottom:1px solid var(--line);position:sticky;top:0;z-index:5}
.brand{display:flex;align-items:center;gap:8px;font-weight:700;font-size:17px;letter-spacing:.01em;text-decoration:none}
.shield{width:22px;height:22px;color:var(--accent)}
.top nav{display:flex;gap:2px;flex:1;overflow-x:auto}
.top nav a{padding:6px 12px;border-radius:7px;text-decoration:none;color:var(--muted);white-space:nowrap}
.top nav a:hover{background:var(--soft);color:var(--ink)}
.top nav a.active{background:var(--soft);color:var(--ink);font-weight:600}
main{max-width:1180px;margin:0 auto;padding:24px 24px 64px}
.title-row{display:flex;align-items:center;gap:12px}
h1{font-size:24px;margin:0;letter-spacing:-.01em}
h2{font-size:15px;margin:0 0 12px;font-weight:650}
.live-dot{font-size:12px;color:var(--ok);display:inline-flex;align-items:center;gap:6px}
.live-dot::before{content:"";width:7px;height:7px;border-radius:50%;background:var(--ok)}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px 20px;margin-top:16px}
.card-head{display:flex;justify-content:space-between;align-items:center;gap:12px}
.card-head h2{margin:0}
.tiles{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin-top:8px}
@media (min-width:1100px){.tiles{grid-template-columns:repeat(6,minmax(0,1fr))}.tiles:has(.tile:nth-child(4):last-child){grid-template-columns:repeat(4,minmax(0,1fr))}}
@media (max-width:560px){.tiles{grid-template-columns:repeat(2,minmax(0,1fr))}}
.tile{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 16px;display:flex;flex-direction:column;gap:2px;min-width:0}
.tile .k{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}
.tile .v{font-size:24px;font-weight:650;font-variant-numeric:tabular-nums;letter-spacing:-.02em}
.tile .sub{font-size:12.5px;color:var(--muted)}
.table-wrap{overflow-x:auto;margin:0 -4px}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);vertical-align:top}
th{font-size:12px;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.04em;white-space:nowrap}
tbody tr:last-child td{border-bottom:0}
.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
tr.revoked td{opacity:.55}
.badge{display:inline-block;padding:1px 8px;border-radius:999px;font-size:12px;font-weight:600;white-space:nowrap;background:var(--soft);color:var(--muted)}
.badge.ok{background:var(--ok-bg);color:var(--ok)}.badge.bad{background:var(--bad-bg);color:var(--bad)}
.badge.warn,.badge.tier-paid{background:var(--warn-bg);color:var(--warn)}
.badge.tier-free{background:var(--ok-bg);color:var(--ok)}
.badge.dim{opacity:.75}
.empty{color:var(--muted);padding:18px 0;margin:0}
.err-text{color:var(--bad);font-size:13px;margin:4px 0 0;overflow-wrap:anywhere}
details{font-size:13px;margin-top:4px}summary{cursor:pointer;color:var(--muted)}
.attempts{margin:6px 0 0;padding-left:20px}
.chart{width:100%;height:auto;display:block}
.bar-up{fill:var(--up)}.bar-hit{fill:var(--hit);opacity:.55}.bar-err{fill:var(--err)}
.axis{stroke:var(--line)}.tick{fill:var(--muted);font-size:10px;font-family:var(--sans)}
.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:12.5px;color:var(--muted);margin-top:8px}
.sw{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px}
.sw.bar-up{background:var(--up)}.sw.bar-hit{background:var(--hit);opacity:.55}.sw.bar-err{background:var(--err)}
.model{border-top:1px solid var(--line);padding:14px 0 4px;margin-top:14px}
.model-head{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
.meters{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px 20px;margin-top:10px}
.meter-head{display:flex;justify-content:space-between;font-size:12.5px;color:var(--muted)}
.track{height:6px;background:var(--soft);border-radius:99px;overflow:hidden;margin-top:4px}
.fill{height:100%;background:var(--accent);border-radius:99px}.fill.warm{background:var(--warn)}.fill.hot{background:var(--bad)}
.arrow{color:var(--muted)}
button,input,select,textarea{font:inherit;color:inherit}
input,select,textarea{width:100%;background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:8px 10px}
input:focus,select:focus,textarea:focus,button:focus-visible,a:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
textarea{resize:vertical;font-family:var(--sans)}
label{display:flex;flex-direction:column;gap:5px;font-size:13px;color:var(--muted);margin-bottom:12px}
label.check{flex-direction:row;align-items:center;gap:8px;color:var(--ink)}
label.check input{width:auto}
button{cursor:pointer;border-radius:8px;border:1px solid var(--line);background:var(--panel);padding:8px 14px;font-weight:600}
button.primary{background:var(--accent);border-color:var(--accent);color:var(--accent-ink)}
button.danger{color:var(--bad);border-color:var(--bad)}
button.ghost{background:transparent;color:var(--muted)}
button.small{padding:3px 9px;font-size:12.5px}
button:disabled{opacity:.6;cursor:progress}
form.inline{display:inline}
.keybox{margin:8px 0 14px}
.keybox summary{cursor:pointer;color:var(--accent);font-weight:600;font-size:13.5px}
.key-row{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0}
.key-row input{flex:1 1 260px;min-width:0}
.grid-form{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:4px 16px;align-items:end}
.grid-form label.check{margin-bottom:20px}
.notice{border-color:var(--accent)}
.copy-row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.secret{padding:8px 10px;overflow-wrap:anywhere;flex:1;min-width:0}
.alert{background:var(--bad-bg);color:var(--bad);padding:10px 12px;border-radius:8px;margin:12px 0}
.kv{display:grid;grid-template-columns:max-content 1fr;gap:8px 20px;margin:0}
.kv dt{color:var(--muted)}.kv dd{margin:0}
.play{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.2fr);gap:16px;align-items:start}
.row{display:grid;grid-template-columns:1fr 1fr auto;gap:12px;align-items:end}
.output{white-space:pre-wrap;overflow-wrap:anywhere;font-family:var(--sans);font-size:15px;margin:0;min-height:160px}
.meta{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:10px;min-height:22px}
body.login{display:grid;place-items:center;min-height:100vh;padding:16px}
.login-card{width:min(380px,100%);margin:0}
.login-card .brand{font-size:22px}.login-card .shield{width:28px;height:28px}
.login-card button{width:100%;margin-top:4px}
@media (max-width:820px){
  .top{flex-wrap:wrap;gap:8px;padding:10px 16px}.top nav{order:3;flex-basis:100%}
  main{padding:16px 16px 48px}.play{grid-template-columns:1fr}.row{grid-template-columns:1fr 1fr}
}
"#;

const JS: &str = r#"
(function(){
  function localize(root){
    root.querySelectorAll('time[data-ts]').forEach(function(t){
      var d=new Date(+t.dataset.ts);
      t.textContent=d.toLocaleString([], {month:'short',day:'numeric',hour:'2-digit',minute:'2-digit',second:'2-digit'});
    });
    root.querySelectorAll('text[data-hour]').forEach(function(t){
      t.textContent=new Date(+t.dataset.hour).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'});
    });
  }
  localize(document);

  document.addEventListener('submit',function(e){
    var m=e.target.dataset&&e.target.dataset.confirm;
    if(m&&!window.confirm(m)){e.preventDefault();}
  });
  document.addEventListener('click',function(e){
    var id=e.target.dataset&&e.target.dataset.copy;
    if(!id)return;
    navigator.clipboard.writeText(document.getElementById(id).textContent).then(function(){e.target.textContent='Copied';});
  });

  var main=document.querySelector('main[data-live]');
  if(main){
    setInterval(function(){
      if(document.hidden||document.querySelector('details[open]'))return;
      fetch(location.href,{credentials:'same-origin'}).then(function(r){return r.ok&&!r.redirected?r.text():null;}).then(function(html){
        if(!html)return;
        var next=new DOMParser().parseFromString(html,'text/html').getElementById('live');
        var live=document.getElementById('live');
        if(next&&live){live.innerHTML=next.innerHTML;localize(live);}
      }).catch(function(){});
    },5000);
  }

  var form=document.getElementById('play-form');
  if(form){
    var out=document.getElementById('play-out'),meta=document.getElementById('play-meta'),btn=document.getElementById('play-send');
    function pill(text,kind){var s=document.createElement('span');s.className='badge '+(kind||'plain');s.textContent=text;meta.appendChild(s);}
    form.addEventListener('submit',async function(e){
      e.preventDefault();
      var f=new FormData(form),messages=[];
      if(f.get('system').trim())messages.push({role:'system',content:f.get('system')});
      messages.push({role:'user',content:f.get('prompt')});
      var stream=!!f.get('stream');
      var body={model:f.get('model'),messages:messages,stream:stream};
      if(f.get('temperature')!=='')body.temperature=parseFloat(f.get('temperature'));
      var headers={'content-type':'application/json'};
      if(f.get('cache'))headers['x-aegis-cache']=f.get('cache');
      out.textContent='';meta.textContent='';btn.disabled=true;btn.textContent='Sending…';
      var t0=performance.now();
      try{
        var r=await fetch('/ui/playground/chat',{method:'POST',headers:headers,body:JSON.stringify(body),credentials:'same-origin'});
        var h=r.headers;
        if(h.get('x-aegis-provider'))pill(h.get('x-aegis-provider')+'/'+h.get('x-aegis-model'));
        if(h.get('x-aegis-tier'))pill(h.get('x-aegis-tier'),'tier-'+h.get('x-aegis-tier'));
        if(h.get('x-aegis-cache'))pill('cache '+h.get('x-aegis-cache'),h.get('x-aegis-cache')==='hit'?'ok':'dim');
        if(h.get('x-aegis-fallback')==='true')pill('fallback','warn');
        var usage=null;
        if(!r.ok){
          var err=await r.json().catch(function(){return null;});
          out.textContent=(err&&err.error&&err.error.message)||('Request failed with status '+r.status);
          pill('error '+r.status,'bad');
        }else if(stream){
          var reader=r.body.getReader(),dec=new TextDecoder(),buf='';
          for(;;){
            var x=await reader.read();if(x.done)break;
            buf+=dec.decode(x.value,{stream:true});
            var lines=buf.split('\n');buf=lines.pop();
            lines.forEach(function(line){
              if(line.indexOf('data:')!==0)return;
              var d=line.slice(5).trim();if(!d||d==='[DONE]')return;
              try{
                var j=JSON.parse(d);
                if(j.error){out.textContent+='\n['+j.error.message+']';pill('stream error','bad');return;}
                if(j.usage)usage=j.usage;
                var c=j.choices&&j.choices[0]&&j.choices[0].delta;
                if(c&&c.content)out.textContent+=c.content;
              }catch(_){}
            });
          }
        }else{
          var j=await r.json();usage=j.usage;
          out.textContent=(j.choices&&j.choices[0]&&j.choices[0].message&&j.choices[0].message.content)||JSON.stringify(j,null,2);
        }
        if(usage)pill(usage.prompt_tokens+' in / '+usage.completion_tokens+' out tokens');
        pill(Math.round(performance.now()-t0)+' ms');
      }catch(err){out.textContent='Could not reach the gateway: '+err.message;}
      btn.disabled=false;btn.textContent='Send';
    });
  }
})();
"#;

//! Aegis — an AI API gateway: one OpenAI-compatible endpoint in front of many
//! providers, with caching, quota-aware routing, fallback and accounting.

pub mod alerts;
pub mod auth;
pub mod cache;
pub mod config;
pub mod db;
pub mod gateway;
pub mod icons;
pub mod providers;
pub mod quota;
pub mod secrets;
pub mod types;
pub mod ui;

use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::routing::{get, post};

pub fn router(state: gateway::Shared) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(gateway::chat_completions))
        .route("/v1/embeddings", post(gateway::embeddings))
        .route("/v1/models", get(gateway::list_models))
        .route("/healthz", get(gateway::healthz))
        .merge(ui::routes())
        .layer(DefaultBodyLimit::max(32 * 1024 * 1024))
        .with_state(state)
}

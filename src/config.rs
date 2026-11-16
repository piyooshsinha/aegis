//! Configuration: providers, models, routes, cache and server settings.
//!
//! Secrets never live in this file. A provider names the environment variable
//! that holds its API key (`api_key_env`); the value is read at startup.

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::collections::HashSet;

pub const KNOWN_CAPABILITIES: &[&str] = &["tools", "vision", "json"];

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    #[serde(default)]
    pub server: ServerCfg,
    #[serde(default)]
    pub auth: AuthCfg,
    #[serde(default)]
    pub cache: CacheCfg,
    #[serde(default)]
    pub providers: Vec<ProviderCfg>,
    #[serde(default)]
    pub routes: Vec<RouteCfg>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ServerCfg {
    pub bind: String,
    pub db_path: String,
    pub retention_days: u64,
}

impl Default for ServerCfg {
    fn default() -> Self {
        Self { bind: "0.0.0.0:8787".into(), db_path: "data/aegis.db".into(), retention_days: 30 }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct AuthCfg {
    /// When false, requests without a key are served as an anonymous,
    /// free-tier-only client. Keep true anywhere reachable by others.
    pub required: bool,
}

impl Default for AuthCfg {
    fn default() -> Self {
        Self { required: true }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CacheScope {
    /// Entries are only shared between requests made with the same client key.
    PerKey,
    /// Entries are shared between all clients.
    Global,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct CacheCfg {
    pub enabled: bool,
    pub ttl_secs: u64,
    pub max_entries: u64,
    /// Only requests with an explicit temperature at or below this are cached.
    pub max_temperature: f64,
    /// Whether to cache requests that do not set a temperature at all
    /// (providers then sample at their default, usually 1.0).
    pub cache_when_temperature_unset: bool,
    pub scope: CacheScope,
}

impl Default for CacheCfg {
    fn default() -> Self {
        Self {
            enabled: true,
            ttl_secs: 3600,
            max_entries: 10_000,
            max_temperature: 0.0,
            cache_when_temperature_unset: false,
            scope: CacheScope::PerKey,
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProviderKind {
    /// Any OpenAI-compatible endpoint: OpenAI, Groq, Ollama, vLLM, llama.cpp.
    Openai,
    Anthropic,
    Gemini,
}

impl ProviderKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Openai => "openai-compatible",
            Self::Anthropic => "anthropic",
            Self::Gemini => "gemini",
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Tier {
    Local,
    Free,
    Paid,
}

impl Tier {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::Free => "free",
            Self::Paid => "paid",
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderCfg {
    pub name: String,
    pub kind: ProviderKind,
    pub base_url: String,
    #[serde(default)]
    pub api_key_env: String,
    pub tier: Tier,
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
    #[serde(default)]
    pub models: Vec<ModelCfg>,
}

fn default_timeout() -> u64 {
    120
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelCfg {
    pub id: String,
    #[serde(default = "default_context")]
    pub context_window: u64,
    #[serde(default)]
    pub max_output: Option<u64>,
    #[serde(default)]
    pub capabilities: Vec<String>,
    /// USD per million input tokens.
    #[serde(default)]
    pub input_per_mtok: f64,
    /// USD per million output tokens.
    #[serde(default)]
    pub output_per_mtok: f64,
    #[serde(default)]
    pub limits: Limits,
}

fn default_context() -> u64 {
    8192
}

impl ModelCfg {
    pub fn has(&self, cap: &str) -> bool {
        self.capabilities.iter().any(|c| c == cap)
    }

    pub fn cost(&self, prompt_tokens: u64, completion_tokens: u64) -> f64 {
        (prompt_tokens as f64 * self.input_per_mtok + completion_tokens as f64 * self.output_per_mtok) / 1_000_000.0
    }
}

/// Provider-imposed quotas. Unset means "not limited by Aegis".
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Limits {
    pub rpm: Option<u64>,
    pub rpd: Option<u64>,
    pub tpm: Option<u64>,
    pub tpd: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RouteCfg {
    pub alias: String,
    #[serde(default)]
    pub description: String,
    /// Ordered `provider/model` targets; the first eligible one is used.
    pub targets: Vec<String>,
}

impl Config {
    pub fn load(path: &str) -> Result<Self> {
        let raw = std::fs::read_to_string(path).with_context(|| format!("reading config {path}"))?;
        Self::parse(&raw)
    }

    pub fn parse(raw: &str) -> Result<Self> {
        let cfg: Config = toml::from_str(raw).context("parsing config")?;
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> Result<()> {
        let mut names = HashSet::new();
        for p in &self.providers {
            if p.name.is_empty() || p.name.contains('/') {
                bail!("provider name {:?} must be non-empty and contain no '/'", p.name);
            }
            if !names.insert(&p.name) {
                bail!("duplicate provider name {:?}", p.name);
            }
            for m in &p.models {
                for c in &m.capabilities {
                    if !KNOWN_CAPABILITIES.contains(&c.as_str()) {
                        bail!("{}/{}: unknown capability {:?} (known: {:?})", p.name, m.id, c, KNOWN_CAPABILITIES);
                    }
                }
            }
        }
        let mut aliases = HashSet::new();
        for r in &self.routes {
            if !aliases.insert(&r.alias) {
                bail!("duplicate route alias {:?}", r.alias);
            }
            if r.targets.is_empty() {
                bail!("route {:?} has no targets", r.alias);
            }
            for t in &r.targets {
                if self.find_target(t).is_none() {
                    bail!("route {:?}: target {:?} does not match any configured provider/model", r.alias, t);
                }
            }
        }
        Ok(())
    }

    /// Resolve `provider/model` to indices into `providers` and its `models`.
    pub fn find_target(&self, target: &str) -> Option<(usize, usize)> {
        let (pname, mid) = target.split_once('/')?;
        let pi = self.providers.iter().position(|p| p.name == pname)?;
        let mi = self.providers[pi].models.iter().position(|m| m.id == mid)?;
        Some((pi, mi))
    }
}

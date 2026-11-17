//! Configuration: providers, models, routes, cache and server settings.
//!
//! Secrets never live in this file. A provider names the environment variable
//! that holds its API key (`api_key_env`); the value is read at startup.

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::collections::HashSet;

/// `embeddings` marks an embedding model: it serves `/v1/embeddings` and is
/// never used for chat.
pub const KNOWN_CAPABILITIES: &[&str] = &["tools", "vision", "json", "embeddings"];

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
    pub alerts: AlertsCfg,
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
    pub semantic: SemanticCfg,
}

/// Semantic caching: a cacheable request whose last user message means the
/// same as an earlier one (and whose every other part is identical) is served
/// the earlier answer. Off by default, because "means the same" is a
/// similarity score, not a guarantee.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct SemanticCfg {
    pub enabled: bool,
    /// `provider/model` of a model with the `embeddings` capability.
    pub embedding_model: String,
    /// Minimum cosine similarity (0 to 1) to count as the same question.
    pub threshold: f64,
    pub max_entries: usize,
    /// Messages shorter than this are matched exactly only; very short
    /// prompts ("yes", "next") are too easy to confuse.
    pub min_chars: usize,
    /// Give up on the embedding call after this long and go upstream.
    pub timeout_ms: u64,
    /// The embedding model is always warmed once at startup. If this is
    /// above zero it is also pinged this often, for servers that unload an
    /// idle model (Ollama does after 5 minutes by default). Leave at 0 for
    /// paid embedding providers.
    pub keep_warm_secs: u64,
}

impl Default for SemanticCfg {
    fn default() -> Self {
        Self { enabled: false, embedding_model: String::new(), threshold: 0.95, max_entries: 2000, min_chars: 16, timeout_ms: 3000, keep_warm_secs: 0 }
    }
}

/// Alerts for things an operator should hear about without watching the
/// dashboard. Always listed on the Alerts page; also posted to a webhook when
/// the environment variable named here holds a URL.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct AlertsCfg {
    /// Name of the environment variable holding the webhook URL (a secret).
    pub webhook_url_env: String,
    /// Warn when a client key has spent this fraction of its daily budget.
    pub budget_warn_at: f64,
    /// Alert when a provider target starts cooling down.
    pub cooldowns: bool,
    /// Do not repeat the same alert more often than this.
    pub min_interval_secs: u64,
}

impl Default for AlertsCfg {
    fn default() -> Self {
        Self { webhook_url_env: "AEGIS_ALERT_WEBHOOK".into(), budget_warn_at: 0.8, cooldowns: true, min_interval_secs: 300 }
    }
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
            semantic: SemanticCfg::default(),
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
    /// Longest a whole request may take.
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
    /// Longest wait to open a connection. A host that is down or unreachable
    /// is skipped after this.
    #[serde(default = "default_connect_timeout")]
    pub connect_timeout_secs: u64,
    /// Longest wait for the first token of a streamed answer before falling
    /// back to the next target. Defaults: 30 s for cloud providers, 90 s for
    /// local ones (a cold model load plus prompt processing on a CPU).
    #[serde(default)]
    pub first_byte_timeout_secs: Option<u64>,
    #[serde(default)]
    pub models: Vec<ModelCfg>,
}

impl ProviderCfg {
    pub fn first_byte_timeout(&self) -> u64 {
        self.first_byte_timeout_secs.unwrap_or(if self.tier == Tier::Local { 90 } else { 30 }).min(self.timeout_secs.max(1))
    }
}

fn default_timeout() -> u64 {
    120
}

fn default_connect_timeout() -> u64 {
    5
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

/// A model choice made in the dashboard, layered over the config file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelOverride {
    pub provider: String,
    /// The model id to send to the provider.
    pub model: String,
    /// `Some(id)`: use `model` in place of the config file's model `id`,
    /// keeping its position in routes and its limits and prices.
    /// `None`: an extra model, added to the provider.
    pub replaces: Option<String>,
    /// Known context window, when the provider's API reports one.
    pub context_window: Option<u64>,
    /// Capabilities for an added model.
    pub capabilities: Vec<String>,
}

impl Config {
    /// This configuration with dashboard model choices applied. Overrides
    /// that no longer fit (unknown provider, a config model that was removed
    /// from the file, a duplicate id) are skipped.
    pub fn with_overrides(&self, overrides: &[ModelOverride]) -> Config {
        let mut cfg = self.clone();
        let mut renames: Vec<(String, String)> = Vec::new();
        for o in overrides {
            let Some(p) = cfg.providers.iter_mut().find(|p| p.name == o.provider) else { continue };
            if o.model.is_empty() || p.models.iter().any(|m| m.id == o.model) {
                continue;
            }
            match &o.replaces {
                Some(old) => {
                    let Some(m) = p.models.iter_mut().find(|m| &m.id == old) else { continue };
                    m.id = o.model.clone();
                    if let Some(cw) = o.context_window {
                        m.context_window = cw;
                    }
                    renames.push((format!("{}/{}", p.name, old), format!("{}/{}", p.name, o.model)));
                }
                None => p.models.push(ModelCfg {
                    id: o.model.clone(),
                    context_window: o.context_window.unwrap_or_else(default_context),
                    max_output: None,
                    capabilities: o.capabilities.iter().filter(|c| KNOWN_CAPABILITIES.contains(&c.as_str())).cloned().collect(),
                    input_per_mtok: 0.0,
                    output_per_mtok: 0.0,
                    limits: Limits::default(),
                }),
            }
        }
        for (from, to) in &renames {
            for r in &mut cfg.routes {
                for t in &mut r.targets {
                    if t == from {
                        *t = to.clone();
                    }
                }
            }
            if &cfg.cache.semantic.embedding_model == from {
                cfg.cache.semantic.embedding_model = to.clone();
            }
        }
        cfg
    }

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
        let sem = &self.cache.semantic;
        if sem.enabled {
            let Some((pi, mi)) = self.find_target(&sem.embedding_model) else {
                bail!("cache.semantic.embedding_model {:?} does not match any configured provider/model", sem.embedding_model);
            };
            if !self.providers[pi].models[mi].has("embeddings") {
                bail!("cache.semantic.embedding_model {:?} needs the \"embeddings\" capability", sem.embedding_model);
            }
            if !(0.5..=1.0).contains(&sem.threshold) {
                bail!("cache.semantic.threshold must be between 0.5 and 1.0");
            }
        }
        if !(0.0..=1.0).contains(&self.alerts.budget_warn_at) {
            bail!("alerts.budget_warn_at must be between 0 and 1");
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

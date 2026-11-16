//! SQLite storage: client keys and the request log.
//!
//! Request rows are written by a background task fed through a channel, so
//! accounting never delays a response.

use anyhow::Result;
use sqlx::Row;
use sqlx::sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePool, SqlitePoolOptions};
use std::time::Duration;
use tokio::sync::mpsc;

#[derive(Clone)]
pub struct Db {
    pool: SqlitePool,
}

#[derive(Debug, Clone, Default)]
pub struct RequestRecord {
    pub request_id: String,
    pub ts: i64,
    pub key_id: Option<i64>,
    pub key_name: String,
    pub requested_model: String,
    pub provider: String,
    pub model: String,
    pub tier: String,
    /// ok | error | rejected | client_closed
    pub status: String,
    pub http_status: i64,
    pub stream: bool,
    /// hit | miss | skip
    pub cache: String,
    pub prompt_tokens: i64,
    pub completion_tokens: i64,
    pub usage_estimated: bool,
    pub cost_usd: f64,
    pub saved_usd: f64,
    pub latency_ms: i64,
    pub ttfb_ms: i64,
    pub fallback: bool,
    /// JSON array describing every target considered.
    pub attempts: String,
    pub error: String,
}

#[derive(Debug, Clone)]
pub struct KeyRow {
    pub id: i64,
    pub name: String,
    pub prefix: String,
    pub key_hash: String,
    pub created_at: i64,
    pub revoked_at: Option<i64>,
    pub rpm: Option<i64>,
    pub daily_budget_usd: Option<f64>,
    pub allow_paid: bool,
    pub allowed_models: String,
}

#[derive(Debug, Clone, Default)]
pub struct Summary {
    pub requests: i64,
    pub cache_hits: i64,
    pub errors: i64,
    pub fallbacks: i64,
    pub prompt_tokens: i64,
    pub completion_tokens: i64,
    pub cost_usd: f64,
    pub saved_usd: f64,
    pub p50_ms: i64,
    pub p95_ms: i64,
}

#[derive(Debug, Clone, Default)]
pub struct HourBucket {
    pub hour: i64,
    pub upstream: i64,
    pub cache_hits: i64,
    pub errors: i64,
}

#[derive(Debug, Clone, Default)]
pub struct ProviderStat {
    pub provider: String,
    pub model: String,
    pub tier: String,
    pub requests: i64,
    pub errors: i64,
    pub tokens: i64,
    pub cost_usd: f64,
    pub avg_latency_ms: f64,
}

const SCHEMA: &[&str] = &[
    "CREATE TABLE IF NOT EXISTS api_keys (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        prefix TEXT NOT NULL,
        key_hash TEXT NOT NULL UNIQUE,
        created_at INTEGER NOT NULL,
        revoked_at INTEGER,
        rpm INTEGER,
        daily_budget_usd REAL,
        allow_paid INTEGER NOT NULL DEFAULT 0,
        allowed_models TEXT NOT NULL DEFAULT ''
    )",
    "CREATE TABLE IF NOT EXISTS requests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        request_id TEXT NOT NULL,
        ts INTEGER NOT NULL,
        key_id INTEGER,
        key_name TEXT NOT NULL,
        requested_model TEXT NOT NULL,
        provider TEXT NOT NULL,
        model TEXT NOT NULL,
        tier TEXT NOT NULL,
        status TEXT NOT NULL,
        http_status INTEGER NOT NULL,
        stream INTEGER NOT NULL,
        cache TEXT NOT NULL,
        prompt_tokens INTEGER NOT NULL,
        completion_tokens INTEGER NOT NULL,
        usage_estimated INTEGER NOT NULL,
        cost_usd REAL NOT NULL,
        saved_usd REAL NOT NULL,
        latency_ms INTEGER NOT NULL,
        ttfb_ms INTEGER NOT NULL,
        fallback INTEGER NOT NULL,
        attempts TEXT NOT NULL,
        error TEXT NOT NULL
    )",
    "CREATE INDEX IF NOT EXISTS idx_requests_ts ON requests(ts)",
];

impl Db {
    pub async fn open(path: &str) -> Result<Self> {
        if let Some(parent) = std::path::Path::new(path).parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        let opts = SqliteConnectOptions::new()
            .filename(path)
            .create_if_missing(true)
            .journal_mode(SqliteJournalMode::Wal)
            .busy_timeout(Duration::from_secs(5));
        let pool = SqlitePoolOptions::new().max_connections(4).connect_with(opts).await?;
        for stmt in SCHEMA {
            sqlx::query(*stmt).execute(&pool).await?;
        }
        Ok(Self { pool })
    }

    // ---- keys ------------------------------------------------------------

    #[allow(clippy::too_many_arguments)]
    pub async fn create_key(
        &self,
        name: &str,
        prefix: &str,
        key_hash: &str,
        rpm: Option<i64>,
        daily_budget_usd: Option<f64>,
        allow_paid: bool,
        allowed_models: &str,
    ) -> Result<i64> {
        let res = sqlx::query(
            "INSERT INTO api_keys (name, prefix, key_hash, created_at, rpm, daily_budget_usd, allow_paid, allowed_models)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(name)
        .bind(prefix)
        .bind(key_hash)
        .bind(crate::types::now_ms())
        .bind(rpm)
        .bind(daily_budget_usd)
        .bind(allow_paid)
        .bind(allowed_models)
        .execute(&self.pool)
        .await?;
        Ok(res.last_insert_rowid())
    }

    pub async fn list_keys(&self) -> Result<Vec<KeyRow>> {
        let rows = sqlx::query(
            "SELECT id, name, prefix, key_hash, created_at, revoked_at, rpm, daily_budget_usd, allow_paid, allowed_models
             FROM api_keys ORDER BY id",
        )
        .fetch_all(&self.pool)
        .await?;
        Ok(rows
            .iter()
            .map(|r| KeyRow {
                id: r.get(0),
                name: r.get(1),
                prefix: r.get(2),
                key_hash: r.get(3),
                created_at: r.get(4),
                revoked_at: r.get(5),
                rpm: r.get(6),
                daily_budget_usd: r.get(7),
                allow_paid: r.get(8),
                allowed_models: r.get(9),
            })
            .collect())
    }

    pub async fn revoke_key(&self, id: i64) -> Result<bool> {
        let res = sqlx::query("UPDATE api_keys SET revoked_at = ? WHERE id = ? AND revoked_at IS NULL")
            .bind(crate::types::now_ms())
            .bind(id)
            .execute(&self.pool)
            .await?;
        Ok(res.rows_affected() > 0)
    }

    // ---- request log -----------------------------------------------------

    pub async fn insert_request(&self, r: &RequestRecord) -> Result<()> {
        sqlx::query(
            "INSERT INTO requests (request_id, ts, key_id, key_name, requested_model, provider, model, tier, status,
                http_status, stream, cache, prompt_tokens, completion_tokens, usage_estimated, cost_usd, saved_usd,
                latency_ms, ttfb_ms, fallback, attempts, error)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&r.request_id)
        .bind(r.ts)
        .bind(r.key_id)
        .bind(&r.key_name)
        .bind(&r.requested_model)
        .bind(&r.provider)
        .bind(&r.model)
        .bind(&r.tier)
        .bind(&r.status)
        .bind(r.http_status)
        .bind(r.stream)
        .bind(&r.cache)
        .bind(r.prompt_tokens)
        .bind(r.completion_tokens)
        .bind(r.usage_estimated)
        .bind(r.cost_usd)
        .bind(r.saved_usd)
        .bind(r.latency_ms)
        .bind(r.ttfb_ms)
        .bind(r.fallback)
        .bind(&r.attempts)
        .bind(&r.error)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn recent_requests(&self, limit: i64) -> Result<Vec<RequestRecord>> {
        let rows = sqlx::query(
            "SELECT request_id, ts, key_id, key_name, requested_model, provider, model, tier, status, http_status,
                stream, cache, prompt_tokens, completion_tokens, usage_estimated, cost_usd, saved_usd, latency_ms,
                ttfb_ms, fallback, attempts, error
             FROM requests ORDER BY id DESC LIMIT ?",
        )
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows
            .iter()
            .map(|r| RequestRecord {
                request_id: r.get(0),
                ts: r.get(1),
                key_id: r.get(2),
                key_name: r.get(3),
                requested_model: r.get(4),
                provider: r.get(5),
                model: r.get(6),
                tier: r.get(7),
                status: r.get(8),
                http_status: r.get(9),
                stream: r.get(10),
                cache: r.get(11),
                prompt_tokens: r.get(12),
                completion_tokens: r.get(13),
                usage_estimated: r.get(14),
                cost_usd: r.get(15),
                saved_usd: r.get(16),
                latency_ms: r.get(17),
                ttfb_ms: r.get(18),
                fallback: r.get(19),
                attempts: r.get(20),
                error: r.get(21),
            })
            .collect())
    }

    pub async fn summary(&self, since_ms: i64) -> Result<Summary> {
        let r = sqlx::query(
            "SELECT COUNT(*),
                COALESCE(SUM(cache = 'hit'), 0),
                COALESCE(SUM(status = 'error'), 0),
                COALESCE(SUM(fallback), 0),
                COALESCE(SUM(prompt_tokens), 0),
                COALESCE(SUM(completion_tokens), 0),
                COALESCE(SUM(cost_usd), 0.0),
                COALESCE(SUM(saved_usd), 0.0)
             FROM requests WHERE ts >= ?",
        )
        .bind(since_ms)
        .fetch_one(&self.pool)
        .await?;
        let lat: Vec<i64> = sqlx::query(
            "SELECT latency_ms FROM requests WHERE ts >= ? AND status = 'ok' AND cache != 'hit'
             ORDER BY latency_ms LIMIT 20000",
        )
        .bind(since_ms)
        .fetch_all(&self.pool)
        .await?
        .iter()
        .map(|r| r.get(0))
        .collect();
        let pct = |p: f64| lat.get(((lat.len() as f64 * p) as usize).min(lat.len().saturating_sub(1))).copied().unwrap_or(0);
        Ok(Summary {
            requests: r.get(0),
            cache_hits: r.get(1),
            errors: r.get(2),
            fallbacks: r.get(3),
            prompt_tokens: r.get(4),
            completion_tokens: r.get(5),
            cost_usd: r.get(6),
            saved_usd: r.get(7),
            p50_ms: pct(0.50),
            p95_ms: pct(0.95),
        })
    }

    pub async fn hourly(&self, since_ms: i64) -> Result<Vec<HourBucket>> {
        let rows = sqlx::query(
            "SELECT ts / 3600000,
                COALESCE(SUM(cache != 'hit' AND status != 'error'), 0),
                COALESCE(SUM(cache = 'hit'), 0),
                COALESCE(SUM(status = 'error'), 0)
             FROM requests WHERE ts >= ? GROUP BY 1 ORDER BY 1",
        )
        .bind(since_ms)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows
            .iter()
            .map(|r| HourBucket { hour: r.get(0), upstream: r.get(1), cache_hits: r.get(2), errors: r.get(3) })
            .collect())
    }

    pub async fn by_provider(&self, since_ms: i64) -> Result<Vec<ProviderStat>> {
        let rows = sqlx::query(
            "SELECT provider, model, tier, COUNT(*),
                COALESCE(SUM(status = 'error'), 0),
                COALESCE(SUM(prompt_tokens + completion_tokens), 0),
                COALESCE(SUM(cost_usd), 0.0),
                COALESCE(AVG(latency_ms), 0.0)
             FROM requests WHERE ts >= ? AND provider != '' AND cache != 'hit'
             GROUP BY provider, model, tier ORDER BY 4 DESC",
        )
        .bind(since_ms)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows
            .iter()
            .map(|r| ProviderStat {
                provider: r.get(0),
                model: r.get(1),
                tier: r.get(2),
                requests: r.get(3),
                errors: r.get(4),
                tokens: r.get(5),
                cost_usd: r.get(6),
                avg_latency_ms: r.get(7),
            })
            .collect())
    }

    /// (target, minute since epoch, requests, tokens) for quota rehydration.
    pub async fn usage_by_minute(&self, since_ms: i64) -> Result<Vec<(String, i64, i64, i64)>> {
        let rows = sqlx::query(
            "SELECT provider || '/' || model, ts / 60000, COUNT(*), COALESCE(SUM(prompt_tokens + completion_tokens), 0)
             FROM requests WHERE ts >= ? AND provider != '' AND cache != 'hit'
             GROUP BY 1, 2 ORDER BY 2",
        )
        .bind(since_ms)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows.iter().map(|r| (r.get(0), r.get(1), r.get(2), r.get(3))).collect())
    }

    /// Spend per client key since `since_ms`.
    pub async fn spend_by_key(&self, since_ms: i64) -> Result<Vec<(i64, f64)>> {
        let rows = sqlx::query(
            "SELECT key_id, COALESCE(SUM(cost_usd), 0.0) FROM requests
             WHERE ts >= ? AND key_id IS NOT NULL GROUP BY key_id",
        )
        .bind(since_ms)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows.iter().map(|r| (r.get(0), r.get(1))).collect())
    }

    pub async fn prune(&self, older_than_ms: i64) -> Result<u64> {
        let res = sqlx::query("DELETE FROM requests WHERE ts < ?").bind(older_than_ms).execute(&self.pool).await?;
        Ok(res.rows_affected())
    }

    /// Start the background writer and return the sender used by handlers.
    pub fn spawn_writer(&self, retention_days: u64) -> mpsc::UnboundedSender<RequestRecord> {
        let (tx, mut rx) = mpsc::unbounded_channel::<RequestRecord>();
        let db = self.clone();
        tokio::spawn(async move {
            while let Some(rec) = rx.recv().await {
                if let Err(e) = db.insert_request(&rec).await {
                    tracing::warn!(error = %e, "failed to write request record");
                }
            }
        });
        if retention_days > 0 {
            let db = self.clone();
            tokio::spawn(async move {
                let mut tick = tokio::time::interval(Duration::from_secs(3600));
                loop {
                    tick.tick().await;
                    let cutoff = crate::types::now_ms() - (retention_days as i64) * 86_400_000;
                    match db.prune(cutoff).await {
                        Ok(n) if n > 0 => tracing::info!(rows = n, "pruned old request records"),
                        Err(e) => tracing::warn!(error = %e, "prune failed"),
                        _ => {}
                    }
                }
            });
        }
        tx
    }
}

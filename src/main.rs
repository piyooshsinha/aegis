use aegis::auth::{KEY_PREFIX, generate_key, hash_key, random_token};
use aegis::config::Config;
use aegis::db::Db;
use aegis::gateway::AppState;
use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "aegis", version, about = "AI API gateway: one OpenAI-compatible endpoint in front of many providers")]
struct Cli {
    /// Path to the TOML configuration file.
    #[arg(long, short, env = "AEGIS_CONFIG", default_value = "aegis.toml", global = true)]
    config: String,
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Run the gateway (default).
    Serve,
    /// Manage client API keys.
    Key {
        #[command(subcommand)]
        action: KeyAction,
    },
    /// Validate the configuration file and exit.
    Check,
    /// Write a consistent copy of the database (safe while the gateway runs).
    Backup {
        /// Folder for the copies.
        #[arg(long, default_value = "data/backups")]
        dir: String,
        /// How many copies to keep; older ones are deleted.
        #[arg(long, default_value_t = 14)]
        keep: usize,
        /// Repeat every this many hours instead of exiting (for a backup container).
        #[arg(long)]
        every_hours: Option<u64>,
    },
}

#[derive(Subcommand)]
enum KeyAction {
    /// Create a key. It is printed once and only its hash is stored.
    Create {
        #[arg(long)]
        name: String,
        /// Requests per minute allowed for this key.
        #[arg(long)]
        rpm: Option<i64>,
        /// Daily spend ceiling in USD.
        #[arg(long)]
        daily_budget: Option<f64>,
        /// Allow fallback to paid providers.
        #[arg(long)]
        allow_paid: bool,
        /// Comma-separated models/aliases this key may use (default: all).
        #[arg(long, default_value = "")]
        models: String,
    },
    List,
    Revoke {
        id: i64,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();

    let cli = Cli::parse();
    let cfg = Config::load(&cli.config)?;

    match cli.command.unwrap_or(Command::Serve) {
        Command::Check => {
            println!("{}: ok — {} providers, {} routes", cli.config, cfg.providers.len(), cfg.routes.len());
            Ok(())
        }
        Command::Key { action } => key_command(&cfg, action).await,
        Command::Backup { dir, keep, every_hours } => backup(&cfg, &dir, keep, every_hours).await,
        Command::Serve => serve(cfg).await,
    }
}

async fn key_command(cfg: &Config, action: KeyAction) -> Result<()> {
    let db = Db::open(&cfg.server.db_path).await?;
    match action {
        KeyAction::Create { name, rpm, daily_budget, allow_paid, models } => {
            let key = generate_key();
            let prefix: String = key.chars().take(KEY_PREFIX.len() + 6).collect();
            let id = db.create_key(&name, &prefix, &hash_key(&key), rpm, daily_budget, allow_paid, &models).await?;
            println!("Created key #{id} for {name}. Store it now; it cannot be shown again:\n{key}");
            println!("(A running gateway picks up CLI-created keys on restart.)");
        }
        KeyAction::List => {
            for k in db.list_keys().await? {
                println!(
                    "#{:<4} {:<24} {}…  {}{}",
                    k.id,
                    k.name,
                    k.prefix,
                    if k.allow_paid { "paid-ok" } else { "free-only" },
                    if k.revoked_at.is_some() { "  REVOKED" } else { "" }
                );
            }
        }
        KeyAction::Revoke { id } => {
            if db.revoke_key(id).await? {
                println!("Revoked key #{id}. Restart the gateway (or revoke from the UI) for it to take effect.");
            } else {
                println!("No active key with id {id}.");
            }
        }
    }
    Ok(())
}

async fn backup(cfg: &Config, dir: &str, keep: usize, every_hours: Option<u64>) -> Result<()> {
    let db = Db::open(&cfg.server.db_path).await?;
    loop {
        match db.backup(std::path::Path::new(dir), keep).await {
            Ok((path, size)) => tracing::info!(path = %path.display(), bytes = size, keep, "database backed up"),
            Err(e) if every_hours.is_some() => tracing::error!("backup failed: {e:#}"),
            Err(e) => return Err(e),
        }
        let Some(h) = every_hours else { return Ok(()) };
        tokio::time::sleep(std::time::Duration::from_secs(h.max(1) * 3600)).await;
    }
}

async fn serve(cfg: Config) -> Result<()> {
    let admin_token = match std::env::var("AEGIS_ADMIN_TOKEN").ok().map(|t| t.trim().to_string()).filter(|t| !t.is_empty()) {
        Some(t) => t,
        None => {
            let t = random_token();
            eprintln!("AEGIS_ADMIN_TOKEN is not set. Generated a one-off admin token for this run:\n  {t}");
            t
        }
    };
    let bind = cfg.server.bind.clone();
    for p in &cfg.providers {
        let env_key = p.api_key_env.is_empty() || std::env::var(&p.api_key_env).is_ok_and(|v| !v.trim().is_empty());
        tracing::info!(provider = %p.name, tier = p.tier.as_str(), models = p.models.len(), env_key, "provider");
    }
    let sealer = match std::env::var(aegis::secrets::SECRET_ENV).ok().filter(|v| !v.trim().is_empty()) {
        Some(secret) => Some(aegis::secrets::Sealer::new(&secret)?),
        None => {
            tracing::info!("{} is not set: provider keys can only come from environment variables", aegis::secrets::SECRET_ENV);
            None
        }
    };
    let state = AppState::build(cfg, admin_token, sealer).await?;
    let app = aegis::router(state);
    let listener = tokio::net::TcpListener::bind(&bind).await.with_context(|| format!("binding {bind}"))?;
    tracing::info!("Aegis listening on http://{bind} (dashboard at /ui)");
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}

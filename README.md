# Aegis — AI API gateway

One OpenAI-compatible endpoint in front of local, free-tier and paid model
providers. Aegis caches deterministic requests, routes around exhausted quotas
and dead providers, tracks tokens and cost, and has an admin dashboard. It is a
single Rust binary; the dashboard is rendered server-side in Rust (`maud`).

Design background: [ai_api_mediator_architecture.md](ai_api_mediator_architecture.md).

## Use it

Point any OpenAI SDK at the gateway and use a route alias as the model:

```bash
curl http://HOST:8787/v1/chat/completions \
  -H "Authorization: Bearer $AEGIS_KEY" -H "content-type: application/json" \
  -d '{"model":"fast-model","messages":[{"role":"user","content":"hello"}]}'
```

`model` can be a route alias (`fast-model`), an explicit target
(`ollama/qwen2.5:7b`), or a bare model id. `GET /v1/models` lists them.

Every response says who answered:

| Header | Meaning |
| --- | --- |
| `x-aegis-provider`, `x-aegis-model`, `x-aegis-tier` | The target that served the request |
| `x-aegis-cache` | `hit`, `semantic` (matched by meaning), `miss`, or `skip` (not cacheable) |
| `x-aegis-fallback` | `true` when the first target of the route did not serve it |
| `x-aegis-attempts` | How many targets were considered |

Request headers: `x-aegis-cache: bypass`, `force` or `exact`, and
`x-aegis-no-fallback: true` to fail instead of falling back.

### Embeddings

`POST /v1/embeddings` takes the OpenAI request shape (`input` as a string or a
list of strings, optional `dimensions`, `encoding_format` `float` or `base64`).
Mark embedding models in the config with `capabilities = ["embeddings"]`; they
are never used for chat. Vectors from different models cannot be compared, so
a route only falls back between targets that serve the same model id as its
first embedding target; other targets are skipped and reported.

## Run it

```bash
export AEGIS_ADMIN_TOKEN=choose-a-long-secret   # dashboard sign-in
export AEGIS_SECRET_KEY=$(openssl rand -hex 32) # lets you save provider keys in the dashboard
cargo run --release -- serve                    # reads ./aegis.toml
cargo run --release -- key create --name my-app # prints a client key once
```

Dashboard: `http://HOST:8787/ui`. Other commands: `aegis check`,
`aegis key list`, `aegis key revoke <id>`. Keys can also be created and
revoked in the dashboard, which takes effect immediately. The Keys page also
breaks usage down by client key (requests, cache share, tokens, spend) over
24 hours, 7 days or 30 days.

### Provider API keys

There are two ways to give a provider its key:

- **Dashboard** (needs `AEGIS_SECRET_KEY`): open Providers, choose "Set API
  key" and paste it. The key is encrypted with XChaCha20-Poly1305 before it is
  written to SQLite, is used from the next request with no restart, and is
  never shown again (only its last four characters are).
- **Environment**: set the variable named by the provider's `api_key_env`
  (for example `GROQ_API_KEY`) before starting.

"Test connection" on the same page sends one tiny request and says whether
the provider accepted the key.

A key saved in the dashboard takes precedence over the environment. Keep
`AEGIS_SECRET_KEY` unchanged: keys saved under a different value cannot be
decrypted and must be entered again.

### Semantic caching

Off by default; turn it on under `[cache.semantic]`. A cacheable request then
also matches a stored answer when its last user message means the same as an
earlier one, judged by cosine similarity of embeddings against `threshold`.
Everything else must be identical: client key scope, model, parameters, system
prompt and earlier turns. Requests with tools or images, and messages shorter
than `min_chars`, are matched exactly only. If the embedding model is slow or
down the request goes upstream as usual.

A similarity score is not a guarantee. "What is 2+3?" and "What is 2+4?" are
close in most embedding models, so keep the threshold high for anything where
a near-match answer would be wrong, or send `x-aegis-cache: exact`. The index
is scanned linearly in memory and saved to SQLite, so it survives a restart;
a purge empties both. Changing `embedding_model` discards the saved entries.
The embedding model is warmed once at startup; `keep_warm_secs` pings it
periodically for servers that unload idle models.

### Alerts

The dashboard's Alerts page lists a client key reaching 80% and 100% of its
daily budget, and a model going into cooldown. Set `AEGIS_ALERT_WEBHOOK` to a
URL to also receive each one as a JSON POST; the payload has `text` and
`content` fields, so Slack and Discord incoming webhooks work unchanged.

## How the design gaps were closed

- **Streaming** is built in, not deferred. All three adapters stream, and the
  gateway re-emits OpenAI-style SSE. Usage is captured from the stream for
  accounting. If a provider fails before its first chunk, the request still
  falls back; after that the error is reported in-band.
- **Caching and non-determinism.** Only requests with `temperature` at or below
  `cache.max_temperature` (default 0), or with a `seed`, are cached. Entries
  expire after `ttl_secs`, can be purged, are partitioned per client key by
  default, and are saved to SQLite so a restart does not empty the cache. Streamed answers are cached once complete and replayed as a stream.
- **Quota tracking** combines local sliding-window counters (requests and
  tokens per minute and per day, against the limits in config) with what
  providers report: `429` + `retry-after` starts a cooldown, and
  `x-ratelimit-remaining-*` headers stop a target before it is exhausted.
  Counters are rebuilt from the request log on restart.
- **Fallback is never silent.** Targets declare capabilities (`tools`,
  `vision`, `json`) and a context window; a target that cannot handle the
  request is skipped. Response headers and the request log record what was
  tried and why. A provider's 4xx rejection is returned as-is, not retried.
- **Auth.** Client keys are random, shown once and stored as SHA-256 hashes,
  with optional requests-per-minute, daily budget, model allow-list and a
  paid-tier permission. Provider keys come from environment variables or are
  stored encrypted from the dashboard. The dashboard needs the admin token.
- **Health.** Three consecutive failures open a circuit breaker with backoff.

## Configuration

See [aegis.toml](aegis.toml); every option is commented there. Model ids,
prices and free-tier limits in it are starting values — check them against
each provider's current documentation.

## Known limits

- Anthropic: sampling parameters are never forwarded (the current API rejects
  them), forced `tool_choice` becomes `auto`, and `response_format` is not
  translated, so Anthropic models are not listed with the `json` capability.
- Gemini: image input must be a `data:` URL.
- Embeddings: Anthropic has no embeddings API. Token-id arrays as `input` are
  refused. Gemini does not report embedding token usage, so it is estimated.
- Storage is SQLite only. Cache and quota state are per process, so run one
  instance.
- Token counts are estimated at 4 characters per token for pre-flight checks,
  and for accounting only when a provider omits usage (marked `~` in the log).

## Deploy

[deploy/compose.yml](deploy/compose.yml) builds the image and runs it with host
networking. The NUC config ([deploy/aegis.nuc.toml](deploy/aegis.nuc.toml)) binds port 9001. Secrets go in `deploy/.env` (see `.env.example`). The compose file also runs a small
Ollama container for the embedding model and a small last-resort chat model;
fetch them once with `docker exec aegis-ollama ollama pull nomic-embed-text`
and `docker exec aegis-ollama ollama pull qwen2.5:3b`.

```bash
docker compose -f deploy/compose.yml up -d --build
```

## Tests

```bash
cargo test
```

The end-to-end suite starts a real gateway against a mock upstream that
succeeds, rate-limits, fails and rejects.

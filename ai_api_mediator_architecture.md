# AI API Mediator (Gateway) Architecture

## 1. High-Level Overview
The Mediator acts as a unified reverse proxy and orchestrator for AI API requests. Client applications send requests to this single service (ideally using a standard format like the OpenAI API specification), and the Mediator handles caching, routing, fallback logic, token tracking, and the actual communication with the target AI providers.

## 2. Core Components

### A. API Layer (Ingress)
*   **Function:** Exposes REST/gRPC endpoints for client applications.
*   **Design:** Implement an OpenAI-compatible API interface (e.g., `/v1/chat/completions`). This allows any existing AI tool or library (like LangChain, LlamaIndex, or standard OpenAI SDKs) to connect to your Rust mediator simply by changing the base URL.

### B. Cache Manager (The "Echo" Layer)
*   **Function:** Intercepts requests to check if a highly similar or exact prompt has been answered recently.
*   **Design:** 
    *   **Exact Match:** Hashing the incoming prompt and parameters (temperature, model type) and checking against an in-memory cache (like `moka`) or Redis.
    *   **Semantic Match (Optional later):** Using a lightweight embedding model to check for prompt similarity.

### C. Router & Cost Optimizer (The "Brain")
*   **Function:** Decides *where* to send the prompt if it's not cached.
*   **Rules Engine:**
    *   **Tier Switching:** Checks the remaining quota of free-tier APIs (e.g., Groq free tier limit, Gemini free tier limit). If the limit is reached, it seamlessly falls back to a paid API (e.g., Claude 3.5 Sonnet, Gemini Pro paid).
    *   **Model Translation:** Maps generic requests (e.g., "fast-model") to specific provider models (e.g., "llama3-8b-8192" on Groq or local Ollama).

### D. Provider Adapters
*   **Function:** Translates the unified internal request into the specific JSON schema required by each provider.
*   **Supported Targets:**
    *   Local LLMs (Ollama, vLLM, Llama.cpp)
    *   Groq API
    *   Anthropic Claude API
    *   Google Gemini API
    *   OpenAI API (as a fallback)

### E. Telemetry & Accounting
*   **Function:** Logs all requests, latency, token usage (prompt + completion tokens), and calculates estimated costs.
*   **Design:** Asynchronous logging to a database (SQLite for local setups, PostgreSQL for production) to avoid blocking the main API response.

---

## 3. Data Flow

1.  **Request:** App sends `POST /v1/chat/completions` to the Mediator.
2.  **Auth & Validation:** Validate client API key and parse the JSON payload.
3.  **Cache Check:** Hash the payload. 
    *   *Cache Hit:* Return stored response immediately.
    *   *Cache Miss:* Proceed to step 4.
4.  **Routing Decision:** Router evaluates rules. (e.g., "Try Groq Free -> If Rate Limited -> Try Local LLM -> If down -> Use Claude 3 Haiku").
5.  **Dispatch:** Adapter formats request for the chosen provider and sends the HTTP request.
6.  **Response Handling:** Stream or buffer the response from the provider.
7.  **Accounting:** Send usage metrics (tokens, cost, latency, provider used) to the database asynchronously via a message channel.
8.  **Cache Update:** Store the new prompt/response pair in the cache.
9.  **Return:** Send the standardized response back to the client application.

---

## 4. Recommended Rust Stack

To build this efficiently, the following Rust ecosystem crates are highly recommended:

*   **Web Framework:** `axum` (Built on Tokio, incredibly fast, great ecosystem, perfect for APIs).
*   **Async Runtime:** `tokio` (The standard for Rust async I/O).
*   **HTTP Client:** `reqwest` (For making outgoing requests to Groq, Claude, Gemini, etc.).
*   **Serialization:** `serde` and `serde_json` (Crucial for parsing and generating API payloads).
*   **Caching:** `moka` (A high-performance concurrent cache for Rust, great for the prompt caching layer) or `redis` crate if you want distributed caching.
*   **Database (Metrics/Tracking):** `sqlx` (Compile-time checked SQL queries). Supports SQLite for easy deployment or Postgres for scale.
*   **Tracing & Observability:** `tracing` and `tracing-subscriber` (For logging request flows, latency, and debugging).
*   **Error Handling:** `anyhow` and `thiserror`.

## 5. Future Expansion Ideas
*   **Streaming Support:** Crucial for UX. You'll need to handle Server-Sent Events (SSE) passing through the mediator from the provider to the client.
*   **Semantic Caching:** Integrating an in-memory vector database (like `qdrant` or a lightweight Rust native one) to cache based on the *meaning* of the prompt, not just exact text matches.
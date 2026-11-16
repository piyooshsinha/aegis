FROM rust:1-bookworm AS build
WORKDIR /src
RUN apt-get update && apt-get install -y --no-install-recommends cmake clang && rm -rf /var/lib/apt/lists/*
COPY Cargo.toml Cargo.lock ./
# Build dependencies first so source edits do not recompile them.
RUN mkdir src && echo 'fn main(){}' > src/main.rs && touch src/lib.rs && cargo build --release --locked && rm -rf src
COPY src ./src
RUN touch src/main.rs src/lib.rs && cargo build --release --locked

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --home /app aegis && mkdir -p /app/data && chown -R aegis /app
WORKDIR /app
COPY --from=build /src/target/release/aegis /usr/local/bin/aegis
USER aegis
ENV AEGIS_CONFIG=/app/aegis.toml
EXPOSE 8787
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s CMD curl -fsS http://127.0.0.1:8787/healthz || exit 1
ENTRYPOINT ["aegis"]
CMD ["serve"]

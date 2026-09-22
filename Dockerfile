# syntax=docker/dockerfile:1
FROM rust:1.97-bookworm AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY src ./src
# The goose git dependency is large; cache the registry and target between builds.
# --locked: the committed Cargo.lock is the build, no silent dependency drift.
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/src/target \
    cargo build --release --locked -j 2 && \
    cp target/release/goose-gateway /goose-gateway

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=build /goose-gateway /usr/local/bin/goose-gateway
# Token cache is mounted at $GOOSE_PATH_ROOT/config/xai_oauth/tokens.json
ENV GOOSE_PATH_ROOT=/goose GOOSE_GATEWAY_HOST=0.0.0.0 GOOSE_GATEWAY_PORT=8791 RUST_LOG=info
EXPOSE 8791
USER nobody
ENTRYPOINT ["goose-gateway"]

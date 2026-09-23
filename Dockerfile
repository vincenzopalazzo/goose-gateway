# syntax=docker/dockerfile:1.7
#
# One fully static (musl) binary on an empty image, plus a CA bundle. No shell, no package
# manager: nothing in the image but what the gateway runs.
#
# Built natively per architecture (tested on arm64; amd64 uses the same Alpine toolchain). A Rust
# build of goose under QEMU emulation is impractically slow, so CI uses a runner of each architecture.

FROM rust:1.97-alpine3.22 AS build
# Native dependencies, linked statically: AWS-LC (rustls), SQLite, tree-sitter and zstd.
RUN apk add --no-cache musl-dev perl make cmake clang linux-headers ca-certificates file
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY src ./src
# The gateway waits on the network, not the CPU, so it is built for size. OPT_LEVEL=3 builds for
# speed instead.
ARG OPT_LEVEL=s
ENV CARGO_PROFILE_RELEASE_OPT_LEVEL=${OPT_LEVEL}
# The goose git dependency is large; cache the registry and target between builds.
# --locked: the committed Cargo.lock is the build, no silent dependency drift.
# -j 2 and no LTO: goose is big enough that more parallelism or LTO exhausts the memory of a
# default Docker Desktop VM.
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/src/target,id=goose-gateway-musl-target \
    cargo build --release --locked -j 2 \
    && cp target/release/goose-gateway /goose-gateway \
    && file /goose-gateway | grep -Eq "static(-pie)? linked|statically linked"
# goose writes under $GOOSE_PATH_ROOT (config, data, state), and some libraries want /tmp.
RUN mkdir -p /rootfs/goose/config /rootfs/tmp \
    && chown -R 1000:1000 /rootfs/goose \
    && chmod 1777 /rootfs/tmp

FROM scratch
COPY --from=build /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/ca-certificates.crt
COPY --from=build /rootfs/ /
COPY --from=build /goose-gateway /usr/local/bin/goose-gateway
# goose reads its config, sign-ins and keys from $GOOSE_PATH_ROOT/config: mount it read-write
# (goose rewrites tokens when it refreshes them).
ENV GOOSE_PATH_ROOT=/goose \
    GOOSE_GATEWAY_HOST=0.0.0.0 \
    GOOSE_GATEWAY_PORT=8791 \
    RUST_LOG=info \
    HOME=/goose \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
EXPOSE 8791
# The usual owner of a bind-mounted ~/.config/goose on Linux, and Umbrel's app user.
USER 1000:1000
ENTRYPOINT ["/usr/local/bin/goose-gateway"]

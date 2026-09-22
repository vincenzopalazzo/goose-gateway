# goose-gateway

A small Rust service that puts an **OpenAI-compatible** endpoint in front of the
[goose SDK](https://github.com/aaif-goose/goose), so the browser app's assistant can run on
an X subscription.

```
browser ──POST /v1/chat/completions──▶ goose-gateway ──goose xai_oauth provider──▶ api.x.ai
```

## Why it exists

- `goose` is a Rust crate; it has no JavaScript or WASM target and cannot be imported into a
  page. This runs it out of process.
- The X subscription credential is an OAuth token that expires. A page cannot renew it, because
  `auth.x.ai` sends no CORS headers. goose's own `xai_oauth` provider renews it, and this
  gateway simply uses that provider, so the refresh logic is goose's, not a copy.
- It speaks the OpenAI wire format, which the app's Grok provider already uses. Pointing that
  provider at the gateway is the only integration; the app carries no goose-specific code.

## Run

```bash
# once, on the host: sign in so a token exists at ~/.config/goose/xai_oauth/tokens.json
goose configure          # choose xai_oauth

# with the rest of the stack
docker compose up -d     # from the repository root

# or natively
cargo run --release      # http://127.0.0.1:8791
```

In the app's assistant drawer choose **goose (local)**, or set the Grok provider's host to
`http://127.0.0.1:8791/v1` with any placeholder token.

## Endpoints

| Route                       | Purpose                                              |
| --------------------------- | ---------------------------------------------------- |
| `POST /v1/chat/completions` | Chat with tools; system, user, assistant, tool roles |
| `GET /v1/models`            | The Grok models offered                              |
| `GET /health`               | `credential: true` once goose can build the provider |

No streaming yet: the app uses non-streaming completions.

## Configuration

| Variable              | Default          | Meaning                                                     |
| --------------------- | ---------------- | ----------------------------------------------------------- |
| `GOOSE_PATH_ROOT`     | goose's default  | goose looks for its config under `<root>/config`             |
| `GOOSE_GATEWAY_HOST`  | `127.0.0.1`      | listen address (`0.0.0.0` inside the container)             |
| `GOOSE_GATEWAY_PORT`  | `8791`           |                                                             |
| `RUST_LOG`            | `info`           |                                                             |

The gateway holds no secret of its own. It reads the same token file goose wrote, and the
provider writes it back when it refreshes, which is why the compose volume is read-write.

## Building

`goose` is not on crates.io — the crate by that name is a load-testing tool — so `Cargo.toml`
pins the repository by revision and repeats the two `[patch.crates-io]` entries goose's
workspace needs. The first build compiles goose's dependency tree and takes a while; the
Dockerfile caches the registry and target directory between builds.

# goose-gateway

An **OpenAI-compatible** HTTP endpoint in front of [goose](https://github.com/aaif-goose/goose),
so a browser app can chat with Grok on an **xAI SuperGrok (X) subscription** instead of an
API key.

```
browser ──POST /v1/chat/completions──▶ goose-gateway ──goose xai_oauth provider──▶ api.x.ai
```

It was written for the assistant in an LDK Server dashboard, but it's a plain OpenAI endpoint
and works with any client that can use a custom base URL.

## Why it exists

- `goose` is a Rust crate with no JavaScript or WASM target, so a page can't import it. This
  runs it out of process.
- A subscription credential is an OAuth token that expires. A page can't renew it, because
  `auth.x.ai` sends no CORS headers. goose's own `xai_oauth` provider renews it, and this
  gateway uses that provider, so the sign-in and refresh logic are goose's, not a copy.
- It speaks the OpenAI wire format, so a client needs no goose-specific code: point its base
  URL at the gateway and send any placeholder API key.

## Run

Sign in once on the host, so a token exists at `~/.config/goose/xai_oauth/tokens.json`:

```bash
goose configure          # choose xai_oauth
```

Then either:

```bash
docker compose up -d     # http://127.0.0.1:8791
cargo run --release      # or natively
```

Point your client at `http://127.0.0.1:8791/v1`.

```bash
curl -N http://127.0.0.1:8791/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"grok-4.7","stream":true,"messages":[{"role":"user","content":"Hello"}]}'
```

## Endpoints

| Route                       | Purpose                                                                 |
| --------------------------- | ----------------------------------------------------------------------- |
| `POST /v1/chat/completions` | Chat with tools; `system`, `user`, `assistant` and `tool` roles         |
| `GET /v1/models`            | The Grok models offered                                                 |
| `GET /health`               | `{"ok":true,"provider":"xai_oauth","credential":bool}`                  |

`credential` is true once goose's token cache holds a refresh token. Without one, chat
requests answer `401` with a message saying to run `goose configure`.

### Streaming

With `"stream": true` the reply comes back as server-sent events in OpenAI's
`chat.completion.chunk` format, relayed straight from goose's message stream:

- Text arrives as `delta.content` pieces as the model writes them.
- Each tool call arrives complete in a single `delta.tool_calls` entry with its `index`, because
  goose assembles tool calls before yielding them.
- A final chunk carries `finish_reason` (`stop` or `tool_calls`). A usage chunk follows when
  `stream_options.include_usage` is set, then `data: [DONE]`.

Sign-in and upstream failures that happen before the first chunk keep their HTTP status
(`401`, `502`). A failure after that arrives in-band as `data: {"error":{...}}` with no
`[DONE]`, so a client can tell the reply is incomplete. When the client disconnects, the
upstream request is cancelled.

## Browser access

The gateway has no login of its own and spends your subscription, so it only answers the
browser origins you allow. Otherwise any website you visit could use it.

- `http(s)://localhost`, `127.0.0.1` and `[::1]` on any port are always allowed.
- Add others in `GOOSE_GATEWAY_ALLOWED_ORIGINS`, comma separated and matched exactly, for
  example `https://my-dashboard.example`. `*` allows every origin; don't use it on a machine
  where you browse the web.
- Requests from other origins get `403`, enforced in the server as well as in CORS. That also
  covers DNS-rebinding pages, which a browser treats as same-origin.
- Requests with no `Origin` header (curl, server-side clients) are accepted. Anything that can
  make them on this machine could read the token file directly anyway.

Keep it bound to loopback (the default, and what `compose.yaml` publishes) unless you put
your own authentication in front of it.

## Configuration

| Variable                        | Default         | Meaning                                                       |
| ------------------------------- | --------------- | ------------------------------------------------------------- |
| `GOOSE_PATH_ROOT`               | goose's default | goose reads its config from `<root>/config`                    |
| `GOOSE_GATEWAY_HOST`            | `127.0.0.1`     | listen address (`0.0.0.0` inside the container)                |
| `GOOSE_GATEWAY_PORT`            | `8791`          |                                                               |
| `GOOSE_GATEWAY_ALLOWED_ORIGINS` | empty           | extra browser origins, see [Browser access](#browser-access)  |
| `GOOSE_CONFIG_DIR`              | `~/.config/goose` | compose only: host directory mounted as goose's config      |
| `RUST_LOG`                      | `info`          |                                                               |

goose's default config directory is `~/.config/goose`, on macOS too. goose settings can also
be overridden by environment variables named after the setting in capitals, for example
`XAI_HOST`.

The gateway holds no secret of its own. It reads the token file goose wrote, and goose writes
it back when it refreshes the access token, which is why the compose volume is read-write.
The provider is always `xai_oauth`; the model comes from each request.

## Building

`goose` isn't on crates.io (the crate by that name is a load-testing tool), so `Cargo.toml`
pins the goose repository by revision and repeats the two `[patch.crates-io]` entries goose's
workspace needs. The first build compiles goose's whole dependency tree and takes a while.

The release profile is deliberately plain and the Dockerfile builds with `-j 2`: thin LTO with
a single codegen unit pushed peak memory past what a default Docker Desktop VM allows.

## License

MIT

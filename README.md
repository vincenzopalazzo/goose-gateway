# goose-gateway

An **OpenAI-compatible** HTTP endpoint in front of [goose](https://github.com/aaif-goose/goose),
so a browser app can chat through any provider goose supports: subscriptions such as **xAI
SuperGrok**, **ChatGPT** and **GitHub Copilot**, API keys (Anthropic, OpenAI, Gemini,
OpenRouter, Mistral…), or local models (Ollama). It also lets that app set providers up:
list them, save a key, or run a subscription sign-in.

```
browser ──POST /v1/chat/completions──▶ goose-gateway ──goose provider──▶ xAI · OpenAI · Anthropic · …
```

It was written for the assistant in an LDK Server dashboard, but it's a plain OpenAI endpoint
and works with any client that can use a custom base URL.

## Why it exists

- `goose` is a Rust crate with no JavaScript or WASM target, so a page can't import it. This
  runs it out of process.
- A subscription credential is an OAuth token that expires. A page can't obtain or renew it
  (`auth.x.ai`, for one, sends no CORS headers). goose's own providers do, and this gateway
  uses them, so the sign-in and refresh logic are goose's, not a copy.
- It speaks the OpenAI wire format, so a client needs no goose-specific code: point its base
  URL at the gateway and send any placeholder API key.

## Run

Either:

```bash
docker compose up -d     # http://127.0.0.1:8791
cargo run --release      # or natively
```

Point your client at `http://127.0.0.1:8791/v1`, then set a provider up through the
[setup API](#provider-setup) or with `goose configure` on the host.

```bash
curl -N http://127.0.0.1:8791/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"grok-4.7","stream":true,"messages":[{"role":"user","content":"Hello"}]}'
```

## Endpoints

| Route                                       | Purpose                                                           |
| ------------------------------------------- | ----------------------------------------------------------------- |
| `POST /v1/chat/completions`                 | Chat with tools; `system`, `user`, `assistant` and `tool` roles   |
| `GET /v1/models?provider=`                  | Models goose knows for a provider                                 |
| `GET /health?provider=`                     | `{"ok":true,"provider":"…","credential":bool,"tools":bool}`       |
| `GET /v1/providers`                         | Every chat provider goose offers, with its setup status           |
| `POST /v1/providers/{id}/config`            | Save a provider's settings (`{"fields":{"KEY":"value"}}`)         |
| `DELETE /v1/providers/{id}/config`          | Remove its settings, or sign out                                  |
| `POST /v1/providers/{id}/sign-in`           | Run a subscription sign-in, streamed as server-sent events        |
| `POST /v1/providers/{id}/sign-in/callback`  | Deliver a sign-in redirect the browser could not (`{"url":"…"}`)  |
| `GET /v1/tools`                             | An MCP server's tools, when one is configured (see [Tools](#tools)) |
| `POST /v1/tools/{name}`                     | Call one with a JSON object of arguments                          |

Chat requests pick a provider with a `"provider"` field (not part of OpenAI's API); without
one the gateway uses `GOOSE_GATEWAY_PROVIDER`, `xai_oauth` by default. `?provider=` works the
same way for `/health` and `/v1/models`. A provider that is not set up answers `401` with a
message saying so, and `credential` is false for it.

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

## Provider setup

`GET /v1/providers` lists goose's chat providers (its "agent" providers, which run their own
tools, are left out):

```json
{ "id": "anthropic", "name": "Anthropic", "method": "single_api_key", "featured": true,
  "sign_in": false, "configured": false, "default_model": "claude-sonnet-4-5", "models": ["…"],
  "fields": [{ "key": "ANTHROPIC_API_KEY", "secret": true, "required": true, "set": false }] }
```

A secret's value is never returned, only whether it is `set`. Saving writes secrets to goose's
secret storage and the rest to its config, as `goose configure` would, so a key saved here is
also what the goose CLI on that machine uses. Removing clears them and any sign-in.

### Subscription sign-in

`POST /v1/providers/{id}/sign-in` runs goose's own `configure_oauth` and streams its progress:

```
data: {"type":"started","provider":"github_copilot"}
data: {"type":"device_code","user_code":"ABCD-1234","verification_uri":"https://github.com/login/device","expires_in":899}
data: {"type":"done","provider":{…}}
```

- **Device code** (GitHub Copilot, Kimi): show the code and link; goose polls until it's
  approved. xAI falls back to a code too if its redirect sign-in fails.
- **Browser redirect** (xAI, ChatGPT, Gemini, Hugging Face): goose opens the provider's sign-in
  page and waits, up to 5 minutes, for the redirect to a `localhost` port.
  - Run natively, goose opens your browser and the redirect reaches it. Nothing else to do.
  - In Docker there is no browser, so the stream carries `{"type":"open_url","url":…}` for the
    client to open. After signing in, the browser is sent to `http://127.0.0.1:…/callback?…`,
    which doesn't load, because goose's listener is inside the container. The client posts that
    address to `/sign-in/callback`, and the gateway replays it to goose inside the container.
    It only accepts a loopback address, and only while that provider's sign-in is running.

The stream ends with `done` (the provider, now set up) or `error`. Only one sign-in runs at a
time (the redirect flows share fixed ports); another answers `409`. Disconnecting cancels it.

goose reports some of these steps only in its log (the page to open when no browser exists,
and xAI's fallback code). While a sign-in runs, a small tracing layer forwards those lines
into the stream.

## Browser access

The gateway has no login of its own and spends your subscription, so it only answers the
browser origins you allow. Otherwise any website you visit could use it.

- `http(s)://localhost`, `127.0.0.1` and `[::1]` on any port are always allowed.
- Add others in `GOOSE_GATEWAY_ALLOWED_ORIGINS`, comma separated and matched exactly, for
  example `https://my-dashboard.example`. `*` allows every origin; don't use it on a machine
  where you browse the web.
- Behind a reverse proxy that serves your page and the gateway from one address, a request
  whose `Origin` is that same address is accepted without listing it, as long as the address
  is an IP, `localhost`, a `.local` name or a `.onion` address. A public DNS name can be
  re-pointed at your machine by an attacker's page (DNS rebinding), so a domain still has to
  be listed. A `.local` name is answered by your local network, so this trusts the LAN,
  which can reach the gateway directly anyway. The proxy must pass the browser's `Host`
  header through.
- Requests from other origins get `403`, enforced in the server as well as in CORS. That also
  covers DNS-rebinding pages, which a browser treats as same-origin.
- Requests with no `Origin` header (curl, server-side clients) are accepted. Anything that can
  make them on this machine could read the token file directly anyway.

Keep it bound to loopback (the default, and what `compose.yaml` publishes) unless you put
your own authentication in front of it: an allowed origin can also save keys and start
sign-ins.

## Tools

Set `GOOSE_GATEWAY_MCP_COMMAND` to a stdio MCP server's command line (program and arguments,
split on spaces) and the gateway serves that server's tools to a client-side agent. It starts
the server on first use, with the gateway's own environment, and starts it again if it exits.

- `GET /v1/tools` returns `{"tools":[{"name","description","parameters","read_only"?,"destructive"?}]}`:
  `parameters` is the tool's JSON Schema, ready to pass as `function.parameters` in a chat
  request, and `read_only`/`destructive` are MCP's hints when the server gives them.
- `POST /v1/tools/{name}` with the arguments (`{}` for none) returns
  `{"output":"…","is_error":bool}`: the tool's text, or its structured result as JSON. A tool
  that fails answers `200` with `is_error: true` and the reason, for the model to read. `502`
  means the server itself failed, `504` that the call took over two minutes; either way the
  call may still have happened, so it is never retried.

The model never runs a tool here. The client sends the tools with its chat request, gets tool
calls back, decides which to run and calls them itself, which is where it should ask its user
before anything that is not read-only. **The gateway does no approval of its own: whatever can
reach it can call every tool.** With tools that move money (such as
[ldk-server-mcp](https://github.com/lightningdevkit/ldk-server/tree/main/ldk-server-mcp)),
reach the gateway only through a login, as the LDK Node bundle does.

For ldk-server-mcp, the command can read the node's API key from its storage directory rather
than an environment variable: point `--config` at a client config naming the network, the
node's address and `[storage.disk] dir_path`, or mount the storage at `$HOME/.ldk-server` and
set `LDK_BASE_URL`.

## Configuration

| Variable                        | Default         | Meaning                                                       |
| ------------------------------- | --------------- | ------------------------------------------------------------- |
| `GOOSE_PATH_ROOT`               | goose's default | goose reads its config from `<root>/config`                    |
| `GOOSE_GATEWAY_HOST`            | `127.0.0.1`     | listen address (`0.0.0.0` inside the container)                |
| `GOOSE_GATEWAY_PORT`            | `8791`          |                                                               |
| `GOOSE_GATEWAY_ALLOWED_ORIGINS` | empty           | extra browser origins, see [Browser access](#browser-access)  |
| `GOOSE_GATEWAY_PROVIDER`        | `xai_oauth`     | provider for requests that don't name one                     |
| `GOOSE_GATEWAY_MCP_COMMAND`     | empty           | stdio MCP server whose tools to serve, see [Tools](#tools)     |
| `GOOSE_CONFIG_DIR`              | `~/.config/goose` | compose only: host directory mounted as goose's config      |
| `RUST_LOG`                      | `info`          |                                                               |

goose's default config directory is `~/.config/goose`, on macOS too. goose settings can also
be overridden by environment variables named after the setting in capitals, for example
`XAI_HOST`.

The gateway holds no secret of its own: sign-ins and keys live in goose's config directory,
and goose rewrites tokens there when it refreshes them, which is why the compose volume is
read-write. This build of goose has no system-keychain support, so keys go to `secrets.yaml`
in that directory; a goose CLI that keeps its keys in the keychain won't see them, and the
other way round.

## Building

`goose` isn't on crates.io (the crate by that name is a load-testing tool), so `Cargo.toml`
pins the goose repository by revision and repeats the two `[patch.crates-io]` entries goose's
workspace needs. The first build compiles goose's whole dependency tree and takes a while.

The release profile is deliberately plain and the Dockerfile builds with `-j 2`: thin LTO with
a single codegen unit pushed peak memory past what a default Docker Desktop VM allows.

### Container image

The image is one statically linked (musl) binary on an empty base with a CA bundle: about
73 MB, with no shell or package manager. It is built for size (`opt-level = "s"`), since the
gateway waits on the network rather than the CPU; pass `--build-arg OPT_LEVEL=3` to build for
speed instead (about 87 MB).

It runs as UID/GID `1000:1000`, so a bind-mounted goose config directory must be writable by
that user; on Linux that is usually already the owner of `~/.config/goose`.

Images are built for amd64 and arm64 on a native runner each (`.github/workflows/image.yml`)
and pushed to `ghcr.io/vincenzopalazzo/goose-gateway` when a `v*` tag is pushed.

## License

MIT

# Configuration

All configuration is read from environment variables (and a local `.env`) by
[`app/config.py`](../app/config.py), coerced into a typed `Settings` object, and validated
**eagerly at startup** — a misconfiguration fails the boot with a clear message naming the
offending key rather than erroring mid-request.

- Env var names are the field names **upper-cased** (case-insensitive), e.g. `stt_provider`
  → `STT_PROVIDER`.
- **Secrets** use `SecretStr`: they never appear in logs, `repr`, or `GET /api/v1/config`
  (which reports only whether each secret is *configured*).
- Swapping a provider is a **config change, not a code change** — the names resolve through
  `app/providers/factory.py`. `mock` is always available for offline runs.
- Copy [`.env.example`](../.env.example) to `.env` for local development. **Never commit real
  secrets.**

## Core (VA-19)

| Env var | Default | Notes |
| --- | --- | --- |
| `APP_NAME` | `voice-ai-agent` | Service name reported at `GET /` |
| `ENVIRONMENT` | `local` | `local` \| `dev` \| `prod`. `dev`/`prod` require the cloud secrets below |
| `PORT` | `8080` | Cloud Run injects `$PORT` at runtime |
| `LOG_LEVEL` | `INFO` | `CRITICAL` \| `ERROR` \| `WARNING` \| `INFO` \| `DEBUG` |
| `API_PREFIX` | `/api/v1` | Prefix for all application endpoints |

## Provider selection (VA-30)

| Env var | Default | Notes |
| --- | --- | --- |
| `STT_PROVIDER` | `deepgram` | Speech-to-text adapter; `mock` for offline |
| `LLM_PROVIDER` | `gemini` | Language model adapter; `mock` for offline |
| `TTS_PROVIDER` | `cartesia` | Text-to-speech adapter; `mock` for offline |
| `REALTIME_PROVIDER` | `openai` | Voice-to-voice adapter for the fast path; `mock` for offline |

Set all four to `mock` to run the whole service with no keys and no network (this is what the
tests and the evaluation harness use by default).

## Deepgram STT (VA-31)

| Env var | Default | Notes |
| --- | --- | --- |
| `DEEPGRAM_API_KEY` | *(empty)* | **Secret.** Required only when `STT_PROVIDER=deepgram` |
| `DEEPGRAM_MODEL` | `nova-3` | STT model |

## Gemini LLM (VA-34 / VA-36)

| Env var | Default | Notes |
| --- | --- | --- |
| `GOOGLE_API_KEY` | *(empty)* | **Secret.** Required only when `LLM_PROVIDER=gemini` |
| `GEMINI_MODEL` | `gemini-2.0-flash` | Flash-tier model |
| `GEMINI_SYSTEM_PROMPT` | *(built-in)* | Base system prompt; grounding instructions are layered on top |
| `GEMINI_ENABLE_PROMPT_CACHING` | `true` | Cache the full document as Gemini cached content so repeat turns aren't re-billed for the large context |

## Cartesia TTS (VA-43)

| Env var | Default | Notes |
| --- | --- | --- |
| `CARTESIA_API_KEY` | *(empty)* | **Secret.** Required only when `TTS_PROVIDER=cartesia` |
| `CARTESIA_MODEL` | `sonic-2` | TTS model |
| `CARTESIA_VOICE_ID` | *(empty)* | Voice id for synthesis |

## OpenAI Realtime (VA-46)

| Env var | Default | Notes |
| --- | --- | --- |
| `OPENAI_API_KEY` | *(empty)* | **Secret.** Required only when `REALTIME_PROVIDER=openai` |
| `OPENAI_REALTIME_MODEL` | `gpt-4o-realtime-preview` | Must be a **beta-protocol** model — the adapter speaks `realtime=v1`; the GA `gpt-realtime` uses a different event schema |
| `OPENAI_VOICE` | `alloy` | Realtime voice |

## Grounding & memory (VA-35 / VA-41)

| Env var | Default | Notes |
| --- | --- | --- |
| `SOURCE_DOC_PATH` | *(empty)* | Path to the source document (`.txt`/`.md`/`.pdf`). Empty ⇒ grounding off. When set, the file must exist and fit the window or the service fails to boot |
| `CONTEXT_WINDOW_TOKENS` | `1000000` | Max document size (token estimate) accepted at load |
| `CONVERSATION_MEMORY_TOKENS` | `2000` | Rolling conversation-memory budget (distinct from the document context) |

## Auth (VA-15)

| Env var | Default | Notes |
| --- | --- | --- |
| `JWT_SECRET_KEY` | *(empty)* | **Secret. Required in `dev`/`prod`** (see below); optional in `local` |

## CORS (VA-16)

| Env var | Default | Notes |
| --- | --- | --- |
| `ALLOWED_ORIGINS` | *(empty)* | Comma-separated origins allowed to call the API from a browser (e.g. `https://app.example.com,https://staging.example.com`). Empty ⇒ **no cross-origin access** — the deny-by-default posture. Origins must include their scheme; wildcards are rejected at startup. The `/ui` dashboard is served same-origin and needs no entry |

## Required outside `local`

When `ENVIRONMENT` is `dev` or `prod`, the keys in `REQUIRED_IN_CLOUD` must be set or the
service refuses to start. Today that is `JWT_SECRET_KEY`; later tickets extend the set (provider
keys in VA-14) and source these values from Secret Manager — the loader contract does not
change. Locally, missing secrets simply disable the features that need them.

## Inspecting the effective config

```bash
curl http://localhost:8080/api/v1/config
# {"app_name":"voice-ai-agent","environment":"local","port":8080,"log_level":"INFO",
#  "api_prefix":"/api/v1","jwt_secret_key_configured":false,"allowed_origins":[]}
```

Secret **values** are never returned — only a configured/not-configured boolean.

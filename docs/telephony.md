# Telephony — inbound phone calls (VA-72 / VA-74)

Let a caller **dial a phone number and talk to the assistant**: the call is bridged, over
Twilio Media Streams, to the traditional pipeline — caller speech → Deepgram (μ-law 8 kHz) →
grounded LLM → TTS → back down the line as μ-law. It answers questions grounded in the
configured source document.

Disabled by default. It is enabled by `TELEPHONY_ENABLED=true` and needs a public URL so
Twilio can reach the service.

## What you need

1. **A Twilio account + a phone number** (Voice-capable) → Account SID + Auth Token.
2. **A public HTTPS URL** for this service — the quickest is [ngrok](https://ngrok.com)
   (`ngrok http 8080`); for something durable, deploy to Cloud Run. Twilio must reach it.
3. **Working provider keys** — at minimum Deepgram (STT), the LLM (Gemini), and Cartesia
   (TTS). Grounded answers need a valid `GOOGLE_API_KEY`.

## Configuration

| Env var | Purpose |
| --- | --- |
| `TELEPHONY_ENABLED` | `true` to mount the `/telephony/*` routes |
| `PUBLIC_BASE_URL` | The public origin Twilio reaches, e.g. `https://ab12cd.ngrok.io` — used to build the media-stream `wss://` URL in the TwiML |
| `TWILIO_AUTH_TOKEN` | When set, the webhook validates Twilio's `X-Twilio-Signature` (recommended) |
| `TWILIO_ACCOUNT_SID` | Your Twilio account SID (for reference/records) |
| `SOURCE_DOC_PATH` | The document answers are grounded in |
| provider keys | `DEEPGRAM_API_KEY`, `GOOGLE_API_KEY`, `CARTESIA_API_KEY` |

## Go live

```bash
# 1) run the service with telephony on (real providers + the document)
docker run --rm -p 8080:8080 --env-file .env \
  -e TELEPHONY_ENABLED=true \
  -e PUBLIC_BASE_URL=https://<your-ngrok-subdomain>.ngrok.io \
  -v "$PWD/source.pdf":/data/source.pdf:ro -e SOURCE_DOC_PATH=/data/source.pdf \
  voice-ai-agent

# 2) expose it (separate terminal)
ngrok http 8080      # copy the https URL into PUBLIC_BASE_URL above, then restart the container

# 3) point your Twilio number at the webhook
#    Twilio Console → Phone Numbers → your number → Voice → "A call comes in":
#      Webhook  (HTTP POST)  https://<your-ngrok-subdomain>.ngrok.io/telephony/voice
```

Then **call the number**. You'll hear the greeting, ask a question about the document, and
get a spoken, grounded answer. Multiple turns work until you hang up.

## How it works

- `POST /telephony/voice` answers with TwiML: `<Connect><Stream url="wss://…/telephony/stream"/>`.
- Twilio opens the WebSocket and streams the caller's audio as base64 **μ-law 8 kHz** frames.
- The bridge feeds that straight to Deepgram (configured `encoding=mulaw`), which segments
  turns via its end-of-turn signal; each utterance is answered by the grounded LLM and spoken
  by the TTS provider; the PCM is resampled to 8 kHz, μ-law-encoded, and streamed back as
  `media` frames.
- The endpoints sit **outside** the API prefix, so the JWT/rate-limit middleware doesn't block
  Twilio; Twilio's request signature is the authentication instead.

## Notes & limits

- **Half-duplex**: while the agent speaks, caller audio buffers and is processed on the next
  turn. Full-duplex barge-in (interrupt the agent by talking over it) is a follow-up — the
  realtime *fast* path already has server-side barge-in.
- **STT**: telephony assumes Deepgram (native μ-law). Other STT providers fall back to their
  default config and may need transcoding.
- **Audio**: μ-law transcoding uses stdlib `audioop`, which is removed in Python 3.13 — install
  the `audioop-lts` backport when upgrading.
- **Outbound calls / Vapi** (VA-73) are not built; this is inbound Twilio only.

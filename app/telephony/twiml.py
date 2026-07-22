"""TwiML response + Twilio request-signature validation (VA-72).

When a call comes in, Twilio POSTs to our voice webhook; we answer with TwiML that tells
Twilio to open a **bidirectional Media Stream** to our WebSocket. From there the media flows
over that socket (see ``app.telephony.stream``).

Twilio signs every webhook request (``X-Twilio-Signature``) with the account auth token; we
validate it so only Twilio can drive the endpoint. Validation is skipped when no auth token
is configured (local testing).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
from xml.sax.saxutils import quoteattr


def build_stream_twiml(websocket_url: str) -> str:
    """TwiML that connects the call to a bidirectional media stream at ``websocket_url``.

    ``<Connect><Stream>`` keeps the call open and streams audio both ways (unlike ``<Start>``,
    which is inbound-only) — that is what lets the agent speak back.
    """
    url = quoteattr(websocket_url)  # XML-escape; wss URLs may carry query params
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response><Connect>"
        f"<Stream url={url} />"
        "</Connect></Response>"
    )


def expected_signature(url: str, params: dict[str, str], auth_token: str) -> str:
    """Compute Twilio's ``X-Twilio-Signature`` for a form POST.

    Twilio concatenates the full request URL with each POST param name+value in **alphabetical
    order by name**, HMAC-SHA1s it with the auth token, and base64-encodes the digest.
    """
    payload = url + "".join(f"{k}{params[k]}" for k in sorted(params))
    digest = hmac.new(auth_token.encode(), payload.encode("utf-8"), hashlib.sha1).digest()
    return base64.b64encode(digest).decode("ascii")


def is_valid_signature(url: str, params: dict[str, str], signature: str, auth_token: str) -> bool:
    """Constant-time check of an inbound ``X-Twilio-Signature`` against the expected value."""
    return hmac.compare_digest(expected_signature(url, params, auth_token), signature or "")

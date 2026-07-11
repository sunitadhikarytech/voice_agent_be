"""Streaming layer.

Owns the server-sent-event contract (transcript.partial/final, answer.delta,
audio.chunk, done) and the shared request/response schemas. Defined in VA-20 and served
by the SSE endpoint in VA-27. This package marks the module boundary.
"""

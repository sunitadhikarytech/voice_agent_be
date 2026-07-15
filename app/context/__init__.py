"""Full-document context (grounding, no RAG).

The entire source document is supplied to the model as context (with prompt caching in VA-36)
rather than retrieved from a vector store. VA-35 loads and size-validates it; VA-37 grounds
answers in it.
"""
from app.context.loader import DocumentContext, DocumentError, estimate_tokens, load_document

__all__ = ["DocumentContext", "DocumentError", "estimate_tokens", "load_document"]

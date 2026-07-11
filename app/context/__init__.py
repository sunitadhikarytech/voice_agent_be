"""Full-document context (grounding, no RAG).

The entire source document is supplied to the model as context (with prompt caching)
rather than retrieved from a vector store. The loader and grounding logic land in
VA-35 / VA-36 / VA-37. This package marks the module boundary.
"""

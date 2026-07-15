"""Traditional pipeline: STT -> LLM -> TTS (document-grounded, tool-capable slow path).

Orchestrated end to end in VA-45. Adapters: Deepgram STT (VA-31), Gemini via ADK (VA-34),
Cartesia TTS (VA-43).
"""
from app.pipelines.traditional.pipeline import TraditionalPipeline

__all__ = ["TraditionalPipeline"]

"""
Enhanced LLM implementations for AgentBoard.

This module provides LLM backends with enhanced features:
- API_LLM: Generic OpenAI-format API client with multi-port detection
- API_DiffusionLLM: DiffusionLLM-specific API client

Both support token counting and are designed for flexible deployment.
"""

from .api_llm import API_LLM
from .api_dllm import API_DiffusionLLM

__all__ = [
    "API_LLM",
    "API_DiffusionLLM",
]

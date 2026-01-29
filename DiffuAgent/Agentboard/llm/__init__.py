"""
LLM backends for AgentBoard.

This module provides various LLM backends for model inference:
- OPENAI_GPT: OpenAI GPT models via Azure or direct API
- OPENAI_GPT_AZURE: OpenAI GPT via Azure
- CLAUDE: Anthropic Claude models
- VLLM: Open-source models via vLLM
- HgModels: HuggingFace models
- API_LLM: Generic OpenAI-format API with multi-port detection
- API_DiffusionLLM: DiffusionLLM-specific API client
"""

# Original LLMs
from .openai_gpt import OPENAI_GPT
from .azure_gpt import OPENAI_GPT_AZURE
from .claude import CLAUDE
from .vllm import VLLM
from .huggingface import HgModels

# Enhanced LLMs
from .enhanced import API_LLM, API_DiffusionLLM

from common.registry import registry

__all__ = [
    # Original LLMs
    "OPENAI_GPT",
    "OPENAI_GPT_AZURE",
    "VLLM",
    "CLAUDE",
    "HgModels",
    # Enhanced LLMs
    "API_LLM",
    "API_DiffusionLLM",
]


def load_llm(name, config):
    llm = registry.get_llm_class(name).from_config(config)
    return llm

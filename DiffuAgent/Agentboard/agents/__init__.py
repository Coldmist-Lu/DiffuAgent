from .vanilla_agent import VanillaAgent
from .react_agent import ReactAgent

# =====[ ADDED: Enhanced agents with modular features ]=====
# Enhanced agents provide: dynamic memory, early exit, history replay
from .enhanced import (
    ReactOnePass,
    ReactMemory,
    ReactMemoryExit,
    ReactHistoryExit,
)
# =======================================================

from common.registry import registry

__all__ = [
    # Original agents
    "VanillaAgent",
    "ReactAgent",
    # =====[ ADDED: Enhanced agents (4 variants) ]=====
    "ReactOnePass",
    "ReactMemory",
    "ReactMemoryExit",
    "ReactHistoryExit",
    # ==============================================
]


def load_agent(name, config, llm_model):
    agent = registry.get_agent_class(name).from_config(llm_model, config)
    return agent

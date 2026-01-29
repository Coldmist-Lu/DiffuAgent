"""
Enhanced agent implementations with modular features.

This module provides pre-configured agent classes combining different features:
- ReactOnePass: Basic ReAct agent (no memory, no early exit)
- ReactMemory: ReAct with dynamic memory
- ReactMemoryExit: ReAct with memory and early exit
- ReactHistoryExit: History replay with early exit (for offline evaluation)

All agents inherit from ReactAgentBaseEnhanced and compose functionality
through mixins (MemoryMixin, VerificationMixin, HistoryMixin).
"""

from .react_onepass import ReactOnePass
from .react_memory import ReactMemory
from .react_memory_exit import ReactMemoryExit
from .react_history_exit import ReactHistoryExit

__all__ = [
    "ReactOnePass",
    "ReactMemory",
    "ReactMemoryExit",
    "ReactHistoryExit",
]

"""
Mixin modules for enhanced agent features.

Mixins provide modular functionality that can be composed into agent classes through inheritance:
- MemoryMixin: Add dynamic memory capabilities
- VerificationMixin: Add early exit verification
- HistoryMixin: Add history replay for offline evaluation
"""

from .memory import MemoryMixin
from .verification import VerificationMixin
from .history import HistoryMixin

__all__ = [
    "MemoryMixin",
    "VerificationMixin",
    "HistoryMixin",
]

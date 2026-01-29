"""
Enhanced tasks with detailed logging and tracking capabilities.

This module provides enhanced versions of embodied AI tasks with:
- Token counting and thought recording
- Complete trajectory tracking
- Dynamic memory logging
- Stop reason tracking (max_steps, early_exit, success)
- Difficulty-based metrics (hard/easy)
- TaskLogger integration for detailed visualization

Usage:
    from tasks.enhanced import EvalalfworldEnhanced, EvalScienceworldEnhanced, EvalBabyaiEnhanced

Or register via registry:
    from common.registry import registry
    AlfWorldTask = registry.get_task_class("alfworld_enhanced")
    ScienceWorldTask = registry.get_task_class("scienceworld_enhanced")
    BabyAITask = registry.get_task_class("babyai_enhanced")
"""

from .base_enhanced import BaseEnhancedTask
from .alfworld_enhanced import EvalalfworldEnhanced
from .scienceworld_enhanced import EvalScienceworldEnhanced
from .babyai_enhanced import EvalBabyaiEnhanced

__all__ = [
    'BaseEnhancedTask',
    'EvalalfworldEnhanced',
    'EvalScienceworldEnhanced',
    'EvalBabyaiEnhanced',
]

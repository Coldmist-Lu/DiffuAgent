"""
Enhanced Agents Module - Modular ReAct Agents with Advanced Features.

This module provides enhanced ReAct agent implementations with modular,
composable features for embodied AI tasks.

## Agent Variants

Four pre-configured agent classes are available:

1. **ReactOnePass** - Basic ReAct agent
   - No memory, no early exit
   - Enhanced prompt formatting and action extraction
   - Use for: Simple ReAct baseline

2. **ReactMemory** - ReAct with dynamic memory
   - Maintains summarized memory of past actions/observations
   - Automatic LLM-based memory updates
   - Use for: Tasks requiring long-term context

3. **ReactMemoryExit** - ReAct with memory and early exit
   - All ReactMemory features
   - Detects stuck/repetitive behavior and exits early
   - Use for: Production with resource optimization

4. **ReactHistoryExit** - History replay with early exit
   - Replays actions from logs (no LLM inference)
   - Tests early exit mechanism offline
   - Use for: Debugging verification logic, testing exit settings

## Usage

### In Configuration Files
```yaml
agent_config:
  name: "ReactMemory"  # or ReactOnePass, ReactMemoryExit, ReactHistoryExit
  init_prompt_path: "prompts/ReactFmtAgentMemory/alfworld_react.json"
  memory_examples: 3
  stored_memory_max: 4
  update_num: 2
  # For ReactMemoryExit and ReactHistoryExit only:
  verification_iter: 1
  verification_format: "strict"
  # For ReactHistoryExit only:
  history_file_path: "/path/to/logs"
```

### In Code
```python
# Direct import
from agents.enhanced import ReactMemory, ReactMemoryExit

# Via registry
from common.registry import registry
agent_class = registry.get_agent_class("ReactMemory")
agent = agent_class.from_config(llm_model, config)
```

## Advanced Customization

To create custom agent variants, you can import base classes and mixins directly:
```python
from agents.enhanced.react_agent_base import ReactAgentBaseEnhanced
from agents.enhanced.mixins import MemoryMixin, VerificationMixin
```
"""

# Import only the 4 agent variants that users will use
from .agents import (
    ReactOnePass,
    ReactMemory,
    ReactMemoryExit,
    ReactHistoryExit,
)

__all__ = [
    "ReactOnePass",
    "ReactMemory",
    "ReactMemoryExit",
    "ReactHistoryExit",
]


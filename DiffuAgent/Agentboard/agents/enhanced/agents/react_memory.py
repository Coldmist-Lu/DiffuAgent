"""
ReactMemory - ReAct Agent with Dynamic Memory.

Enhanced ReAct agent that maintains a dynamic memory summary of
past actions and observations, updated periodically using LLM.
"""
from common.registry import registry

from agents.enhanced.react_agent_base import ReactAgentBaseEnhanced
from agents.enhanced.mixins.memory import MemoryMixin
from agents.enhanced.utils.utils import format_history


@registry.register_agent("ReactMemory")
class ReactMemory(ReactAgentBaseEnhanced, MemoryMixin):
    """
    ReAct agent with dynamic memory capabilities.

    This agent extends ReactOnePass with:
    - Dynamic memory summarization
    - Periodic memory updates using LLM
    - Concise history representation in prompts

    Features:
    - Automatically stores actions and observations
    - Summarizes memory when threshold is reached
    - Uses summarized memory in prompts for better context

    Configuration:
        agent_config:
            init_prompt_path: "prompts/ReactFmtAgentMemory/alfworld_react.json"
            memory_examples: 3
            stored_memory_max: 4      # Trigger update after 4 items
            update_num: 2             # Use 2 items per update

    Example:
        agent = ReactMemory(
            llm_model,
            init_prompt_path="prompts/ReactFmtAgentMemory/...",
            stored_memory_max=4,
            update_num=2
        )
    """

    def __init__(self,
                 llm_model,
                 init_prompt_path=None,
                 memory_examples: int = 3,
                 stored_memory_max: int = 4,
                 update_num: int = 2):
        """
        Initialize ReactMemory agent.

        Args:
            llm_model: LLM model for inference
            init_prompt_path: Path to prompt template JSON
            memory_examples: Number of examples in prompt
            stored_memory_max: Items to store before memory update
            update_num: Items to use for each memory update
        """
        # Initialize base class
        ReactAgentBaseEnhanced.__init__(
            self,
            llm_model=llm_model,
            init_prompt_path=init_prompt_path,
            memory_examples=memory_examples
        )

        # Memory configuration (used by MemoryMixin)
        self.memory_config = {
            "stored_memory_max": stored_memory_max,
            "update_num": update_num,
        }

    def _get_history_str(self):
        """
        Get formatted history with dynamic memory.

        Overrides base class to use dynamic memory summary instead
        of raw history.

        Returns:
            str: Formatted history with memory summary
        """
        return format_history(self.memory, self.dynamic_memory)

    @classmethod
    def from_config(cls, llm_model, config):
        """
        Create agent from configuration.

        Args:
            llm_model: LLM model instance
            config: Configuration dict

        Returns:
            ReactMemory instance
        """
        init_prompt_path = config.get("init_prompt_path", None)
        memory_examples = config.get("memory_examples", 3)
        stored_memory_max = config.get("stored_memory_max", 4)
        update_num = config.get("update_num", 2)

        return cls(
            llm_model=llm_model,
            init_prompt_path=init_prompt_path,
            memory_examples=memory_examples,
            stored_memory_max=stored_memory_max,
            update_num=update_num
        )

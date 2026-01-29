"""
ReactOnePass - Basic Enhanced ReAct Agent.

A simple ReAct agent without dynamic memory or early exit mechanisms.
Provides enhanced prompt formatting and action extraction compared to
the original react_agent.
"""
from common.registry import registry

from agents.enhanced.react_agent_base import ReactAgentBaseEnhanced


@registry.register_agent("ReactOnePass")
class ReactOnePass(ReactAgentBaseEnhanced):
    """
    Basic enhanced ReAct agent without memory or early exit.

    This agent provides:
    - Standard ReAct prompt formatting
    - Improved action extraction
    - Task-specific command formatting

    Compared to the original react_agent:
    - Better structured code
    - More robust action parsing
    - Cleaner separation of concerns

    Configuration:
        agent_config:
            init_prompt_path: "prompts/ReactFmtAgent/alfworld_react.json"
            memory_examples: 3

    Example:
        agent = ReactOnePass(llm_model, init_prompt_path="prompts/...")
    """

    def __init__(self,
                 llm_model,
                 init_prompt_path=None,
                 memory_examples: int = 3):
        """
        Initialize ReactOnePass agent.

        Args:
            llm_model: LLM model for inference
            init_prompt_path: Path to prompt template JSON file
            memory_examples: Number of examples to include in prompt
        """
        super().__init__(
            llm_model=llm_model,
            init_prompt_path=init_prompt_path,
            memory_examples=memory_examples
        )

    @classmethod
    def from_config(cls, llm_model, config):
        """
        Create agent from configuration.

        Args:
            llm_model: LLM model instance
            config: Configuration dict

        Returns:
            ReactOnePass instance
        """
        init_prompt_path = config.get("init_prompt_path", None)
        memory_examples = config.get("memory_examples", 3)

        return cls(
            llm_model=llm_model,
            init_prompt_path=init_prompt_path,
            memory_examples=memory_examples
        )

"""
ReactHistoryExit - History Replay with Early Exit.

Agent for offline evaluation of early exit mechanisms using historical
trajectories. Replays actions from logs without LLM inference, allowing
quick evaluation of verification effectiveness.
"""
from common.registry import registry

from agents.enhanced.react_agent_base import ReactAgentBaseEnhanced
from agents.enhanced.mixins.history import HistoryMixin
from agents.enhanced.mixins.verification import VerificationMixin
from agents.enhanced.mixins.memory import MemoryMixin
from agents.enhanced.utils.utils import format_history


@registry.register_agent("ReactHistoryExit")
class ReactHistoryExit(ReactAgentBaseEnhanced, HistoryMixin, MemoryMixin, VerificationMixin):
    """
    Agent for offline early exit evaluation using historical trajectories.

    This agent replays actions from previously saved logs to evaluate
    the effectiveness of early exit mechanisms without re-running
    the full agent workflow with LLM inference.

    Use cases:
    - Test different verification settings (strict vs modest)
    - Evaluate early exit impact on performance
    - Debug verification logic
    - Compare exit points across different tasks

    Features:
    - Loads historical trajectory from logs
    - Replays actions without LLM calls
    - Runs verification at specified intervals
    - Exits when verification triggers

    Configuration:
        agent_config:
            init_prompt_path: "prompts/ReactFmtAgentMemory/alfworld_react.json"
            memory_examples: 3
            stored_memory_max: 4
            update_num: 2
            verification_iter: 1       # Check every N steps
            verification_format: "strict"  # or "modest"
            history_file_path: "/path/to/logs"  # Directory containing logs/

    Example:
        agent = ReactHistoryExit(
            llm_model,
            init_prompt_path="prompts/ReactFmtAgentMemory/...",
            history_file_path="/path/to/runs/exp1",
            stored_memory_max=4,
            update_num=2,
            verification_iter=1,
            verification_format="strict"
        )
    """

    def __init__(self,
                 llm_model,
                 init_prompt_path=None,
                 memory_examples: int = 3,
                 stored_memory_max: int = 4,
                 update_num: int = 2,
                 verification_iter: int = 1,
                 verification_format: str = "strict",
                 history_file_path: str = ""):
        """
        Initialize ReactHistoryExit agent.

        Args:
            llm_model: LLM model (still needed for memory/verification)
            init_prompt_path: Path to prompt template JSON
            memory_examples: Number of examples in prompt
            stored_memory_max: Items to store before memory update
            update_num: Items to use for each memory update
            verification_iter: Check every N steps
            verification_format: "strict" or "modest"
            history_file_path: Path to directory containing logs/
        """
        # Set verification configuration BEFORE initializing base class
        # (so reset_extended() can access it)
        self.verification_iter = verification_iter
        self.verification_format = verification_format

        # Memory configuration
        self.memory_config = {
            "stored_memory_max": stored_memory_max,
            "update_num": update_num,
        }

        # Initialize base class (this will call reset() -> reset_extended())
        ReactAgentBaseEnhanced.__init__(
            self,
            llm_model=llm_model,
            init_prompt_path=init_prompt_path,
            memory_examples=memory_examples
        )

        # History configuration
        self.history_file_path = history_file_path

    def _get_history_str(self):
        """
        Get formatted history with dynamic memory.

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
            ReactHistoryExit instance
        """
        init_prompt_path = config.get("init_prompt_path", None)
        memory_examples = config.get("memory_examples", 3)
        stored_memory_max = config.get("stored_memory_max", 4)
        update_num = config.get("update_num", 2)
        verification_iter = config.get("verification_iter", 1)
        verification_format = config.get("verification_format", "strict")
        history_file_path = config.get("history_file_path", "")

        return cls(
            llm_model=llm_model,
            init_prompt_path=init_prompt_path,
            memory_examples=memory_examples,
            stored_memory_max=stored_memory_max,
            update_num=update_num,
            verification_iter=verification_iter,
            verification_format=verification_format,
            history_file_path=history_file_path
        )

"""
ReactMemoryExit - ReAct Agent with Memory and Early Exit.

Enhanced ReAct agent with dynamic memory and early exit mechanism.
Detects when agent is stuck or task is complete and exits early.
"""
from common.registry import registry

from agents.enhanced.react_agent_base import ReactAgentBaseEnhanced
from agents.enhanced.mixins.memory import MemoryMixin
from agents.enhanced.mixins.verification import VerificationMixin
from agents.enhanced.utils.utils import format_history


@registry.register_agent("ReactMemoryExit")
class ReactMemoryExit(ReactAgentBaseEnhanced, MemoryMixin, VerificationMixin):
    """
    ReAct agent with dynamic memory and early exit verification.

    Combines features of ReactMemory and verification mechanism to:
    - Maintain dynamic memory summaries
    - Detect stuck/repetitive behavior
    - Exit early when task is complete or infeasible

    Features:
    - All ReactMemory features
    - Periodic verification checks (every N steps)
    - Configurable verification strictness (modest/strict)
    - Sets exit_flag when verification triggers
    - Support for auxiliary LLM (for memory and verification)

    Configuration:
        agent_config:
            init_prompt_path: "prompts/ReactFmtAgentMemory/alfworld_react.json"
            memory_examples: 3
            stored_memory_max: 4
            update_num: 2
            verification_iter: 1       # Check every N steps
            verification_format: "strict"  # or "modest"
            auxiliary_llm: "llada"     # Optional: auxiliary LLM for memory/verification

    Example:
        # Single LLM for both actions and memory/verification
        agent = ReactMemoryExit(
            llm_model,
            init_prompt_path="prompts/ReactFmtAgentMemory/...",
            stored_memory_max=4,
            update_num=2,
            verification_iter=1,
            verification_format="strict"
        )

        # Main LLM for actions, auxiliary LLM for memory/verification
        agent = ReactMemoryExit(
            llm_model=qwen3_model,
            auxiliary_llm_model=llada_model,
            init_prompt_path="prompts/ReactFmtAgentMemory/...",
            verification_iter=1,
            verification_format="modest"
        )
    """

    def __init__(self,
                 llm_model,
                 auxiliary_llm_model=None,
                 init_prompt_path=None,
                 memory_examples: int = 3,
                 stored_memory_max: int = 4,
                 update_num: int = 2,
                 verification_iter: int = 1,
                 verification_format: str = "strict"):
        """
        Initialize ReactMemoryExit agent.

        Args:
            llm_model: Main LLM model for action generation
            auxiliary_llm_model: Optional auxiliary LLM for memory/verification
            init_prompt_path: Path to prompt template JSON
            memory_examples: Number of examples in prompt
            stored_memory_max: Items to store before memory update
            update_num: Items to use for each memory update
            verification_iter: Check every N steps (0 to disable)
            verification_format: "strict" or "modest"
        """
        # Store auxiliary LLM (used by MemoryMixin and VerificationMixin)
        self.auxiliary_llm_model = auxiliary_llm_model

        # Set verification configuration BEFORE initializing base class
        # (so reset_extended() can access it)
        self.verification_iter = verification_iter
        self.verification_format = verification_format

        # Memory configuration (used by MemoryMixin)
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
            llm_model: Main LLM model instance
            config: Configuration dict (may contain auxiliary_llm and llm_config_all)

        Returns:
            ReactMemoryExit instance
        """
        init_prompt_path = config.get("init_prompt_path", None)
        memory_examples = config.get("memory_examples", 3)
        stored_memory_max = config.get("stored_memory_max", 4)
        update_num = config.get("update_num", 2)
        verification_iter = config.get("verification_iter", 1)
        verification_format = config.get("verification_format", "strict")

        # Handle auxiliary LLM
        auxiliary_llm_model = None
        auxiliary_llm_name = config.get("auxiliary_llm")

        if auxiliary_llm_name:
            from llm import load_llm

            # Get llm_config_all from config (passed by eval_modular.py)
            llm_config_all = config.get("llm_config_all", {})

            # Get the specific auxiliary LLM config
            if "llm" in llm_config_all and auxiliary_llm_name in llm_config_all["llm"]:
                aux_llm_config = llm_config_all["llm"][auxiliary_llm_name]
                auxiliary_llm_model = load_llm(aux_llm_config["name"], aux_llm_config)
                print(f"[ReactMemoryExit] Loaded auxiliary LLM: {auxiliary_llm_name}")
            else:
                print(f"[ReactMemoryExit] WARNING: auxiliary_llm '{auxiliary_llm_name}' not found in config")

        return cls(
            llm_model=llm_model,
            auxiliary_llm_model=auxiliary_llm_model,
            init_prompt_path=init_prompt_path,
            memory_examples=memory_examples,
            stored_memory_max=stored_memory_max,
            update_num=update_num,
            verification_iter=verification_iter,
            verification_format=verification_format
        )

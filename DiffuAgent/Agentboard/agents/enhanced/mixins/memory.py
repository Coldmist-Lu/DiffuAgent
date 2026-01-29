"""
Memory Mixin for Dynamic Memory Management.

This mixin adds dynamic memory capabilities to agents by:
1. Initializing DynamicMemory in reset_extended()
2. Storing observations in update_extended()
3. Storing actions in run_extended()

Usage:
    class MyAgent(ReactAgentBase, MemoryMixin):
        pass  # Automatically gets memory functionality
"""
from ..utils.dynamic_memory import DynamicMemory
from ..utils.logging import get_logger

# Module logger
logger = get_logger(__name__)


class MemoryMixin:
    """
    Mixin to add dynamic memory functionality to agents.

    Requires the agent class to have:
    - llm_model: LLM model instance
    - memory_config: Dict with 'stored_memory_max' and 'update_num' keys

    Provides:
    - self.dynamic_memory: DynamicMemory instance
    - Automatic memory updates in reset_extended(), update_extended(), run_extended()
    """

    def reset_extended(self):
        """
        Initialize dynamic memory system.

        This method should be called via super() in agent's reset_extended().
        """
        # Get auxiliary LLM if available (for memory summarization)
        aux_llm = getattr(self, 'auxiliary_llm_model', None)

        # Initialize DynamicMemory if memory_config is available
        if hasattr(self, 'memory_config'):
            self.dynamic_memory = DynamicMemory(
                llm_model_main=self.llm_model,
                llm_model_aux=aux_llm,  # Use auxiliary LLM for memory summarization
                **self.memory_config
            )
            aux_info = f" (with auxiliary: {aux_llm.__class__.__name__})" if aux_llm else ""
            logger.info(f"DynamicMemory initialized{aux_info}: max={self.memory_config.get('stored_memory_max')}, "
                       f"update_num={self.memory_config.get('update_num')}")
        else:
            # Use default config
            self.dynamic_memory = DynamicMemory(
                llm_model_main=self.llm_model,
                llm_model_aux=aux_llm
            )
            logger.debug("DynamicMemory initialized with defaults")

        # Initialize last_action tracker
        self.last_action = ""

        # Call next mixin's reset_extended() (e.g., VerificationMixin)
        if hasattr(super(), 'reset_extended'):
            super().reset_extended()

    def update_extended(self, obs: str):
        """
        Store observation in dynamic memory.

        This method should be called via super() in agent's update_extended().

        Args:
            obs: Observation string to store
        """
        if hasattr(self, 'dynamic_memory'):
            self.dynamic_memory.store(f"Observation: {obs}")

        # Call next mixin's update_extended() (e.g., VerificationMixin)
        if hasattr(super(), 'update_extended'):
            super().update_extended(obs)

    def run_extended(self, action):
        """
        Store action in dynamic memory.

        This method should be called via super() in agent's run_extended().

        Args:
            action: Action string to store (formatted as "Thought: ...\nAction: ...")
        """
        if hasattr(self, 'dynamic_memory'):
            self.dynamic_memory.store(f"Action: {action}")

        # Update last_action tracker
        self.last_action = action

"""
Verification Mixin for Early Exit Mechanism.

This mixin adds early exit verification to agents by:
1. Initializing Verification in reset_extended()
2. Running verification checks in update_extended()
3. Setting exit_flag when verification triggers

Usage:
    class MyAgent(ReactAgentBase, VerificationMixin):
        pass  # Automatically gets verification functionality
"""
from ..utils.verification import Verification
from ..utils.logging import get_logger

# Module logger
logger = get_logger(__name__)


class VerificationMixin:
    """
    Mixin to add early exit verification to agents.

    Requires the agent class to have:
    - llm_model: LLM model instance
    - verification_iter: Check frequency (check every N steps)
    - prompt_dict: Dict with 'system_msg' and 'instruction' keys
    - env_info: Dict with 'goal' key
    - memory: Agent's memory list
    - steps: Current step count

    Provides:
    - self.verification_module: Verification instance
    - self.exit_flag: Boolean flag set to True when verification triggers
    - Automatic verification checks in reset_extended() and update_extended()
    """

    def reset_extended(self):
        """
        Initialize verification module.

        This method should be called via super() in agent's reset_extended().
        """
        # Reset exit flag for new episode
        self.exit_flag = False

        # Initialize Verification if verification_iter is set
        if hasattr(self, 'verification_iter') and self.verification_iter > 0:
            # Get auxiliary LLM if available (for verification)
            aux_llm = getattr(self, 'auxiliary_llm_model', None)

            self.verification_module = Verification(
                llm_model_main=self.llm_model,
                llm_model_aux=aux_llm,  # Use auxiliary LLM for verification
                verify_format=getattr(self, 'verification_format', 'strict')
            )
            self.verification_module.init_verify()

            aux_info = f" (with auxiliary: {aux_llm.__class__.__name__})" if aux_llm else ""
            logger.info(f"Verification initialized{aux_info}: iter={self.verification_iter}, "
                       f"format={getattr(self, 'verification_format', 'strict')}")

    def update_extended(self, obs: str):
        """
        Run verification check at specified intervals.

        This method should be called via super() in agent's update_extended().

        Args:
            obs: Observation string (unused by verification but passed for consistency)
        """
        # Check if verification should run this step
        should_verify = (
            hasattr(self, 'verification_iter') and
            self.verification_iter > 0 and
            hasattr(self, 'verification_module') and
            self.steps % self.verification_iter == 0
        )

        if should_verify:
            # Run verification check
            self.verification_module.verify(
                sys_mess=self.prompt_dict.get("system_msg", "You are a helpful assistant."),
                instruction=self.prompt_dict.get("instruction", ""),
                goal=self.env_info["goal"],
                memory=self.memory
            )

            # Update exit_flag
            self.exit_flag = self.verification_module.exit_flag

            if self.exit_flag:
                logger.warning(f"Early exit triggered at step {self.steps}")
            else:
                logger.info(f"Verification passed at step {self.steps}")

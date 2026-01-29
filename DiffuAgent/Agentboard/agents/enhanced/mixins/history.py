"""
History Mixin for History Replay.

This mixin adds history replay functionality to agents by:
1. Loading historical trajectory from logs in reset_extended()
2. Using historical actions instead of LLM generation in agent_call()

This is useful for offline evaluation of early exit mechanisms without
re-running the full agent workflow.

Usage:
    class MyAgent(ReactAgentBase, HistoryMixin):
        pass  # Automatically gets history replay functionality
"""
import os
from ..utils.utils import load_multiple_json_objects
from ..utils.logging import get_logger

# Module logger
logger = get_logger(__name__)


class HistoryMixin:
    """
    Mixin to add history replay functionality to agents.

    Requires the agent class to have:
    - history_file_path: Path to directory containing logs/
    - task_id: Formatted as "{task_name}_{env_id}" (e.g., "alfworld_0")
    - steps: Current step count

    Provides:
    - self.trajectory: Dict containing historical trajectory
    - Automatic action retrieval from history in agent_call()

    Note:
    - This mixin overrides agent_call() to return historical actions
    - Should be combined with VerificationMixin for early exit experiments
    """

    def reset_extended(self):
        """
        Load historical trajectory from logs.

        This method should be called via super() in agent's reset_extended().
        """
        if not hasattr(self, 'history_file_path') or not self.history_file_path:
            logger.debug("History replay disabled (no history_file_path)")
            return

        try:
            # Parse task_id
            task, task_index = self.task_id.split("_")
            log_file = os.path.join(self.history_file_path, "logs", f"{task}.jsonl")

            # Load history log
            self.history_log = load_multiple_json_objects(log_file)
            self.trajectory = self.history_log[int(task_index)]["trajectory"]

            logger.info(f"History loaded: {len(self.trajectory)} turns from {log_file}")

        except Exception as e:
            logger.warning(f"Error loading history: {e}")
            logger.debug("History replay disabled, falling back to LLM generation")
            self.history_log = None
            self.trajectory = None

    def agent_call(self, input_message):
        """
        Get action from history instead of LLM generation.

        This method should be called in place of the normal agent_call().

        Args:
            input_message: Input message (ignored, for compatibility)

        Returns:
            tuple: (success, response, thought, action, token_cnt)
        """
        if not hasattr(self, 'trajectory') or self.trajectory is None:
            # Fall back to LLM generation if no history
            logger.debug("No history available, using LLM generation")
            return super().agent_call(input_message)

        try:
            # Get action from history
            step_info = self.trajectory.get(f"Interaction Turn {self.steps}")

            if step_info is None:
                logger.debug(f"No history found for step {self.steps}")
                success, thought, action, token_cnt = False, "", "[No Action Found]", 0
            else:
                thought = step_info.get("Thought", "")
                action = step_info.get("Action", "[No Action Found]")
                token_cnt = step_info.get("Token", 0)
                success = True

                logger.debug(f"History replay step {self.steps}: Thought={thought[:50]}..., Action={action}")

            response = f"Thought: {thought}\nAction: {action}"

            return success, response, thought, action, token_cnt

        except Exception as e:
            logger.error(f"Error retrieving from history: {e}")
            return False, "", "[No Action Found]", 0

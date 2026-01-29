"""
Base class and shared utilities for enhanced tasks.

This module provides common functionality for enhanced task implementations including:
- TaskLogger integration
- Trajectory management
- Token/Thought tracking
- Memory logging
- Difficulty-based metrics
"""
import copy
from ..base_task import BaseTask
from utils.logging.logger import TaskLogger
from utils.logging.agent_logger import AgentLogger

logger = AgentLogger(__name__)


class BaseEnhancedTask(BaseTask):
    """
    Base class for enhanced tasks with logging and tracking capabilities.

    Features:
    - TaskLogger integration for detailed logging
    - Token counting and thought recording
    - Complete trajectory tracking
    - Dynamic memory logging
    - Difficulty-based metrics (hard/easy)
    - Exit reason tracking
    """

    def setup_logger(self, task_name, log_path, max_num_steps, baseline_dir):
        """
        Initialize TaskLogger for the task.

        Args:
            task_name: Name of the task
            log_path: Path to store logs
            max_num_steps: Maximum number of steps per episode
            baseline_dir: Directory containing baseline results

        Returns:
            TaskLogger instance
        """
        self.agentboard = TaskLogger(
            task_name=task_name,
            log_path=log_path,
            max_num_steps=max_num_steps,
            baseline_dir=baseline_dir
        )
        return self.agentboard

    def init_trajectory(self, goal, init_ob):
        """
        Initialize trajectory list with goal and initial observation.

        Args:
            goal: Task goal/description
            init_ob: Initial observation from environment

        Returns:
            List: Initialized trajectory
        """
        trajectory = []
        trajectory.append({"Goal": goal, "id": 0})
        trajectory.append({"Observation": init_ob, "id": 0})
        return trajectory

    def action_dict_process(self, action, step_id, trajectory, logger, extra_details):
        """
        Process action dict that contains token and thought information.

        This method handles the case where agent.run() returns a dict with:
        - 'token': number of tokens used
        - 'action': the actual action string
        - 'thought': reasoning/thought process
        - 'response': raw response details

        Args:
            action: Action dict with keys 'token', 'action', 'thought', 'response'
            step_id: Current step ID
            trajectory: Trajectory list to append to
            logger: Logger instance
            extra_details: Dict to store extra information (like exit_details)

        Returns:
            tuple: (token_count, action_string)
        """
        trajectory.append({"Thought": action['thought'], "id": step_id})
        trajectory.append({"Token": action['token'], "id": step_id})
        logger.info("Step {:02} - Thought: {}".format(step_id, action['thought']))
        extra_details['exit_details'] = action['response']

        # Restore action
        return action['token'], action['action']

    def log_memory_if_available(self, step_id, logger):
        """
        Log agent's dynamic memory if available.

        Args:
            step_id: Current step ID
            logger: Logger instance
        """
        try:
            logger.info(f"Mem - {self.agent.dynamic_memory.display()}")
        except:
            pass

    def calculate_difficulty_metrics(self, srs, scores, grounding_accs, difficulties):
        """
        Calculate metrics separated by difficulty level (hard/easy).

        Args:
            srs: List of success rates (0 or 1)
            scores: List of progress rates
            grounding_accs: List of grounding accuracies
            difficulties: List of difficulty strings ('hard' or 'easy')

        Returns:
            dict: Dictionary with keys:
                - sr: Overall success rate
                - pr: Overall progress rate
                - gr: Overall grounding rate
                - hard_sr: Hard task success rate
                - hard_pr: Hard task progress rate
                - easy_sr: Easy task success rate
                - easy_pr: Easy task progress rate
        """
        sr = sum(srs) * 1.0 / len(srs)
        pr = sum(scores) * 1.0 / len(scores)
        gr = sum(grounding_accs) * 1.0 / len(grounding_accs)

        # Calculate hard task metrics
        hard_sr = [sr for sr, difficulty in zip(srs, difficulties) if difficulty == "hard"]
        hard_sr = sum(hard_sr) / len(hard_sr) if len(hard_sr) > 0 else 0

        hard_pr = [pr for pr, difficulty in zip(scores, difficulties) if difficulty == "hard"]
        hard_pr = sum(hard_pr) / len(hard_pr) if len(hard_pr) > 0 else 0

        # Calculate easy task metrics
        easy_sr = [sr for sr, difficulty in zip(srs, difficulties) if difficulty == "easy"]
        easy_sr = sum(easy_sr) / len(easy_sr) if len(easy_sr) > 0 else 0

        easy_pr = [pr for pr, difficulty in zip(scores, difficulties) if difficulty == "easy"]
        easy_pr = sum(easy_pr) / len(easy_pr) if len(easy_pr) > 0 else 0

        return {
            "sr": sr,
            "pr": pr,
            "gr": gr,
            "hard_sr": hard_sr,
            "hard_pr": hard_pr,
            "easy_sr": easy_sr,
            "easy_pr": easy_pr
        }

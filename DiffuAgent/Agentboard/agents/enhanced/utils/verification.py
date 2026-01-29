"""
Verification Module for Early Exit Mechanism.

This module provides early exit functionality for agents by detecting
when the agent is stuck or the task is complete.
"""
from .logging import get_logger

# Module logger
logger = get_logger(__name__)

VERIFY_PROMPT = {
    "modest": """
Evaluate the agent's recent history and consider:
	1.	Whether the agent appears stuck or making little meaningful progress despite repeated attempts.
	2.	Whether the task seems complete or no longer feasible to pursue.

If you have good reason to believe further steps are unlikely to help, you may output "YES" to suggest stopping. Otherwise, output "NO" and continue exploring.
""",

    "strict": """
Evaluate the current history of the agent's actions and determine if it meets any of the following conditions:
	1.	The recent steps show repetitive actions or the agent appears to be stuck in a loop.
 	2.	The agent repeatedly checks for valid actions but fails to make meaningful progress toward the objective.
 	3.	The agent's recent actions suggest the task is complete and no further steps are necessary.
 	4.	The task is no longer achievable due to high difficulty or significant deviation from the expected course.

If any of the above conditions are met, output "YES". Otherwise, output "NO" to indicate the agent should continue exploring.
"""
}


class Verification:
    """
    Early exit verification module.

    Monitors agent behavior and determines when to exit early by detecting:
    - Stuck/repetitive behavior
    - Task completion
    - Infeasible tasks

    Args:
        llm_model_main: Main LLM model (typically for action generation)
        llm_model_aux: Auxiliary LLM model for verification (optional)
                       If None, uses llm_model_main for verification
        verify_format: Verification strictness - "modest" or "strict" (default: "strict")

    Example:
        # Single LLM for both actions and verification
        verifier = Verification(llm_model_main, verify_format="modest")

        # Main LLM for actions, auxiliary LLM for verification
        verifier = Verification(llm_model_main, llm_model_aux=dllm_model, verify_format="modest")

        verifier.init_verify()
        # ... after some steps ...
        verifier.verify(sys_msg, instruction, goal, memory)
        if verifier.exit_flag:
            print("Agent should exit early")
    """

    def __init__(self, llm_model_main, llm_model_aux=None, verify_format="strict"):
        """
        Initialize verification module.

        Args:
            llm_model_main: Main LLM model (for actions)
            llm_model_aux: Auxiliary LLM model (for verification, optional)
            verify_format: "modest" (lenient) or "strict" (stringent)
        """
        self.llm_model_main = llm_model_main
        # Use auxiliary LLM for verification if provided, otherwise use main LLM
        self.llm_model = llm_model_aux if llm_model_aux else llm_model_main
        self.token_cnt = 0  # Token count per scenario, not total
        self.verify_format = verify_format
        assert self.verify_format in ['modest', 'strict'], \
            f"verify_format must be 'modest' or 'strict', got {self.verify_format}"

    def _convert_memory2str(self, memory):
        """
        Convert memory list to string.

        Args:
            memory: List of (type, content) tuples

        Returns:
            str: Formatted memory string
        """
        return "\n".join(
            f"{mem[0]}: {mem[1]}" if mem[0] != "Action" else mem[1]
            for mem in memory
        )

    def _prompt_verify(self, instruction: str, goal: str, history_str: str):
        """
        Build verification prompt.

        Args:
            instruction: Task instruction
            goal: Agent's goal
            history_str: Agent's action history as string

        Returns:
            str: Complete verification prompt
        """
        return f"""
You will be given a historical scenario in which you are placed in a specific environment with a designated objective to accomplish.

### Task Description:
{instruction}

### Your Objective:
{goal}

### Your Current History:
{self._convert_memory2str(history_str)}

### Instructions:
{VERIFY_PROMPT[self.verify_format]}

Include a short explanation in your response.
"""

    def init_verify(self):
        """Initialize verification for a new scenario. Resets exit_flag."""
        self.exit_flag = False

    def verify(self, sys_mess, instruction, goal, memory):
        """
        Verify if agent should exit early.

        Sets self.exit_flag to True if verification determines agent should stop.

        Args:
            sys_mess: System message for LLM
            instruction: Task instruction
            goal: Agent's current goal
            memory: Agent's memory list
        """
        prompt = self._prompt_verify(instruction, goal, memory)
        message = [
            {"role": "system", "content": sys_mess},
            {"role": "user", "content": prompt}
        ]

        _, response = self.llm_model.generate(message)

        # Log which LLM was used for verification
        llm_type = "auxiliary" if self.llm_model != self.llm_model_main else "main"
        logger.debug(f"Verification (using {llm_type} LLM) Response: {response}")

        if isinstance(response, tuple):
            self.token_cnt += response[-1]
            response = response[0]

        if "YES" in response:
            self.exit_flag = True
            logger.info(f"Early exit triggered (using {llm_type} LLM): {response[:100]}...")
        else:
            self.exit_flag = False

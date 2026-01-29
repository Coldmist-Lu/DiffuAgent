"""
Base class for enhanced ReAct agents.

This module provides ReactAgentBase and ReactAgentBaseEnhanced, which extends ReactAgentBase
with common prompt formatting and action extraction logic used across
all enhanced agent variants.

Enhanced agents should inherit from this class and optionally add
functionality via mixins (MemoryMixin, VerificationMixin, HistoryMixin).
"""
import json
from typing_extensions import override

from agents.base_agent import BaseAgent
from common.registry import registry
from agents.enhanced.utils.utils import (
    extract_think_action,
    format_example,
    format_history,
    format_commands,
    format_system_msg
)
from agents.enhanced.utils.logging import get_logger

# Module logger
logger = get_logger(__name__)


# Standard ReAct prompt template
QUERY = """
{instruction}
The past actions and observations have been summarized in the memory, which provides you with the essential context of what has happened so far.

{example}

Now, it's your turn. You should perform thoughts and actions to accomplish the goal, guided by the memory that summarizes past actions and observations to provide essential context. Your response should use the following format:

Thought: <your thoughts>
Action: <your next action>

Your task is: {goal}
{init_obs}

{history_str}

{commands_str}
"""


@registry.register_agent("ReactAgentBase")
class ReactAgentBase(BaseAgent):
    """Base class for ReAct agents with core functionality."""
    def __init__(self, llm_model):
        super().__init__()
        self.llm_model = llm_model

    def reset(self, goal, init_obs, init_act=None, env=None):
        """Reset agent with new task."""
        self.env_info = {
            "task_name": self.task_id.split("_")[0] if self.task_id else "",
            "env_id": self.task_id.split("_")[-1] if self.task_id else "",
            "goal": goal,
            "init_obs": init_obs,
        }

        self.memory = [('Observation', self.env_info["init_obs"])]
        self.steps = 0
        self.done = False
        self.env = env
        self.goal = self.env_info["goal"]

        # Call extended reset for mixins (if exists)
        if hasattr(self, 'reset_extended'):
            self.reset_extended()

    def update(self, action='', state=''):
        """Update agent with action and observation."""
        obs = state
        self.steps += 1
        self.memory.append(("Observation", obs))

        # Call extended update for mixins (if exists)
        if hasattr(self, 'update_extended'):
            self.update_extended(obs)

    def run(self, init_prompt_dict=None):
        """Run one step of the agent."""
        input_message = self.make_prompt(init_prompt_dict)
        success, response, thought, action, token_cnt = self.agent_call(input_message)

        action_dict = {
            "response": response,
            "thought": thought,
            "action": action,
            "token": token_cnt
        }

        self.memory.append(
            ("Action", f"Thought: {thought}\nAction: {action}")
        )

        # Call extended run for mixins (if exists)
        if hasattr(self, 'run_extended'):
            self.run_extended(action)

        return success, action_dict

    @classmethod
    def from_config(cls, llm_model, config):
        """Create agent from config."""
        return cls(llm_model)


class ReactAgentBaseEnhanced(ReactAgentBase):
    """
    Enhanced base class for ReAct agents with common formatting logic.

    This class provides:
    - Standard prompt construction with memory and examples
    - ReAct format action extraction
    - Configurable memory examples
    - Task-specific command formatting

    Subclasses should:
    1. Call super().__init__() to initialize parent
    2. Optionally add mixins (MemoryMixin, VerificationMixin, HistoryMixin)
    3. Override make_prompt() if custom formatting is needed
    4. Override agent_call() if custom inference is needed

    Attributes:
        llm_model: LLM model instance
        init_prompt_path: Path to prompt template JSON file
        memory_examples: Number of memory examples to include in prompt (default: 3)
        prompt_dict: Dict containing system_msg, instruction, examples
        last_action: Previous action string (for command formatting)
    """

    def __init__(self,
                 llm_model,
                 init_prompt_path=None,
                 memory_examples: int = 3):
        """
        Initialize enhanced ReAct agent.

        Args:
            llm_model: LLM model for inference
            init_prompt_path: Path to JSON file with prompt templates
            memory_examples: Number of memory examples to include (0 for none)
        """
        super().__init__(llm_model)

        # Load prompt templates
        self.prompt_dict = {
            "system_msg": "You are a helpful assistant.",
        }
        if init_prompt_path is not None:
            with open(init_prompt_path, 'r') as f:
                self._update_prompt_dict(json.load(f))

        self.memory_examples = memory_examples
        self.last_action = ""

    def _update_prompt_dict(self, init_prompt_dict):
        """
        Update prompt dictionary with loaded templates.

        Args:
            init_prompt_dict: Dict from JSON file with system_msg, instruction, examples
        """
        if init_prompt_dict is not None:
            self.prompt_dict.update(init_prompt_dict)

    @override
    def make_prompt(self, init_prompt_dict=None):
        """
        Build prompt for ReAct agent.

        Constructs a complete prompt with:
        - System message
        - Instruction
        - Examples (with memory demonstrations)
        - Goal
        - Initial observation
        - Memory/history
        - Available commands

        Args:
            init_prompt_dict: Optional dict to update prompt_dict

        Returns:
            list: OpenAI API message format
        """
        # Update with runtime prompt dict
        self._update_prompt_dict(init_prompt_dict)

        # Assert required keys are present
        assert all(k in self.prompt_dict for k in ["system_msg", "instruction", "examples"]), \
            f"Missing required keys in prompt_dict. Have: {list(self.prompt_dict.keys())}"

        # Build prompt content
        query_dict = {
            "instruction": self.prompt_dict["instruction"],
            "example": format_example(self.prompt_dict["examples"], self.memory_examples),
            "goal": self.env_info["goal"],
            "init_obs": self.env_info["init_obs"],
            "history_str": self._get_history_str(),
            "commands_str": format_commands(self.env_info["task_name"], self.env, self.last_action)
        }

        # Format messages
        messages = [
            {
                "role": "system",
                "content": format_system_msg(
                    self.env_info["task_name"],
                    self.prompt_dict["system_msg"],
                    self.last_action
                )
            },
            {
                "role": "user",
                "content": QUERY.format(**query_dict)
            }
        ]

        return messages

    def _get_history_str(self):
        """
        Get formatted history string.

        This method can be overridden by subclasses or mixins to provide
        custom history formatting (e.g., with dynamic memory).

        Returns:
            str: Formatted history string
        """
        # Default: show recent memory without summarization
        recent_memory = self.memory[-10:] if len(self.memory) > 10 else self.memory
        history_str = "Recent History:\n"
        for mem in recent_memory:
            if mem[0] == "Action":
                history_str += mem[-1] + "\n"
            else:
                history_str += f"Observation: {mem[-1]}\n"
        return history_str

    @override
    def agent_call(self, input_message):
        """
        Call LLM and extract thought/action from response.

        Implements standard ReAct format parsing with optional validation.

        Args:
            input_message: OpenAI API message format

        Returns:
            tuple: (success, response, thought, action, token_cnt)
        """
        # Retry up to 3 times if action parsing fails
        for iteration in range(3):
            success, responses = self.llm_model.generate(input_message)

            # Split response and token count
            if isinstance(responses, tuple):
                token_cnt = responses[-1]
                response = responses[0]
            else:
                response = responses
                token_cnt = 0

            # Extract thought and action
            use_commands_check = (self.env_info["task_name"] == "alfworld")
            thought, action = extract_think_action(
                response,
                env=self.env if use_commands_check else None,
                use_commands_check=use_commands_check
            )

            if action != "[No Action Found]":
                break

            logger.warning(f"Invalid generation detected (attempt {iteration + 1}): {response[:100]}...")

        # Store last action for command formatting
        self.last_action = action

        return success, response, thought, action, token_cnt

    @classmethod
    def from_config(cls, llm_model, config):
        """
        Create agent instance from configuration dict.

        Args:
            llm_model: LLM model instance
            config: Configuration dict with keys:
                - init_prompt_path: Path to prompt template JSON
                - memory_examples: Number of memory examples (default: 3)

        Returns:
            ReactAgentBaseEnhanced instance
        """
        init_prompt_path = config.get("init_prompt_path", None)
        memory_examples = config.get("memory_examples", 3)

        return cls(
            llm_model=llm_model,
            init_prompt_path=init_prompt_path,
            memory_examples=memory_examples
        )

"""
Utility functions for enhanced agents.
"""
import json
import re
import os
import difflib
from .logging import get_logger

# Module logger
logger = get_logger(__name__)


def load_multiple_json_objects(file_path):
    """
    Load multiple JSON objects from a JSONL file.

    Args:
        file_path: Path to JSONL file (one JSON object per line)

    Returns:
        list: List of parsed JSON objects

    Example:
        data = load_multiple_json_objects("logs/alfworld.jsonl")
        # Returns [{"id": 0, ...}, {"id": 1, ...}, ...]
    """
    objects = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                objects.append(json.loads(line.strip()))
    return objects


def extract_think_action(response: str, env=None, use_commands_check: bool = False):
    """
    Extract thought and action from ReAct-formatted response.

    Args:
        response: LLM response string
        env: Environment instance (for valid action checking)
        use_commands_check: If True, validate and correct actions using environment

    Returns:
        tuple: (thought, action) - Extracted thought and action strings
               Returns ("[No Thought Found]", "[No Action Found]") on failure
    """
    # Split thought and action from ReAct format
    match = re.search(r"\s*Thought:\s*(.*?)\s*Action:\s*(.*)", response, re.DOTALL)
    if match:
        thought = match.group(1).strip()
        action_full = match.group(2).strip()
        action = action_full.split('\n')[0].strip()
    else:
        return "[No Thought Found]", "[No Action Found]"

    # Parse action from vanilla agent format
    try:
        origin_action = action
        if 'action' in action.lower():
            action_temp = action.split('\n')
            for act in action_temp:
                if 'action' in act.lower() and ':' in act:  # "action:" case
                    action_temp = ':'.join(act.split(':')[1:])
                    if action_temp != "":
                        action = action_temp
                        break
                if 'action' in act.lower() and 'is to' in act:  # "action is to ..." case
                    action_temp = act.split('is to')[1]
                    if action_temp != "":
                        action = action_temp
                        break

        if action.strip() == "":
            action = origin_action.split('\n')[0]

        action = action.strip()
        action = action.strip("'/")
        action = action.split('\n')[0]

        # Optional: Validate and correct action using environment
        if use_commands_check and env is not None:
            action = find_most_similar_action(action, env.get_action_space())

    except Exception as e:
        logger.debug(f"Error parsing action: {e}")
        return thought, "[No Action Found]"

    return thought, action


def find_most_similar_action(pred_action, valid_actions):
    """
    Find the most similar valid action to predicted action.

    Uses sequence matching to find the best match from valid_actions.
    Prints a warning if the predicted action is corrected.

    Args:
        pred_action: Predicted action string
        valid_actions: List of valid action strings

    Returns:
        str: Best matching action (either pred_action if valid, or most similar valid action)
    """
    if pred_action in valid_actions:
        return pred_action

    # Calculate similarity scores (0-1)
    similarities = [
        (a, difflib.SequenceMatcher(None, pred_action, a).ratio())
        for a in valid_actions
    ]
    # Select highest score
    best_action, best_score = max(similarities, key=lambda x: x[1])

    if best_score > 0.5:  # Only correct if similarity is reasonable
        logger.debug(f"Action '{pred_action}' corrected to '{best_action}' (similarity: {best_score:.2f})")
        return best_action
    else:
        logger.warning(f"Action '{pred_action}' not found (best: {best_action} @ {best_score:.2f})")
        return pred_action


def format_example(example: dict, memory_examples: int) -> str:
    """
    Format example prompt with memory demonstrations.

    Args:
        example: Example dictionary with task, memory examples, observations
        memory_examples: Number of memory examples to include (0 to show none)

    Returns:
        str: Formatted example string
    """
    examples_str_list = []
    if memory_examples > 0:
        examples_str_list.append(example["task"])
        for i in range(1, memory_examples + 1):
            examples_str_list.extend([
                f"Example {i}: ",
                "Memory: " + example[str(i)]["memory"],
                example[str(i)]["assistant"],
                "Observation: " + example[str(i)]["observation"],
                "",
            ])
    return "\n".join(examples_str_list)


def format_history(memory, dynamic_memory) -> str:
    """
    Format agent history with memory and recent actions.

    Args:
        memory: Agent's memory list (actions and observations)
        dynamic_memory: DynamicMemory instance

    Returns:
        str: Formatted history string
    """
    history_str = "Memory: " + dynamic_memory.display() + "\n\n"

    # Add action-based storage
    if dynamic_memory.len_store() > 0:
        history_str += "History of Last Steps: \n"
    else:
        return history_str  # Don't display if no storage

    # Display the last actions
    for mem in memory[-dynamic_memory.len_store():]:
        if mem[0] == "Action":
            history_str += mem[-1] + "\n"
        else:
            history_str += "Observation: " + mem[-1] + "\n"

    return history_str


def format_commands(task_name: str, env, last_action: str = "") -> str:
    """
    Format available commands for the agent.

    Args:
        task_name: Name of the task (e.g., "babyai", "alfworld")
        env: Environment instance
        last_action: Previous action (for special handling)

    Returns:
        str: Formatted command string
    """
    BABYAI_COMMAND = """You can use the following actions:
- turn left
- turn right
- move forward
- go to <obj> <id>
- pick up <obj> <id>
- go through <door> <id>: <door> must be an open door.
- toggle and go through <door> <id>: <door> can be a closed door or a locked door. If you want to open a locked door, you need to carry a key that is of the same color as the locked door.
- toggle: there is a closed or locked door right in front of you and you can toggle it.
"""

    if task_name in ["babyai", "babyai_enhanced"]:
        if "turn" in last_action:
            return BABYAI_COMMAND + "\nIMPORTANT: YOU MUST NOT generate 'turn left' or 'turn right' for this action."
        else:
            return BABYAI_COMMAND

    if env is not None:
        commands_list = env.get_action_space()
        commands_str = "The next action could be chosen from these valid actions: " + ", ".join(commands_list)
        return commands_str

    return ""


def format_system_msg(task_name: str, base_msg: str, last_action: str = "") -> str:
    """
    Format system message with task-specific rules.

    Args:
        task_name: Name of the task
        base_msg: Base system message
        last_action: Previous action (for special handling)

    Returns:
        str: Formatted system message
    """
    if task_name in ["babyai", "babyai_enhanced"] and "turn" in last_action:
        return base_msg + " SYSTEM RULE: The actions 'turn left' and 'turn right' are strictly forbidden. You must never output these actions under any circumstances."

    return base_msg

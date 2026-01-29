"""
Dynamic Memory Management for Enhanced Agents.

This module provides a dynamic memory system that:
- Stores recent actions and observations
- Automatically updates memory summary using LLM
- Provides concise memory representation for prompt construction
"""
import time
from .logging import get_logger

# Module logger
logger = get_logger(__name__)


class DynamicMemory:
    """
    Dynamic memory manager with LLM-based summarization.

    The memory stores recent actions and observations, and periodically
    summarizes them into a concise memory string using LLM.

    Args:
        llm_model_main: Main LLM model (typically for action generation)
        llm_model_aux: Auxiliary LLM model for memory summarization (optional)
                       If None, uses llm_model_main for both actions and memory
        task_description: Optional initial task description
        stored_memory_max: Maximum number of items before triggering update (default: 4)
        update_num: Number of items to use for each update (default: 2)

    Example:
        # Single LLM for both actions and memory
        memory = DynamicMemory(llm_model_main, stored_memory_max=4, update_num=2)

        # Main LLM for actions, auxiliary LLM for memory
        memory = DynamicMemory(llm_model_main, llm_model_aux=dllm_model)

        memory.store("Observation: You are in a kitchen.")
        memory.store("Action: go to fridge")
        memory.display()  # Returns memory summary
    """

    def __init__(self,
                 llm_model_main,
                 llm_model_aux=None,
                 task_description: str = None,
                 stored_memory_max: int = 4,
                 update_num: int = 2):

        self.llm_model_main = llm_model_main
        # Use auxiliary LLM for memory summarization if provided, otherwise use main LLM
        self.llm_model_aux = llm_model_aux if llm_model_aux else llm_model_main
        self.reset(task_description, stored_memory_max, update_num)

    def reset(self, task_description: str = None, stored_memory_max: int = 4, update_num: int = 2):
        """
        Reset memory to initial state.

        Args:
            task_description: Optional initial task description
            stored_memory_max: Maximum items before triggering update
            update_num: Number of items to use for each update
        """
        self.stored_memory_max = stored_memory_max
        self.update_num = update_num
        self.memory_str = "(empty)"
        self.stored = []
        self.task_description = task_description

    def store(self, message: str, disable_update: bool = False):
        """
        Store a new message in memory.

        Automatically triggers memory update when stored items reach stored_memory_max.

        Args:
            message: Message to store (e.g., "Observation: ..." or "Action: ...")
            disable_update: If True, skip LLM update (useful for history replay)
        """
        self.stored.append(message)

        # Auto update when reaching threshold
        if len(self.stored) >= self.stored_memory_max:
            self.update(disable=disable_update)

    def display(self):
        """
        Get current memory summary string.

        Returns:
            str: Current memory summary
        """
        return self.memory_str

    def len_store(self):
        """
        Get number of items currently in storage.

        Returns:
            int: Number of stored items waiting for update
        """
        return len(self.stored)

    def update(self, disable: bool = False):
        """
        Update memory summary using LLM.

        Takes the oldest `update_num` items from storage and uses LLM to
        incorporate them into the memory summary.

        Args:
            disable: If True, skip LLM call and just clear storage
        """
        stored_string = "\n".join(self.stored[:self.update_num])

        message = [
            {
                "role": "system",
                "content": (
                    "You are a memory updater.\n"
                    "Update the memory_str to reflect what the agent has done and learned so far.\n"
                    "Include important actions taken, locations visited, and key observations.\n"
                    "Keep the summary concise, chronological, and consistent.\n"
                    "Do not invent new facts or omit relevant past actions.\n"
                    "Write the memory in third-person, concise past tense, like a mission log."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Memory_str: {self.memory_str}\n"
                    f"Recent_steps: {stored_string}\n\n"
                    "Please output the updated Memory_str only — a short narrative summary of what has been done and observed so far.\n"
                    "No explanations or formatting other than plain text.\n"
                    "Memory_str: "
                )
            },
        ]

        if disable is False:
            # Retry up to 3 times on failure
            for i in range(3):
                try:
                    # Use auxiliary LLM for memory summarization
                    success, response = self.llm_model_aux.generate(message)
                    self.memory_str = response[0]
                    llm_type = "auxiliary" if self.llm_model_aux != self.llm_model_main else "main"
                    logger.info(f"Memory updated (using {llm_type} LLM): success={success}, "
                               f"new_memory={self.memory_str[:50]}...")
                    break
                except Exception as e:
                    logger.warning(f"Memory update failed (attempt {i+1}): {e}")
                    time.sleep(5)
        else:
            logger.debug("Memory update disabled (manual mode)")

        # Clear processed items from storage
        self.stored = self.stored[self.update_num:]

import json
from typing import Any, Dict, List, Tuple
import requests
import time
import re
from copy import deepcopy

class Selector():

    def __init__(self, llm):
        self.llm = llm

    def build_selector_prompt(self, functions: List[Dict[str, Any]], user_message: List[Dict], history_omit: int=0) -> str:
        """
        Build a prompt for a tool-selector agent that selects functions that are potentially useful.
        2) missing: capabilities/tools that are not provided but might be needed (to help argue/justify)

        Selector output format (STRICT):
        {
        "selected": ["func_name_1", "func_name_2"],
        "missing": ["capability_or_tool_name_1", "capability_or_tool_name_2"]
        }

        Notes:
        - "selected" may be [].
        - "missing" may be [].
        - "missing" MUST NOT include any function name that exists in the provided functions list.
        """
        cleaned_functions = json.dumps(self._extract_function_meta(functions), ensure_ascii=False, indent=2)
        history = self._extract_history_str(user_message, omit=history_omit)

        system_instructions = """
You are a tool selector for a function-calling agent.

Task:
Given a user message ([User Message]), the previous tool call([Tool Call]) and its results([Tool Execution Results], You must select a minimum of **3 distinct functions** from the provided list.

Rules:
- Output at least 3 function names, and no more than 10 functions.
- Use ONLY names from the provided function list.
- Output ONLY function names. No explanations or extra text.
- Prioritize the [USER MESSAGE] above all else; use previous tool calls and results only as supplementary context.
"""

        input_prompt = (
            "Functions:\n"
            "{functions}\n\n"
            "{history}\n\n"
            "Selected Functions:"
        )

        message = [
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": input_prompt.format(functions=cleaned_functions, history=history)}
        ]

        return message

    def run_selector(self, 
                     functions: List[Dict],
                     user_message: List[Dict],
                     retry: int=3,
                     ):

        if len(functions) <= 3: # disable selector if len(functions) <= 3.
            return [fun["name"] for fun in functions], 0

        # length control
        history_omit_num = 0
        while history_omit_num <= 20:
            selector_message = self.build_selector_prompt(
                functions=functions,
                user_message=user_message,
                history_omit=history_omit_num
            )

            token_len = self.llm.num_tokens_from_messages(selector_message, quiet=True)

            if token_len > self.llm.context_length:
                print(f"context is cut!! original_length: {token_len}")
                history_omit_num += 1
            else:
                break  # ✅ 核心：一旦合法就停
        try:
            response = self.llm.chat_completion(selector_message, max_tokens=64, quiet=True)
            selected_funcs = self._post_process(response.text, functions)
            token = response.num_token

        except:
            print("selector calling failure. return no selection.")
            return [], 0

        return selected_funcs, token

    def _post_process(self, response: str, functions):
        """
        Parse selected functions
        Returns:
            selected_functions: List[str]
        """
        
        return [fun["name"] for fun in functions if fun['name'] in response]


    def _extract_function_meta(self, functions):
        """
        Input: list of function schemas
        Output: list of {name, description}
        """
        return [
            {
                "name": f.get("name"),
                "description": f.get("description")
            }
            for f in functions
            if "name" in f and "description" in f
        ]

    def _extract_history_str(self, message, omit: int = 0):
        history_str = ""
        skipped_tool_results = 0  # 已忽略的 tool execution result 数量

        for m in message:
            content = m["content"] if len(m["content"]) < 512 else m["content"][:512] # cut content length

            # ---- Tool execution result ----
            if m["role"] == "user" and "[Tool Execution Result]" in content:
                if skipped_tool_results < omit:
                    skipped_tool_results += 1
                    continue  # 忽略前 omit 个 tool result

                history_str += (
                    content
                    .strip()
                    .rstrip("Please modify the functions or parameters based on this.")
                    + "\n"
                )

            # ---- Normal user message ----
            elif m["role"] == "user":
                history_str = "[User Message]\n" + content + "\n\n"

            # ---- Assistant tool call ----
            elif m["role"] == "assistant":
                history_str += "[Tool Call]\n" + content.split("\n")[0] + "\n" # 只取第一行，防止 主 Agent 过长导致炸显存

        return history_str

def filter_message(message: dict, functions: list, selected_funcs: list):
    instruction, _functions = split_system_message(message)
    instruction += "\nYou must NOT include <think> or any reasoning. Your response must ONLY contain a list of function calls in the required format."

    message_new = deepcopy(message)

    if len(selected_funcs) > 0:
        functions_return = filter_func(functions, selected_funcs)
        message_new[0]["content"] = instruction + "\n" + str(functions_return) + "\n"
    else:
        message_new[0]["content"] = instruction + "\n" + _functions + "\n" # keep original

    return message_new

def filter_func(functions: list, selected_funcs: list=[]):
    return [func for func in functions if func["name"] in selected_funcs]

def split_system_message(message: dict):

    def split_by_target_line(text: str):
        lines = text.split("\n")

        for i, line in enumerate(lines):
            # Look for the start of function definitions array
            # Handle cases where "[" might have leading whitespace
            stripped_line = line.strip()
            if stripped_line.startswith("[") or stripped_line.startswith("[{"):
                before = "\n".join(lines[:i])
                after = "\n".join(lines[i:])
                return before, after

        raise ValueError("No line starting with '[' (function definitions) found")

    # cut system message
    for m in message:
        if m["role"] == "system":
            instruction, functions = split_by_target_line(m["content"])

            return instruction, functions
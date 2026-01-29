from typing import Any, Dict, List, Tuple
import time

class Format_Editor():

    def __init__(self, llm):
        self.llm = llm

    def user_message_to_history(self, message):
        history = []
        for m in message:
            if m["role"] == "system":
                continue
            history.append(f"{m['role'].upper()}: {m['content']}")
            
        return "\n".join(history)

    def build_editor_prompt(self, message: list[dict[str, str]], response: str) -> str:
        history = self.user_message_to_history(message)
        
        system_prompt = """\
You are a strict tool-call format auditor and repairer.

Your task:
Task: Repair or validate a broken tool-call and output a final call that strictly follows TOOL_CALL_FORMAT. 

Rules:
1.	If the tool-call is already valid and correct, output UNCHANGED.
2.	If the tool-call is textual explanations, output NO_VALID_TOOL_CALLS.
3.  If the tool-call contains both explanations and tool-calls, remove the explanations and correct the tool-calls.
4.	If the tool-call does not conform to TOOL_CALL_FORMAT, repair any format or schema errors and output the corrected tool-call only; do not invent functions or parameters.

TOOL_CALL_FORMAT:
[func_name1(param_name1=param_value1, param_name2=param_value2, ...), func_name2(param_name3=param_value3, ...)]

examples:
BROKEN_TOOL_CALL 1:
[cd(folder="academic_venture")]
Output 1:
UNCHANGED

BROKEN_TOOL_CALL 2:
```cd(folder="academic_venture")```
Output 2:
[cd(folder="academic_venture")]

BROKEN_TOOL_CALL 3:
{"cd": {"folder": "academic_venture"}}
Output 3:
[cd(folder="academic_venture")]

BROKEN_TOOL_CALL 4:
The task is now complete.
Output 4:
NO_VALID_TOOL_CALLS

BROKEN_TOOL_CALL 4:
The task is now complete. The final tool-call is {"ls": {}}
Output 4:
[ls()]
"""
        
        user_prompt = """\
BROKEN_TOOL_CALL (to be audited and possibly corrected):
{response}

Now produce the final output according to the rules above. No explanations, markdown, or extra text.

Output:
"""

        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt.format(response=response)}
        ]

        return message


    def build_regenerate_tool_call_prompt(
            self,
            message: list[dict[str, str]],
            functions: list[dict],
        ) -> List[Dict[str, Any]]:

        history = self.user_message_to_history(message)
        functions_str = str(functions)
    
        system_prompt = """\
You are a strict tool-call generator for a fixed tool-call grammar.

Context:
A previous tool-call was rejected as INVALID_TOOL_CALL. You must generate a NEW tool-call that satisfies the user's true requirement.

You may internally reason about the interaction history and why the prior call failed, but you MUST NOT output any analysis.

Hard rules:
1) Output MUST strictly follow TOOL_CALL_FORMAT, with no extra text, explanations, or markdown.
2) You MUST use INTERACTION HISTORY to infer the actual user requirement and any constraints from prior attempts.
   - If the history contains tool execution errors or validation errors, you must replan and avoid repeating the same mistake.
3) Only use function names and parameter names permitted by AVAILABLE_TOOLS.
4) Do NOT invent parameters. Do NOT omit required parameters.
5) If multiple tool calls are needed, output them as a list in the exact TOOL_CALL_FORMAT.
6) If no tool-call is required or no valid tool-call can be inferred, output exactly:
   NO_VALID_TOOL_CALLS!
"""

        user_prompt = """\
INTERACTION HISTORY:
{interaction_history}

TOOL_CALL_FORMAT (authoritative):
[func_name1(param_name1=param_value1, param_name2=param_value2, ...), func_name2(param_name3=param_value3, ...)]

AVAILABLE_TOOLS:
{functions_str}

Now generate a NEW valid tool-call that best fulfills the user requirement, strictly following TOOL_CALL_FORMAT.
"""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt.format(
                interaction_history=history,
                functions_str=functions_str
            )}
        ]

    def run_formateditor(self,
                         messages: List[Dict[str, str]],
                         functions: List[Dict],
                         agent_response: str,
                         ):

        # editor_message = self.build_editor_prompt(message=messages, response=agent_response)

        while True:
            editor_message = self.build_editor_prompt(
                message=messages,
                response=agent_response
            )

            token_len = self.llm.num_tokens_from_messages(editor_message, quiet=True)

            # ✅ 新增停止条件：只剩一行（或空）
            if agent_response.count("\n") == 0:
                if len(agent_response) > 10000: # 只保留前 10000 个字符
                    agent_response = agent_response[:10000]
                break

            if token_len > self.llm.context_length:
                agent_response = "\n".join(agent_response.splitlines()[:-1])
            else:
                break  # 一旦合法就停

        response = self.llm.chat_completion(editor_message, max_tokens=64, quiet=True)

        # breakpoint()
        if "MISSING" in response.text or "UNCHANGED" in response.text or "NO_VALID_TOOL_CALLS" in response.text: # missing case
            return agent_response

        return response.text

        # if "MISSING" in response.text: # missing case
        #     return agent_response

        # if "NO_VALID_TOOL_CALLS" not in response.text: # formatted case
        #     return response.text
                
        # return agent_response
        # # regenerate case
        # regenerate_message = self.build_regenerate_tool_call_prompt(message=messages, functions=functions)

        # response = self.llm.chat_completion(regenerate_message)
        # print("POST-HOC: REGENERATOR: ", response.text)

        # if "NO_VALID_TOOL_CALLS" not in response.text:
        #     return response.text

        # r
        # eturn agent_response
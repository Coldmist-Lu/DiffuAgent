"""
DiffuAgent Handler Classes

This module provides unified handler classes for all DiffuAgent variants.
Handlers are composed of base functionality + mixins for different features.

Handler naming:
{Backend}{Feature}Handler

Backends: LLM, DLLM
Features: (none), Selector, Editor, SelectorEditor

Examples:
- LLMHandler: Base LLM handler
- SelectorLLMHandler: LLM + function selection
- EditorLLMHandler: LLM + format editing
- SelectorEditorLLMHandler: LLM + both features
- DLLMHandler: Base DLLM handler
- SelectorDLLMHandler: DLLM + function selection
- EditorDLLMHandler: DLLM + format editing
- SelectorEditorDLLMHandler: DLLM + both features
"""

import json
from typing import Any
from overrides import override

from bfcl_eval.model_handler.api_inference.diffuagent.base import DiffuagentBaseHandler
from bfcl_eval.model_handler.api_inference.diffuagent.mixins import (
    SelectorMixin,
    EditorMixin,
    SelectorEditorMixin,
)


class LLMHandler(DiffuagentBaseHandler):
    """
    Base LLM handler with chat-based formatting.

    This handler provides the foundation for LLM-based inference.
    """

    def __init__(
        self,
        model_name: str,
        temperature: float,
        registry_name: str,
        is_fc_model: bool,
        bos_token: str = "<s>",
        eos_token: str = "</s>",
    ) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model, backend="llm")
        self.bos_token = bos_token
        self.eos_token = eos_token

    @override
    def _format_prompt(self, messages, function):
        """Format messages with simple role-based markers."""
        formatted_prompt = ""
        for m in messages:
            role = m["role"].upper()
            formatted_prompt += f"[{role}]" + m["content"] + f"[/{role}]"
        return formatted_prompt

    @override
    def _add_execution_results_prompting(
        self, inference_data: dict, execution_results: list[str], model_response_data: dict
    ) -> dict:
        """Add execution results as user messages for LLM variants."""
        for execution_result, decoded_model_response in zip(
            execution_results, model_response_data["model_responses_decoded"]
        ):
            inference_data["message"].append(
                {
                    "role": "user",
                    "content": (
                        f"[Tool Execution Result]\n"
                        f"Tool name: {decoded_model_response}\n"
                        f"Output:\n{execution_result}\n"
                        "Please modify the functions or parameters based on this."
                    ),
                }
            )
        return inference_data

    @override
    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        """Parse LLM response, extracting reasoning content if present."""
        # Handle both SimpleNamespace (from our backend wrappers) and raw API response
        if hasattr(api_response, 'text'):
            # Response from backend wrapper (SimpleNamespace)
            model_response = api_response.text
            input_token = 0
            output_token = 0
        else:
            # Response from direct API call (OpenAI format)
            model_response = api_response.choices[0].message.content
            input_token = api_response.usage.prompt_tokens
            output_token = api_response.usage.completion_tokens

        reasoning_content = ""
        cleaned_response = model_response
        if "<|think|>" in model_response:
            parts = model_response.split("<|think|>")
            reasoning_content = (
                parts[0].rstrip("\n").split("<|think|>")[-1].lstrip("\n")
                if len(parts) > 1
                else ""
            )
            cleaned_response = parts[-1].lstrip("\n")

        return {
            "model_responses": cleaned_response,
            "reasoning_content": reasoning_content,
            "input_token": input_token,
            "output_token": output_token,
        }


class SelectorLLMHandler(SelectorMixin, LLMHandler):
    """LLM handler with function selection capability."""
    pass


class EditorLLMHandler(EditorMixin, LLMHandler):
    """LLM handler with format editing capability."""
    pass


class SelectorEditorLLMHandler(SelectorEditorMixin, LLMHandler):
    """LLM handler with both function selection and format editing."""
    pass


class DLLMHandler(DiffuagentBaseHandler):
    """
    Base DLLM handler.

    This handler provides the foundation for DLLM-based inference.
    Uses function calling schema for tool interactions.
    """

    def __init__(
        self,
        model_name: str,
        temperature: float,
        registry_name: str,
        is_fc_model: bool,
        dtype: str = "bfloat16",
    ) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model, backend="dllm", dtype=dtype)

        # Detect which DLLM variant is being used
        self.dllm_name = None
        for dllm_name in ["Llada", "Dream", "Fdllmv2", "Dllmvar"]:
            if dllm_name.lower() in self.model_name:
                self.dllm_name = dllm_name
                break

        if self.dllm_name is None:
            raise ValueError(f"Unknown DLLM variant in model name: {self.model_name}")

    @override
    def _format_prompt(self, messages, function):
        """Format messages with function schema for DLLM."""
        formatted_prompt = json.dumps({
            "messages": messages,
            "functions": function
        }, ensure_ascii=False)
        return formatted_prompt


class SelectorDLLMHandler(SelectorMixin, DLLMHandler):
    """DLLM handler with function selection capability."""
    pass


class EditorDLLMHandler(EditorMixin, DLLMHandler):
    """DLLM handler with format editing capability."""
    pass


class SelectorEditorDLLMHandler(SelectorEditorMixin, DLLMHandler):
    """DLLM handler with both function selection and format editing."""
    pass

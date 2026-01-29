"""
DiffuAgent Feature Mixins

This module contains mixin classes that add specific capabilities to the base handler:
- SelectorMixin: Adds function selection capability (runs BEFORE chat_completion)
- EditorMixin: Adds format editing capability (runs AFTER chat_completion)

These mixins should be used together in _query_prompting to ensure correct execution order.
"""

from bfcl_eval.model_handler.api_inference.utils.selector_utils import (
    Selector,
    filter_message,
    filter_func,
)
from bfcl_eval.model_handler.api_inference.utils.fmeditor_utils import Format_Editor


class SelectorMixin:
    """
    Mixin that adds function selection capability.

    This mixin runs BEFORE chat_completion to filter relevant functions.

    Usage:
        class MyHandler(SelectorMixin, DiffuagentBaseHandler):
            pass
    """

    def _initialize_features(self):
        """Initialize the Selector component."""
        super()._initialize_features()

        # Use feature_backend for selector (configured in base.__init__)
        if self.feature_backend == "llm":
            if self.llm is None:
                raise ValueError("LLM backend not available for Selector")
            self.selector = Selector(llm=self.llm)
            self._selector_backend = "llm"
        elif self.feature_backend == "dllm":
            if self.dllm is None:
                raise ValueError(
                    f"DLLM backend not available for Selector (server may not be running on {self.dllm_base_url})"
                )
            self.selector = Selector(llm=self.dllm)
            self._selector_backend = "dllm"
        else:
            raise ValueError(f"Unknown feature backend: {self.feature_backend}")

    def _query_prompting(self, inference_data: dict):
        """
        Query backend with selector (before chat_completion).

        Execution order:
        1. Print user message
        2. Run selector to filter relevant functions (BEFORE chat_completion)
        3. Call chat_completion to get response
        """
        function: list[dict] = inference_data["function"]
        message: list[dict] = inference_data["message"]

        # STEP 0: Print user message first (before everything)
        print(f"  ▶ {message[-1]['role'].upper()}: {message[-1]['content']}")

        # STEP 1: Run selector BEFORE chat_completion (filter functions)

        # Get request info based on actual backend used by selector
        if self._selector_backend == "llm":
            request_url = self.llm.urls["chat"]
            model_path = self.llm.model_path
        elif self._selector_backend == "dllm":
            request_url = self.dllm.urls["generate"]
            model_path = self.dllm.model_name
        else:
            request_url = "Unknown"
            model_path = "Unknown"

        print(f"  ▶ Selector Request: {request_url}")
        print(f"    • Backend Model: {model_path}")
        print(f"    • Input: {len(function)} functions")

        selected_functions, _ = self.selector.run_selector(
            functions=function, user_message=message
        )

        print(f"  ✓ Selector: {len(selected_functions)} functions selected")
        print(f"    • Result: {selected_functions}\n")

        # Filter messages using selected functions
        message = filter_message(
            message=message,
            functions=function,
            selected_funcs=selected_functions
        )

        # Update inference_data with filtered messages
        inference_data["message"] = message

        # STEP 2: Call parent's _query_prompting to get response
        return super()._query_prompting(inference_data)


class EditorMixin:
    """
    Mixin that adds format editing capability.

    This mixin runs AFTER chat_completion to fix format issues.

    Usage:
        class MyHandler(EditorMixin, DiffuagentBaseHandler):
            pass
    """

    def _initialize_features(self):
        """Initialize the Format_Editor component."""
        super()._initialize_features()

        # Use feature_backend for editor (configured in base.__init__)
        if self.feature_backend == "llm":
            if self.llm is None:
                raise ValueError("LLM backend not available for Editor")
            self.editor = Format_Editor(llm=self.llm)
            self._editor_backend = "llm"
        elif self.feature_backend == "dllm":
            if self.dllm is None:
                raise ValueError(
                    f"DLLM backend not available for Editor (server may not be running on {self.dllm_base_url})"
                )
            self.editor = Format_Editor(llm=self.dllm)
            self._editor_backend = "dllm"
        else:
            raise ValueError(f"Unknown feature backend: {self.feature_backend}")

    def _query_prompting(self, inference_data: dict):
        """
        Query backend with editor (after chat_completion).

        Execution order:
        1. Call chat_completion to get response
        2. Run editor to fix format (AFTER chat_completion)
        """
        function: list[dict] = inference_data["function"]

        # STEP 1: Call parent's _query_prompting to get response
        response, latency = super()._query_prompting(inference_data)

        # STEP 2: Run editor AFTER chat_completion (fix format)

        # Get request info based on actual backend used by editor
        if self._editor_backend == "llm":
            request_url = self.llm.urls["chat"]
            model_path = self.llm.model_path
        elif self._editor_backend == "dllm":
            request_url = self.dllm.urls["generate"]
            model_path = self.dllm.model_name
        else:
            request_url = "Unknown"
            model_path = "Unknown"

        print(f"  ▶ Editor Request: {request_url}")
        print(f"    • Backend Model: {model_path}")
        print(f"    • Original length: {len(response.text)}")

        # Run editor to fix format
        original_response = response.text
        edited_response = self.editor.run_formateditor(
            messages=inference_data["message"],
            functions=function,
            agent_response=original_response,
        )

        print(f"  ✓ Editor: Edited length: {len(edited_response)}")
        print(f"    • Result: {edited_response}")

        # Update response with edited content
        response.text = edited_response

        return response, latency


class SelectorEditorMixin(SelectorMixin, EditorMixin):
    """
    Mixin that combines both Selector and Editor capabilities.

    This mixin:
    1. Runs selector BEFORE querying (filters relevant functions)
    2. Runs editor AFTER querying (fixes format issues)

    The key is that both operations happen in _query_prompting at the right time.

    Usage:
        class MyHandler(SelectorEditorMixin, DiffuagentBaseHandler):
            pass
    """

    def _initialize_features(self):
        """Initialize both Selector and Editor components."""
        # Use feature_backend for both selector and editor (configured in base.__init__)
        if self.feature_backend == "llm":
            if self.llm is None:
                raise ValueError("LLM backend not available for Selector/Editor")
            backend = self.llm
            self._selector_backend = "llm"
            self._editor_backend = "llm"
        elif self.feature_backend == "dllm":
            if self.dllm is None:
                raise ValueError(
                    f"DLLM backend not available for Selector/Editor (server may not be running on {self.dllm_base_url})"
                )
            backend = self.dllm
            self._selector_backend = "dllm"
            self._editor_backend = "dllm"
        else:
            raise ValueError(f"Unknown feature backend: {self.feature_backend}")

        self.selector = Selector(llm=backend)
        self.editor = Format_Editor(llm=backend)

    def _query_prompting(self, inference_data: dict):
        """
        Query backend with selector (before) and editor (after).

        Execution order:
        1. Print user message
        2. Run selector to filter relevant functions (BEFORE chat_completion)
        3. Call chat_completion to get response
        4. Run editor to fix format (AFTER chat_completion)
        """
        function: list[dict] = inference_data["function"]
        message: list[dict] = inference_data["message"]

        # STEP 0: Print user message first (before everything)
        print(f"  ▶ {message[-1]['role'].upper()}: {message[-1]['content']}")

        # STEP 1: Run selector BEFORE chat_completion (filter functions)

        # Get request info based on actual backend used by selector
        if self._selector_backend == "llm":
            request_url = self.llm.urls["chat"]
            model_path = self.llm.model_path
        elif self._selector_backend == "dllm":
            request_url = self.dllm.urls["generate"]
            model_path = self.dllm.model_name
        else:
            request_url = "Unknown"
            model_path = "Unknown"

        print(f"  ▶ Selector Request: {request_url}")
        print(f"    • Backend Model: {model_path}")
        print(f"    • Input: {len(function)} functions")

        selected_functions, _ = self.selector.run_selector(
            functions=function, user_message=message
        )

        print(f"  ✓ Selector: {len(selected_functions)} functions selected")
        print(f"    • Result: {selected_functions}\n")

        # Filter messages using selected functions
        message = filter_message(
            message=message,
            functions=function,
            selected_funcs=selected_functions
        )

        # Update inference_data with filtered messages
        inference_data["message"] = message

        # STEP 2: Call DiffuagentBaseHandler's _query_prompting directly
        # Skip SelectorMixin and EditorMixin to avoid duplicate execution
        from bfcl_eval.model_handler.api_inference.diffuagent.base import DiffuagentBaseHandler
        response, latency = DiffuagentBaseHandler._query_prompting(self, inference_data)

        # STEP 3: Run editor AFTER chat_completion (fix format)

        # Get request info based on actual backend used by editor
        if self._editor_backend == "llm":
            request_url = self.llm.urls["chat"]
            model_path = self.llm.model_path
        elif self._editor_backend == "dllm":
            request_url = self.dllm.urls["generate"]
            model_path = self.dllm.model_name
        else:
            request_url = "Unknown"
            model_path = "Unknown"

        print(f"  ▶ Editor Request: {request_url}")
        print(f"    • Backend Model: {model_path}")
        print(f"    • Original length: {len(response.text)}")

        # Filter functions (no selection, just filter)
        filtered_functions = filter_func(function, selected_functions)

        # Run editor to fix format
        original_response = response.text
        edited_response = self.editor.run_formateditor(
            messages=message,
            functions=filtered_functions,
            agent_response=original_response,
        )

        print(f"  ✓ Editor: Edited length: {len(edited_response)}")
        print(f"    • Result: {edited_response}")

        # Update response with edited content
        response.text = edited_response

        return response, latency

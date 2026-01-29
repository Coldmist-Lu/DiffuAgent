"""
DiffuAgent Base Handler

This module contains the unified base handler for all DiffuAgent variants.
It provides common functionality for batch inference, error handling, and prompting.
"""

import os
import json
import traceback
from typing import Optional, Literal, Any
from datetime import datetime
from tqdm import tqdm
from overrides import EnforceOverrides, final, override

from bfcl_eval.constants.eval_config import RESULT_PATH
from bfcl_eval.constants.enums import ModelStyle, ReturnFormat
from bfcl_eval.model_handler.base_handler import BaseHandler
from bfcl_eval.model_handler.api_inference.utils.llm_utils import (
    resolve_model_and_context_length,
    calculate_leftover_tokens,
)
from bfcl_eval.model_handler.utils import (
    default_decode_ast_prompting,
    default_decode_execute_prompting,
    system_prompt_pre_processing_chat_model,
)
from bfcl_eval.utils import _func_doc_language_specific_pre_processing as func_doc_language_specific_pre_processing

# Backend imports
from bfcl_eval.model_handler.api_inference.utils.request_llm import REQUEST_LLM
from bfcl_eval.model_handler.api_inference.utils.request_dllm import REQUEST_DLLM
from bfcl_eval.model_handler.api_inference.utils.debug_utils import debug


BackendType = Literal["llm", "dllm"]


class DiffuagentBaseHandler(BaseHandler, EnforceOverrides):
    """
    Unified base handler for all DiffuAgent variants.

    This class provides:
    - Common initialization and model setup
    - Batch inference logic
    - Error handling and recovery
    - Standard prompting methods

    Subclasses should override:
    - _format_prompt() for model-specific formatting
    - _query_prompting() for backend-specific queries
    - _parse_query_response_prompting() for response parsing

    Features can be added via mixins:
    - SelectorMixin: Add function selection capability
    - EditorMixin: Add format editing capability
    """

    def __init__(
        self,
        model_name: str,
        temperature: float,
        registry_name: str,
        is_fc_model: bool,
        backend: BackendType = "llm",
        dtype: str = "bfloat16",
    ) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model)
        self.model_name_huggingface = model_name
        self.model_style = ModelStyle.GORILLA  # Using GORILLA for OSS models
        self.dtype = dtype

        # Parse model_name to determine backends
        # Examples:
        # - "qwen3-8b": main_backend=LLM, feature_backend=LLM
        # - "qwen3-8b-llada": main_backend=LLM, feature_backend=DLLM (llada)
        # - "llada": main_backend=DLLM, feature_backend=DLLM
        # - "qwen3-8b-dream": main_backend=LLM, feature_backend=DLLM (dream)

        DLLM_VARIANTS = ["Llada", "Dream", "Fdllmv2", "Dllmvar"]

        # Extract actual model name (last part after '/')
        actual_model_name = self.model_name.split('/')[-1]

        # Check if actual_model_name is a pure DLLM variant
        is_pure_dllm = actual_model_name.lower() in [v.lower() for v in DLLM_VARIANTS]

        if is_pure_dllm:
            # Model is pure DLLM (e.g., diffuagent-chatbase/llada)
            self.backend = "dllm"
            self.feature_backend = "dllm"
            self.dllm_name = actual_model_name  # Use the actual model name
        elif '-' in actual_model_name:
            # Check if model_name has DLLM suffix (e.g., "qwen3-8b-llada")
            dllm_variant = None
            for variant in DLLM_VARIANTS:
                if variant.lower() in actual_model_name.lower():
                    dllm_variant = variant
                    break

            if dllm_variant is not None:
                # Model has DLLM suffix (e.g., qwen3-8b-llada)
                # Main agent uses LLM, features use DLLM
                self.backend = "llm"
                self.feature_backend = "dllm"
                self.dllm_name = dllm_variant
            else:
                # Model is LLM without DLLM suffix (e.g., qwen3-8b)
                self.backend = "llm"
                self.feature_backend = "llm"
                self.dllm_name = None
        else:
            # Model is LLM without DLLM suffix (e.g., qwen3-8b)
            self.backend = "llm"
            self.feature_backend = "llm"
            self.dllm_name = None

        # Model setup
        self.model_path_or_id = None
        self.tokenizer = None
        self.max_context_length = None

        # Main agent backend configuration
        # Support both new naming (MAIN_AGENT_*) and legacy naming (VLLM_API_KEY)
        self.api_key = os.getenv("MAIN_AGENT_API_KEY") or os.getenv("VLLM_API_KEY", None)
        self.base_url = os.getenv("MAIN_AGENT_BASE_URL") or os.getenv("VLLM_BASE_URL", None)

        # Features backend configuration (Selector/Editor)
        # Support both new naming (FEATURES_*) and legacy naming (DLLM_API_KEY)
        self.dllm_api_key = os.getenv("FEATURES_API_KEY") or os.getenv("DLLM_API_KEY", None)
        self.dllm_base_url = os.getenv("FEATURES_BASE_URL") or os.getenv("DLLM_BASE_URL", None)

        # Backend instances (initialized in batch_inference)
        self.llm = None
        self.dllm = None

        # Feature components (set by mixins)
        self.selector = None
        self.editor = None

        # Initialize backend on first inference call
        self._backend_initialized = False

        # Print configuration summary
        print(f"\n  🔧 DiffuAgent Configuration:")
        print(f"     ├─ Model Name: {self.model_name}")
        print(f"     ├─ Actual Model: {self.model_name.split('/')[-1]}")
        print(f"     ├─ Main Agent Backend: {self.backend.upper()}")
        if self.base_url:
            print(f"     │  └─ URL: {self.base_url}")
        print(f"     ├─ Feature Backend: {self.feature_backend.upper()}")
        if self.dllm_base_url:
            print(f"     │  └─ URL: {self.dllm_base_url}")
        if self.dllm_name:
            print(f"     └─ DLLM Variant: {self.dllm_name}")
        else:
            print(f"     └─ DLLM Variant: None")
        print()

    def _extract_vllm_model_name(self, model_config_name: str) -> str:
        """
        Extract the actual model name from DiffuAgent config name.

        Examples:
            "diffuagent-chatbase/qwen3-8b" -> "qwen3-8b"
            "diffuagent-chatbase/qwen3-8b-llada" -> "qwen3-8b-llada"
            "diffuagent-selector-chatbase/ministral-8b" -> "ministral-8b"
        """
        # Get the last part after "/"
        return model_config_name.split('/')[-1]

    def _ensure_backend_initialized(self):
        """Initialize backend if not already initialized."""
        if not self._backend_initialized:
            # Try to get local_model_path from environment first
            # Support both new naming (MAIN_AGENT_MODEL_PATH) and legacy naming (VLLM_MODEL_PATH)
            local_model_path = os.getenv("MAIN_AGENT_MODEL_PATH") or os.getenv("VLLM_MODEL_PATH", None)

            # Only load tokenizer if we have a valid local_model_path
            # Otherwise skip tokenizer loading to avoid HuggingFace download
            if local_model_path and os.path.isdir(local_model_path):
                self.model_path_or_id, self.tokenizer, self.max_context_length = (
                    resolve_model_and_context_length(
                        local_model_path=local_model_path,
                        model_name_huggingface=self.model_name_huggingface,
                    )
                )
            else:
                # Skip tokenizer loading, use default values
                self.model_path_or_id = None
                self.tokenizer = None
                self.max_context_length = 262144  # Default large context length

                # Save local_model_path for later use in _initialize_backend
                # This allows using the full path even without tokenizer
                if local_model_path:
                    self.model_path_or_id = local_model_path

            # Initialize backend
            self._initialize_backend()
            # Initialize features (selector, editor)
            self._initialize_features()
            self._backend_initialized = True

    @override
    def inference(
        self, test_entry: dict, include_input_log: bool, exclude_state_log: bool
    ):
        """
        Handle single inference request for API-based models.

        Returns:
            tuple: (model_responses, metadata) where model_responses is the
                   raw result and metadata is a dict with additional info.
        """
        # Ensure backend is initialized
        self._ensure_backend_initialized()

        # Use existing multi-threaded inference logic
        result_dict = self._multi_threaded_inference(test_entry, include_input_log, exclude_state_log)

        # Extract model_responses and metadata from result_dict
        model_responses = result_dict.get("result")
        metadata = {k: v for k, v in result_dict.items() if k not in ["id", "result"]}

        return model_responses, metadata

    @override
    def decode_ast(self, result, language: ReturnFormat, has_tool_call_tag: bool):
        # DiffuAgent only handles non-tool-call scenarios
        return default_decode_ast_prompting(result, language)

    @override
    def decode_execute(self, result, has_tool_call_tag: bool):
        # DiffuAgent only handles non-tool-call scenarios
        return default_decode_execute_prompting(result)

    def batch_inference(
        self,
        test_entries: list[dict],
        num_gpus: int,
        gpu_memory_utilization: float,
        backend: str,
        skip_server_setup: bool,
        local_model_path: Optional[str],
        include_input_log: bool,
        exclude_state_log: bool,
        update_mode: bool,
        result_dir=RESULT_PATH,
    ):
        """
        Unified batch inference for all backends and variants.
        """
        # Initialize model and tokenizer
        self.model_path_or_id, self.tokenizer, self.max_context_length = (
            resolve_model_and_context_length(
                local_model_path=local_model_path,
                model_name_huggingface=self.model_name_huggingface,
            )
        )

        # Initialize backend
        self._initialize_backend()

        # Initialize features (selector, editor)
        self._initialize_features()

        # Run inference
        with tqdm(
            total=len(test_entries),
            desc=f"Generating results for {self.model_name}",
        ) as pbar:
            for test_case in test_entries:
                result = self._multi_threaded_inference(
                    test_case,
                    include_input_log,
                    exclude_state_log,
                )
                self.write(result, result_dir, update_mode=update_mode)
                pbar.update()

    def _initialize_backend(self):
        """Initialize the appropriate backend(s) based on configuration."""
        # DEBUG: Show handler type and features
        has_selector = hasattr(self, 'selector') and self.selector is not None
        has_editor = hasattr(self, 'editor') and self.editor is not None
        print(f"\n  🔧 Handler Info: {self.__class__.__name__}")
        print(f"     ├─ Selector: {'✓' if has_selector else '✗'}")
        print(f"     ├─ Editor:   {'✓' if has_editor else '✗'}")
        print(f"     ├─ Main Agent Backend: {self.backend.upper()}")
        print(f"     └─ Feature Backend: {self.feature_backend.upper()}\n")

        # Initialize main agent backend
        if self.backend == "llm":
            # Determine VLLM model identifier:
            # 1. Use model_path_or_id if it's a valid absolute path (from --local-model-path)
            # 2. Otherwise extract from config name (e.g., "diffuagent-chatbase/qwen3-8b" -> "qwen3-8b")
            if self.model_path_or_id and (
                self.model_path_or_id.startswith('/') or
                self.model_path_or_id.startswith('.')
            ):
                # Full path provided (e.g., "/model/ModelScope/Qwen/Qwen3-8B")
                vllm_model_identifier = self.model_path_or_id
            else:
                # Extract short name from config
                vllm_model_identifier = self._extract_vllm_model_name(self.model_name)

            self.llm = REQUEST_LLM(
                model_path=vllm_model_identifier,
                base_url=self.base_url,
                api_key=self.api_key,
                context_length=self.max_context_length,
            )
            self.llm.check_server_availability()
        elif self.backend == "dllm":
            self.dllm = self._check_and_init_dllm()
        else:
            raise ValueError(f"Unknown main backend: {self.backend}")

        # Initialize feature backend (for selector/editor) if different from main backend
        if self.feature_backend != self.backend:
            if self.feature_backend == "dllm":
                self.dllm = self._check_and_init_dllm()
            elif self.feature_backend == "llm":
                # LLM should already be initialized if main backend is DLLM
                if self.llm is None:
                    if self.model_path_or_id and (
                        self.model_path_or_id.startswith('/') or
                        self.model_path_or_id.startswith('.')
                    ):
                        vllm_model_identifier = self.model_path_or_id
                    else:
                        vllm_model_identifier = self._extract_vllm_model_name(self.model_name)

                    self.llm = REQUEST_LLM(
                        model_path=vllm_model_identifier,
                        base_url=self.base_url,
                        api_key=self.api_key,
                        context_length=self.max_context_length,
                    )
                    self.llm.check_server_availability()
            else:
                raise ValueError(f"Unknown feature backend: {self.feature_backend}")

    def _initialize_features(self):
        """
        Initialize feature components (selector, editor).
        This method is called by mixins to set up their components.
        """
        pass

    def _check_and_init_dllm(self):
        """Check if DLLM is available and initialize it."""
        # Get DLLM variant name from handler (set by DLLMHandler.__init__)
        if not hasattr(self, 'dllm_name') or self.dllm_name is None:
            raise ValueError(f"DLLM name not set. Model: {self.model_name}")

        dllm = REQUEST_DLLM(
            model_name=self.dllm_name,  # DLLM expects variant name like "Llada", "Dream"
            base_url=self.dllm_base_url,
            api_key=self.dllm_api_key,
        )
        try:
            dllm.check_server_availability()
            return dllm
        except Exception:
            return None

    @final
    def _multi_threaded_inference(
        self, test_case, include_input_log: bool, exclude_state_log: bool
    ):
        """
        Wrapper for error handling during inference.
        """
        assert type(test_case["function"]) is list

        try:
            if "multi_turn" in test_case["id"]:
                model_responses, metadata = self.inference_multi_turn_prompting(
                    test_case, include_input_log, exclude_state_log
                )
            else:
                model_responses, metadata = self.inference_single_turn_prompting(
                    test_case, include_input_log
                )
        except Exception as e:
            print("-" * 100)
            print(
                "❗️❗️ Error occurred during inference. Maximum retries reached for rate limit or other error. Continuing to next test case."
            )
            print(f"❗️❗️ Test case ID: {test_case['id']}, Error: {str(e)}")
            traceback.print_exc(limit=10)
            print("-" * 100)

            model_responses = f"Error during inference: {str(e)}"
            metadata = {"traceback": traceback.format_exc()}

        result_to_write = {
            "id": test_case["id"],
            "result": model_responses,
        }
        result_to_write.update(metadata)

        return result_to_write

    #### Prompting methods ####

    def _format_prompt(self, messages, function):
        """
        Format messages and functions into a model-specific prompt.
        Must be implemented by subclasses.
        """
        raise NotImplementedError(
            "Subclasses must implement _format_prompt for their specific format."
        )

    @override
    def _query_prompting(self, inference_data: dict):
        """
        Query the backend (LLM or DLLM) for response.
        Can be overridden for custom behavior.
        """
        function: list[dict] = inference_data["function"]
        message: list[dict] = inference_data["message"]

        # Print user message (only if not already printed by SelectorMixin)
        has_selector = hasattr(self, 'selector') and self.selector is not None
        if not has_selector:
            print(f"  ▶ {message[-1]['role'].upper()}: {message[-1]['content']}")

        # Filter messages (may be modified by mixins)
        message = self._filter_messages(message, function)

        # Format prompt for logging
        formatted_prompt = self._format_prompt(message, function)
        inference_data["inference_input_log"] = {"formatted_prompt": formatted_prompt}

        # Get token count and query backend
        # Don't use quiet mode - let backend wrappers show their debug output
        if self.backend == "llm":
            response = self._query_llm(message, quiet=False)
        elif self.backend == "dllm":
            response = self._query_dllm(message, function, quiet=False)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        # Print response
        if "</think>" in response.text:
            response.text = response.text.split("</think>", 1)[-1].strip()
        print(f"  ✓ ASSISTANT: {response.text}\n")

        # Log to file
        self._log_inference(formatted_prompt, response.json)

        # Return response wrapper (SimpleNamespace) instead of response.object
        # This allows _parse_query_response_prompting to work with both LLM and DLLM
        return response, response.latency

    def _filter_messages(self, messages, function):
        """
        Filter messages before querying.
        Can be overridden by mixins (e.g., SelectorMixin).
        Base implementation returns messages unchanged.
        """
        # Base handler doesn't filter messages
        # SelectorMixin overrides this to filter functions
        return messages

    def _query_llm(self, messages, quiet: bool = False):
        """Query LLM backend."""
        input_token_count = self.llm.num_tokens_from_messages(messages=messages, quiet=True)
        leftover_tokens_count = calculate_leftover_tokens(
            self.max_context_length, input_token_count
        )

        response = self.llm.chat_completion(
            messages=messages,
            temperature=self.temperature,
            max_tokens=leftover_tokens_count,
            quiet=quiet,
        )

        return response

    def _query_dllm(self, messages, function, quiet: bool = False):
        """Query DLLM backend."""
        input_token_count = self.dllm.num_tokens_from_messages(messages=messages, quiet=True)

        # For DLLM, limit leftover tokens to prevent OOM:
        # 1. Maximum 256 tokens (also enforced in request_dllm.py)
        # 2. calculate_leftover_tokens may return 1000 for overflow, so we cap it here
        # 3. The request_dllm.py will further ensure safety margin of 128 tokens
        leftover_tokens_count = min(
            256,
            calculate_leftover_tokens(self.max_context_length, input_token_count)
        )

        response = self.dllm.chat_completion(
            messages=messages,
            temperature=self.temperature,
            max_tokens=leftover_tokens_count,
            quiet=quiet,
        )

        return response

    def _log_inference(self, formatted_prompt, response_json):
        """Log inference to file."""
        logger_dir = os.path.join(os.environ["BFCL_PROJECT_ROOT"], "logger")
        if os.path.exists(logger_dir) is False:
            os.makedirs(logger_dir, exist_ok=True)
        logger_path = os.path.join(
            logger_dir, self.model_name.replace("/", "_") + ".jsonl"
        )
        with open(logger_path, "a", encoding="utf-8") as f:
            json.dump(
                {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "formatted_prompt": formatted_prompt,
                    "response": response_json,
                },
                f,
                ensure_ascii=False,
            )
            f.write("\n")

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        test_category: str = test_entry["id"].rsplit("_", 1)[0]

        functions = func_doc_language_specific_pre_processing(functions, test_category)

        test_entry["question"][0] = system_prompt_pre_processing_chat_model(
            test_entry["question"][0], functions, test_category
        )

        return {"message": [], "function": functions}

    @override
    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        """
        Parse API response into standard format.
        Can be overridden for custom response formats.
        """
        # Check if response is from LLM/DLLM backend (SimpleNamespace with 'text' attribute)
        if hasattr(api_response, 'text'):
            # Response from our backend wrappers (SimpleNamespace)
            return {
                "model_responses": api_response.text,
                "input_token": 0,  # Token counts not available in wrapper
                "output_token": 0,
            }
        else:
            # Response from direct API call (original format with choices)
            return {
                "model_responses": api_response.choices[0].text,
                "input_token": api_response.usage.prompt_tokens,
                "output_token": api_response.usage.completion_tokens,
            }

    @override
    def add_first_turn_message_prompting(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(first_turn_message)
        return inference_data

    @override
    def _add_next_turn_user_message_prompting(
        self, inference_data: dict, user_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(user_message)
        return inference_data

    @override
    def _add_assistant_message_prompting(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        inference_data["message"].append(
            {"role": "assistant", "content": model_response_data["model_responses"]}
        )
        return inference_data

    @override
    def _add_execution_results_prompting(
        self,
        inference_data: dict,
        execution_results: list[str],
        model_response_data: dict,
    ) -> dict:
        """
        Add execution results to messages.
        Can be overridden for custom formats (e.g., in LLM variants).
        """
        for execution_result, decoded_model_response in zip(
            execution_results, model_response_data["model_responses_decoded"]
        ):
            inference_data["message"].append(
                {
                    "role": "tool",
                    "name": decoded_model_response,
                    "content": execution_result,
                }
            )

        return inference_data

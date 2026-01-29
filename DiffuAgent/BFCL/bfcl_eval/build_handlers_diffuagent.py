"""
DiffuAgent Model Configuration Builder

This module builds model configurations for all DiffuAgent variants.

Handler Naming:
{Backend}{Feature}Handler

Example Usage:
    from bfcl_eval.build_handlers_diffuagent import add_diffuagent_model_configs
    configs = add_diffuagent_model_configs()
"""

from bfcl_eval.constants.model_config import ModelConfig
from bfcl_eval.model_handler.api_inference.diffuagent import (
    LLMHandler,
    SelectorLLMHandler,
    EditorLLMHandler,
    SelectorEditorLLMHandler,
    DLLMHandler,
    SelectorDLLMHandler,
    EditorDLLMHandler,
    SelectorEditorDLLMHandler,
)


def build_config(config_name: str, url: str, model_handler) -> ModelConfig:
    """Build a single ModelConfig."""
    return ModelConfig(
        model_name=config_name,
        display_name=config_name,
        url=url,
        org="External LLM",
        license="Apache 2.0",
        model_handler=model_handler,
        input_price=None,
        output_price=None,
        is_fc_model=False,
        underscore_to_dot=False,
    )


def add_diffuagent_model_configs():
    """
    Build all DiffuAgent model configurations.

    Returns:
        dict: Mapping of config names to ModelConfig objects
    """
    # LLM models
    LLM_MODELS = {
        "ministral": {
            "display_name": "ministral-8b",
            "url": "https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512",
        },
        "qwen": {
            "display_name": "qwen3-8b",
            "url": "https://huggingface.co/Qwen/Qwen3-8B",
        },
    }

    # Agent settings -> handler mapping
    AGENT_HANDLERS = {
        "chatbase": {
            "llm": LLMHandler,
            "dllm": DLLMHandler,
        },
        "selector-chatbase": {
            "llm": SelectorLLMHandler,
            "dllm": SelectorDLLMHandler,
        },
        "editor-chatbase": {
            "llm": EditorLLMHandler,
            "dllm": EditorDLLMHandler,
        },
        "selector-editor-chatbase": {
            "llm": SelectorEditorLLMHandler,
            "dllm": SelectorEditorDLLMHandler,
        },
    }

    # DLLM variants (wedlm removed as requested)
    DLLM_VARIANTS = ["llada", "dream", "fdllmv2", "dllmvar"]

    configs = {}

    # Generate configurations
    for setting_key, handlers in AGENT_HANDLERS.items():
        prefix = f"diffuagent-{setting_key}"

        # LLM-only: "{prefix}/{llm_display_name}"
        for llm_info in LLM_MODELS.values():
            config_name = f"{prefix}/{llm_info['display_name']}"
            configs[config_name] = build_config(
                config_name=config_name,
                url=llm_info["url"],
                model_handler=handlers["llm"],
            )

        # LLM + DLLM: "{prefix}/{llm_display_name}-{dllm_variant}"
        # Uses dllm handler (main agent = LLM, features = DLLM)
        for llm_info in LLM_MODELS.values():
            for dllm_variant in DLLM_VARIANTS:
                config_name = f"{prefix}/{llm_info['display_name']}-{dllm_variant}"
                configs[config_name] = build_config(
                    config_name=config_name,
                    url=llm_info["url"],
                    model_handler=handlers["dllm"],
                )

        # Pure DLLM: "{prefix}/{dllm_variant}"
        # For all settings (chatbase, selector-chatbase, editor-chatbase, selector-editor-chatbase)
        # Uses the same handler class (e.g., SelectorDLLMHandler for selector-chatbase)
        for dllm_variant in DLLM_VARIANTS:
            config_name = f"{prefix}/{dllm_variant}"
            configs[config_name] = build_config(
                config_name=config_name,
                url="",
                model_handler=handlers["dllm"],
            )

    print(f"Generated {len(configs)} DiffuAgent configurations")
    return configs

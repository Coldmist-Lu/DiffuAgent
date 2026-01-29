"""
DiffuAgent Handlers Module

This module provides unified handler classes for all DiffuAgent variants.

Architecture:
- base.py: Unified base handler (DiffuagentBaseHandler)
- mixins.py: Feature mixins (SelectorMixin, EditorMixin, SelectorEditorMixin)
- handlers.py: All handler variants composed from base + mixins

Handler Classes:
    LLMHandler: Base LLM handler
    SelectorLLMHandler: LLM + function selection
    EditorLLMHandler: LLM + format editing
    SelectorEditorLLMHandler: LLM + both features

    DLLMHandler: Base DLLM handler
    SelectorDLLMHandler: DLLM + function selection
    EditorDLLMHandler: DLLM + format editing
    SelectorEditorDLLMHandler: DLLM + both features

Usage:
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
"""

from bfcl_eval.model_handler.api_inference.diffuagent.handlers import (
    LLMHandler,
    SelectorLLMHandler,
    EditorLLMHandler,
    SelectorEditorLLMHandler,
    DLLMHandler,
    SelectorDLLMHandler,
    EditorDLLMHandler,
    SelectorEditorDLLMHandler,
)

__all__ = [
    "LLMHandler",
    "SelectorLLMHandler",
    "EditorLLMHandler",
    "SelectorEditorLLMHandler",
    "DLLMHandler",
    "SelectorDLLMHandler",
    "EditorDLLMHandler",
    "SelectorEditorDLLMHandler",
]

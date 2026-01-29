import os
from transformers import AutoConfig, AutoTokenizer
from typing import Any, Dict, List

def normalize_tools_schema(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert "simple tools schema" -> "OpenAI-like tools schema".

    Input (your current):
      [
        {
          "name": "find_concert",
          "description": "...",
          "parameters": {"type": "dict", "properties": {...}, "required": [...]}
        }
      ]

    Output (normalized):
      [
        {
          "type": "function",
          "function": {
            "name": "find_concert",
            "description": "...",
            "parameters": {"type": "object", "properties": {...}, "required": [...]}
          }
        }
      ]

    Also:
    - If a tool is already in {"type":"function","function":{...}} format, it will be kept,
      but parameters.type "dict" will still be normalized to "object".
    - Ensures `parameters` exists; if missing, creates a minimal object schema.
    """
    if not isinstance(tools, list):
        raise TypeError(f"`tools` must be a list, got {type(tools).__name__}")

    normalized: List[Dict[str, Any]] = []

    for i, t in enumerate(tools):
        if not isinstance(t, dict):
            raise TypeError(f"tools[{i}] must be dict, got {type(t).__name__}")

        # Case 1: already OpenAI-like
        if t.get("type") == "function" and isinstance(t.get("function"), dict):
            fn = dict(t["function"])  # shallow copy
            params = fn.get("parameters")
            if not isinstance(params, dict):
                params = {"type": "object", "properties": {}, "required": []}
            else:
                params = dict(params)
                # normalize jsonschema "type"
                if params.get("type") == "dict":
                    params["type"] = "object"
                elif "type" not in params:
                    params["type"] = "object"

            fn["parameters"] = params
            normalized.append({"type": "function", "function": fn})
            continue

        # Case 2: your simple schema
        name = t.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"tools[{i}].name must be a non-empty string")

        description = t.get("description", "")
        if description is None:
            description = ""
        if not isinstance(description, str):
            raise TypeError(f"tools[{i}].description must be a string")

        params = t.get("parameters")
        if params is None:
            params = {"type": "object", "properties": {}, "required": []}
        if not isinstance(params, dict):
            raise TypeError(f"tools[{i}].parameters must be a dict")

        params = dict(params)  # copy
        if params.get("type") == "dict":
            params["type"] = "object"
        elif "type" not in params:
            params["type"] = "object"

        normalized.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": params,
                },
            }
        )

    return normalized

def calculate_leftover_tokens(max_context_length: int, input_token_count: int) -> int:
    """
    根据已有输入 token 数量和最大上下文长度，计算还能请求的 token 数量。
    默认最多请求 4096 个 token，如果输入已经超过上下文长度，返回 1000。

    参数:
        max_context_length: 模型最大上下文长度
        input_token_count: 当前输入 token 数量

    返回:
        leftover_tokens_count: 可请求的 token 数量
    """
    if max_context_length < input_token_count + 2:
        leftover_tokens_count = 1000
    else:
        leftover_tokens_count = min(
            4096,
            max_context_length - input_token_count - 2,
        )
    return leftover_tokens_count

def resolve_model_and_context_length(
    *,
    local_model_path: str | None,
    model_name_huggingface: str,
    default_context_length: int = 262144,
    trust_remote_code: bool = True,
):
    """
    Resolve model path / id, load tokenizer & config if possible,
    and robustly determine max context length.

    Returns
    -------
    model_path_or_id : str
    tokenizer        : AutoTokenizer | None
    max_context_len  : int
    """

    # -------- 1. Resolve model source --------
    if local_model_path is not None:
        if not os.path.isdir(local_model_path):
            raise ValueError(
                f"local_model_path '{local_model_path}' does not exist or is not a directory."
            )

        required_files = ["config.json", "tokenizer_config.json"]
        for file_name in required_files:
            if not os.path.exists(os.path.join(local_model_path, file_name)):
                raise ValueError(
                    f"Required file '{file_name}' not found in '{local_model_path}'."
                )

        model_path_or_id = local_model_path
        load_kwargs = dict(
            pretrained_model_name_or_path=model_path_or_id,
            local_files_only=False,
            trust_remote_code=trust_remote_code,
        )
    else:
        model_path_or_id = model_name_huggingface
        load_kwargs = dict(
            pretrained_model_name_or_path=model_path_or_id,
            trust_remote_code=trust_remote_code,
        )

    # -------- 2. Robust context length resolution --------
    tokenizer = None
    max_context_length = default_context_length

    try:
        # tokenizer 是最优先信号
        tokenizer = AutoTokenizer.from_pretrained(**load_kwargs)
        if getattr(tokenizer, "model_max_length", None):
            if tokenizer.model_max_length < 10**8:
                max_context_length = tokenizer.model_max_length
    except Exception:
        pass  # 允许失败，继续

    try:
        # config 是第二优先信号
        config = AutoConfig.from_pretrained(**load_kwargs)

        for attr in (
            "max_position_embeddings",
            "seq_length",
            "n_positions",
            "context_length",
        ):
            if hasattr(config, attr):
                value = getattr(config, attr)
                if isinstance(value, int) and value > 0:
                    max_context_length = value
                    break
    except Exception:
        pass  # 允许失败，继续

    return model_path_or_id, tokenizer, max_context_length

import json
import re
from typing import Any, Dict, List, Tuple

# toolcall transfer

_TOOL_CALL_RE = re.compile(
    r"""
    (?P<name>[A-Za-z_][A-Za-z0-9_]*)      # function name
    \s*\(\s*
    (?P<args>\{.*?\})                    # JSON object args
    \s*\)
    """,
    re.VERBOSE | re.DOTALL,
)

def _py_literal(v: Any) -> str:
    """Convert JSON value to a Python-literal string."""
    if v is None:
        return "None"
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        # json.dumps gives valid escaping with double quotes
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, list):
        return "[" + ", ".join(_py_literal(x) for x in v) + "]"
    if isinstance(v, dict):
        # dict literal with JSON-style keys
        items = ", ".join(f"{json.dumps(k, ensure_ascii=False)}: {_py_literal(val)}" for k, val in v.items())
        return "{" + items + "}"
    # fallback
    return json.dumps(str(v), ensure_ascii=False)

def ministral_toolcalls_to_bfcl(text: str) -> str:
    """
    Convert Ministral outputs like:
      [TOOL_CALLS]find_concert({"location": "Chicago, IL", "price": 100, "genre": "Rock"})
    Into BFCL format:
      [find_concert(location="Chicago, IL", price=100, genre="Rock")]
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    # Strip leading tag if present
    s = text.strip()
    if s.startswith("[TOOL_CALLS]"):
        s = s[len("[TOOL_CALLS]"):].strip()

    matches = list(_TOOL_CALL_RE.finditer(s))
    if not matches:
        # If no tool calls detected, return original or empty list depending on your eval needs
        return "[]"

    calls: List[str] = []
    for m in matches:
        name = m.group("name")
        args_json = m.group("args")
        try:
            args: Dict[str, Any] = json.loads(args_json)
        except json.JSONDecodeError:
            # If the model produced invalid JSON, you can choose to fail hard or return []
            return "[]"

        # Preserve key order as emitted (Python 3.7+ keeps insertion order)
        kv = ", ".join(f"{k}={_py_literal(v)}" for k, v in args.items())
        calls.append(f"{name}({kv})")

    return "[" + ", ".join(calls) + "]"
# DiffuAgent Environment Variables

## Overview

DiffuAgent uses a flexible backend architecture where different components can use different backends:
- **Main Agent**: The primary chat completion model
- **Features**: Selector and Editor components

## Environment Variables

### New Naming (Recommended)

Environment variables are named based on **purpose** rather than backend type:

```bash
# Main Agent Backend (used for primary inference)
export MAIN_AGENT_API_KEY="your-api-key"
export MAIN_AGENT_BASE_URL="http://localhost:23456"
export MAIN_AGENT_MODEL_PATH="/path/to/model"  # Optional, for local models

# Features Backend (used for Selector and Editor)
export FEATURES_API_KEY="your-api-key"
export FEATURES_BASE_URL="http://localhost:23450"
```

### Legacy Naming (Still Supported)

For backward compatibility, the old variable names are still supported:

```bash
# LLM Backend (equivalent to MAIN_AGENT_*)
export VLLM_API_KEY="your-api-key"
export VLLM_BASE_URL="http://localhost:23456"
export VLLM_MODEL_PATH="/path/to/model"  # Optional, for local models

# DLLM Backend (equivalent to FEATURES_*)
export DLLM_API_KEY="your-api-key"
export DLLM_BASE_URL="http://localhost:23450"
```

### Priority

If both new and legacy variables are set, **new variables take priority**:
- `MAIN_AGENT_API_KEY` > `VLLM_API_KEY`
- `MAIN_AGENT_BASE_URL` > `VLLM_BASE_URL`
- `MAIN_AGENT_MODEL_PATH` > `VLLM_MODEL_PATH`
- `FEATURES_API_KEY` > `DLLM_API_KEY`
- `FEATURES_BASE_URL` > `DLLM_BASE_URL`

## Configuration Examples

### Example 1: Pure LLM (qwen3-8b)
- Main Agent: LLM
- Features: LLM
```bash
export MAIN_AGENT_BASE_URL="http://localhost:23456"
export MAIN_AGENT_MODEL_PATH="/model/ModelScope/Qwen/Qwen3-8B"
export FEATURES_BASE_URL="http://localhost:23456"  # Same as main agent
```

### Example 2: Mixed Mode (qwen3-8b-llada)
- Main Agent: LLM
- Features: DLLM
```bash
export MAIN_AGENT_BASE_URL="http://localhost:23456"  # For LLM
export MAIN_AGENT_MODEL_PATH="/model/ModelScope/Qwen/Qwen3-8B"
export FEATURES_BASE_URL="http://localhost:23450"     # For DLLM
```

### Example 3: Pure DLLM (llada)
- Main Agent: DLLM
- Features: DLLM
```bash
export MAIN_AGENT_BASE_URL="http://localhost:23450"  # For DLLM
export FEATURES_BASE_URL="http://localhost:23450"    # Same as main agent
```

## Migration Guide

To migrate from legacy to new naming:

```bash
# Old (still works)
export VLLM_API_KEY="your-api-key"
export VLLM_BASE_URL="http://localhost:23456"
export VLLM_MODEL_PATH="/model/ModelScope/Qwen/Qwen3-8B"
export DLLM_API_KEY="your-api-key"
export DLLM_BASE_URL="http://localhost:23450"

# New (recommended)
export MAIN_AGENT_API_KEY="your-api-key"
export MAIN_AGENT_BASE_URL="http://localhost:23456"
export MAIN_AGENT_MODEL_PATH="/model/ModelScope/Qwen/Qwen3-8B"
export FEATURES_API_KEY="your-api-key"
export FEATURES_BASE_URL="http://localhost:23450"

# Or use both - new names take priority
export VLLM_BASE_URL="http://localhost:23456"
export MAIN_AGENT_BASE_URL="http://localhost:23456"  # This one is used
```

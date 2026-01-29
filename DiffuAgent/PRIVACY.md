# Privacy and Configuration Guide

This document explains how to configure DiffuAgent without exposing sensitive information.

## Environment Variables

### Required Environment Variables

All sensitive information (API keys, URLs, paths) should be set via environment variables:

#### BFCL Testing

```bash
# Main Agent Backend (LLM)
export MAIN_AGENT_API_KEY="your-api-key-here"
export MAIN_AGENT_BASE_URL="http://localhost:23456"
export MAIN_AGENT_MODEL_PATH="/path/to/your/model"

# Features Backend (DLLM/LLM for Selector/Editor)
export FEATURES_API_KEY="your-api-key-here"
export FEATURES_BASE_URL="http://localhost:23450"

# BFCL Project Root
export BFCL_PROJECT_ROOT="/workspace/DiffuAgent_TMP/unified_envs/gorilla/berkeley-function-call-leaderboard"
```

#### Legacy Environment Variables (Still Supported)

```bash
# Equivalent to MAIN_AGENT_*
export VLLM_API_KEY="your-api-key-here"
export VLLM_BASE_URL="http://localhost:23456"
export VLLM_MODEL_PATH="/path/to/your/model"

# Equivalent to FEATURES_*
export DLLM_API_KEY="your-api-key-here"
export DLLM_BASE_URL="http://localhost:23450"
```

#### Agentboard

```bash
# LLM Configuration
export VLLM_API_KEY="your-api-key-here"
export VLLM_BASE_URL="http://localhost:23456"
export VLLM_BASE_URL_1="http://localhost:23460"  # Optional
export VLLM_BASE_URL_2="http://localhost:23465"  # Optional

# DLLM Configuration
export DLLM_API_KEY="your-api-key-here"
export DLLM_BASE_URL="http://localhost:23450"
```

## Files Modified for Privacy

### 1. **BFCL/bfcl_eval/model_handler/api_inference/utils/request_dllm.py**
- Removed hardcoded API key
- Now uses: `os.getenv("DLLM_API_KEY") or os.getenv("FEATURES_API_KEY")`
- Base URL uses: `os.getenv("DLLM_BASE_URL") or os.getenv("FEATURES_BASE_URL")`

### 2. **BFCL/tests/test_run_ids.sh**
- Removed hardcoded API keys
- Removed hardcoded workspace path
- Removed hardcoded model path
- All values now use environment variables with fallback defaults:
  ```bash
  export MAIN_AGENT_API_KEY="${MAIN_AGENT_API_KEY:-your-api-key-here}"
  export BFCL_PROJECT_ROOT="${BFCL_PROJECT_ROOT:-/default/path}"
  export MODEL_PATH="${MAIN_AGENT_MODEL_PATH:-/default/model/path}"
  ```

### 3. **Agentboard/llm/enhanced/api_dllm.py**
- Removed hardcoded API key
- Now uses: `os.getenv("DLLM_API_KEY", "your-api-key-here")`
- Base URL uses: `os.getenv("DLLM_BASE_URL", "http://localhost:23450/")`

### 4. **Agentboard/llm/enhanced/api_llm.py**
- Removed hardcoded API keys
- Now uses: `os.getenv("VLLM_API_KEY", "your-api-key-here")`
- Base URLs use: `os.getenv("VLLM_BASE_URL", "http://localhost:23456/")`

## Security Checklist

✅ **No hardcoded API keys in code**
✅ **No hardcoded workspace paths in code**
✅ **No hardcoded personal paths in code**
✅ **All sensitive values use environment variables**
✅ **Fallback defaults are example values only**

## .gitignore

The following files/directories are excluded from version control:

- `.env` - Environment configuration file
- `unified_envs/` - Local working directory
- `__pycache__/` - Python cache files
- `.DS_Store` - macOS system files
- `logger/` - Log files
- `result/` - Test results
- `score/` - Evaluation scores

## Usage Example

```bash
# Set your environment variables
export MAIN_AGENT_API_KEY="sk-..."
export MAIN_AGENT_BASE_URL="http://your-server:23456"
export MAIN_AGENT_MODEL_PATH="/models/qwen3-8b"

export FEATURES_API_KEY="sk-..."
export FEATURES_BASE_URL="http://your-dllm-server:23450"

# Run tests
bash DiffuAgent/BFCL/tests/test_run_ids.sh
```

## Notes

- Default values in code are **examples only** and should be replaced with environment variables
- Never commit actual API keys or credentials to the repository
- Use `.env` files for local development (already in .gitignore)
- For production, use environment variable management (e.g., Kubernetes secrets, AWS Secrets Manager)

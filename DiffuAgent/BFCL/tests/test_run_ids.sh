#!/bin/bash
################################################################################
# Test DiffuAgent with Test Case IDs
################################################################################

source /usr/local/miniconda3/bin/activate bfcl2

# Configuration - modify these paths according to your environment
export BFCL_PROJECT_ROOT="${BFCL_PROJECT_ROOT:-/path/to/DiffuAgent/unified_envs/gorilla/berkeley-function-call-leaderboard}"
export CUDA_VISIBLE_DEVICES="0"

export DEBUG_DIFFUAGENT=0

# Model Configuration
MODEL_PATH="${MAIN_AGENT_MODEL_PATH:-/model/ModelScope/Qwen/Qwen3-8B}"
MODEL_NAME="diffuagent-chatbase/qwen3-8b"

# VLLM Server Configuration
# Set API keys via environment variables or replace with your actual keys
export MAIN_AGENT_API_KEY="${MAIN_AGENT_API_KEY:-your-api-key-here}"
export MAIN_AGENT_BASE_URL="${MAIN_AGENT_BASE_URL:-http://localhost:23456}"
export MAIN_AGENT_MODEL_PATH="$MODEL_PATH"

export FEATURES_API_KEY="${FEATURES_API_KEY:-your-api-key-here}"
export FEATURES_BASE_URL="${FEATURES_BASE_URL:-http://localhost:23450}"

# Change to BFCL directory
cd ${BFCL_PROJECT_ROOT}

echo "=========================================="
echo "Testing Model: ${MODEL_NAME}"
echo "=========================================="

# Step 1: Generate responses
echo "[Step 1] Generating responses..."
bfcl generate \
  --model "$MODEL_NAME" \
  --local-model-path "$MODEL_PATH" \
  --run-ids \
  --allow-overwrite

echo "=========================================="
echo "Test completed!"
echo "=========================================="

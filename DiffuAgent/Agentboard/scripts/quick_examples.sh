#!/bin/bash
# AgentBoard Quick Examples
# This script demonstrates how to use AgentBoard with different models and agent configurations

# Make sure PROJECT_PATH is set
# export PROJECT_PATH=/path/to/AgentBoard

# ==============================================================================
# Part 1: LLM Agent Configuration Examples
# ==============================================================================

echo "========================================"
echo "LLM Agent Configuration Examples"
echo "========================================"

# Example 1: OnePass (Basic ReAct without memory)
echo ""
echo "Example 1: OnePass (Basic ReAct)"
cd agentboard && python eval_modular.py \
    --cfg-path ${PROJECT_PATH}/agentboard/configs/experiments/onepass.yaml \
    --model qwen3 \
    --tasks alfworld_enhanced \
    --max_num_steps 30 \
    --log_path ${PROJECT_PATH}/outputs/qwen3_onepass
cd ..

# Example 2: Memory (ReAct with dynamic memory)
echo ""
echo "Example 2: Memory (ReAct with dynamic memory)"
cd agentboard && python eval_modular.py \
    --cfg-path ${PROJECT_PATH}/agentboard/configs/experiments/memory.yaml \
    --model qwen3 \
    --tasks alfworld_enhanced \
    --max_num_steps 30 \
    --log_path ${PROJECT_PATH}/outputs/qwen3_memory
cd ..

# Example 3: Memory Exit (with strict early stopping)
echo ""
echo "Example 3: Memory Exit (with strict early stopping)"
cd agentboard && python eval_modular.py \
    --cfg-path ${PROJECT_PATH}/agentboard/configs/experiments/memory_exit.yaml \
    --model qwen3 \
    --tasks alfworld_enhanced \
    --max_num_steps 30 \
    --log_path ${PROJECT_PATH}/outputs/qwen3_memory_exit
cd ..

# Example 4: Memory Exit Modest (with modest early stopping)
echo ""
echo "Example 4: Memory Exit Modest (with modest early stopping)"
cd agentboard && python eval_modular.py \
    --cfg-path ${PROJECT_PATH}/agentboard/configs/experiments/memory_exit_modest.yaml \
    --model qwen3 \
    --tasks alfworld_enhanced \
    --max_num_steps 30 \
    --log_path ${PROJECT_PATH}/outputs/qwen3_memory_exit_modest
cd ..

# ==============================================================================
# Part 2: LLM + dLLM Collaboration Examples
# ==============================================================================

echo ""
echo "========================================"
echo "LLM + dLLM Collaboration Examples"
echo "========================================"

# Example 5: Qwen3 as main, Llada as auxiliary
echo ""
echo "Example 5: Qwen3 as main, Llada (dLLM) as auxiliary"
cd agentboard && python eval_modular.py \
    --cfg-path ${PROJECT_PATH}/agentboard/configs/experiments/collaboration_llm_dllm.yaml \
    --model qwen3 \
    --tasks alfworld_enhanced \
    --max_num_steps 30 \
    --log_path ${PROJECT_PATH}/outputs/collaboration_qwen3_llada
cd ..

# Example 6: Ministral as main, Dream as auxiliary
echo ""
echo "Example 6: Ministral as main, Dream (dLLM) as auxiliary"
cd agentboard && python eval_modular.py \
    --cfg-path ${PROJECT_PATH}/agentboard/configs/experiments/collaboration_llm_dllm.yaml \
    --model ministral \
    --tasks alfworld_enhanced \
    --max_num_steps 30 \
    --log_path ${PROJECT_PATH}/outputs/collaboration_ministral_dream
cd ..

echo ""
echo "========================================"
echo "All examples completed!"
echo "========================================"
echo ""
echo "Summary:"
echo "  Main LLM models: qwen3, ministral"
echo "  Auxiliary dLLM models: llada, dream, fdllm, dvar"
echo "  Available agents: onepass, memory, memory_exit, memory_exit_modest"
echo ""
echo "  Architecture: LLM generates actions, dLLM processes memory"
echo ""
echo "Verification format recommendations:"
echo "  - LLM models (qwen3, ministral): use memory_exit (strict)"
echo "  - dLLM models (llada, dream, fdllm, dvar): use memory_exit_modest"
echo ""
echo "Available collaboration configs:"
echo "  - collaboration_llm_dllm.yaml  (LLM main + dLLM auxiliary, uses modest)"
echo "  - collaboration_qwen3_ministral.yaml  (LLM + LLM, for comparison)"
echo "  - collaboration_qwen3_qwen3.yaml  (Same LLM for both)"
echo ""
echo "Available tasks: alfworld_enhanced, scienceworld_enhanced, babyai_enhanced"
echo "For more information, see configs/base/llms.yaml and configs/base/agents.yaml"

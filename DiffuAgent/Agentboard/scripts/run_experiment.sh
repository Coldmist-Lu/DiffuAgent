#!/bin/bash
# AgentBoard Experiment Runner
# This script provides a convenient interface for running experiments

set -e  # Exit on error

# Default values
EXPERIMENT_CONFIG=""
MODEL=""
TASKS="all"
MAX_STEPS=30
USE_WANDB=false
LOG_PATH=""
PROJECT_NAME=""

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Run AgentBoard experiments with modular or legacy configs.

Options:
    -c, --config PATH       Experiment config file (required)
    -m, --model NAME        Model name from config (required)
    -t, --tasks TASKS       Tasks to run (default: all)
                            Available: alfworld, scienceworld, babyai
                            Use "all" for all tasks
    -s, --max_steps NUM     Maximum steps per episode (default: 30)
    -w, --wandb             Enable wandb logging
    -l, --log_path PATH     Output log path
    -p, --project NAME      Wandb project name
    --legacy                Force legacy config format
    -h, --help              Display this help message

Examples:
    # Run Qwen3 with memory on all tasks
    $0 --config configs/experiments/qwen3_memory_all.yaml --model qwen3

    # Run Dream on AlfWorld only with 40 steps
    $0 --config configs/experiments/dream_onepass_alfworld.yaml \\
       --model dream --tasks alfworld --max_steps 40

    # Run with custom log path
    $0 --config configs/experiments/ministral_memory_exit_modest.yaml \\
       --model ministral --log_path \$PROJECT_PATH/outputs/my_exp

    # Run with legacy config
    $0 --config scripts/config_old.yaml --model qwen3 --legacy

Available Models (from base/llms.yaml):
    - qwen3 (Standard LLM - Qwen3-8B)
    - ministral (Standard LLM - Ministral-3-8B-Instruct)
    - llada (DiffusionLLM - Llada)
    - dream (DiffusionLLM - Dream)
    - fdllm (DiffusionLLM - FastDLLM-v2)
    - dvar (DiffusionLLM - DiffusionLLM-Var)

Available Agents (from base/agents.yaml):
    - onepass (Basic ReAct without memory)
    - memory (ReAct with dynamic memory)
    - memory_exit (ReAct with memory + strict early exit)
    - memory_exit_modest (ReAct with memory + modest early exit)
    - history_exit (History replay + early exit, for offline testing)

EOF
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            EXPERIMENT_CONFIG="$2"
            shift 2
            ;;
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -t|--tasks)
            TASKS="$2"
            shift 2
            ;;
        -s|--max_steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        -w|--wandb)
            USE_WANDB=true
            shift
            ;;
        -l|--log_path)
            LOG_PATH="$2"
            shift 2
            ;;
        -p|--project)
            PROJECT_NAME="$2"
            shift 2
            ;;
        --legacy)
            USE_LEGACY="--legacy"
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Error: Unknown option $1"
            usage
            ;;
    esac
done

# Check required arguments
if [[ -z "$EXPERIMENT_CONFIG" ]]; then
    echo "Error: --config is required"
    usage
fi

if [[ -z "$MODEL" ]]; then
    echo "Error: --model is required"
    usage
fi

# Set PROJECT_PATH from environment if not already set
if [[ -z "$PROJECT_PATH" ]]; then
    echo "Warning: PROJECT_PATH not set, using current directory"
    PROJECT_PATH=$(pwd)
fi

# Build command
# Note: In Git repo structure, eval_modular.py is in the root directory
# Run from the agentboard directory using PROJECT_PATH
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CMD="cd ${PROJECT_PATH}/agentboard && python eval_modular.py"
CMD="$CMD --cfg-path ${PROJECT_PATH}/agentboard/configs/experiments/$(basename $EXPERIMENT_CONFIG)"
CMD="$CMD --model $MODEL"
CMD="$CMD --tasks $TASKS"
CMD="$CMD --max_num_steps $MAX_STEPS"

if [[ "$USE_WANDB" == "true" ]]; then
    CMD="$CMD --wandb"
fi

if [[ -n "$LOG_PATH" ]]; then
    CMD="$CMD --log_path $LOG_PATH"
fi

if [[ -n "$PROJECT_NAME" ]]; then
    CMD="$CMD --project_name $PROJECT_NAME"
fi

if [[ -n "$USE_LEGACY" ]]; then
    CMD="$CMD $USE_LEGACY"
fi

# Display experiment info
echo "=========================================="
echo "AgentBoard Experiment Runner"
echo "=========================================="
echo "Config:    $EXPERIMENT_CONFIG"
echo "Model:     $MODEL"
echo "Tasks:     $TASKS"
echo "Max Steps: $MAX_STEPS"
echo "Wandb:     $USE_WANDB"
if [[ -n "$LOG_PATH" ]]; then
    echo "Log Path:  $LOG_PATH"
fi
echo "=========================================="
echo ""

# Run command
eval $CMD

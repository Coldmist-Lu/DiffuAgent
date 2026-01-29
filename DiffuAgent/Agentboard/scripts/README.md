# AgentBoard Scripts

This directory contains scripts for running AgentBoard experiments.

## Quick Start

The easiest way to get started is to use `quick_examples.sh`:

```bash
export PROJECT_PATH=/path/to/AgentBoard
cd scripts
./quick_examples.sh
```

## Available Scripts

### `quick_examples.sh`
Demonstrates the core functionality of AgentBoard with 6 examples:

**LLM Examples:**
1. **LLM + OnePass**: Basic ReAct without memory
2. **LLM + Memory**: ReAct with dynamic memory
3. **LLM + Memory + EarlyExit**: ReAct with memory and early stopping

**DLLM Examples:**
4. **DLLM + OnePass**: Basic ReAct without memory
5. **DLLM + Memory**: ReAct with dynamic memory
6. **DLLM + Memory + EarlyExit**: ReAct with memory and early stopping

This script shows that:
- Any LLM (Qwen3, GPT, etc.) can use: `onepass`, `memory`, `memory_exit`
- Any DLLM (Dream, Llada, FastDLLM, etc.) can use: `onepass`, `memory`, `memory_exit`

### `run_experiment.sh`
Advanced experiment runner with command-line arguments:

```bash
./run_experiment.sh \
  --config configs/experiments/qwen3_memory_all.yaml \
  --model qwen3 \
  --tasks alfworld \
  --max_steps 30
```

## Configuration Files

Experiment configurations are stored in `../configs/experiments/`:

### LLM Configurations
- `llm_onepass.yaml` - Basic ReAct for standard LLMs
- `llm_memory.yaml` - ReAct with memory for standard LLMs
- `llm_memory_exit.yaml` - ReAct with memory + early exit for standard LLMs

### DLLM Configurations
- `dllm_onepass.yaml` - Basic ReAct for DiffusionLLMs
- `dllm_memory.yaml` - ReAct with memory for DiffusionLLMs
- `dllm_memory_exit.yaml` - ReAct with memory + early exit for DiffusionLLMs

### Custom Configurations
You can create your own experiment configs by referencing models from `configs/base/llms.yaml` and agents from `configs/base/agents.yaml`.

## Example: Creating a Custom Experiment

1. Create a new config file in `configs/experiments/my_experiment.yaml`:

```yaml
llm: [qwen3]
agent: memory_exit
env: [alfworld_enhanced, scienceworld_enhanced]

run:
  max_num_steps: 30
  wandb: false
  log_path: ${PROJECT_PATH}/outputs/my_experiment
```

2. Run it:

```bash
cd agentboard
python eval_modular.py \
  --cfg-path ../configs/experiments/my_experiment.yaml \
  --model qwen3 \
  --tasks all
```

## Available Models

See `configs/base/llms.yaml` for the complete list:
- **LLMs**: qwen3, ministral, gpt
- **DLLMs**: dream, llada, fastdllm, dllmvar

## Available Agents

See `configs/base/agents.yaml` for the complete list:
- **onepass**: Basic ReAct without memory
- **memory**: ReAct with dynamic memory
- **memory_exit**: ReAct with memory + strict early exit
- **memory_exit_modest**: ReAct with memory + modest early exit

## Available Environments

- **alfworld**: Original ALFWorld tasks
- **alfworld_enhanced**: ALFWorld with enhanced logging
- **scienceworld**: Original ScienceWorld tasks
- **scienceworld_enhanced**: ScienceWorld with enhanced logging
- **babyai**: Original BabyAI tasks
- **babyai_enhanced**: BabyAI with enhanced logging

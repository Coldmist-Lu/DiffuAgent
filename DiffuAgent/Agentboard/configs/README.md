# AgentBoard Modular Configuration System

A modular, maintainable configuration system for AgentBoard experiments.

## Overview

The modular config system separates concerns into three layers:
- **Base Configs**: Reusable definitions for LLMs, agents, and environments
- **Experiment Configs**: Minimal configs that reference base definitions
- **Presets**: Pre-configured combinations for common scenarios

This approach eliminates duplication and makes experiments easy to create and modify.

## Directory Structure

```
configs/
├── base/
│   ├── llms.yaml      # All LLM model definitions
│   ├── agents.yaml    # All agent presets
│   └── envs.yaml      # All environment definitions
├── experiments/
│   ├── qwen3_memory_all.yaml
│   ├── ministral_memory_exit_modest.yaml
│   └── ...            # Your experiment configs
└── README.md
```

## Quick Start

### 1. Define Your Experiment

Create a minimal experiment config file:

```yaml
# configs/experiments/my_experiment.yaml
llm: qwen3              # Which LLM to use
agent: memory           # Which agent preset
env: [alfworld, scienceworld]  # Which environments

run:
  max_num_steps: 30
  log_path: ${PROJECT_PATH}/outputs/my_exp
```

### 2. Run Your Experiment

```bash
# Using the wrapper script (recommended)
./scripts/run_experiment.sh \
    --config configs/experiments/my_experiment.yaml \
    --model qwen3

# Or directly with Python
python eval_modular.py \
    --cfg-path configs/experiments/my_experiment.yaml \
    --model qwen3 \
    --tasks all
```

## Available Components

### LLM Models (in `base/llms.yaml`)

| Name | Type | Description |
|------|------|-------------|
| `ministral` | api_llm | Ministral-3-8B-Instruct-2512 |
| `qwen3` | api_llm | Qwen3-8B |
| `llada` | api_dllm | Llada DiffusionLLM |
| `dream` | api_dllm | Dream DiffusionLLM |
| `fastdllm` | api_dllm | FastDLLM-v2 |
| `dllmvar` | api_dllm | DiffusionLLM-Var |

### Agent Presets (in `base/agents.yaml`)

| Name | Description | Key Features |
|------|-------------|--------------|
| `onepass` | Basic ReAct agent | No memory, no early exit |
| `memory` | ReAct with dynamic memory | LLM-based memory summarization |
| `memory_exit` | Memory + strict early exit | Verification every 5 steps |
| `memory_exit_modest` | Memory + modest early exit | More lenient verification |
| `history_exit` | History replay + early exit | For offline testing only |

### Environments (in `base/envs.yaml`)

| Name | Description |
|------|-------------|
| `alfworld` | Household tasks |
| `scienceworld` | Science tasks |
| `babyai` | Navigation tasks |

## Experiment Config Examples

### Example 1: Single Task, Single Model

```yaml
# configs/experiments/dream_alfworld.yaml
llm: dream
agent: onepass
env: alfworld

run:
  max_num_steps: 40
  log_path: ${PROJECT_PATH}/outputs/dream_alfworld
```

Run it:
```bash
python eval_modular.py \
    --cfg-path configs/experiments/dream_alfworld.yaml \
    --model dream \
    --tasks alfworld
```

### Example 2: Multiple Tasks

```yaml
# configs/experiments/qwen3_all_tasks.yaml
llm: qwen3
agent: memory
env: [alfworld, scienceworld, babyai]

run:
  max_num_steps: 30
  log_path: ${PROJECT_PATH}/outputs/qwen3_all
```

### Example 3: Custom Agent Configuration

Override preset parameters:

```yaml
# configs/experiments/custom_memory.yaml
llm: qwen3

# Use preset but override specific parameters
agent:
  preset: memory_exit
  overrides:
    stored_memory_max: 16  # Override default 12
    verification_iter: 8   # Override default 5

env: [alfworld, scienceworld]

run:
  max_num_steps: 30
```

### Example 4: Multiple Models (Advanced)

```yaml
# configs/experiments/compare_llms.yaml
llm: [qwen3, dream]  # Test multiple LLMs
agent: memory
env: [alfworld]

run:
  max_num_steps: 30
  log_path: ${PROJECT_PATH}/outputs/llm_comparison
```

Run with specific model:
```bash
python eval_modular.py \
    --cfg-path configs/experiments/compare_llms.yaml \
    --model qwen3  # Specify which model from list
```

## Adding New Components

### Add a New LLM

Edit `configs/base/llms.yaml`:

```yaml
my_new_llm:
  name: api_llm  # or api_dllm
  engine: /path/to/model
  temperature: 0.1
  max_tokens: 128
  context_length: 32000
  return_token: true
```

Then use it:
```yaml
llm: my_new_llm
```

### Add a New Agent Preset

Edit `configs/base/agents.yaml`:

```yaml
my_custom_agent:
  name: ReactMemoryExit
  stored_memory_max: 20
  update_num: 10
  verification_iter: 3
  verification_format: modest
```

### Add a New Environment

Edit `configs/base/envs.yaml`:

```yaml
my_env:
  name: my_env
  # ... environment-specific config
  label_path: ${PROJECT_PATH}/data/my_env/test.jsonl
  init_prompt_path: ${PROJECT_PATH}/prompts/my_env.json
```

## Command-Line Options

```bash
python eval_modular.py \
    --cfg-path CONFIG_FILE       # Path to experiment config \
    --model MODEL_NAME           # Model name from config \
    --tasks TASK1 TASK2 ...     # Tasks to run (or "all") \
    --max_num_steps 30          # Max steps per episode \
    --log_path PATH             # Output directory \
    --wandb                     # Enable Weights & Biases \
    --project_name NAME         # Wandb project name \
    --legacy                    # Force legacy config format
```

## Using the Wrapper Script

The `run_experiment.sh` script provides a convenient interface:

```bash
./scripts/run_experiment.sh \
    --config configs/experiments/qwen3_memory_all.yaml \
    --model qwen3 \
    --tasks "alfworld scienceworld" \
    --max_steps 30 \
    --log_path ${PROJECT_PATH}/outputs/my_exp
```

Show help:
```bash
./scripts/run_experiment.sh --help
```

## Backward Compatibility

The system **fully supports legacy configs** (old single-file configs). Just use the `--legacy` flag or let the script auto-detect:

```bash
# Auto-detection (default)
python eval_modular.py --cfg-path scripts/old_config.yaml --model qwen3

# Force legacy mode
python eval_modular.py --cfg-path scripts/old_config.yaml --model qwen3 --legacy
```

## Best Practices

1. **Keep experiment configs minimal**: Only specify what differs from defaults
2. **Use descriptive names**: Name experiment configs like `{model}_{agent}_{tasks}.yaml`
3. **Version control**: Commit experiment configs to track what you ran
4. **Document experiments**: Add comments in YAML files explaining the purpose
5. **Use presets**: Don't repeat agent configurations, use presets from `base/agents.yaml`

## Migration from Legacy Configs

To migrate a legacy config to modular format:

1. Extract LLM definitions to `base/llms.yaml`
2. Extract environment definitions to `base/envs.yaml`
3. Create agent preset in `base/agents.yaml` if needed
4. Create minimal experiment config referencing the above

Example transformation:

**Legacy config** (139 lines, lots of duplication):
```yaml
llm:
  qwen3-8b:
    name: api_request
    engine: /model/Qwen3-8B
    ...
  mistral-24b:
    name: api_request
    engine: /model/Mistral-24B
    ...

agent:
  name: ReactAgentMemoryNew
  stored_memory_max: 12
  ...

env:
  alfworld:
    name: alfworld
    ...
  scienceworld:
    name: scienceworld
    ...
```

**Modular config** (8 lines, no duplication):
```yaml
llm: qwen3
agent: memory
env: [alfworld, scienceworld]
run:
  max_num_steps: 30
```

The modular approach is **17x shorter** and eliminates all duplication!

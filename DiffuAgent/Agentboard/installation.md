# Installation Guide

This guide provides step-by-step instructions for setting up the DiffuAgent Agentboard components.

## Prerequisites

- Git
- Python 3.8+
- wget (for downloading data)

## Step 1: Clone this repository

```bash
git clone https://github.com/Coldmist-Lu/DiffuAgent.git
cd DiffuAgent
```

## Step 2: Create unified_envs working directory

```bash
mkdir -p /path/to/unified_envs
cd /path/to/unified_envs
```

## Step 3: Sparse checkout AgentBoard

Using modern sparse checkout method to download only the `agentboard` folder from the original AgentBoard repository:

```bash
# Clone with sparse filter (partial clone)
git clone --depth 1 --filter=blob:none --sparse https://github.com/hkust-nlp/AgentBoard.git
cd AgentBoard

# Set sparse checkout to only include agentboard folder
git sparse-checkout set agentboard
```

This downloads only the `agentboard` folder (~36MB) instead of the full repository.

## Step 4: Download required data

Download and extract the data files needed for AgentBoard:

```bash
# Download data from HuggingFace
wget https://huggingface.co/datasets/hkust-nlp/agentboard/resolve/main/data.tar.gz

# Extract to agentboard directory
tar -xzvf data.tar.gz -C ./agentboard/

# Optional: Remove the downloaded tar file to save space
rm data.tar.gz
```

## Step 5: Merge with DiffuAgent enhanced code

Now you have the base AgentBoard code in `/path/to/unified_envs/AgentBoard/agentboard/`. Merge it with the enhanced code from `DiffuAgent/Agentboard/`:

```bash
# Set PROJECT_PATH
export PROJECT_PATH=/path/to/unified_envs/AgentBoard

# Simple one-command merge: copy all enhanced code over base code
# Note: current directory should be /path/to/unified_envs/AgentBoard/
cp -r /path/to/DiffuAgent/Agentboard/* ./agentboard/
```

This merges enhanced agents, LLMs, tasks, prompts, configs, and scripts with the base AgentBoard code.

## Step 6: Manual Configuration Fixes

⚠️ **IMPORTANT**: If using AlfWorld environment, you need to fix relative paths in the base configuration file.

Edit `agentboard/environment/alfworld/base_config.yaml` and convert relative paths to absolute paths by prepending the full path to your AgentBoard directory.

**Example**: Change `./data/alfworld/` → `/path/to/unified_envs/AgentBoard/agentboard/data/alfworld/`

## Running Experiments

After setup, you can run experiments using the provided scripts:

```bash
cd ${PROJECT_PATH}/agentboard

# Run a quick example
bash scripts/quick_examples.sh

# Or run with specific config
python eval_modular.py \
    --cfg-path ${PROJECT_PATH}/agentboard/configs/experiments/onepass.yaml \
    --model qwen3 \
    --tasks alfworld_enhanced \
    --max_num_steps 30 \
    --log_path ${PROJECT_PATH}/outputs/qwen3_onepass
```

## Next Steps

- See `configs/README.md` for configuration details
- See `scripts/README.md` for usage examples
- See `prompts/` directory for available prompt templates

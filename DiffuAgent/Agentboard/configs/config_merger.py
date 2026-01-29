"""
Modular Config Merger for AgentBoard

This utility merges base configuration files with experiment-specific configs
to create complete, ready-to-use configurations.

Usage:
    from configs.config_merger import load_merged_config

    # Load and merge all configs
    config = load_merged_config("experiments/my_experiment.yaml")

    # Access merged config
    llm_config = config["llm"]["qwen3"]
    agent_config = config["agent"]
    env_config = config["env"]["alfworld"]
"""
import os
import yaml
import re
from typing import Dict, Any


def path_constructor(loader, node):
    """YAML constructor for environment variable expansion."""
    path_matcher = re.compile(r'\$\{([^}^{]+)\}')
    value = node.value
    match = path_matcher.match(value)
    if match:
        env_var = match.group()[2:-1]
        return os.environ.get(env_var, "") + value[match.end():]
    return value


def deep_merge(base: Dict, override: Dict) -> Dict:
    """
    Deep merge two dictionaries.

    Args:
        base: Base dictionary
        override: Dictionary with overrides

    Returns:
        Merged dictionary
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def load_yaml_file(file_path: str) -> Dict[str, Any]:
    """
    Load a YAML file with environment variable expansion.

    Args:
        file_path: Path to YAML file

    Returns:
        Parsed YAML dictionary
    """
    # Add environment variable support
    path_matcher = re.compile(r'\$\{([^}^{]+)\}')
    yaml.add_implicit_resolver('!path', path_matcher)
    yaml.add_constructor('!path', path_constructor)

    with open(file_path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_merged_config(experiment_config_path: str, base_dir: str = None) -> Dict[str, Any]:
    """
    Load and merge base configurations with experiment-specific config.

    Args:
        experiment_config_path: Path to experiment config file
        base_dir: Directory containing base configs (default: configs/base/)

    Returns:
        Merged configuration dictionary

    Example experiment config:
        # experiments/qwen3_memory_alfworld.yaml
        llm: [qwen3]
        agent: memory
        env: [alfworld, scienceworld]
        run:
          max_num_steps: 30
          log_path: ${PROJECT_PATH}/outputs/my_experiment
    """
    if base_dir is None:
        # Default base_dir relative to experiment config
        experiment_dir = os.path.dirname(experiment_config_path)
        base_dir = os.path.join(experiment_dir, "..", "base")
        base_dir = os.path.abspath(base_dir)

    # Load experiment config
    exp_config = load_yaml_file(experiment_config_path)

    # Initialize merged config
    merged = {
        "llm": {},
        "agent": {},
        "env": {},
        "run": {}
    }

    # Load LLM configs
    if "llm" in exp_config:
        llms_config = load_yaml_file(os.path.join(base_dir, "llms.yaml"))
        llm_list = exp_config["llm"]

        if isinstance(llm_list, str):
            llm_list = [llm_list]

        for llm_name in llm_list:
            if llm_name in llms_config:
                merged["llm"][llm_name] = llms_config[llm_name]
            else:
                print(f"Warning: LLM '{llm_name}' not found in base config")

    # Load agent config
    if "agent" in exp_config:
        agents_config = load_yaml_file(os.path.join(base_dir, "agents.yaml"))
        agent_preset = exp_config["agent"]

        if isinstance(agent_preset, str):
            # Reference to preset
            if agent_preset in agents_config:
                merged["agent"] = agents_config[agent_preset]
            else:
                print(f"Warning: Agent preset '{agent_preset}' not found in base config")
        elif isinstance(agent_preset, dict):
            # Custom agent config with optional preset reference
            if "preset" in agent_preset:
                preset_name = agent_preset["preset"]
                if preset_name in agents_config:
                    merged["agent"] = deep_merge(
                        agents_config[preset_name],
                        agent_preset.get("overrides", {})
                    )
            else:
                merged["agent"] = agent_preset

    # Load environment configs
    if "env" in exp_config:
        envs_config = load_yaml_file(os.path.join(base_dir, "envs.yaml"))
        env_list = exp_config["env"]

        if isinstance(env_list, str):
            env_list = [env_list]

        for env_name in env_list:
            if env_name in envs_config:
                merged["env"][env_name] = envs_config[env_name]
            else:
                print(f"Warning: Environment '{env_name}' not found in base config")

    # Merge run config
    if "run" in exp_config:
        merged["run"] = deep_merge(
            {
                "max_num_steps": 30,
                "wandb": False,
                "project_name": "eval-test",
                "baseline_dir": "data/baseline_results",
                "log_path": "${PROJECT_PATH}/outs/agentboard"
            },
            exp_config["run"]
        )

    return merged


def save_full_config(merged_config: Dict[str, Any], output_path: str):
    """
    Save merged config to a standalone YAML file.

    This is useful for debugging or for compatibility with the original
    eval_main_lqy.py which expects a single config file.

    Args:
        merged_config: Merged configuration dictionary
        output_path: Path to save the full config
    """
    with open(output_path, "w") as f:
        yaml.dump(merged_config, f, default_flow_style=False, sort_keys=False)

    print(f"Full config saved to: {output_path}")

"""
AgentBoard Evaluation Script with Modular Config Support

This script supports both legacy single-file configs and new modular configs.
It automatically detects which format is being used.

Usage:
    # With modular config (new)
    python eval_modular.py --cfg-path configs/experiments/qwen3_memory_all.yaml --tasks all --model qwen3

    # With legacy config (old, still works)
    python eval_modular.py --cfg-path scripts/config_old.yaml --tasks all --model qwen3
"""
import sys
import os
import re
import wandb
import warnings
import yaml
import json
import argparse
from dotenv import load_dotenv
from tasks import load_task
from llm import load_llm
from utils.logging.agent_logger import AgentLogger
from utils.logging.logger import SummaryLogger

# Import modular config support
try:
    from configs.config_merger import load_merged_config
    MODULAR_CONFIG_AVAILABLE = True
except ImportError:
    MODULAR_CONFIG_AVAILABLE = False
    print("Warning: Modular config system not available. Only legacy configs supported.")


logger = AgentLogger(__name__)
warnings.filterwarnings("ignore")

TASKS = ["alfworld_enhanced", "scienceworld_enhanced", "babyai_enhanced"]


def parse_args():
    parser = argparse.ArgumentParser(description="AgentBoard Evaluation")

    parser.add_argument("--cfg-path", required=True, help="Path to configuration file")
    parser.add_argument("--tasks", required=True, type=str, nargs='+', help="Tasks to run")
    parser.add_argument("--model", required=True, help="Model name from config")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--log_path", default='', help="Output log path")
    parser.add_argument("--project_name", default='', help="Wandb project name")
    parser.add_argument("--baseline_dir", default='', help="Baseline directory for comparison")
    parser.add_argument("--max_num_steps", type=int, default=30, help="Max steps per episode")
    parser.add_argument("--continue_exp", default='', help="Continue from existing results")
    parser.add_argument("--legacy", action="store_true", help="Force legacy config loading")

    args = parser.parse_args()
    return args


def is_modular_config(cfg_path):
    """
    Detect if config file is modular format.

    Modular configs typically have simple keys like 'llm', 'agent', 'env' at top level
    without extensive nested configurations.
    """
    try:
        with open(cfg_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        # Check for modular indicators
        top_keys = set(config.keys())

        # Modular configs have these simple keys
        modular_indicators = {'llm', 'agent', 'env', 'run'}

        # If top level matches modular pattern and is relatively simple
        if top_keys.issubset(modular_indicators):
            # Further check: values should be simple (strings or lists), not deeply nested
            for key, value in config.items():
                if key == 'run' and isinstance(value, dict):
                    continue  # run config can be a dict
                if isinstance(value, dict) and len(value) > 5:
                    return False  # Too complex, likely legacy
            return True

        return False
    except Exception as e:
        logger.warning(f"Error detecting config type: {e}")
        return False


def load_config(cfg_path, args):
    """
    Load configuration (modular or legacy).

    Args:
        cfg_path: Path to config file
        args: Command line arguments

    Returns:
        tuple: (llm_config, agent_config, env_config, run_config, llm_config_all)
    """
    # Determine if modular or legacy
    use_legacy = args.legacy

    if not use_legacy and MODULAR_CONFIG_AVAILABLE:
        use_modular = is_modular_config(cfg_path)
        if use_modular:
            logger.info(f"Loading modular config from: {cfg_path}")
            config = load_merged_config(cfg_path)
        else:
            logger.info(f"Loading legacy config from: {cfg_path}")
            config = load_legacy_config(cfg_path, args)
    else:
        logger.info(f"Loading legacy config from: {cfg_path}")
        config = load_legacy_config(cfg_path, args)

    # Extract configs
    llm_config_all = config.get("llm", {})
    agent_config = config.get("agent", {})
    env_config = config.get("env", {})
    run_config = config.get("run", {})

    # Get specific model config
    llm_config = llm_config_all.get(args.model, {})
    if not llm_config:
        raise ValueError(f"Model '{args.model}' not found in config. Available: {list(llm_config_all.keys())}")

    # Override with command line args
    if args.log_path:
        run_config["log_path"] = args.log_path
    if args.project_name:
        run_config["project_name"] = args.project_name
    if args.baseline_dir:
        run_config["baseline_dir"] = args.baseline_dir
    if args.wandb:
        run_config["wandb"] = True
    if args.max_num_steps:
        run_config["max_num_steps"] = args.max_num_steps

    return llm_config, agent_config, env_config, run_config, llm_config_all


def load_legacy_config(cfg_path, args):
    """Load legacy single-file config format."""
    path_matcher = re.compile(r'\$\{([^}^{]+)\}')
    yaml.add_implicit_resolver('!path', path_matcher)

    def path_constructor(loader, node):
        value = node.value
        match = path_matcher.match(value)
        if match:
            env_var = match.group()[2:-1]
            return os.environ.get(env_var, "") + value[match.end():]
        return value

    yaml.add_constructor('!path', path_constructor)

    with open(cfg_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config


def check_log_paths_are_ready(log_dir, baseline_dir):
    """Ensure log directories exist."""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(os.path.join(log_dir, "logs")):
        os.makedirs(os.path.join(log_dir, "logs"))

    if not os.path.exists(baseline_dir):
        os.makedirs(baseline_dir)

    if not os.path.exists(os.path.join(log_dir, 'all_results.txt')):
        with open(os.path.join(log_dir, 'all_results.txt'), "w") as f:
            f.write("")

    return True


def main():
    load_dotenv()

    PROJECT_PATH = os.environ.get("PROJECT_PATH", "")
    if PROJECT_PATH:
        sys.path.append(PROJECT_PATH)
        # Change working directory to PROJECT_PATH so relative paths in config files work correctly
        os.chdir(PROJECT_PATH)
        logger.info(f"Project path: {PROJECT_PATH}")
        logger.info(f"Working directory changed to: {os.getcwd()}")
    else:
        logger.warning("PROJECT_PATH environment variable not set")

    args = parse_args()
    llm_config, agent_config, env_config, run_config, llm_config_all = load_config(args.cfg_path, args)

    # Load LLM
    logger.info("Loading language model...")
    llm = load_llm(llm_config["name"], llm_config)
    logger.info("Language model loaded")

    # Initialize wandb
    if not run_config.get("wandb", False):
        os.environ["WANDB_MODE"] = "disabled"
        wandb.init(mode="disabled")
        logger.info("Wandb disabled")
    else:
        wandb.init(
            project=run_config.get("project_name", "eval-test"),
            name=f"{llm_config['name']}_{llm_config.get('engine', 'unknown')}",
            notes="AgentBoard evaluation",
            config={
                'llm_config': llm_config,
                'agent_config': agent_config,
                'env_config': env_config,
                'run_config': run_config
            }
        )

    # Setup logging
    log_dir = run_config.get("log_path", "outputs/agentboard")
    baseline_path = run_config.get('baseline_dir', 'data/baseline_results_details')

    assert check_log_paths_are_ready(log_dir, baseline_path)

    agentboard = SummaryLogger(baseline_dir=baseline_path, log_path=log_dir)

    # Load existing results
    log_history = {}
    try:
        for line in open(os.path.join(log_dir, 'all_results.txt'), "r"):
            if line.strip():
                result = json.loads(line.strip())
                if "_summary" not in result:
                    task_name = result.get("task_name", "")
                    if task_name:
                        log_history[task_name] = result
    except FileNotFoundError:
        logger.info("No existing results found")

    logger.info(f"Previously completed tasks: {list(log_history.keys())}")

    # Run evaluation
    task_names = args.tasks if args.tasks != ["all"] else TASKS

    for task_name in task_names:
        if task_name not in env_config:
            logger.warning(f"Task '{task_name}' not in config, skipping")
            continue

        # Skip if already done
        if task_name in log_history:
            logger.info(f"Task {task_name} already evaluated, skipping")
            result = log_history[task_name]
            agentboard.log_run_result(
                task_name,
                result.get("success_rate", 0),
                result.get("progress_rate", 0),
                result.get("grounding_acc", 0),
                result.get("success_rate_hard", 0),
                result.get("success_rate_easy", 0),
                result.get("progress_rate_hard", 0),
                result.get("progress_rate_easy", 0)
            )
            continue

        logger.info(f"Starting task: {task_name}")

        # Configure agent for this task
        agent_task_config = agent_config.copy()
        for key in env_config[task_name]:
            if key in ["check_actions", "check_inventory", "init_prompt_path"]:
                agent_task_config[key] = env_config[task_name][key]

        # Pass llm_config_all if auxiliary_llm is configured
        # This allows the agent to load the auxiliary LLM model
        if "auxiliary_llm" in agent_task_config:
            agent_task_config["llm_config_all"] = {"llm": llm_config_all}
            logger.info(f"Providing llm_config_all to agent (auxiliary_llm: {agent_task_config['auxiliary_llm']})")

        # Load and run task
        if 'tool' in task_name:
            task = load_task('tool', run_config, llm_config, agent_task_config, env_config[task_name], llm=llm)
        else:
            task = load_task(task_name, run_config, llm_config, agent_task_config, env_config[task_name], llm=llm)

        success_rates, progress_rates, grounding_accs, score_state_records, \
            easy_sr, hard_sr, easy_pr, hard_pr = task.evaluate()

        success_rate = sum(success_rates) / len(success_rates) if success_rates else 0
        progress_rate = sum(progress_rates) / len(progress_rates) if progress_rates else 0
        grounding_acc = sum(grounding_accs) / len(grounding_accs) if grounding_accs else 0

        logger.finish(
            f"Task {task_name} | SR: {success_rate:.3f}, PR: {progress_rate:.3f}, "
            f"Easy SR: {easy_sr:.3f}, Hard SR: {hard_sr:.3f}, "
            f"Easy PR: {easy_pr:.3f}, Hard PR: {hard_pr:.3f}, GA: {grounding_acc:.3f}"
        )

        agentboard.log_run_result(task_name, success_rate, progress_rate, grounding_acc, hard_sr, easy_sr, hard_pr, easy_pr)

    logger.info("All tasks completed")
    agentboard.log_summary()


if __name__ == "__main__":
    main()

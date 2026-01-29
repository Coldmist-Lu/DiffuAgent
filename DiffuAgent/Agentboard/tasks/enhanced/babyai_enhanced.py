"""
Enhanced BabyAI task with detailed logging and tracking.

This enhanced version includes:
- Token counting and thought recording
- Complete trajectory tracking
- Dynamic memory logging
- Stop reason tracking
- Difficulty-based metrics (hard/easy)
"""
import json
import numpy as np

from llm import load_llm
from agents import load_agent
from environment import load_environment
from common.registry import registry

from utils.logging.logger import TaskLogger
from utils.logging.agent_logger import AgentLogger
from .base_enhanced import BaseEnhancedTask

logger = AgentLogger(__name__)


@registry.register_task("babyai_enhanced")
class EvalBabyaiEnhanced(BaseEnhancedTask):

    def __init__(self,
                 llm_name="gpt",
                 llm_config=None,
                 agent_name="POMDPAgent",
                 agent_config=None,
                 env_config=None,
                 max_num_steps=20,
                 llm=None,
                 baseline_dir=None,
                 log_path=None):
        super().__init__()

        # Initialize llm and agent
        if llm is None:
            llm = load_llm(llm_name, llm_config)
        self.agent = load_agent(agent_name, agent_config, llm)

        self.env_num_per_task = env_config.get("env_num_per_task", 1)
        self.seed = env_config.get("seed", 1234)
        self.game_level = env_config.get("game_level", [])
        self.label_path = env_config.get("label_path", None)

        self.env_configs = self.get_all_environment_configs()
        self.max_num_steps = max_num_steps

        self.baseline_dir = baseline_dir

        # Setup enhanced logger
        self.setup_logger(
            task_name="babyai",
            log_path=log_path,
            max_num_steps=self.max_num_steps,
            baseline_dir=self.baseline_dir
        )

    def load_annotation(self, path):
        """Load annotation file with subgoals and difficulty."""
        all_annotations = []
        difficulty = []
        with open(path, 'r') as f:
            for line in f:
                if line.strip() == '':
                    continue
                line = json.loads(line.strip())
                if "subgoals" in line and "subgoals_1" not in line:
                    all_annotations.append(line["subgoals"])
                else:
                    annotation = []
                    for key in line:
                        if "subgoals" in key:
                            annotation.append(line[key])
                    all_annotations.append(annotation)

                if "difficulty" in line:
                    difficulty.append(line["difficulty"])
                else:
                    raise ValueError("No difficulty in annotation file")
        return all_annotations, difficulty

    def load_seq(self, path):
        """Load sequence file."""
        all_seqs = []
        with open(path, 'r') as f:
            for line in f:
                if line.strip() == '':
                    continue
                all_seqs.append(line.strip())
        return all_seqs

    def get_all_environment_configs(self):
        """Generate all environment configurations."""
        iter_num = 0
        env_configs = []
        self.seeds = range(self.seed, self.seed + self.env_num_per_task)
        obs_to_reward_list, difficulties = self.load_annotation(self.label_path)
        assert len(self.game_level) * self.env_num_per_task == len(obs_to_reward_list)
        for level in self.game_level:
            for seed in self.seeds:
                env_configs.append({
                    "game_level": level,
                    "seed": seed,
                    "obs_to_reward": obs_to_reward_list[iter_num],
                    "difficulty": difficulties[iter_num]
                })
                iter_num += 1
        return env_configs

    def evaluate_env(self, id):
        """Evaluate a single environment instance."""

        env = load_environment("babyai", self.env_configs[id])
        init_obs = env._get_obs()
        goal = env._get_goal()
        self.agent.task_id = f"babyai_{id}"
        self.agent.reset(goal, init_obs)

        logger.goal("Example {} | Goal: {}".format(id, self.agent.goal))
        logger.info("Step {:02} - Message: {}".format(0, init_obs))

        max_steps = self.max_num_steps
        reward = 0
        grounding_acc_count = 0
        score_change_record = []
        last_reward = 0

        token_cnt = 0
        exit_reason = "max_steps"

        # Initialize trajectory
        trajectory = self.init_trajectory(goal=goal, init_ob=init_obs)

        extra_details = {}

        for step_id in range(max_steps):
            # Log memory if available
            self.log_memory_if_available(step_id, logger)

            success, action = self.agent.run()

            # Process dict action (with token and thought)
            if isinstance(action, dict):
                _token, action = self.action_dict_process(
                    action, step_id=step_id, trajectory=trajectory, logger=logger, extra_details=extra_details
                )
                token_cnt += _token

            if not success or getattr(self.agent, "exit_flag", False) is True:
                exit_reason = "early_exit"
                break

            if isinstance(action, tuple):
                logger.info("Step {:02} - Thought: {}".format(step_id, action[-1]))
                action = action[0]

            logger.info("Step {:02} - Action: {}".format(step_id, action))
            trajectory.append({"Action": action, "id": step_id})

            state, reward, done, infos = env.step(action)

            trajectory.append({"Observation": state, "id": step_id})
            trajectory.append({"Progress Rate": reward, "id": step_id})

            if infos.get("action_is_valid", False):
                grounding_acc_count += 1

            if reward > last_reward:
                score_change_record.append((step_id, reward))
            last_reward = reward

            logger.info("Step {:02} - Observation: {}".format(step_id, state))
            logger.info("Step {:02} - Progress Rate: {}\n".format(step_id, reward))

            self.agent.update(action, state)

            if done:
                progress_rate = reward

                env_details = {
                    "task_name": env.game_name,
                    "goal": self.agent.goal,
                    "difficulty": env.difficulty
                }
                extra_details.update({
                    "steps": step_id + 1,
                    "avg_tokens": token_cnt / (step_id + 1),
                    "exit_reason": "success",
                })
                self.agentboard.log_example(
                    id, True, progress_rate, grounding_acc_count / (step_id + 1),
                    score_change_record, env_details, trajectory, extra=extra_details
                )

                return True, progress_rate, step_id + 1, grounding_acc_count / (step_id + 1), score_change_record

        env_details = {
            "goal": self.agent.goal,
            "task_name": env.game_name,
            "difficulty": env.difficulty
        }
        try:
            example_prompt = self.agent.get_example_prompt()
        except:
            example_prompt = None

        progress_rate = reward
        extra_details.update({
            "steps": step_id + 1,
            "avg_tokens": token_cnt / (step_id + 1),
            "exit_reason": exit_reason,
        })
        self.agentboard.log_example(
            id, False, progress_rate, grounding_acc_count / (step_id + 1),
            score_change_record, env_details, trajectory, example_prompt, extra=extra_details
        )

        return False, progress_rate, step_id + 1, grounding_acc_count / (step_id + 1), score_change_record

    def evaluate(self):
        """Evaluate all BabyAI examples."""
        num_envs = len(self.env_configs)
        success_rate = []
        all_progress_rates = []
        score_state_records = []
        grounding_accs = []
        difficulties = []

        for id in range(num_envs):
            success, progress_rate, steps, grounding_acc_count, score_change_record = self.evaluate_env(id)
            all_progress_rates.append(progress_rate)
            grounding_accs.append(grounding_acc_count)
            score_state_records.append(score_change_record)
            difficulties.append(self.env_configs[id]["difficulty"])

            if success:
                success_rate.append(1)
            else:
                success_rate.append(0)

            logger.finish("Example {} | Success: {} , Progress Rate: {} , Steps: {}\n".format(id, success, progress_rate, steps))

        # Calculate all metrics
        metrics = self.calculate_difficulty_metrics(success_rate, all_progress_rates, grounding_accs, difficulties)

        self.agentboard.log_summary(
            metrics['sr'], metrics['pr'], metrics['gr'],
            score_state_records, metrics['hard_sr'], metrics['hard_pr'],
            metrics['easy_sr'], metrics['easy_pr']
        )

        return (
            success_rate, all_progress_rates, grounding_accs, score_state_records,
            metrics['easy_sr'], metrics['hard_sr'],
            metrics['easy_pr'], metrics['hard_pr']
        )

    @classmethod
    def from_config(cls,
                    run_config,
                    llm_config,
                    agent_config,
                    env_config,
                    llm=None):
        """Create task instance from configuration."""
        # Accept both 'babyai' and 'babyai_enhanced' as valid environment names
        env_name = env_config.get("name", "babyai")
        if env_name not in ["babyai", "babyai_enhanced"]:
            raise ValueError(f"Expected env_name to be 'babyai' or 'babyai_enhanced', got '{env_name}'")

        max_num_steps = run_config.get("max_num_steps", 20)
        baseline_dir = run_config.get("baseline_dir", "data/baseline_results")
        llm_name = llm_config.get("name", "gpt")
        agent_name = agent_config.get("name", "POMDPAgent")
        log_path = run_config.get("log_path", None)

        return cls(
            llm_name=llm_name,
            llm_config=llm_config,
            agent_name=agent_name,
            agent_config=agent_config,
            env_config=env_config,
            max_num_steps=max_num_steps,
            llm=llm,
            baseline_dir=baseline_dir,
            log_path=log_path
        )

"""
Enhanced ScienceWorld task with detailed logging and tracking.

This enhanced version includes:
- Token counting and thought recording
- Complete trajectory tracking
- Dynamic memory logging
- Stop reason tracking
- Difficulty-based metrics (hard/easy)
"""
import os
import random
from llm import load_llm
from agents import load_agent
from environment import load_environment
from common.registry import registry

from utils.logging.agent_logger import AgentLogger
from .base_enhanced import BaseEnhancedTask

logger = AgentLogger(__name__)


@registry.register_task("scienceworld_enhanced")
class EvalScienceworldEnhanced(BaseEnhancedTask):
    def __init__(self,
                 llm_name="gpt",
                 llm_config=None,
                 agent_name="GPTAgent",
                 agent_config=None,
                 env_config=None,
                 run_config=None,
                 llm=None,
                 baseline_dir=None,
                 log_path=None):
        super().__init__()

        if llm is None:
            llm = load_llm(llm_name, llm_config)
        self.agent = load_agent(agent_name, agent_config, llm)
        self.simplefied = env_config.get("simplefied", False)
        seed = env_config.get("seed", 42)
        self.set_seed(seed)
        self.simplification_str = self.build_simplification_str()
        self.env_cfg = env_config

        self.max_num_steps = run_config.get("max_num_steps", 30)
        self.context_length = llm_config.get("context_length")

        self.baseline_dir = baseline_dir

        # Setup enhanced logger
        self.setup_logger(
            task_name="scienceworld",
            log_path=log_path,
            max_num_steps=self.max_num_steps,
            baseline_dir=self.baseline_dir
        )

    def build_simplification_str(self):
        """Build simplification string for ScienceWorld environment."""
        simplifications = list()
        simplifications.append("selfWateringFlowerPots")
        simplifications.append("openContainers")
        simplifications.append("openDoors")
        simplifications.append("noElectricalAction")
        return ",".join(simplifications)

    def set_seed(self, seed):
        """Set random seed for reproducibility."""
        random.seed(seed)

    def evaluate_env(self, index, task_name, var, modified_goal):
        """Evaluate a single environment instance."""

        self.env.load(task_name, var, simplificationStr=self.simplification_str)
        initialObs, initialDict = self.env.reset()
        init_obs = initialObs + f"\n{self.env.inventory()}"

        self.agent.task_id = f"scienceworld_{index}"
        self.agent.reset(goal=modified_goal, init_obs=init_obs, env=self.env)
        reward = 0.
        last_reward = 0.

        logger.info("Step {:02} - Observation: {}".format(0, init_obs))
        grounding_acc_count = 0
        score_change_record = []
        isDone = False

        token_cnt = 0
        exit_reason = "max_steps"

        # Initialize trajectory
        trajectory = self.init_trajectory(goal=modified_goal, init_ob=init_obs)

        extra_details = {}

        for i in range(self.max_num_steps):
            # Log memory if available
            self.log_memory_if_available(i, logger)

            success, action = self.agent.run()

            # Process dict action (with token and thought)
            if isinstance(action, dict):
                _token, action = self.action_dict_process(
                    action, step_id=i, trajectory=trajectory, logger=logger, extra_details=extra_details
                )
                token_cnt += _token

            logger.info("Step {:02} - Action: {}".format(i, action))
            trajectory.append({"Action": action, "id": i})

            if not success or getattr(self.agent, "exit_flag", False) is True:
                exit_reason = "early_exit"
                break

            observation, reward, isDone, info = self.env.step(action)
            if action in self.env.get_action_space(abstract=False):
                grounding_acc_count += 1

            logger.info("Step {:02} - Observation: {}".format(i, observation))
            logger.info("Step {:02} - Progress Rate: {}\n".format(i, reward))

            trajectory.append({"Observation": observation, "id": i})
            trajectory.append({"Progress Rate": reward, "id": i})

            if reward > last_reward:
                score_change_record.append((i, reward))
            last_reward = reward

            if isDone:
                env_details = {
                    "task_name": task_name,
                    "goal": self.agent.goal,
                    "difficulty": self.env.difficulty
                }
                extra_details.update({
                    "steps": i + 1,
                    "avg_tokens": token_cnt / (i + 1),
                    "exit_reason": "success",
                })
                self.agentboard.log_example(
                    index, True, 1.0, grounding_acc_count / (i + 1),
                    score_change_record, env_details, trajectory, extra=extra_details
                )

                return 1.0, True, grounding_acc_count / (i + 1), score_change_record, i

            self.agent.update(action=action, state=observation)

        env_details = {
            "task_name": task_name,
            "goal": self.agent.goal,
            "difficulty": self.env.difficulty
        }
        try:
            example_prompt = self.agent.get_example_prompt()
        except:
            example_prompt = None

        progress_rate = reward

        extra_details.update({
            "steps": i + 1,
            "avg_tokens": token_cnt / (i + 1),
            "exit_reason": exit_reason,
        })

        self.agentboard.log_example(
            index, isDone, progress_rate, grounding_acc_count / (i + 1),
            score_change_record, env_details, trajectory, example_prompt, extra=extra_details
        )

        return progress_rate, isDone, grounding_acc_count / (i + 1), score_change_record, i

    def evaluate(self):
        """Evaluate all ScienceWorld examples."""
        scores = []
        self.env = load_environment("scienceworld", self.env_cfg)
        labels = self.env.labels
        count = 0
        scores = []
        score_state_records = []
        grounding_accs = []
        srs = []
        difficulties = []

        for index, (k, v) in enumerate(labels.items()):
            task_name = v["task_name"]
            var = v["var"]
            modified_goal = v["modified_goal"]

            logger.goal("Example {} | Goal: {}".format(index, f"task_name: {task_name}, var: {var}, {modified_goal}"))
            score, done, grounding_acc, score_change_record, num_steps = self.evaluate_env(
                index, task_name, var, modified_goal
            )

            difficulties.append(self.env.difficulty)
            logger.finish("Example {} | Success: {} , Progress Rate: {} , Steps: {}\n".format(index, done, score, num_steps + 1))
            count += 1
            if done:
                srs.append(1.0)
            else:
                srs.append(0.0)
            scores.append(score)
            grounding_accs.append(grounding_acc)
            score_state_records.append(score_change_record)

        # Calculate all metrics
        metrics = self.calculate_difficulty_metrics(srs, scores, grounding_accs, difficulties)

        self.agentboard.log_summary(
            metrics['sr'], metrics['pr'], metrics['gr'],
            score_state_records, metrics['hard_sr'], metrics['hard_pr'],
            metrics['easy_sr'], metrics['easy_pr']
        )

        return (
            srs, scores, grounding_accs, score_state_records,
            metrics['easy_sr'], metrics['hard_sr'],
            metrics['easy_pr'], metrics['hard_pr']
        )

    def _grounding_fn(self, action):
        """Check if action is valid."""
        valid_actions = self.env.GetValidActions()
        return "check valid actions" if action not in valid_actions else action

    @classmethod
    def from_config(cls,
                    run_config,
                    llm_config,
                    agent_config,
                    env_config,
                    llm=None):
        """Create task instance from configuration."""
        llm_name = llm_config.get("name", "gpt")
        agent_name = agent_config.get("name", "GPTAgent")
        baseline_dir = run_config.get("baseline_dir", "data/baseline_results")
        log_path = run_config.get("log_path", None)

        return cls(
            llm_name=llm_name,
            llm_config=llm_config,
            agent_name=agent_name,
            agent_config=agent_config,
            env_config=env_config,
            run_config=run_config,
            llm=llm,
            baseline_dir=baseline_dir,
            log_path=log_path
        )

"""
Enhanced AlfWorld task with detailed logging and tracking.

This enhanced version includes:
- Token counting and thought recording
- Complete trajectory tracking
- Dynamic memory logging
- Stop reason tracking
- Difficulty-based metrics (hard/easy)
"""
import json
from agents import load_agent
from environment import load_environment
from llm import load_llm
from common.registry import registry
import copy

from utils.logging.agent_logger import AgentLogger
from .base_enhanced import BaseEnhancedTask

logger = AgentLogger(__name__)


prefixes = {
    'pick_and_place': 'put',
    'pick_clean_then_place': 'clean',
    'pick_heat_then_place': 'heat',
    'pick_cool_then_place': 'cool',
    'look_at_obj': 'examine',
    'pick_two_obj': 'puttwo'
}


@registry.register_task("alfworld_enhanced")
class EvalalfworldEnhanced(BaseEnhancedTask):
    def __init__(self,
                 llm_config=None,
                 agent_name='agent_name',
                 max_num_steps=30,
                 num_exams=134,
                 init_prompt_path='prompts/alfworld_base.json',
                 agent_config=None,
                 env_config=None,
                 llm=None,
                 baseline_dir=None,
                 log_path=None):
        super().__init__()

        # Initialize llm and agent
        if llm is None:
            llm = load_llm(llm_config.get("name", "gpt"), llm_config)
        self.agent = load_agent(agent_name, agent_config, llm)

        with open(init_prompt_path, 'r') as f:
            self.prompts = json.load(f)
        self.env_cfg = env_config
        self.max_num_steps = max_num_steps
        self.num_exams = num_exams

        self.baseline_dir = baseline_dir

        # Setup enhanced logger
        self.setup_logger(
            task_name="alfworld",
            log_path=log_path,
            max_num_steps=self.max_num_steps,
            baseline_dir=self.baseline_dir
        )

    def parseAction(self, action):
        """Parse and normalize action string."""
        action = action.strip()
        if "put" in action:
            if " in " in action:
                action = action.replace(" in ", ' in/on ')
            elif " on " in action:
                action = action.replace(" on ", ' in/on ')
        if action.endswith('.'):
            action = action[:-1].strip()
        return action

    def evaluate_env(self, index, ob='', examples=None):
        """Evaluate a single environment instance."""

        init_ob = ob.split('\n')[0]
        goal = ob.split('\n')[1].split("Your task is to:")[1].strip()

        self.agent.task_id = f"alfworld_{index}"
        self.agent.reset(goal=goal, init_obs=init_ob, env=self.env)

        logger.goal("Example {} | Goal: {}".format(index, self.agent.goal))
        init_prompt_dict = copy.deepcopy(self.prompts)
        init_prompt_dict['examples'] = examples
        reward = 0.
        last_reward = 0.
        done = False
        grounding_acc_count = 0
        score_change_record = []
        logger.info("Step {:02} - Message: {}".format(0, init_ob))

        # Initialize token counting
        token_cnt = 0
        exit_reason = "max_steps"

        # Initialize trajectory
        trajectory = self.init_trajectory(goal=self.agent.goal, init_ob=init_ob)

        extra_details = {}

        for i in range(0, self.max_num_steps):
            # Log memory if available
            self.log_memory_if_available(i, logger)

            success, action = self.agent.run(init_prompt_dict=init_prompt_dict)

            # Process dict action (with token and thought)
            if isinstance(action, dict):
                _token, action = self.action_dict_process(
                    action, step_id=i, trajectory=trajectory, logger=logger, extra_details=extra_details
                )
                token_cnt += _token

            if not success or getattr(self.agent, "exit_flag", False) is True:
                exit_reason = "early_exit"
                break

            action = self.parseAction(action)
            if action in self.env.get_action_space():
                grounding_acc_count += 1.0

            logger.info("Step {:02} - Action: {}".format(i, action))
            trajectory.append({"Action": action, "id": i})

            observation, reward, done, info = self.env.step(action)
            logger.info("Step {:02} - Observation: {}".format(i, observation))

            if "Task accomplished!" in observation and reward < 1.0:
                raise Exception("Task accomplished error")

            logger.info("Step {:02} - Progress Rate: {}\n".format(i, reward))

            trajectory.append({"Observation": observation, "id": i})
            trajectory.append({"Progress Rate": reward, "id": i})

            if reward > last_reward:
                score_change_record.append((i, reward))
            last_reward = reward
            self.agent.update(action=action, state=observation)

            if done:
                game_name = self.env.cur_task_name.split('/')[0]
                env_details = {
                    "task_name": game_name,
                    "goal": self.agent.goal,
                    "difficulty": self.env.difficulty
                }
                extra_details.update({
                    "steps": i + 1,
                    "avg_tokens": token_cnt / (i + 1),
                    "exit_reason": "success",
                })
                self.agentboard.log_example(
                    index, True, reward, grounding_acc_count / (i + 1),
                    score_change_record, env_details, trajectory, extra=extra_details
                )

                return 1.0, True, grounding_acc_count / (i + 1), score_change_record, i

        # Handle unsuccessful scenario
        game_name = self.env.cur_task_name.split('/')[0]
        env_details = {
            "task_name": game_name,
            "goal": self.agent.goal,
            "difficulty": self.env.difficulty
        }

        progress_rate = reward

        try:
            example_prompt = self.agent.get_example_prompt()
        except:
            example_prompt = None

        extra_details.update({
            "steps": i + 1,
            "avg_tokens": token_cnt / (i + 1),
            "exit_reason": exit_reason,
        })

        self.agentboard.log_example(
            index, done, progress_rate, grounding_acc_count / (i + 1),
            score_change_record, env_details, trajectory, example_prompt, extra=extra_details
        )

        return progress_rate, done, grounding_acc_count / (i + 1), score_change_record, i

    def evaluate(self):
        """Evaluate all AlfWorld examples."""
        self.env = load_environment('alfworld', self.env_cfg)
        scores = []
        score_state_records = []
        grounding_accs = []
        srs = []
        difficulties = []

        for id in range(self.num_exams):
            ob, info = self.env.reset()
            ob = '\n'.join(ob[0].split('\n\n')[1:])
            name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])
            difficulties.append(self.env.difficulty)

            for i, (k, v) in enumerate(prefixes.items()):
                if name.startswith(k):
                    examples = self.prompts['examples'][v]
                    score, is_done, grounding_acc, score_change_record, steps = self.evaluate_env(ob=ob, examples=examples, index=id)
                    if is_done:
                        srs.append(1.0)
                    else:
                        srs.append(0.0)
                    scores.append(score)
                    grounding_accs.append(grounding_acc)
                    score_state_records.append(score_change_record)
                    logger.finish("Example {} | Success: {} , Progress Rate: {} , Steps: {}\n".format(id, is_done, score, steps))

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
        if action not in self.env.GetValidActions():
            logger.debug(f"Invalid action detected: {action}")
            return "check valid actions"
        else:
            return action

    @classmethod
    def from_config(cls,
                    run_config,
                    llm_config,
                    agent_config,
                    env_config,
                    llm=None):
        """Create task instance from configuration."""
        agent_name = agent_config.get("name", "GPTAgent")
        init_prompt_path = agent_config.get("init_prompt_path", 'prompts/alfworld_in_context_learning.json')
        max_num_steps = run_config.get("max_num_steps", 30)
        baseline_dir = run_config.get("baseline_dir", "data/baseline_results")
        num_exams = run_config.get("num_exam", 134)
        log_path = run_config.get("log_path", None)
        return cls(
            llm_config=llm_config,
            agent_name=agent_name,
            max_num_steps=max_num_steps,
            num_exams=num_exams,
            init_prompt_path=init_prompt_path,
            agent_config=agent_config,
            env_config=env_config,
            llm=llm,
            baseline_dir=baseline_dir,
            log_path=log_path
        )

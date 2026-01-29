from .webshop import EvalWebshop
from .alfworld import Evalalfworld
# from .webbrowse import EvalWebBrowse  # Commented out due to beartype compatibility issue
from .babyai import EvalBabyai
from .pddl import EvalPddl
from .scienceworld import EvalScienceworld
from .jericho import EvalJericho
from .tool import EvalTool

# =====[ ADDED: Import enhanced tasks with detailed logging ]=====
# Enhanced versions provide: exit_reason tracking, token counting,
# thought recording, complete trajectory, difficulty metrics (hard/easy)
from .enhanced import EvalalfworldEnhanced, EvalScienceworldEnhanced, EvalBabyaiEnhanced
# ================================================================

from common.registry import registry

__all__ = [
    "Evalalfworld",
    "EvalBabyai",
    "EvalPddl",
    # "EvalWebBrowse",  # Commented out due to beartype compatibility issue
    "EvalWebshop",
    "EvalJericho",
    "EvalTool",
    "EvalWebshop",
    "EvalScienceworld",
    # =====[ ADDED: Enhanced tasks ]=====
    "EvalalfworldEnhanced",
    "EvalScienceworldEnhanced",
    "EvalBabyaiEnhanced"
    # ===================================
]


def load_task(name, run_config, llm_config, agent_config, env_config, llm=None):
    task = registry.get_task_class(name).from_config(run_config, llm_config, agent_config, env_config, llm=llm)
    return task

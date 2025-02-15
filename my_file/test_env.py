import argparse
import time
from typing import List
import furniture_bench
from furniture_bench.envs.furniture_sim_env import FurnitureSimEnv
from src.behavior.base import Actor  # noqa
import torch
from omegaconf import OmegaConf, DictConfig
from src.eval.rollout import calculate_success_rate
from src.behavior import get_actor
from src.common.tasks import furniture2idx, task_timeout
from src.common.files import trajectory_save_dir
from src.gym import get_env
from src.dataset import get_normalizer
import os
import time
from ipdb import set_trace as bp  # noqa
import wandb
from wandb import Api
from wandb.sdk.wandb_run import Run
from src.common.context import suppress_all_output

parser = argparse.ArgumentParser()
parser.add_argument("--run-id", type=str, required=False, nargs="*")
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--n-envs", type=int, default=1)
parser.add_argument("--n-rollouts", type=int, default=1)
parser.add_argument("--randomness", type=str, default="low")
parser.add_argument(
    "--furniture",
    "-f",
    type=str,
    choices=["one_leg", "lamp", "round_table", "desk", "square_table", "cabinet"],
    required=True,
)
parser.add_argument("--n-parts-assemble", type=int, default=None)

parser.add_argument("--save-rollouts", action="store_true")
parser.add_argument("--save-failures", action="store_true")
parser.add_argument("--store-full-resolution-video", action="store_true")

parser.add_argument("--wandb", action="store_true")
parser.add_argument("--leaderboard", action="store_true")

# Define what should be done if the success rate fields are already present
parser.add_argument(
    "--if-exists",
    type=str,
    choices=["skip", "overwrite", "append", "error"],
    default="error",
)
parser.add_argument(
    "--run-state",
    type=str,
    default=None,
    choices=["running", "finished", "failed", "crashed"],
    nargs="*",
)

# For batch evaluating runs from a sweep or a project
parser.add_argument("--sweep-id", type=str, default=None)
parser.add_argument("--project-id", type=str, default=None)

parser.add_argument("--continuous-mode", action="store_true")
parser.add_argument(
    "--continuous-interval",
    type=int,
    default=60,
    help="Pause interval before next evaluation",
)
parser.add_argument("--ignore-currently-evaluating-flag", action="store_true")

parser.add_argument("--visualize", action="store_true")
parser.add_argument("--store-video-wandb", action="store_true")
parser.add_argument("--eval-top-k", type=int, default=None)
parser.add_argument(
    "--action-type", type=str, default="pos", choices=["delta", "pos"]
)
parser.add_argument("--prioritize-fewest-rollouts", action="store_true")
parser.add_argument("--multitask", action="store_true")
parser.add_argument("--compress-pickles", action="store_true")
parser.add_argument("--max-rollouts", type=int, default=None)
parser.add_argument("--verbose", "-v", action="store_true")
parser.add_argument("--max-rollout-steps", type=int, default=None)
parser.add_argument("--no-april-tags", action="store_true")
parser.add_argument("--record", action="store_true")
parser.add_argument("--pose", action="store_true")
# Parse the arguments
args = parser.parse_args()

env: FurnitureSimEnv = get_env(
    gpu_id=args.gpu,
    furniture=args.furniture,
    num_envs=args.n_envs,
    randomness=args.randomness,
    max_env_steps=5_000,
    resize_img=False,
    act_rot_repr="rot_6d",
    ctrl_mode="osc",
    action_type=args.action_type,
    april_tags=not args.no_april_tags,
    verbose=args.verbose,
    headless=not args.visualize,
    record=args.record
)

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
with suppress_all_output(True):
        obs = env.reset()

while True:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
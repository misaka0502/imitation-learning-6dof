import furniture_bench
import gym
from src.gym import get_env
import torch
# env: FurnitureSimEnv = get_env(
    
#     gpu_id=7,
#     furniture=args.furniture,
#     num_envs=args.n_envs,
#     randomness=args.randomness,
#     max_env_steps=5_000,
#     resize_img=False,
#     act_rot_repr="rot_6d",
#     ctrl_mode="osc",
#     action_type=args.action_type,
#     april_tags=not args.no_april_tags,
#     verbose=args.verbose,
#     headless=not args.visualize,
#     save_camera_input=True
# )

env = gym.make(
    "FurnitureSimFull-v0",
    furniture="one_leg",
    num_envs=1,
    
)

ac = torch.tensor(env.action_space.sample()).float().to('cuda')
ob, rew, done, _ = env.step(ac)
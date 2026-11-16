"""Train an RL agent on the chemotherapy environment.

Usage:
    python train.py --exp E1 --algo PPO --steps 200000 --seed 0 --config tuning/best_PPO.json
    python train.py --exp E3 --algo PPO --hist 7 --name PPO-hist ...   (stack 7 days of observations)
    (omit --config to use the hand-picked defaults in agents.py)
"""
import argparse
import json
import os

import torch

from agents import load_config, make_model
from experiments import EXPERIMENTS

torch.set_num_threads(1)

p = argparse.ArgumentParser()
p.add_argument("--algo", choices=["PPO", "DQN"], default="PPO")
p.add_argument("--steps", type=int, default=300_000)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--exp", choices=list(EXPERIMENTS), default="E1")
p.add_argument("--hist", type=int, default=0, help="days of observation history to stack")
p.add_argument("--config", default=None, help="JSON file with hyperparameters")
p.add_argument("--name", default=None, help="label used in file names (default: algo)")
args = p.parse_args()

os.makedirs(f"models/{args.exp}", exist_ok=True)
os.makedirs(f"logs/{args.exp}", exist_ok=True)
tag = f"{args.exp}/{args.name or args.algo}_s{args.seed}"
cfg = load_config(args.algo, args.config)
print(tag, json.dumps(cfg))

model = make_model(args.algo, cfg, args.seed, EXPERIMENTS[args.exp],
                   monitor_dir=f"logs/{tag}", hist=args.hist)
model.learn(total_timesteps=args.steps)
model.save(f"models/{tag}.zip")
print("saved", tag)

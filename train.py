"""Train an RL agent on the chemotherapy environment.

Usage:
    python train.py --algo PPO --steps 300000 --seed 0 --config tuning/best_PPO.json
    (omit --config to use the hand-picked defaults in agents.py)
"""
import argparse
import json
import os

import torch

from agents import load_config, make_model

torch.set_num_threads(1)

p = argparse.ArgumentParser()
p.add_argument("--algo", choices=["PPO", "DQN"], default="PPO")
p.add_argument("--steps", type=int, default=300_000)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--var", type=float, default=0.2, help="patient variability")
p.add_argument("--config", default=None, help="JSON file with hyperparameters")
p.add_argument("--name", default=None, help="label used in file names (default: algo)")
args = p.parse_args()

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)
tag = f"{args.name or args.algo}_var{args.var}_s{args.seed}"
cfg = load_config(args.algo, args.config)
print(tag, json.dumps(cfg))

model = make_model(args.algo, cfg, args.seed, dict(patient_variability=args.var),
                   monitor_dir=f"logs/{tag}")
model.learn(total_timesteps=args.steps)
model.save(f"models/{tag}.zip")
print("saved", tag)

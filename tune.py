"""Hyperparameter search with Optuna.

Each trial trains an agent for a reduced budget and scores it by the mean return
on VALIDATION patients (seeds 5000+), which are disjoint from the TEST patients
(seeds 10000+) used in evaluate.py, so the reported test results are not tuned on.

Usage: python tune.py --algo PPO --trials 16 --steps 100000
"""
import argparse
import json
import os

import numpy as np
import optuna
import torch

from agents import make_model
from baselines import run_policy
from chemo_env import ChemoEnv
from experiments import EXPERIMENTS

torch.set_num_threads(1)
VAL_SEED0, N_VAL = 5_000, 50

p = argparse.ArgumentParser()
p.add_argument("--algo", choices=["PPO", "DQN"], required=True)
p.add_argument("--trials", type=int, default=16)
p.add_argument("--steps", type=int, default=100_000)
p.add_argument("--exp", choices=list(EXPERIMENTS), default="E1")
args = p.parse_args()


def sample(trial):
    common = dict(
        learning_rate=trial.suggest_float("learning_rate", 1e-4, 2e-3, log=True),
        gamma=trial.suggest_categorical("gamma", [0.95, 0.98, 0.99, 0.995]),
        net=trial.suggest_categorical("net", ["small", "medium"]),
    )
    if args.algo == "PPO":
        return dict(common,
                    n_steps=trial.suggest_categorical("n_steps", [128, 256, 512]),
                    batch_size=trial.suggest_categorical("batch_size", [64, 128, 256]),
                    ent_coef=trial.suggest_float("ent_coef", 1e-4, 5e-2, log=True),
                    clip_range=trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3]),
                    n_epochs=trial.suggest_categorical("n_epochs", [5, 10, 20]))
    return dict(common,
                batch_size=trial.suggest_categorical("batch_size", [64, 128, 256]),
                exploration_fraction=trial.suggest_float("exploration_fraction", 0.1, 0.5),
                exploration_final_eps=trial.suggest_categorical("exploration_final_eps", [0.01, 0.02, 0.05]),
                target_update_interval=trial.suggest_categorical("target_update_interval", [500, 1000, 5000]),
                train_freq=trial.suggest_categorical("train_freq", [2, 4, 8]))


def objective(trial):
    cfg = sample(trial)
    model = make_model(args.algo, cfg, seed=trial.number, env_kwargs=EXPERIMENTS[args.exp])
    model.learn(total_timesteps=args.steps)
    env = ChemoEnv(**EXPERIMENTS[args.exp])
    runs = [run_policy(env, None, seed=VAL_SEED0 + i, sb3_model=model) for i in range(N_VAL)]
    trial.set_user_attr("cure_rate", float(np.mean([r["outcome"] == "cure" for r in runs])))
    trial.set_user_attr("tox_rate", float(np.mean([r["outcome"] == "toxicity" for r in runs])))
    return float(np.mean([r["return"] for r in runs]))


os.makedirs("tuning", exist_ok=True)
study = optuna.create_study(
    direction="maximize", study_name=args.algo,
    storage=f"sqlite:///tuning/{args.algo}.db", load_if_exists=True,
    sampler=optuna.samplers.TPESampler(seed=0, n_startup_trials=6),
)
# trial 0 = the hand-picked defaults, so tuned vs. default is a fair comparison
from agents import DEFAULTS
if len(study.trials) == 0:
    study.enqueue_trial(DEFAULTS[args.algo])
study.optimize(objective, n_trials=args.trials)

best = study.best_trial
print("best value", best.value, best.params, best.user_attrs)
json.dump(best.params, open(f"tuning/best_{args.algo}.json", "w"), indent=2)
study.trials_dataframe().to_csv(f"tuning/trials_{args.algo}.csv", index=False)

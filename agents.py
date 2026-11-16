"""Model factories shared by training and tuning scripts."""
import json

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.wrappers import FrameStackObservation
from stable_baselines3.common.monitor import Monitor

from chemo_env import ChemoEnv

NETS = {"small": [64, 64], "medium": [128, 128]}

DEFAULTS = {
    "PPO": dict(learning_rate=3e-4, n_steps=256, batch_size=256, gamma=0.99,
                ent_coef=0.01, clip_range=0.2, n_epochs=10, net="small"),
    "DQN": dict(learning_rate=5e-4, batch_size=128, gamma=0.99,
                exploration_fraction=0.3, exploration_final_eps=0.02,
                target_update_interval=1000, train_freq=4, net="small"),
}


def load_config(algo, path=None):
    cfg = dict(DEFAULTS[algo])
    if path:
        cfg.update(json.load(open(path)))
    return cfg


def make_env(env_kwargs, hist=0):
    """A single evaluation env; with hist>0 the last `hist` observations are stacked."""
    env = ChemoEnv(**env_kwargs)
    return FrameStackObservation(env, stack_size=hist) if hist else env


def make_model(algo, cfg, seed, env_kwargs, monitor_dir=None, hist=0):
    cfg = dict(cfg)
    policy_kwargs = dict(net_arch=NETS[cfg.pop("net")])
    wrapper = (lambda e: FrameStackObservation(e, stack_size=hist)) if hist else None
    if algo == "PPO":
        env = make_vec_env(ChemoEnv, n_envs=8, seed=seed, env_kwargs=env_kwargs,
                           monitor_dir=monitor_dir, wrapper_class=wrapper)
        return PPO("MlpPolicy", env, seed=seed, verbose=0, device="cpu",
                   policy_kwargs=policy_kwargs, **cfg)
    env = ChemoEnv(**env_kwargs)
    env = Monitor(env, monitor_dir) if monitor_dir else env
    env = wrapper(env) if hist else env
    env.reset(seed=seed)
    return DQN("MlpPolicy", env, buffer_size=100_000, learning_starts=5_000, seed=seed,
               verbose=0, device="cpu", policy_kwargs=policy_kwargs, **cfg)

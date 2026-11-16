"""Tabular Q-learning on a discretized version of the observation.

Observation -> grid cell (full observation, E1/E2):
    log-tumor  : 12 bins over [-1, 0]   (cure level ... full tumor)
    healthy N  :  8 bins over [0.5, 1]
    drug C     :  6 bins over [0, 2]
    time       :  3 bins over [0, 1]
Partial observation (E3) replaces drug C by days-since-scan (7 bins) and adds the
previous dose (3 bins).

Usage: python qlearning.py --exp E1 --episodes 20000 --seed 0
"""
import argparse
import os

import numpy as np

from chemo_env import ChemoEnv
from experiments import EXPERIMENTS

_LOGT = np.linspace(-1.0, 0.0, 13)[1:-1]
_N = np.linspace(0.5, 1.0, 9)[1:-1]
_TIME = np.linspace(0.0, 1.0, 4)[1:-1]
EDGES = {
    4: [_LOGT, _N, np.linspace(0.0, 2.0, 7)[1:-1], _TIME],
    5: [_LOGT, _N, (np.arange(1, 7) - 0.5) / 7, _TIME, np.array([0.25, 0.75])],
}


def discretize(obs):
    return tuple(int(np.digitize(o, e)) for o, e in zip(obs, EDGES[len(obs)]))


class QAgent:
    def __init__(self, obs_dim=4, n_actions=3, Q=None):
        shape = tuple(len(e) + 1 for e in EDGES[obs_dim])
        self.Q = np.zeros(shape + (n_actions,)) if Q is None else Q

    def predict(self, obs, deterministic=True):  # same interface as SB3 models
        return int(np.argmax(self.Q[discretize(obs)])), None

    def save(self, path):
        np.save(path, self.Q)

    @classmethod
    def load(cls, path):
        return cls(Q=np.load(path))


def train(episodes=20000, seed=0, exp="E1", alpha=0.1, gamma=0.99, eps_end=0.05, log_path=None):
    rng = np.random.default_rng(seed)
    env = ChemoEnv(**EXPERIMENTS[exp])
    agent = QAgent(env.observation_space.shape[0], env.action_space.n)
    Q = agent.Q
    log = []
    for ep in range(episodes):
        eps = max(eps_end, 1.0 - ep / (0.8 * episodes))
        obs, _ = env.reset(seed=int(rng.integers(1_000_000)))
        s, done, ret, steps = discretize(obs), False, 0.0, 0
        while not done:
            a = int(rng.integers(Q.shape[-1])) if rng.random() < eps else int(np.argmax(Q[s]))
            obs, r, term, trunc, _ = env.step(a)
            s2 = discretize(obs)
            target = r + (0.0 if term else gamma * np.max(Q[s2]))
            Q[s + (a,)] += alpha * (target - Q[s + (a,)])
            s, done = s2, term or trunc
            ret += r
            steps += 1
        log.append((ep, steps, ret))
    if log_path:
        np.savetxt(log_path, np.array(log), delimiter=",", header="episode,length,return", comments="")
    return agent


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exp", choices=list(EXPERIMENTS), default="E1")
    args = p.parse_args()
    os.makedirs(f"models/{args.exp}", exist_ok=True)
    os.makedirs(f"logs/{args.exp}", exist_ok=True)
    tag = f"{args.exp}/QL_s{args.seed}"
    agent = train(args.episodes, args.seed, args.exp, log_path=f"logs/{tag}.csv")
    agent.save(f"models/{tag}.npy")
    print("saved", tag)

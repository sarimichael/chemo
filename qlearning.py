"""Tabular Q-learning on a discretized version of the observation.

Observation -> grid cell:
    log-tumor  : 12 bins over [-1, 0]   (cure level ... full tumor)
    healthy N  :  8 bins over [0.5, 1]
    drug C     :  6 bins over [0, 2]
    time       :  3 bins over [0, 1]

Usage: python qlearning.py --episodes 20000 --seed 0
"""
import argparse
import os

import numpy as np

from chemo_env import ChemoEnv

EDGES = [
    np.linspace(-1.0, 0.0, 13)[1:-1],
    np.linspace(0.5, 1.0, 9)[1:-1],
    np.linspace(0.0, 2.0, 7)[1:-1],
    np.linspace(0.0, 1.0, 4)[1:-1],
]
SHAPE = tuple(len(e) + 1 for e in EDGES)


def discretize(obs):
    return tuple(int(np.digitize(o, e)) for o, e in zip(obs, EDGES))


class QAgent:
    def __init__(self, n_actions=3, Q=None):
        self.Q = np.zeros(SHAPE + (n_actions,)) if Q is None else Q

    def predict(self, obs, deterministic=True):  # same interface as SB3 models
        return int(np.argmax(self.Q[discretize(obs)])), None

    def save(self, path):
        np.save(path, self.Q)

    @classmethod
    def load(cls, path):
        return cls(Q=np.load(path))


def train(episodes=5000, seed=0, var=0.2, alpha=0.1, gamma=0.99, eps_end=0.05, log_path=None):
    rng = np.random.default_rng(seed)
    env = ChemoEnv(patient_variability=var)
    agent = QAgent(env.action_space.n)
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
    p.add_argument("--var", type=float, default=0.2)
    args = p.parse_args()
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    tag = f"QL_var{args.var}_s{args.seed}"
    agent = train(args.episodes, args.seed, args.var, log_path=f"logs/{tag}.csv")
    agent.save(f"models/{tag}.npy")
    print("saved", tag)

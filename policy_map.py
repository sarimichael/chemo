"""Visualize what the learned E1 agents do: greedy dose over (tumor, healthy cells),
for a few drug levels, at day 30. Also reports how often each dose is used on test patients.

Usage: python policy_map.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from stable_baselines3 import DQN, PPO

from agents import make_env
from baselines import run_policy
from experiments import EXPERIMENTS
from qlearning import QAgent

agents = {
    "PPO": PPO.load("models/E1/PPO_s0.zip", device="cpu"),
    "DQN": DQN.load("models/E1/DQN_s0.zip", device="cpu"),
    "QL": QAgent.load("models/E1/QL_s0.npy"),
}
logT = np.linspace(-3, np.log10(0.5), 80)     # tumor from cure level to 0.5
Ns = np.linspace(0.5, 1.0, 60)               # healthy cells from the toxicity limit to normal
drugs = [0.0, 0.5, 1.0]
cmap = ListedColormap(["#f2f2f2", "#9ecae1", "#08519c"])

fig, axes = plt.subplots(len(agents), len(drugs), figsize=(11, 8.6), sharex=True, sharey=True)
for i, (name, ag) in enumerate(agents.items()):
    for j, C in enumerate(drugs):
        A = np.zeros((len(Ns), len(logT)))
        for a, n in enumerate(Ns):
            for b, lt in enumerate(logT):
                obs = np.array([lt / 3, n, C, 30 / 90], dtype=np.float32)
                A[a, b] = ag.predict(obs, deterministic=True)[0]
        ax = axes[i, j]
        ax.imshow(A, origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=2,
                  extent=[logT[0], logT[-1], Ns[0], Ns[-1]])
        if i == 0:
            ax.set_title(f"drug level C = {C}")
        if j == 0:
            ax.set_ylabel(f"{name}\nhealthy cells N")
        if i == len(agents) - 1:
            ax.set_xlabel("tumor log10(T)")
handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(k)) for k in range(3)]
fig.legend(handles, ["dose 0 (also: state never visited, for QL)", "dose 0.5", "dose 1.0"], loc="lower center", ncol=3, frameon=False)
fig.suptitle("Greedy dose chosen by each E1 agent (seed 0) at day 30", y=0.995)
fig.tight_layout(rect=(0, 0.04, 1, 0.97))
fig.savefig("figures/E1_policy_map.png", dpi=130)

# dose usage on the test patients
env = make_env(EXPERIMENTS["E1"])
print("dose usage on 300 E1 test patients (share of treatment days)")
for name, ag in agents.items():
    doses = np.concatenate([run_policy(env, None, seed=10_000 + k, sb3_model=ag)["dose"] for k in range(300)])
    print(f"{name:4s}  none {np.mean(doses == 0):.2f}  low {np.mean(doses == 0.5):.2f}  high {np.mean(doses == 1.0):.2f}")

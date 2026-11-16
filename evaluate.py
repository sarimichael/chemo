"""Evaluate baselines and all trained agents (every seed) on the same TEST patients.

Test patients use env seeds 10000..10299 and are never used in training or tuning.
Writes results/<exp>_summary.csv, results/<exp>_per_seed.csv and figures/<exp>_*.png.

Usage: python evaluate.py --exp E1
"""
import argparse
import glob
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import DQN, PPO

from agents import make_env
from baselines import BASELINES, run_policy
from experiments import EXPERIMENTS, HIST
from qlearning import QAgent

ap = argparse.ArgumentParser()
ap.add_argument("--exp", choices=list(EXPERIMENTS), default="E1")
EXP = ap.parse_args().exp
N_PAT, TEST_SEED0 = 300, 10_000
VAR = EXPERIMENTS[EXP]["patient_variability"]
os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)
plain_env = make_env(EXPERIMENTS[EXP])
hist_env = make_env(EXPERIMENTS[EXP], HIST)
env = plain_env


def load(path):
    if path.endswith(".npy"):
        return QAgent.load(path)
    return (PPO if "PPO" in os.path.basename(path) else DQN).load(path, device="cpu")


def evaluate(policy=None, model=None, env=plain_env):
    runs = [run_policy(env, policy, seed=TEST_SEED0 + i, sb3_model=model) for i in range(N_PAT)]
    oc = np.array([r["outcome"] for r in runs])
    cured = [r["days"] for r in runs if r["outcome"] == "cure"]
    return runs, dict(
        cure=100 * np.mean(oc == "cure"),
        toxicity=100 * np.mean(oc == "toxicity"),
        days_to_cure=np.mean(cured) if cured else np.nan,
        drug=np.mean([sum(r["dose"]) for r in runs]),
        min_healthy=np.mean([min(r["N"]) for r in runs]),
        ret=np.mean([r["return"] for r in runs]),
    )


rows, examples = [], {}
for name, pol in BASELINES.items():
    runs, m = evaluate(policy=pol)
    rows.append(dict(method=name, seed=0, **m))
    examples[name] = runs[0]

models = sorted(glob.glob(f"models/{EXP}/*_s*.zip") + glob.glob(f"models/{EXP}/*_s*.npy"))
for path in models:
    label, seed = re.match(r".*/(.+)_s(\d+)\.", path).groups()
    runs, m = evaluate(model=load(path), env=hist_env if label.endswith("-hist") else plain_env)
    rows.append(dict(method=label, seed=int(seed), **m))
    if int(seed) == 0:
        examples[label] = runs[0]
    print(f"{label:12s} seed {seed}: cure {m['cure']:.1f}%  tox {m['toxicity']:.1f}%  return {m['ret']:.2f}")

per_seed = pd.DataFrame(rows)
per_seed.to_csv(f"results/{EXP}_per_seed.csv", index=False)
agg = per_seed.groupby("method", sort=False).agg(["mean", "std"]).drop(columns="seed")
agg.columns = [f"{a}_{b}" for a, b in agg.columns]
agg["n_seeds"] = per_seed.groupby("method", sort=False).size()
agg.to_csv(f"results/{EXP}_summary.csv")

print("\nmethod          cure%          tox%           days        drug   min_N  return")
for meth, r in agg.iterrows():
    sd = lambda k: f"±{r[k + '_std']:.1f}" if r["n_seeds"] > 1 else "     "
    print(f"{meth:14s} {r.cure_mean:5.1f}{sd('cure'):6s}  {r.toxicity_mean:5.1f}{sd('toxicity'):6s}  "
          f"{r.days_to_cure_mean:5.1f}  {r.drug_mean:6.1f}  {r.min_healthy_mean:5.2f}  {r.ret_mean:6.2f}{sd('ret')}")

# ---------- Figure 1: outcome bars (mean ± std over seeds) ----------
fig, ax = plt.subplots(figsize=(10, 4.4))
names = list(agg.index)
x = np.arange(len(names))
ax.bar(x - 0.2, agg.cure_mean, 0.4, yerr=agg.cure_std.fillna(0), capsize=3, label="cured %", color="#27ae60")
ax.bar(x + 0.2, agg.toxicity_mean, 0.4, yerr=agg.toxicity_std.fillna(0), capsize=3, label="toxicity %", color="#c0392b")
ax.set_xticks(x, names, rotation=20)
ax.set_ylim(0, 105)
ax.set_title(f"{EXP}: outcomes on {N_PAT} unseen test patients (±{int(VAR*100)}% parameters); error bars = std over seeds")
ax.legend()
fig.tight_layout()
fig.savefig(f"figures/{EXP}_outcomes.png", dpi=130)

# ---------- Figure 2: example trajectories ----------
show = [s for s in ["Weekly 3on/4off", "Threshold rule", "QL", "DQN", "PPO", "PPO-hist"] if s in examples]
fig, axes = plt.subplots(3, len(show), figsize=(3.8 * len(show), 8.5), sharex=True, sharey="row")
for j, name in enumerate(show):
    tr = examples[name]
    days = np.arange(len(tr["T"]))
    axes[0, j].semilogy(days, tr["T"], color="#c0392b")
    axes[0, j].axhline(1e-3, ls="--", c="gray", lw=0.8)
    axes[0, j].set_title(f"{name}\n({tr['outcome']}, day {tr['days']})", fontsize=11)
    axes[1, j].plot(days, tr["N"], color="#2471a3")
    axes[1, j].axhline(0.5, ls="--", c="gray", lw=0.8)
    axes[2, j].bar(days[1:], tr["dose"], color="#7d3c98", width=0.9)
    axes[2, j].set_xlabel("day")
axes[0, 0].set_ylabel("tumor (log)")
axes[1, 0].set_ylabel("healthy cells")
axes[2, 0].set_ylabel("daily dose")
fig.suptitle(f"{EXP}: same test patient under each policy (dashed = cure level / toxicity limit)")
fig.tight_layout()
fig.savefig(f"figures/{EXP}_trajectories.png", dpi=130)

# ---------- Figure 3: learning curves (return vs env steps, mean ± std over seeds) ----------
def curve_sb3(path):  # a monitor file (DQN) or a directory of monitor files (PPO, 8 envs)
    files = [path] if path.endswith(".monitor.csv") else glob.glob(f"{path}/*.monitor.csv")
    dfs = [pd.read_csv(f, skiprows=1) for f in files]
    df = pd.concat(dfs).sort_values("t")
    return df["l"].cumsum().values, df["r"].values


def curve_ql(csv):
    df = pd.read_csv(csv)
    return df["length"].cumsum().values, df["return"].values


grid = defaultdict(list)
for d in sorted(glob.glob(f"logs/{EXP}/*_s*")):
    label = re.match(r".*/(.+)_s\d+", d).group(1)
    is_ql = d.endswith(".csv") and not d.endswith(".monitor.csv")
    steps, ret = curve_ql(d) if is_ql else curve_sb3(d)
    grid[label].append((steps, ret))

fig, ax = plt.subplots(figsize=(8, 4.5))
colors = {"PPO": "#1f77b4", "DQN": "#ff7f0e", "QL": "#2ca02c", "PPO-hist": "#08306b", "DQN-hist": "#a63603", "DQN-default": "#fdae6b"}
for label, runs in grid.items():
    xmax = min(s[-1] for s, _ in runs)
    xs = np.linspace(0, xmax, 200)
    ys = []
    for s, r in runs:
        sm = pd.Series(r).rolling(200, min_periods=1).mean().values
        ys.append(np.interp(xs, s, sm))
    ys = np.array(ys)
    ax.plot(xs, ys.mean(0), label=f"{label} (n={len(runs)})", color=colors.get(label))
    ax.fill_between(xs, ys.mean(0) - ys.std(0), ys.mean(0) + ys.std(0), alpha=0.2, color=colors.get(label))
ax.axhline(agg.loc["Threshold rule", "ret_mean"], ls="--", c="gray", lw=1, label="Threshold rule")
ax.set_xlabel("environment steps (days simulated)")
ax.set_ylabel("episode return (rolling mean)")
ax.set_title(f"{EXP}: learning curves, mean ± std over seeds")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(f"figures/{EXP}_learning_curves.png", dpi=130)
print("figures saved")

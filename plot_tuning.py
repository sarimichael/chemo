"""Plot the Optuna search: validation return of every trial (trial 0 = hand-picked defaults)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for ax, algo in zip(axes, ["PPO", "DQN"]):
    df = pd.read_csv(f"tuning/trials_{algo}.csv")
    df = df[df.state == "COMPLETE"]
    ax.plot(df.number, df.value, "o", color="#7f8c8d", label="trial")
    ax.plot(df.number, df.value.cummax(), "-", color="#2c3e50", label="best so far")
    ax.plot(0, df.value.iloc[0], "s", color="#e67e22", ms=9, label="defaults (trial 0)")
    ax.set_title(f"{algo}: {len(df)} trials, 120k steps each")
    ax.set_xlabel("trial")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("mean return, 50 validation patients")
axes[0].legend(fontsize=9)
fig.tight_layout()
fig.savefig("figures/tuning.png", dpi=130)
print("saved figures/tuning.png")

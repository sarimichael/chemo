# Adaptive Chemotherapy Dosing with Reinforcement Learning

Final project, Introduction to Reinforcement Learning course.

An RL agent decides a daily chemotherapy dose for a simulated patient. It has to
eradicate the tumor without letting healthy cells drop below a toxicity limit.
Patients differ: every episode draws new model parameters (±20%), so a fixed
schedule that works for one patient poisons another. The question is whether an
adaptive, state-dependent policy beats standard fixed protocols across a
population of unseen patients.

## Model

All quantities are normalized. `T` = tumor, `N` = healthy cells, `C` = drug concentration, `u` = daily dose.

```
dT/dt = r_T·T·(1 − T) − a_T·(1 − e^(−C))·T
dN/dt = r_N·N·(1 − N) − a_N·(1 − e^(−C))·N
dC/dt = u − γ·C
```

Base parameters: `r_T=0.10, r_N=0.30, a_T=0.50, a_N=0.25, γ=0.50`. Per patient,
`r_T, r_N, a_T, a_N` are each scaled by `U(0.8, 1.2)`. The ODEs are integrated with RK4 (20 substeps per day).

## MDP

| | |
|---|---|
| Step | 1 day; horizon 90 days |
| State | `[log10(T)/3, N, C, t/90]` |
| Actions | dose `u ∈ {0, 0.5, 1.0}` |
| Reward | `log10(T_prev) − log10(T)` (tumor log-reduction) `− 0.05·u` |
| Cure | `T < 1e-3` → `+10`, episode ends |
| Toxicity | `N < 0.5` → `−10`, episode ends |
| Timeout | day 90 → `−3 · (log10(T) + 3)` for residual tumor |

## Methods

* **Baselines:** no treatment, maximum dose every day, constant low dose, a weekly protocol (3 days on, 4 off), and a threshold rule (dose only while `N > 0.75`).
* **Tabular Q-learning:** state discretized into a 12×8×6×3 grid (`qlearning.py`).
* **DQN** and **PPO** from Stable-Baselines3, with hyperparameters tuned by Optuna (`tune.py`).

Every agent is trained with 5 seeds.

## Data splits (env seeds)

* **Training:** random seeds drawn during training
* **Validation (tuning only):** 5000–5049
* **Test (all reported results):** 10000–10299

## Reproduce

```bash
pip install -r requirements.txt

# 1. hyperparameter tuning (~30 min on 2 cores)
python tune.py --algo PPO --trials 16 --steps 120000
python tune.py --algo DQN --trials 12 --steps 120000

# 2. train 5 seeds of each agent
for s in 0 1 2 3 4; do
  python train.py --algo PPO --steps 300000 --seed $s --config tuning/best_PPO.json
  python train.py --algo DQN --steps 300000 --seed $s --config tuning/best_DQN.json
  python qlearning.py --episodes 20000 --seed $s
done

# 3. evaluate everything on the test patients and make figures
python evaluate.py
```

Or run the whole pipeline with `bash run_all.sh`.

## Files

| File | Contents |
|---|---|
| `chemo_env.py` | Gymnasium environment |
| `baselines.py` | Hand-crafted policies and the episode runner |
| `agents.py` | SB3 model factory and default hyperparameters |
| `qlearning.py` | Tabular Q-learning |
| `tune.py` | Optuna search |
| `train.py` | Train PPO/DQN |
| `evaluate.py` | Test-set evaluation, tables (`results/`), figures (`figures/`) |

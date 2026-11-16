# Adaptive Chemotherapy Dosing with Reinforcement Learning

Final project, Introduction to Reinforcement Learning course.

An RL agent chooses a daily chemotherapy dose for a simulated patient. It must
eradicate the tumor without letting healthy cells drop below a toxicity limit.
Patients differ: every episode draws new model parameters, so a fixed schedule
that works for one patient poisons another. We compare tabular Q-learning, DQN
and PPO against fixed and rule-based protocols on held-out patients, in three
settings of increasing difficulty.

## Model

All quantities are normalized. `T` is the tumor, `N` the healthy cells, `C` the drug concentration and `u` the daily dose.

```
dT/dt = r_T·T·(1 − T) − a_T·(1 − e^(−C))·T
dN/dt = r_N·N·(1 − N) − a_N·(1 − e^(−C))·N
dC/dt = u − γ·C
```

* **Base parameters:** `r_T=0.10, r_N=0.30, a_T=0.50, a_N=0.25, γ=0.50`.
* **Per patient:** `r_T, r_N, a_T, a_N` are each scaled by `U(1−v, 1+v)`.
* **Integration:** RK4 with 20 substeps per day.

## MDP

| | |
|---|---|
| Step | 1 day; horizon 90 days |
| Actions | dose `u ∈ {0, 0.5, 1.0}` |
| State (full observation) | `[log10(T)/3, N, C, t/90]` |
| Reward | `log10(T_prev) − log10(T) − 0.05·u` |
| Cure | `T < 1e-3` → `+10`, episode ends |
| Toxicity | `N < 0.5` → `−10`, episode ends |
| Timeout | day 90 → `−3·(log10(T)+3)` |

## Experiments (`experiments.py`)

| | Variability `v` | Observation |
|---|---|---|
| E1 | 0.2 | full state |
| E2 | 0.4 | full state |
| E3 | 0.4 | partial: weekly tumor scan with noise σ=0.1 on log10(T), daily blood count with noise σ=0.02, drug level hidden; the observation is `[scan, N̂, days since scan/7, t/90, previous dose]` |

## Data splits (env seeds)

| Split | Seeds | Used for |
|---|---|---|
| Training | random | training only |
| Validation | 5000–5049 | Optuna tuning only |
| Test | 10000–10299 | every reported number |

## Results (test patients; mean ± std over seeds)

| | E1 cure / toxicity | E2 cure / toxicity | E3 cure / toxicity |
|---|---|---|---|
| Weekly 3on/4off | 61.0 / 39.0 | 56.0 / 42.7 | 56.0 / 42.7 |
| Threshold rule | 65.3 / 0.0 | 56.7 / 1.0 | 60.7 / 1.3 |
| Q-learning | 98.3±0.8 / 1.2±1.1 | 82.4±0.8 / 10.4±1.4 | 75.6±0.8 / 17.1±2.5 |
| DQN (tuned) | 94.0±4.1 / 5.1±4.3 | 69.3±6.7 / 29.7±6.1 | 67.6±1.5 / 30.3±2.8 |
| PPO (tuned) | **98.4±0.4 / 0.0±0.0** | **86.8±0.7 / 8.6±1.6** | **82.7±3.4 / 7.0±3.7** |
| PPO-hist (7-day stack) | – | – | 75.9±5.6 / 21.2±7.2 |
| DQN-hist (7-day stack) | – | – | 65.0±13.1 / 35.0±13.1 |

* **Seeds:** 5 per agent in E1, 3 in E2 and E3.
* **E2 reference:** 87.0% of E2 test patients are curable by at least one of 19 simple rules tuned per patient in hindsight (`oracle.py`).
* **More:** full tables are in `results/` and figures in `figures/`.

## Reproduce

```bash
pip install -r requirements.txt
bash run_all.sh          # about 2 h on a 2-core CPU, no GPU needed
```

Single steps:

```bash
python tune.py --algo PPO --trials 16 --steps 120000          # tuning on E1 validation patients
python train.py --exp E1 --algo PPO --seed 0 --steps 200000 --config tuning/best_PPO.json
python train.py --exp E3 --algo PPO --hist 7 --name PPO-hist --seed 0 --steps 300000 --config tuning/best_PPO.json
python qlearning.py --exp E2 --episodes 20000 --seed 0
python evaluate.py --exp E3                                    # tables + figures
```

## Files

| File | Contents |
|---|---|
| `chemo_env.py` | Gymnasium environment (full and partial observation) |
| `experiments.py` | settings of E1, E2, E3 |
| `baselines.py` | hand-crafted policies and the episode runner |
| `agents.py` | SB3 model factory, default hyperparameters, history stacking |
| `qlearning.py` | tabular Q-learning |
| `tune.py`, `plot_tuning.py` | Optuna search and its figure |
| `train.py` | training of PPO / DQN |
| `evaluate.py` | test-set evaluation → `results/<exp>_*.csv`, `figures/<exp>_*.png` |
| `policy_map.py` | policy maps and dose usage of the E1 agents |
| `oracle.py` | per-patient hindsight reference |
| `run_all.sh` | full pipeline |

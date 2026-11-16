#!/usr/bin/env bash
# Full pipeline: tuning -> 5 seeds per agent -> evaluation
set -e
python tune.py --algo PPO --trials 16 --steps 120000
python tune.py --algo DQN --trials 12 --steps 120000
for s in 0 1 2 3 4; do
  python train.py --algo PPO --steps 300000 --seed $s --config tuning/best_PPO.json
  python train.py --algo DQN --steps 300000 --seed $s --config tuning/best_DQN.json
  python qlearning.py --episodes 20000 --seed $s
done
python evaluate.py

#!/usr/bin/env bash
# Full pipeline: tuning -> training (all experiments, several seeds) -> evaluation
set -e
# 1. hyperparameter tuning on E1 validation patients
python tune.py --algo PPO --trials 16 --steps 120000
python tune.py --algo DQN --trials 12 --steps 120000
python plot_tuning.py

# 2. E1: 5 seeds per agent (+ untuned DQN, 3 seeds)
for s in 0 1 2 3 4; do
  python train.py --exp E1 --algo PPO --steps 200000 --seed $s --config tuning/best_PPO.json
  python train.py --exp E1 --algo DQN --steps 200000 --seed $s --config tuning/best_DQN.json
  python qlearning.py --exp E1 --episodes 20000 --seed $s
done
for s in 0 1 2; do
  python train.py --exp E1 --algo DQN --steps 200000 --seed $s --name DQN-default
done

# 3. E2 and E3: 3 seeds per agent
for s in 0 1 2; do
  for e in E2 E3; do
    python train.py --exp $e --algo PPO --steps 300000 --seed $s --config tuning/best_PPO.json
    python train.py --exp $e --algo DQN --steps 300000 --seed $s --config tuning/best_DQN.json
    python qlearning.py --exp $e --episodes 20000 --seed $s
  done
  python train.py --exp E3 --algo PPO --hist 7 --name PPO-hist --steps 300000 --seed $s --config tuning/best_PPO.json
  python train.py --exp E3 --algo DQN --hist 7 --name DQN-hist --steps 300000 --seed $s --config tuning/best_DQN.json
done

# 4. evaluation on the test patients
for e in E1 E2 E3; do python evaluate.py --exp $e; done
python policy_map.py
python oracle.py --exp E2

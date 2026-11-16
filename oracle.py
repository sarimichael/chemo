"""Hindsight reference for E2/E3: for each test patient, try a family of simple rules
and keep the best outcome for THAT patient. A patient counts as curable if any rule
cures it. This is not a policy (it needs to know the outcome in advance); it gives a
lower bound on how many test patients can be cured at all.

Rule family: dose `d` while healthy cells N > `n_hi`, else 0 (the threshold rule, tuned
per patient), plus constant dose 0.5.

Usage: python oracle.py --exp E2
"""
import argparse

import numpy as np

from agents import make_env
from baselines import run_policy
from experiments import EXPERIMENTS

ap = argparse.ArgumentParser()
ap.add_argument("--exp", default="E2")
EXP = ap.parse_args().exp
# use the full-observation env so the rule sees the true N
kw = dict(EXPERIMENTS[EXP])
for k in ("scan_every", "scan_noise", "blood_noise", "hide_drug"):
    kw.pop(k, None)
env = make_env(kw)

rules = [lambda o, t: 1]
for n_hi in np.arange(0.55, 0.96, 0.05):
    for a in (1, 2):
        rules.append(lambda o, t, n_hi=n_hi, a=a: a if o[1] > n_hi else 0)

curable = 0
for i in range(300):
    outcomes = [run_policy(env, r, seed=10_000 + i)["outcome"] for r in rules]
    curable += "cure" in outcomes
print(f"{EXP}: {curable}/300 test patients ({100 * curable / 300:.1f}%) are cured by at least one rule in the family")

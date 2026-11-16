"""Hand-crafted dosing policies used as baselines."""
import numpy as np


def no_treatment(obs, t):
    return 0


def max_dose(obs, t):  # "maximum tolerated dose" every day
    return 2


def constant_low(obs, t):
    return 1


def cyclic(obs, t, on=3, off=4):  # weekly protocol: 3 days on, 4 days off
    return 2 if (t % (on + off)) < on else 0


def threshold(obs, t, n_hi=0.75):  # dose only while healthy cells are high enough
    return 2 if obs[1] > n_hi else 0


BASELINES = {
    "No treatment": no_treatment,
    "Max dose": max_dose,
    "Constant low": constant_low,
    "Weekly 3on/4off": cyclic,
    "Threshold rule": threshold,
}


def run_policy(env, policy, seed=None, sb3_model=None):
    obs, info = env.reset(seed=seed)
    st = env.unwrapped.state
    traj = {"T": [st[0]], "N": [st[1]], "C": [st[2]], "dose": []}
    total, t, done = 0.0, 0, False
    while not done:
        if sb3_model is not None:
            a, _ = sb3_model.predict(obs, deterministic=True)
        else:
            a = policy(obs, t)
        obs, r, term, trunc, info = env.step(a)
        total += r
        t += 1
        done = term or trunc
        for k in ("T", "N", "C"):
            traj[k].append(info[k])
        traj["dose"].append(info["dose"])
    traj["outcome"] = info["outcome"]
    traj["return"] = total
    traj["days"] = t
    return traj

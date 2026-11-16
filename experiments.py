"""The three experimental settings. Each maps to ChemoEnv keyword arguments."""

EXPERIMENTS = {
    # E1: full observation, moderate patient variability (+-20%)
    "E1": dict(patient_variability=0.2),
    # E2: full observation, wide patient variability (+-40%)
    "E2": dict(patient_variability=0.4),
    # E3: wide variability AND clinic-like partial observation:
    #     tumor scanned weekly with noise, noisy daily blood count, drug level hidden
    "E3": dict(patient_variability=0.4, scan_every=7, scan_noise=0.1, blood_noise=0.02, hide_drug=True),
}
HIST = 7  # days of observation history stacked for the "-hist" agents

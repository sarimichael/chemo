"""
Chemotherapy dosing environment (Gymnasium).

Model (all quantities normalized):
    T : tumor cell population (fraction of carrying capacity), logistic growth
    N : normal (healthy) cell population, logistic regrowth
    C : drug concentration in the body, first-order elimination

    dT/dt = r_T * T * (1 - T)   - a_T * (1 - exp(-C)) * T
    dN/dt = r_N * N * (1 - N)   - a_N * (1 - exp(-C)) * N
    dC/dt = u                   - gamma * C

The agent picks a daily dose u from a discrete set. The episode ends when
the tumor is eradicated (cure), normal cells fall below a toxicity threshold
(treatment failure), or the horizon is reached.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

DEFAULT_PARAMS = dict(
    r_T=0.10,     # tumor growth rate [1/day]
    r_N=0.30,     # normal-cell regrowth rate [1/day]
    a_T=0.50,     # drug kill rate on tumor
    a_N=0.25,     # drug kill rate on normal cells
    gamma=0.50,   # drug elimination rate [1/day]
)


class ChemoEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        doses=(0.0, 0.5, 1.0),
        horizon=90,
        T0=0.30,
        N_min=0.50,
        cure_level=1e-3,
        dose_cost=0.05,
        fail_penalty=10.0,
        cure_bonus=10.0,
        patient_variability=0.0,  # e.g. 0.2 -> each parameter scaled by U(0.8, 1.2)
        substeps=20,
        params=None,
    ):
        super().__init__()
        self.doses = np.asarray(doses, dtype=np.float64)
        self.horizon = horizon
        self.T0 = T0
        self.N_min = N_min
        self.cure_level = cure_level
        self.dose_cost = dose_cost
        self.fail_penalty = fail_penalty
        self.cure_bonus = cure_bonus
        self.var = patient_variability
        self.substeps = substeps
        self.base_params = dict(DEFAULT_PARAMS if params is None else params)

        self.action_space = spaces.Discrete(len(self.doses))
        # obs: [log10(T) scaled, N, C, time fraction]
        self.observation_space = spaces.Box(
            low=np.array([-1.5, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([0.5, 1.0, 5.0, 1.0], dtype=np.float32),
        )

    # ---------- dynamics ----------
    def _deriv(self, y, u):
        T, N, C = y
        p = self.p
        kill = 1.0 - np.exp(-C)
        dT = p["r_T"] * T * (1.0 - T) - p["a_T"] * kill * T
        dN = p["r_N"] * N * (1.0 - N) - p["a_N"] * kill * N
        dC = u - p["gamma"] * C
        return np.array([dT, dN, dC])

    def _integrate_day(self, u):
        h = 1.0 / self.substeps
        y = self.state
        for _ in range(self.substeps):  # RK4
            k1 = self._deriv(y, u)
            k2 = self._deriv(y + 0.5 * h * k1, u)
            k3 = self._deriv(y + 0.5 * h * k2, u)
            k4 = self._deriv(y + h * k3, u)
            y = y + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            y[0] = max(y[0], 1e-12)
        self.state = y

    def _obs(self):
        T, N, C = self.state
        logT = np.log10(max(T, 1e-12)) / 3.0  # cure level 1e-3 -> -1
        return np.array([logT, N, C, self.t / self.horizon], dtype=np.float32)

    # ---------- gym API ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.p = dict(self.base_params)
        if self.var > 0:
            for k in ("r_T", "a_T", "a_N", "r_N"):
                self.p[k] *= self.np_random.uniform(1 - self.var, 1 + self.var)
        self.state = np.array([self.T0, 1.0, 0.0])
        self.t = 0
        return self._obs(), {"params": self.p}

    def step(self, action):
        u = self.doses[int(action)]
        T_prev = self.state[0]
        self._integrate_day(u)
        self.t += 1
        T, N, C = self.state

        # reward: log-reduction of tumor, minus drug cost
        reward = (np.log10(T_prev) - np.log10(T)) - self.dose_cost * u
        terminated = False
        outcome = None
        if N < self.N_min:
            reward -= self.fail_penalty
            terminated, outcome = True, "toxicity"
        elif T < self.cure_level:
            reward += self.cure_bonus
            terminated, outcome = True, "cure"
        truncated = self.t >= self.horizon and not terminated
        if truncated:
            outcome = "timeout"
            reward -= 3.0 * max(np.log10(T) + 3.0, 0)  # penalize residual tumor
        info = {"T": T, "N": N, "C": C, "dose": u, "outcome": outcome}
        return self._obs(), float(reward), terminated, truncated, info

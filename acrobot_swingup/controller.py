"""Energy shaping followed by local LQR; every actuator command is bounded."""
import numpy as np
from acrobot_comparison.env import advance_torque, measurements
from acrobot_feasibility.study import design, errors

DT = 0.05
TARGET_ENERGY = 19.6


def energy(state):
    t1, t2, v1, v2 = state.T
    d1, d2 = 3.5 + np.cos(t2), 1.25 + 0.5 * np.cos(t2)
    return 0.5 * d1 * v1**2 + d2 * v1 * v2 + 0.625 * v2**2 - 14.7 * np.cos(t1) - 4.9 * np.cos(t1 + t2)


def swing_torque(state, config):
    t1, t2, v1, v2 = state.T
    gap = energy(state) - TARGET_ENERGY
    if config["family"] == "power":
        torque = -config["ke"] * gap * v2 - config["kp"] * np.sin(t2) - config["kd"] * v2
    else:
        # Collocated partial feedback linearization, following Spong's formulation.
        acceleration = -config["kp"] * t2 - config["kd"] * v2 + config["ke"] * v1 * gap
        d1, d2 = 3.5 + np.cos(t2), 1.25 + 0.5 * np.cos(t2)
        phi2 = 4.9 * np.cos(t1 + t2 - np.pi / 2)
        phi1 = -0.5 * v2**2 * np.sin(t2) - v1 * v2 * np.sin(t2) + 14.7 * np.cos(t1 - np.pi / 2) + phi2
        torque = (1.25 - d2**2 / d1) * acceleration - d2 / d1 * phi1 + phi2 + 0.5 * v1**2 * np.sin(t2)
    return np.clip(torque, -config["limit"], config["limit"])


class HybridController:
    def __init__(self, configs, count):
        self.configs, self.count = configs, count
        self.gain, _ = design(DT)
        self.balancing = np.zeros(len(configs) * count, dtype=bool)

    def __call__(self, state):
        error = errors(state)
        lqr = -error @ self.gain
        near = (np.abs(error[:, :2]).max(1) < 0.25) & (np.abs(error[:, 2:]).max(1) < 1.0)
        stay = (np.abs(error[:, :2]).max(1) < 0.5) & (np.abs(error[:, 2:]).max(1) < 2)
        torque = np.empty(len(state))
        for i, config in enumerate(self.configs):
            sl = slice(i*self.count, (i+1)*self.count)
            self.balancing[sl] = (self.balancing[sl] & stay[sl]) | (near[sl] & (np.abs(lqr[sl]) < config["limit"]))
            torque[sl] = np.where(self.balancing[sl], np.clip(lqr[sl], -config["limit"], config["limit"]), swing_torque(state[sl], config))
        return torque


def rollout(configs, seed=41000, count=8, seconds=60, physics_dt=0.01, capture=False):
    initial = np.random.default_rng(seed).uniform(-0.1, 0.1, (count, 4))
    state = np.tile(initial, (len(configs), 1))
    controller = HybridController(configs, count)
    longest, streak, max_height = (np.zeros(len(state)) for _ in range(3))
    traces, actions = [], []
    for step in range(round(seconds / DT)):
        torque = controller(state)
        if capture:
            traces.append(state.copy())
            actions.append(torque.copy())
        for _ in range(round(DT / physics_dt)):
            state = advance_torque(state, torque, physics_dt)
            _, height, upright = measurements(state)
            streak = np.where(upright, streak + physics_dt, 0)
            longest = np.maximum(longest, streak)
            max_height = np.maximum(max_height, height)
    results = []
    for i, config in enumerate(configs):
        sl = slice(i*count, (i+1)*count)
        results.append({"config": config, "seed": seed, "episodes": count, "seconds": seconds,
                        "hold_5s_rate": float(np.mean(longest[sl] >= 5 - 1e-8)),
                        "final_5s_rate": float(np.mean(streak[sl] >= 5 - 1e-8)),
                        "mean_longest_hold": float(np.mean(longest[sl])),
                        "mean_max_height": float(np.mean(max_height[sl])),
                        "per_episode_hold": longest[sl].tolist()})
    return results, (np.asarray(traces), np.asarray(actions))


def candidate_configs(limit=1.0):
    configs = []
    for family in ("power", "pfl"):
        for ke in ([0.03, 0.1, 0.3, 1., 3.] if family == "power" else [0.01, 0.03, 0.1, 0.3, 1.]):
            for kp in ([0, 0.1, 0.5] if family == "power" else [0.5, 2., 8.]):
                for kd in ([0, 0.05] if family == "power" else [0.1, 1.]):
                    configs.append(dict(family=family, ke=ke, kp=kp, kd=kd, limit=limit))
    return configs

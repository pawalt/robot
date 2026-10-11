"""Batched Gymnasium Acrobot dynamics with an upright-balance objective.

The dynamics algebra follows Gymnasium's BSD-licensed AcrobotEnv (book variant).
Changes: 50 ms control, no height termination, fixed-duration task, dense reward.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

DT = 0.05
HORIZON = 400


def derivatives(state, torque):
    t1, t2, v1, v2 = state.T
    d1 = 3.5 + np.cos(t2)
    d2 = 1.25 + 0.5 * np.cos(t2)
    phi2 = 4.9 * np.cos(t1 + t2 - np.pi / 2)
    phi1 = (-0.5 * v2**2 * np.sin(t2) - v2 * v1 * np.sin(t2)
            + 14.7 * np.cos(t1 - np.pi / 2) + phi2)
    a2 = (torque + d2 / d1 * phi1 - 0.5 * v1**2 * np.sin(t2) - phi2) / (1.25 - d2**2 / d1)
    a1 = -(d2 * a2 + phi1) / d1
    return np.column_stack((v1, v2, a1, a2))


def advance(state, actions, dt=DT):
    return advance_torque(state, np.asarray(actions) - 1.0, dt)


def advance_torque(state, torque, dt=DT):
    """Same dynamics with directly specified elbow torque (N m)."""
    torque = np.asarray(torque)
    k1 = derivatives(state, torque)
    k2 = derivatives(state + dt * k1 / 2, torque)
    k3 = derivatives(state + dt * k2 / 2, torque)
    k4 = derivatives(state + dt * k3, torque)
    result = state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    result[:, :2] = (result[:, :2] + np.pi) % (2 * np.pi) - np.pi
    result[:, 2:] = np.clip(result[:, 2:], [-4 * np.pi, -9 * np.pi], [4 * np.pi, 9 * np.pi])
    return result


def measurements(state):
    t1, t2, v1, v2 = state.T
    height = (-np.cos(t1) - np.cos(t1 + t2) + 2) / 4
    # Both links must be vertical, and both absolute angular speeds small.
    upright = ((-np.cos(t1) > np.cos(np.deg2rad(10)))
               & (-np.cos(t1 + t2) > np.cos(np.deg2rad(10)))
               & (np.abs(v1) < 0.5) & (np.abs(v1 + v2) < 0.5))
    speed2 = v1**2 + (v1 + v2)**2
    reward = DT * (height + 2 * upright - 0.05 * height**4 * speed2)
    return reward, height, upright


def observations(state, step, horizon):
    t1, t2, v1, v2 = state.T
    return np.column_stack((np.cos(t1), np.sin(t1), np.cos(t2), np.sin(t2),
                            v1 / (4 * np.pi), v2 / (9 * np.pi),
                            np.full(len(state), 1 - step / horizon))).astype(np.float32)


def sample_starts(rng, count, mode="mixed"):
    state = rng.uniform(-0.1, 0.1, (count, 4))
    near = (rng.random(count) < 0.75) if mode == "mixed" else np.full(count, mode == "balance")
    state[near, :2] *= 0.8
    state[near, 0] += np.pi
    state[:, :2] = (state[:, :2] + np.pi) % (2 * np.pi) - np.pi
    return state


class BalanceAcrobot(gym.Env):
    """Seven observations: Acrobot state with normalized speeds, plus time left.

    This is a finite 20-second task; its known horizon is a true termination.
    Supplying time left avoids hiding a variable that determines future return.
    """
    metadata = {"render_modes": []}

    def __init__(self, horizon=HORIZON):
        self.horizon = horizon
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-1.0, 1.0, (7,), dtype=np.float32)
        self.state = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = sample_starts(self.np_random, 1, (options or {}).get("mode", "mixed"))
        self.steps = 0
        return observations(self.state, 0, self.horizon)[0], {}

    def step(self, action):
        if self.state is None or self.steps >= self.horizon:
            raise RuntimeError("Reset before stepping")
        if not self.action_space.contains(action):
            raise ValueError("Invalid action")
        self.state = advance(self.state, [action])
        self.steps += 1
        reward, height, upright = measurements(self.state)
        return observations(self.state, self.steps, self.horizon)[0], float(reward[0]), self.steps == self.horizon, False, {"height": float(height[0]), "upright": bool(upright[0])}

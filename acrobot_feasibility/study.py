"""Sampled-data LQR on the same nonlinear Acrobot dynamics as the RL study."""

import numpy as np
from scipy.linalg import expm, solve_discrete_are

from acrobot_comparison.env import advance_torque, derivatives, measurements

PHYSICS_DT = 0.005
DURATION = 20.0
COUNT = 128
SEED = 30000
SCALES = (0.05, 0.25, 1.0, 2.0)
LIMITS = (0.5, 1.0, 2.0)
CONTROLLERS = ("continuous", "discrete3", "discrete9", "passive")


def errors(state):
    result = state.copy()
    result[:, 0] = (result[:, 0] - np.pi + np.pi) % (2 * np.pi) - np.pi
    result[:, 1] = (result[:, 1] + np.pi) % (2 * np.pi) - np.pi
    return result


def design(dt):
    equilibrium = np.array([[np.pi, 0., 0., 0.]])
    epsilon = 1e-6
    columns = []
    for axis in range(4):
        shift = np.zeros((1, 4))
        shift[0, axis] = epsilon
        columns.append(((derivatives(equilibrium + shift, np.zeros(1))
                         - derivatives(equilibrium - shift, np.zeros(1))) / (2 * epsilon))[0])
    a = np.column_stack(columns)
    b = ((derivatives(equilibrium, np.ones(1) * epsilon)
          - derivatives(equilibrium, -np.ones(1) * epsilon)) / (2 * epsilon)).T
    # Exact zero-order-hold discretization of the linearized continuous system.
    augmented = np.zeros((5, 5))
    augmented[:4, :4], augmented[:4, 4:] = a, b
    discrete = expm(augmented * dt)
    ad, bd = discrete[:4, :4], discrete[:4, 4:]
    q, r = np.diag([10., 10., 1., 1.]), np.array([[1.]])
    p = solve_discrete_are(ad, bd, q, r)
    gain = np.linalg.solve(r + bd.T @ p @ bd, bd.T @ p @ ad)
    controllability = np.column_stack([np.linalg.matrix_power(a, i) @ b for i in range(4)])
    return gain[0], {"A": a.tolist(), "B": b.tolist(), "K": gain[0].tolist(),
                     "controllability_rank": int(np.linalg.matrix_rank(controllability)),
                     "open_loop_eigenvalues": np.linalg.eigvals(a).real.tolist(),
                     "closed_loop_eigenvalue_magnitudes": np.abs(np.linalg.eigvals(ad - bd @ gain)).tolist(),
                     "Q_diagonal": np.diag(q).tolist(), "R": 1.0}


def starting_states(scale, count=COUNT):
    state = np.random.default_rng(SEED).uniform(-1, 1, (count, 4))
    state *= np.array([0.08, 0.08, 0.1, 0.1]) * scale
    state[:, 0] += np.pi
    return state


def control(state, gain, kind, limit):
    desired = -errors(state) @ gain
    if kind == "passive":
        return np.zeros(len(state)), desired
    bounded = np.clip(desired, -limit, limit)
    if kind == "discrete3":
        bounded = np.round(bounded / limit) * limit
    elif kind == "discrete9":
        bounded = np.round(bounded / (limit / 4)) * (limit / 4)
    elif kind != "continuous":
        raise ValueError(kind)
    return bounded, desired


def simulate(dt, configurations, count=COUNT, physics_dt=PHYSICS_DT, capture=False):
    gain, linear = design(dt)
    substeps = round(dt / physics_dt)
    assert np.isclose(substeps * physics_dt, dt)
    state = np.concatenate([starting_states(c["scale"], count) for c in configurations])
    streak = np.zeros(len(state))
    longest = np.zeros(len(state))
    balanced_time = np.zeros(len(state))
    saturation = np.zeros(len(state))
    torque_squared = np.zeros(len(state))
    tail_error = np.zeros(len(state))
    trace = [{"time": 0.0, "states": state.tolist(), "torques": [0.] * len(state)}] if capture else []
    total_physics_steps = round(DURATION / physics_dt)
    for step in range(round(DURATION / dt)):
        torque = np.empty(len(state))
        for index, config in enumerate(configurations):
            sl = slice(index * count, (index + 1) * count)
            torque[sl], desired = control(state[sl], gain, config["controller"], config["limit"])
            saturation[sl] += (np.abs(desired) > config["limit"]) * dt if config["controller"] != "passive" else 0
        torque_squared += torque**2 * dt
        for substep in range(substeps):
            state = advance_torque(state, torque, physics_dt)
            upright = measurements(state)[2]
            streak = np.where(upright, streak + physics_dt, 0)
            longest = np.maximum(longest, streak)
            balanced_time += upright * physics_dt
            tick = step * substeps + substep + 1
            if tick > total_physics_steps - round(5 / physics_dt):
                e = errors(state)
                tail_error += (e[:, 0]**2 + (e[:, 0] + e[:, 1])**2) * physics_dt / 10
            if capture and tick % round(0.05 / physics_dt) == 0:
                trace.append({"time": tick * physics_dt, "states": state.tolist(), "torques": torque.tolist()})
    rows = []
    for index, config in enumerate(configurations):
        sl = slice(index * count, (index + 1) * count)
        rows.append({**config, "dt": dt, "episodes": count,
                     "hold_5s_rate": float(np.mean(longest[sl] >= 5 - 1e-8)),
                     "final_5s_rate": float(np.mean(streak[sl] >= 5 - 1e-8)),
                     "mean_longest_hold": float(np.mean(longest[sl])),
                     "balanced_fraction": float(np.mean(balanced_time[sl] / DURATION)),
                     "saturation_fraction": float(np.mean(saturation[sl] / DURATION)),
                     "rms_torque": float(np.sqrt(np.mean(torque_squared[sl] / DURATION))),
                     "final_angle_rms_degrees": float(np.rad2deg(np.sqrt(np.mean(tail_error[sl])))),
                     "per_episode_final_5s": (streak[sl] >= 5 - 1e-8).tolist()})
    return {"dt": dt, "linearization": linear, "conditions": rows}, trace


def run_interval(dt):
    configs = [{"controller": c, "limit": limit, "scale": scale}
               for c in CONTROLLERS for limit in LIMITS for scale in SCALES
               if c != "passive" or limit == 1.0]
    return simulate(dt, configs)[0]

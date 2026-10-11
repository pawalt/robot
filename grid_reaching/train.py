"""Run with: uv run python -m grid_reaching.train"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .env import GridReachingEnv

DEFAULT_OUTPUT = Path(__file__).parent / "artifacts"


def q_update(q, state, action, reward, next_state, terminated, alpha, gamma):
    # A time limit stops the rollout, but does not make the next state terminal.
    future = 0.0 if terminated else float(q[next_state].max())
    target = reward + gamma * future
    q[state, action] += alpha * (target - q[state, action])


def train(env, episodes=5000, seed=0, alpha=0.1, gamma=0.95):
    if episodes < 1:
        raise ValueError("episodes must be positive")
    rng = np.random.default_rng(seed)
    q = np.zeros((env.observation_space.n, env.action_space.n))
    history = []
    for episode in range(episodes):
        # Explore heavily initially; decay to 5% over the first 80% of training.
        epsilon = max(0.05, 1.0 - 0.95 * episode / max(1, 0.8 * episodes))
        state, _ = env.reset(seed=seed if episode == 0 else None)
        total_reward = 0.0
        while True:
            if rng.random() < epsilon:
                action = int(rng.integers(env.action_space.n))
            else:
                # Random tie-breaking avoids preferring 'up' in an empty table.
                best = np.flatnonzero(q[state] == q[state].max())
                action = int(rng.choice(best))
            next_state, reward, terminated, truncated, _ = env.step(action)
            q_update(q, state, action, reward, next_state, terminated, alpha, gamma)
            total_reward += reward
            state = next_state
            if terminated or truncated:
                break
        history.append((episode + 1, epsilon, total_reward, env.steps, terminated))
    return q, history


def evaluate(env, q=None, seed=10000):
    """Test every non-goal start; q=None uses uniformly random actions."""
    rng = np.random.default_rng(seed)
    successes, lengths, returns, excess_steps = [], [], [], []
    for start in range(env.goal):
        state, _ = env.reset(seed=seed + start, options={"start_state": start})
        total_reward = 0.0
        while True:
            action = int(rng.integers(4)) if q is None else int(q[state].argmax())
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        successes.append(terminated)
        lengths.append(env.steps)
        returns.append(total_reward)
        if terminated:
            row, col = divmod(start, env.size)
            shortest = 2 * (env.size - 1) - row - col
            excess_steps.append(env.steps - shortest)
    return {
        "episodes": len(successes),
        "success_rate": float(np.mean(successes)),
        "mean_steps": float(np.mean(lengths)),
        "mean_return": float(np.mean(returns)),
        "mean_extra_steps_on_success": (
            float(np.mean(excess_steps)) if excess_steps else None
        ),
    }


def policy_text(q, size):
    cells = [GridReachingEnv.ARROWS[int(row.argmax())] for row in q]
    cells[-1] = "G"
    return "\n".join(" ".join(cells[i : i + size]) for i in range(0, len(cells), size))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=int, default=10)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    env = GridReachingEnv(size=args.size)
    q, history = train(env, episodes=args.episodes, seed=args.seed)
    results = {
        "training": {"episodes": args.episodes, "seed": args.seed,
                     "size": env.size, "max_steps": env.max_steps,
                     "alpha": 0.1, "gamma": 0.95},
        "evaluation_seed": 10000,
        "random": evaluate(env),
        "learned": evaluate(env, q),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez(args.output / "policy.npz", q=q, size=env.size, max_steps=env.max_steps)
    with (args.output / "training.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["episode", "epsilon", "return", "steps", "success"])
        writer.writerows(history)
    (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    (args.output / "policy.txt").write_text(policy_text(q, env.size) + "\n")
    print(json.dumps(results, indent=2))
    print("\nGreedy policy (^ > v <; G = goal):\n" + policy_text(q, env.size))
    print(f"\nSaved policy and training results to {args.output}")
    env.close()


if __name__ == "__main__":
    main()

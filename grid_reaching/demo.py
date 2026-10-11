"""Load a trained policy and print its path from the top-left corner."""

import argparse

import numpy as np

from .env import GridReachingEnv
from .train import DEFAULT_OUTPUT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default=str(DEFAULT_OUTPUT / "policy.npz"))
    parser.add_argument("--start", type=int, default=0, help="row * size + column")
    args = parser.parse_args()
    with np.load(args.policy, allow_pickle=False) as data:
        q = data["q"]
        env = GridReachingEnv(size=int(data["size"]),
                              max_steps=int(data["max_steps"]), render_mode="ansi")
    if q.shape != (env.observation_space.n, 4) or not np.isfinite(q).all():
        raise ValueError("Invalid Q-table")
    state, _ = env.reset(options={"start_state": args.start})
    print(env.render())
    path = [divmod(state, env.size)]
    total_reward = 0.0
    while True:
        action = int(q[state].argmax())
        state, reward, terminated, truncated, _ = env.step(action)
        path.append(divmod(state, env.size))
        total_reward += reward
        if terminated or truncated:
            break
    print("\nPath (row, column): " + " -> ".join(map(str, path)))
    print(f"Reached goal: {terminated}; steps: {env.steps}; return: {total_reward}")
    env.close()


if __name__ == "__main__":
    main()

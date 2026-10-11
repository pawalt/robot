"""A discrete point robot with a fixed target in the bottom-right corner."""

import gymnasium as gym
from gymnasium import spaces


class GridReachingEnv(gym.Env):
    metadata = {"render_modes": ["ansi"], "render_fps": 4}
    # Coordinates are (row, column), with row zero at the top.
    MOVES = ((-1, 0), (0, 1), (1, 0), (0, -1))
    ARROWS = ("^", ">", "v", "<")

    def __init__(self, size=10, max_steps=200, render_mode=None):
        if size < 2 or max_steps < 1:
            raise ValueError("size must be >= 2 and max_steps must be positive")
        if render_mode not in (None, "ansi"):
            raise ValueError("render_mode must be None or 'ansi'")
        self.size = size
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.observation_space = spaces.Discrete(size * size)
        self.action_space = spaces.Discrete(4)
        self.goal = size * size - 1
        self.state = None
        self.steps = 0
        self.done = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        start = (options or {}).get("start_state")
        if start is None:
            start = int(self.np_random.integers(self.goal))
        if not self.observation_space.contains(start) or start == self.goal:
            raise ValueError("start_state must be a non-goal cell")
        self.state = int(start)
        self.steps = 0
        self.done = False
        return self.state, {}

    def step(self, action):
        if self.done:
            raise RuntimeError("Call reset() before stepping a new episode")
        if not self.action_space.contains(action):
            raise ValueError("action must be 0=up, 1=right, 2=down, or 3=left")
        row, col = divmod(self.state, self.size)
        dr, dc = self.MOVES[action]
        row = min(self.size - 1, max(0, row + dr))
        col = min(self.size - 1, max(0, col + dc))
        self.state = row * self.size + col
        self.steps += 1
        terminated = self.state == self.goal
        truncated = self.steps >= self.max_steps and not terminated
        self.done = terminated or truncated
        reward = 10.0 if terminated else -1.0
        return self.state, reward, terminated, truncated, {}

    def render(self):
        if self.render_mode != "ansi":
            return None
        cells = ["."] * self.observation_space.n
        cells[self.goal] = "G"
        if self.state is not None:
            cells[self.state] = "R"
        return "\n".join(
            " ".join(cells[i : i + self.size])
            for i in range(0, len(cells), self.size)
        )

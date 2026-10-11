import unittest

import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env

from .env import GridReachingEnv
from .train import evaluate, q_update, train


class GridReachingTests(unittest.TestCase):
    def test_gymnasium_contract(self):
        env = gym.make("Robot/GridReaching-v0", render_mode="ansi")
        check_env(env.unwrapped, skip_render_check=False)
        env.close()

    def test_boundary_goal_and_time_limit(self):
        env = GridReachingEnv(size=3, max_steps=2)
        env.reset(options={"start_state": 0})
        self.assertEqual(env.step(0)[:4], (0, -1.0, False, False))
        self.assertEqual(env.step(3)[:4], (0, -1.0, False, True))
        env.reset(options={"start_state": 7})
        self.assertEqual(env.step(1)[:4], (8, 10.0, True, False))
        with self.assertRaises(RuntimeError):
            env.step(0)

    def test_terminal_vs_truncated_bootstrap(self):
        q = np.zeros((2, 4))
        q[1] = 100.0
        q_update(q, 0, 0, 10, 1, True, 1.0, 0.9)
        self.assertEqual(q[0, 0], 10.0)
        q_update(q, 0, 1, -1, 1, False, 1.0, 0.9)
        self.assertEqual(q[0, 1], 89.0)

    def test_learning_reaches_every_start_optimally(self):
        env = GridReachingEnv(size=4)
        q, _ = train(env, episodes=1500, seed=7)
        result = evaluate(env, q)
        self.assertEqual(result["success_rate"], 1.0)
        self.assertEqual(result["mean_extra_steps_on_success"], 0.0)


if __name__ == "__main__":
    unittest.main()

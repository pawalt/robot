import unittest

import numpy as np
import torch
from gymnasium.envs.classic_control.acrobot import AcrobotEnv
from gymnasium.utils.env_checker import check_env

from .env import BalanceAcrobot, DT, advance, measurements
from .train import gae, grouped_advantages


class ComparisonTests(unittest.TestCase):
    def test_batched_dynamics_match_gymnasium(self):
        rng = np.random.default_rng(5)
        state = rng.uniform(-2, 2, (12, 4))
        envs = [AcrobotEnv() for _ in state]
        for env, s in zip(envs, state):
            env.dt = DT
            env.state = s.copy()
        for _ in range(30):
            actions = rng.integers(3, size=len(state))
            state = advance(state, actions)
            for index, (env, action) in enumerate(zip(envs, actions)):
                env.step(int(action))
                np.testing.assert_allclose(state[index], env.state, atol=1e-10)

    def test_gym_contract(self):
        check_env(BalanceAcrobot(), skip_render_check=True)

    def test_reward_distinguishes_balance_and_flyby(self):
        state = np.array([[np.pi, 0, 0, 0], [np.pi, 0, 2, 0], [0, 0, 0, 0]])
        reward, height, upright = measurements(state)
        np.testing.assert_array_equal(upright, [True, False, False])
        np.testing.assert_allclose(height, [1, 1, 0])
        self.assertGreater(reward[0], reward[1])
        self.assertGreater(reward[1], reward[2])

    def test_gae_terminal_target(self):
        rewards = torch.tensor([[1.0], [2.0], [3.0]])
        values = torch.tensor([[5.0], [4.0], [9.0]])
        advantages, targets = gae(rewards, values, lam=1)
        torch.testing.assert_close(targets, torch.tensor([[6.0], [5.0], [3.0]]))
        torch.testing.assert_close(advantages, targets - values)

    def test_group_baseline_and_identical_returns(self):
        rewards = torch.tensor([[1., 2., 3., 5., 5., 5.], [1., 2., 3., 5., 5., 5.]])
        advantages, zero_fraction = grouped_advantages(rewards, group_size=3)
        self.assertAlmostEqual(zero_fraction, 0.5)
        torch.testing.assert_close(advantages[:, 3:], torch.zeros((2, 3)))
        self.assertLess(float(advantages[0, 0]), 0)
        self.assertGreater(float(advantages[0, 2]), 0)
        torch.testing.assert_close(advantages[0], advantages[1])


if __name__ == "__main__":
    unittest.main()

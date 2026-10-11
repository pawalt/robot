import unittest
import numpy as np
from acrobot_comparison.env import advance, advance_torque
from .study import control, errors, design


class FeasibilityTests(unittest.TestCase):
    def test_original_actions_have_identical_dynamics(self):
        state = np.array([[0.1, 0.2, 0.3, 0.4]] * 3)
        np.testing.assert_allclose(advance(state, [0, 1, 2]), advance_torque(state, [-1, 0, 1]))

    def test_wrapped_upright_error(self):
        np.testing.assert_allclose(errors(np.array([[np.pi, 0, 0, 0], [-np.pi, 0, 0, 0]])), 0, atol=1e-12)

    def test_discrete_torque_bounds(self):
        state = np.array([[np.pi, 0, 0, 0], [np.pi+0.1, 0, 0, 0]])
        for kind in ("continuous", "discrete3", "discrete9"):
            torque, _ = control(state, np.array([12, 1, 1, 1]), kind, 1)
            self.assertTrue(np.all(np.abs(torque) <= 1))
            self.assertEqual(torque[0], 0)

    def test_linear_model_is_stabilized(self):
        _, result = design(0.05)
        self.assertEqual(result["controllability_rank"], 4)
        self.assertLess(max(result["closed_loop_eigenvalue_magnitudes"]), 1)


if __name__ == "__main__":
    unittest.main()

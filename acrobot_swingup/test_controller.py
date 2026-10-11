import unittest
import numpy as np
from acrobot_comparison.env import derivatives
from .controller import energy, TARGET_ENERGY, swing_torque


class ControllerTests(unittest.TestCase):
    def test_energy_power_identity(self):
        state = np.array([[0.3, -0.7, 0.4, -0.2]])
        torque = np.array([0.6])
        ds = derivatives(state, torque)
        h = 1e-6
        power = (energy(state+h*ds)-energy(state-h*ds))/(2*h)
        np.testing.assert_allclose(power, torque * state[:, 3], atol=1e-7)

    def test_upright_energy(self):
        self.assertAlmostEqual(float(energy(np.array([[np.pi, 0, 0, 0]]))[0]), TARGET_ENERGY)

    def test_bounded_controller(self):
        for family in ("power", "pfl"):
            torque = swing_torque(np.array([[1., 2., 3., 4.]]), dict(family=family, ke=1., kp=1., kd=1., limit=1.))
            self.assertLessEqual(abs(torque[0]), 1)

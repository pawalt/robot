import unittest
import numpy as np
from .radial_robust import initial_states,CONDITIONS

class StartingStatesTests(unittest.TestCase):
    def test_bounds_and_velocity_only_cases(self):
        x=initial_states(42,32)
        self.assertEqual(x.shape,(32*len(CONDITIONS),4))
        for i,(_,a,v) in enumerate(CONDITIONS):
            block=x[i*32:(i+1)*32]
            self.assertTrue(np.all(np.abs(block)<=np.array([a,a,v,v])))
            self.assertTrue(np.all(np.any(block!=0,axis=1)))
        np.testing.assert_array_equal(x[-64:,:2],np.zeros((64,2)))
        np.testing.assert_array_equal(x,initial_states(42,32))
        self.assertFalse(np.array_equal(x,initial_states(43,32)))

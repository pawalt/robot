import unittest
import numpy as np
from acrobot_comparison.env import advance_torque
from .trajectory import step_raw, TrajectoryController


class TrajectoryTests(unittest.TestCase):
    def test_unwrapped_integrator_matches_environment(self):
        x=np.array([[3.14,.2,.3,-.4],[.2,-.6,.1,.2]])
        u=np.array([.8,-.5])
        actual=step_raw(x,u)
        for _ in range(2): x=advance_torque(x,u,.025)
        actual[:,:2]=(actual[:,:2]+np.pi)%(2*np.pi)-np.pi
        np.testing.assert_allclose(actual,x,atol=1e-12)

    def test_optimizer_dynamics_match_replay(self):
        from .optimal import symbolic_step
        state=np.array([.4,-.7,.3,-.2])
        actual=np.array(symbolic_step()(state,.8)).ravel()
        expected=step_raw(state[None,:],np.array([.8]),substeps=5)[0]
        np.testing.assert_allclose(actual,expected,atol=1e-12)

    def test_tracker_wrap_reset_and_handoff(self):
        tracker=TrajectoryController(np.array([[np.pi,0,0,0],[np.pi,0,0,0]]),
                                     np.array([.2]),np.array([[1.,1.,1.,1.]]))
        self.assertAlmostEqual(tracker(np.array([-np.pi,0,0,0])),.2)
        self.assertAlmostEqual(tracker(np.array([-np.pi,0,0,0])),0.)
        tracker.reset()
        self.assertEqual(tracker(np.array([0.,0,10.,10.])), -1.)

    def test_capture_handoff_latches(self):
        tracker=TrajectoryController(np.tile([np.pi,0.,0.,0.],(4,1)),
                np.full(3,.5),np.zeros((3,4)),capture_region={
                    'lyapunov_matrix':np.eye(4),'empirical_level':.01})
        self.assertEqual(float(tracker(np.array([np.pi,0.,0.,0.]))),0.)
        self.assertTrue(tracker.balancing)
        tracker(np.array([np.pi+.3,0.,0.,0.]))
        self.assertTrue(tracker.balancing)
        tracker.reset()
        self.assertFalse(tracker.balancing)

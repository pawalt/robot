import unittest
import numpy as np
from .reward_sweep import compute_reward,potential
from .train import DT
from acrobot_comparison.env import measurements

class RewardTests(unittest.TestCase):
    def test_baseline_matches_previous(self):
        x=np.array([[0.,0.,1.,2.],[np.pi,0.,0.,0.]])
        _,h,u=measurements(x)
        np.testing.assert_allclose(compute_reward(x,x,np.eye(4),'baseline',1),DT*(h+5*u-.01*(x[:,2]**2+x[:,3]**2)))
    def test_braking_allows_bottom_momentum(self):
        x=np.array([[0.,0.,1.,2.]])
        self.assertGreater(compute_reward(x,x,np.eye(4),'braking',1)[0],compute_reward(x,x,np.eye(4),'baseline',1)[0])
    def test_potential_telescopes(self):
        states=np.array([[0.,0.,0.,0.],[1.,.1,2.,0.],[np.pi,0.,0.,0.]])
        terms=[(compute_reward(states[i:i+1],states[i+1:i+2],np.eye(4),'capture',2)-compute_reward(states[i:i+1],states[i+1:i+2],np.eye(4),'braking',2))[0] for i in range(2)]
        expected=5*(.999**2*potential(states[-1:],np.eye(4),2)[0]-potential(states[:1],np.eye(4),2)[0])
        self.assertAlmostEqual(terms[0]+.999*terms[1],expected)

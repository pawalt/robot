import unittest
import numpy as np
import torch
from .policy import Actor,features
from .train import advantages

class NeuralTests(unittest.TestCase):
    def test_periodic_state_features(self):
        x=np.array([[.3,-.7,.1,.4]])
        shifted=x.copy();shifted[:,:2]+=2*np.pi
        np.testing.assert_allclose(features(x),features(shifted),atol=1e-6)

    def test_continuing_gae_bootstraps(self):
        rewards=torch.tensor([[1.],[2.]])
        adv,targets=advantages(rewards,torch.zeros_like(rewards),torch.tensor([3.]),gamma=1,lam=1)
        torch.testing.assert_close(targets,torch.tensor([[6.],[5.]]))

    def test_actor_has_no_time_or_teacher_input_and_bounds_torque(self):
        actor=Actor()
        with torch.no_grad():actor.net[-1].bias.fill_(100)
        action=actor.act([[0,0,0,0],[np.pi,0,0,0]])
        np.testing.assert_equal(action,np.ones(2))

    def test_curriculum_start_uses_requested_remaining_time(self):
        from .curriculum import starts
        reference=np.zeros((601,4));reference[:,0]=np.arange(601)
        x=starts(np.random.default_rng(0),reference,2.,3,noise=0)
        np.testing.assert_equal(x[:,0],[560,560,560])
        x=starts(np.random.default_rng(0),reference,30.,3,noise=0)
        np.testing.assert_equal(x[:,0],[0,0,0])

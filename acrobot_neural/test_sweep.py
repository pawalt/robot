import unittest
import torch
from .policy import Actor
from .sweep import make_actor,flat,put

class SweepTests(unittest.TestCase):
    def test_larger_and_expert_networks_preserve_initial_function(self):
        torch.manual_seed(4);base=Actor();obs=torch.randn(100,6)
        for arm in ['wide256','wide512','residual256','experts']:
            actor=make_actor(arm,base)
            torch.testing.assert_close(actor(obs),base(obs),atol=1e-6,rtol=1e-5)

    def test_parameter_restore(self):
        actor=Actor();params=list(actor.parameters());before=flat(params)
        put(params,torch.zeros_like(before));put(params,before)
        torch.testing.assert_close(flat(params),before)

import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from .radial import RadialActor,Deployment,ExportedActor

class RadialTests(unittest.TestCase):
    def test_export_is_self_contained_and_clips(self):
        actor=RadialActor(np.zeros((2,6)),np.zeros((2,6)),np.array([3.,3.]),.03)
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'deployed.pt'
            torch.jit.trace(Deployment(actor),torch.zeros(2,4)).save(str(path))
            loaded=ExportedActor(path)
            np.testing.assert_allclose(loaded.act(np.zeros((7,4))),np.ones(7))
    def test_gaussian_mixture(self):
        actor=RadialActor(np.array([[0.]*6,[1.]*6]),np.zeros((2,6)),np.array([-.5,.5]),.1)
        with torch.no_grad():
            np.testing.assert_allclose(actor(torch.tensor([[0.]*6,[1.]*6])).numpy(),[-.5,.5],atol=1e-6)

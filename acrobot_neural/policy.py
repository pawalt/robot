"""Deployment graph: state features, neural layers, and torque clipping only."""
import numpy as np
import torch
from torch import nn


def features(state):
    state=np.atleast_2d(state)
    return np.column_stack([np.sin(state[:,0]),np.cos(state[:,0]),np.sin(state[:,1]),
                            np.cos(state[:,1]),state[:,2]/4,state[:,3]/8]).astype(np.float32)


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(6,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),
                               nn.Linear(128,128),nn.Tanh(),nn.Linear(128,1))
        self.log_std=nn.Parameter(torch.tensor(-2.5))

    def forward(self,obs):
        return self.net(obs).squeeze(-1)

    def act(self,state):
        with torch.no_grad():
            return self(torch.from_numpy(features(state))).clamp(-1,1).numpy()

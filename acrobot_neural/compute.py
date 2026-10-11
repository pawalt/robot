"""Small remote throughput benchmark; timings exclude image startup."""
import time
import numpy as np
import torch
from .policy import Actor,features
from .train import advance


def benchmark():
    rows=[]
    for threads,count in [(1,64),(1,256),(4,256),(4,512)]:
        torch.set_num_threads(threads);torch.manual_seed(90000)
        actor=Actor();rng=np.random.default_rng(90000);x=rng.uniform(-.1,.1,(count,4))
        opt=torch.optim.Adam(actor.parameters(),lr=1e-4)
        # Warm up both inference and backward paths before timing.
        for _ in range(10):
            obs=torch.from_numpy(features(x));loss=actor(obs).square().mean();opt.zero_grad();loss.backward();opt.step()
        before=time.perf_counter()
        for _ in range(200):x=advance(x,actor.act(x))
        rollout=time.perf_counter()-before
        obs=torch.randn(4096,6);target=torch.randn(4096)
        before=time.perf_counter()
        for _ in range(50):
            loss=(actor(obs)-target).square().mean();opt.zero_grad();loss.backward();opt.step()
        training=time.perf_counter()-before
        row=dict(threads=threads,environments=count,rollout_transitions_per_second=200*count/rollout,
                 supervised_samples_per_second=50*4096/training,rollout_seconds=rollout,training_seconds=training)
        rows.append(row);print(row,flush=True)
    return rows

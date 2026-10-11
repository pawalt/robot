"""Teacher distillation, on-policy PPO, and separate neural-only evaluation."""
import json
import time
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from acrobot_comparison.env import advance_torque, measurements
from acrobot_feasibility.study import design, errors
from acrobot_swingup.trajectory import TrajectoryController
from .policy import Actor,features

DT=.05
SOURCE='20261011T160551-2c0f97e7'


def advance(x,u,substeps=5):
    for _ in range(substeps): x=advance_torque(x,u,DT/substeps)
    return x


def capture_samples(rng,count,p,level=10.):
    direction=rng.normal(size=(count,4));direction/=np.linalg.norm(direction,axis=1)[:,None]
    offset=direction@np.linalg.inv(np.linalg.cholesky(p).T).T
    return np.array([np.pi,0,0,0])+offset*np.sqrt(level)*rng.random((count,1))**.25


def evaluate(actor,p,seed=72000,count=32,mode='balance',seconds=45,physics_steps=10,capture=False,capture_index=0):
    rng=np.random.default_rng(seed)
    if mode=='balance': x=capture_samples(rng,count,p)
    elif mode=='bottom':
        x=rng.uniform(-.01,.01,(count,4));x[0]=0
    else: x=np.tile([np.pi,0.,0.,0.],(count,1))
    streak=np.zeros(count);longest=streak.copy();pre_kick=np.zeros(count,dtype=bool)
    trace=[];actions=[]
    for t in range(round(seconds/DT)):
        if mode.startswith('kick') and t==100:
            pre_kick=measurements(x)[2]
            scale=float(mode.split('_')[1]);x[:,2:]+=rng.uniform(-scale,scale,(count,2))
            streak[:]=0;longest[:]=0
        u=actor.act(x)
        if capture:trace.append(x[capture_index].copy());actions.append(u[capture_index])
        for _ in range(physics_steps):
            x=advance_torque(x,u,DT/physics_steps)
            streak=np.where(measurements(x)[2],streak+DT/physics_steps,0)
            longest=np.maximum(longest,streak)
    return dict(mode=mode,episodes=count,seed=seed,seconds=seconds,physics_dt=DT/physics_steps,
                final_10s_rate=float(np.mean(streak>=10-1e-8)),mean_longest_hold=float(longest.mean()),
                per_episode_final_hold=streak.tolist(),pre_kick_balanced_rate=float(pre_kick.mean()) if mode.startswith('kick') else None), (trace,actions)


def balance_data(rng,p,count=50000):
    x=capture_samples(rng,count,p,level=30.)
    # Mix recoverable states with tiny states; precision near equilibrium matters.
    x[:count//3]=np.array([np.pi,0,0,0])+rng.normal(size=(count//3,4))*.003
    gain,_=design(DT)
    return features(x),np.clip(-errors(x)@gain,-1.2,1.2).astype(np.float32)


def imitate(actor,obs,targets,steps,lr=3e-4):
    obs=torch.from_numpy(obs);targets=torch.from_numpy(targets)
    opt=torch.optim.Adam(actor.parameters(),lr=lr)
    equilibrium=torch.from_numpy(features([[np.pi,0,0,0]]))
    for step in range(steps):
        idx=torch.randint(len(obs),(1024,))
        loss=(actor(obs[idx])-targets[idx]).square().mean()+.2*actor(equilibrium).square().mean()
        opt.zero_grad();loss.backward();opt.step()
        if (step+1)%2000==0:print('Imitation',step+1,'loss',float(loss.detach()),flush=True)
    return float(loss.detach())


def demonstrations(source,rng,count=24):
    observations=[];labels=[]
    # Start with one coherent trajectory; different teachers can disagree at the same state.
    teacher=TrajectoryController.load(source/'controller-0.npz')
    x=rng.uniform(-.025,.025,(count,4));x[0]=0
    for t in range(len(teacher.actions)+240):
        u=teacher(x)
        observations.append(features(x));labels.append(u.astype(np.float32))
        x=advance(x,u,10)
    # Label local perturbations along the reference with its feedback correction.
    for _ in range(24):
        idx=rng.integers(0,len(teacher.actions),1024)
        x=teacher.states[idx]+rng.normal(size=(1024,4))*np.array([.01,.01,.025,.025])
        e=x-teacher.states[idx];e[:,:2]=(e[:,:2]+np.pi)%(2*np.pi)-np.pi
        u=np.clip(teacher.actions[idx]-np.sum(e*teacher.gains[idx],axis=1),-1,1)
        observations.append(features(x));labels.append(u.astype(np.float32))
    return np.concatenate(observations),np.concatenate(labels),teacher.states


def advantages(rewards,values,bootstrap,gamma=.999,lam=.98):
    result=torch.zeros_like(rewards);carry=torch.zeros_like(bootstrap)
    for t in reversed(range(len(rewards))):
        following=bootstrap if t==len(rewards)-1 else values[t+1]
        carry=rewards[t]+gamma*following-values[t]+gamma*lam*carry
        result[t]=carry
    return result,result+values


def ppo(actor,p,reference,bc_obs,bc_targets,updates,report):
    rng=np.random.default_rng(73000)
    critic=nn.Sequential(nn.Linear(6,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,1))
    optimizer=torch.optim.Adam(actor.parameters(),lr=3e-5)
    value_optimizer=torch.optim.Adam(critic.parameters(),lr=1e-3)
    anchor_obs=torch.from_numpy(bc_obs);anchor_targets=torch.from_numpy(bc_targets)
    for update in range(1,updates+1):
        count=64;horizon=256
        # Continuing-task chunks use a value bootstrap; no fall termination.
        x=reference[rng.integers(0,len(reference),count)].copy()
        x+=rng.normal(size=x.shape)*np.array([.01,.01,.02,.02])
        x[:16]=capture_samples(rng,16,p)
        x[16:24]=rng.uniform(-.02,.02,(8,4))
        obsbuf=[];rawbuf=[];logbuf=[];rbuf=[];vbuf=[]
        for t in range(horizon):
            if t==100:
                x[:16,2:]+=rng.uniform(-min(.5,.05+update*.01),min(.5,.05+update*.01),(16,2))
            obs=torch.from_numpy(features(x))
            with torch.no_grad():
                dist=Normal(actor(obs),actor.log_std.clamp(-4,-1).exp())
                raw=dist.sample();logp=dist.log_prob(raw);value=critic(obs).squeeze(-1)
            x=advance(x,raw.clamp(-1,1).numpy())
            _,height,upright=measurements(x)
            reward=DT*(height+5*upright-.01*(x[:,2]**2+x[:,3]**2))
            obsbuf.append(obs);rawbuf.append(raw);logbuf.append(logp);vbuf.append(value)
            rbuf.append(torch.tensor(reward,dtype=torch.float32))
        with torch.no_grad(): bootstrap=critic(torch.from_numpy(features(x))).squeeze(-1)
        adv,targets=advantages(torch.stack(rbuf),torch.stack(vbuf),bootstrap)
        obs=torch.stack(obsbuf).flatten(0,1);raw=torch.stack(rawbuf).flatten();old=torch.stack(logbuf).flatten()
        adv=adv.flatten();adv=(adv-adv.mean())/adv.std().clamp_min(1e-8);targets=targets.flatten()
        stop=False
        for _ in range(4):
            for idx in torch.randperm(len(raw)).split(1024):
                dist=Normal(actor(obs[idx]),actor.log_std.clamp(-4,-1).exp())
                logratio=dist.log_prob(raw[idx])-old[idx];ratio=logratio.exp()
                if float(((ratio-1)-logratio).mean().detach())>.02:stop=True
                if not stop:
                    anchor=torch.randint(len(anchor_obs),(512,))
                    bc=(actor(anchor_obs[anchor])-anchor_targets[anchor]).square().mean()
                    loss=-torch.minimum(ratio*adv[idx],ratio.clamp(.8,1.2)*adv[idx]).mean()+.5*bc
                    optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(actor.parameters(),.5);optimizer.step()
                loss=(critic(obs[idx]).squeeze(-1)-targets[idx]).square().mean()
                value_optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(critic.parameters(),1);value_optimizer.step()
        if update%10==0 or update==updates:report(update,dict(training_steps=update*count*horizon,training_reward=float(torch.stack(rbuf).sum(0).mean())))


def run(directory,source,commit,updates=60):
    torch.set_num_threads(1);torch.manual_seed(0);rng=np.random.default_rng(70000)
    p=np.load(source/'controller-0.npz')['capture_matrix']
    actor=Actor();history=[];start=time.monotonic()
    result=dict(id=directory.name,status='running',source=SOURCE,architecture='6 → 128 tanh → 128 tanh → 128 tanh → 1; clipped torque',
                runtime='Neural actor only; no teacher, trajectory, clock, or LQR at inference',history=history)
    def save(stage,metrics=None):
        torch.save(actor.state_dict(),directory/f'{stage}.pt')
        scores=[evaluate(actor,p,count=16,seed=71000,mode=m,seconds=40)[0] for m in ['balance','bottom','kick_0.1']]
        point=dict(stage=stage,elapsed_seconds=time.monotonic()-start,evaluations=scores,**(metrics or {}))
        history.append(point);(directory/'results.json').write_text(json.dumps(result,indent=2));commit()
        print('CHECKPOINT',stage,[(s['mode'],s['final_10s_rate']) for s in scores],flush=True)
    obs,targets=balance_data(rng,p)
    loss=imitate(actor,obs,targets,8000)
    save('balance',{'imitation_loss':loss})
    demo_obs,demo_targets,reference=demonstrations(source,rng)
    # Keep balance labels half of the imitation dataset to limit forgetting.
    chosen=rng.integers(0,len(obs),len(demo_obs))
    combined_obs=np.concatenate([demo_obs,obs[chosen]])
    combined_targets=np.concatenate([demo_targets,targets[chosen]])
    loss=imitate(actor,combined_obs,combined_targets,12000,lr=1e-4)
    save('imitation',{'imitation_loss':loss,'demonstration_samples':len(demo_obs)})
    ppo(actor,p,reference,combined_obs,combined_targets,updates,lambda u,m:save(f'ppo-{u}',m))
    result['status']='trained';(directory/'results.json').write_text(json.dumps(result,indent=2));commit()
    return result


def consolidate(directory,source,previous,commit,updates=60):
    """Protect the already learned balance skill with stronger rehearsal."""
    torch.set_num_threads(1);torch.manual_seed(1);rng=np.random.default_rng(75000)
    p=np.load(source/'controller-0.npz')['capture_matrix']
    actor=Actor();actor.load_state_dict(torch.load(previous/'imitation.pt',weights_only=True))
    obs,targets=balance_data(rng,p)
    demo_obs,demo_targets,reference=demonstrations(source,rng)
    history=[];start=time.monotonic()
    result=dict(id=directory.name,status='running',source=SOURCE,parent_run=previous.name,
                architecture='6 → 128 tanh → 128 tanh → 128 tanh → 1; clipped torque',
                runtime='Neural actor only',revision='Balance rehearsal: 95% of supervised samples',history=history)
    def save(stage,metrics=None):
        torch.save(actor.state_dict(),directory/f'{stage}.pt')
        scores=[evaluate(actor,p,count=16,seed=71000,mode=m,seconds=40)[0] for m in ['balance','bottom','kick_0.1']]
        history.append(dict(stage=stage,elapsed_seconds=time.monotonic()-start,evaluations=scores,**(metrics or {})))
        (directory/'results.json').write_text(json.dumps(result,indent=2));commit()
        print('CONSOLIDATION',stage,[(s['mode'],s['final_10s_rate']) for s in scores],flush=True)
    # Keep immutable copies of the preceding stages for the same audit procedure.
    import shutil
    shutil.copyfile(previous/'balance.pt',directory/'balance.pt')
    actor.load_state_dict(torch.load(previous/'balance.pt',weights_only=True));save('balance')
    actor.load_state_dict(torch.load(previous/'imitation.pt',weights_only=True))
    idx=rng.integers(0,len(obs),len(demo_obs)*19)
    combined_obs=np.concatenate([demo_obs,obs[idx]])
    combined_targets=np.concatenate([demo_targets,targets[idx]])
    loss=imitate(actor,combined_obs,combined_targets,8000,lr=5e-5)
    save('imitation',{'imitation_loss':loss})
    ppo(actor,p,reference,combined_obs,combined_targets,updates,lambda u,m:save(f'ppo-{u}',m))
    result['status']='trained';(directory/'results.json').write_text(json.dumps(result,indent=2));commit()
    return result

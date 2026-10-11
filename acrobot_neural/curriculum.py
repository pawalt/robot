"""45-second neural-only PPO episodes with a gated backward curriculum."""
import json
import time
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from acrobot_comparison.env import measurements
from .policy import Actor,features
from .train import advance,capture_samples,balance_data,advantages,evaluate,DT,SOURCE

BACK_SECONDS=[1.,2.,4.,6.,8.,12.,16.,20.,25.,30.]


def starts(rng,reference,back,count,noise=.003):
    index=max(0,len(reference)-1-round(back/DT))
    x=np.tile(reference[index],(count,1))
    x+=rng.uniform(-noise,noise,x.shape)
    x[0]=reference[index]
    return x


def test_suffix(actor,reference,back,seed=82000,count=32,substeps=10,capture=False):
    x=starts(np.random.default_rng(seed),reference,back,count)
    streak=np.zeros(count);longest=streak.copy();trace=[];actions=[]
    for _ in range(900):
        u=actor.act(x)
        if capture:trace.append(x[0].copy());actions.append(u[0])
        for _ in range(substeps):
            from acrobot_comparison.env import advance_torque
            x=advance_torque(x,u,DT/substeps)
            streak=np.where(measurements(x)[2],streak+DT/substeps,0)
            longest=np.maximum(longest,streak)
    return dict(mode=f'suffix_{back:g}s',episodes=count,seed=seed,seconds=45.,
                final_10s_rate=float(np.mean(streak>=10-1e-8)),mean_longest_hold=float(longest.mean()),
                per_episode_final_hold=streak.tolist(),pre_kick_balanced_rate=None), (trace,actions)


def run(directory,source,previous,commit,updates=40,count=256,threads=4,
        seed=2,start_back=1.,initial_checkpoint='balance',audit_seed=None):
    torch.set_num_threads(threads);torch.manual_seed(seed);rng=np.random.default_rng(80998+seed)
    actor=Actor();actor.load_state_dict(torch.load(previous/f'{initial_checkpoint}.pt',weights_only=True))
    data=np.load(source/'controller-0.npz');reference=data['states'];p=data['capture_matrix']
    teacher_actions=data['actions'];teacher_gains=data['gains']
    critic=nn.Sequential(nn.Linear(6,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,1))
    opt=torch.optim.Adam(actor.parameters(),lr=3e-5);vopt=torch.optim.Adam(critic.parameters(),lr=1e-3)
    bal_obs,bal_targets=balance_data(rng,p)
    bal_obs=torch.from_numpy(bal_obs);bal_targets=torch.from_numpy(bal_targets)
    stage=BACK_SECONDS.index(start_back);passes=0;history=[];begin=time.perf_counter();rollout_time=0.;optimization_time=0.
    result=dict(id=directory.name,status='running',kind='backward_curriculum',source=SOURCE,
                architecture='6 → 128 tanh → 128 tanh → 128 tanh → 1; clipped torque',
                runtime='Neural actor only',history=history,
                config=dict(episode_seconds=45,horizon=900,environments=count,threads=threads,allocated_cpus=4,
                            allocated_memory_mb=8192,gpu=None,updates=updates,gamma=.999,gae_lambda=.995,
                            stage_back_seconds=BACK_SECONDS,advance_success=.8,advance_balance=.9,
                            consecutive_checks=2,check_every_updates=2,seed=seed,
                            warm_start=str(previous/initial_checkpoint),start_back=start_back,audit_seed=audit_seed))
    def checkpoint(update,metrics=None):
        nonlocal passes,stage
        name=f'ppo-{update}';torch.save(actor.state_dict(),directory/f'{name}.pt')
        torch.set_num_threads(1)
        back=BACK_SECONDS[stage]
        suffix=test_suffix(actor,reference,back)[0]
        balance=evaluate(actor,p,seed=82000,count=32,mode='balance',seconds=45)[0]
        # Evaluate bottom separately; never infer full swing-up from a suffix success.
        bottom=evaluate(actor,p,seed=82000,count=32,mode='bottom',seconds=45)[0]
        successful=suffix['final_10s_rate']>=.8 and balance['final_10s_rate']>=.9
        passes=passes+1 if successful else 0
        point=dict(stage=name,curriculum_back_seconds=back,elapsed_seconds=time.perf_counter()-begin,
                   training_steps=update*count*900,rollout_seconds=rollout_time,optimization_seconds=optimization_time,
                   evaluations=[balance,bottom,suffix],**(metrics or {}))
        history.append(point)
        if passes>=2 and stage<len(BACK_SECONDS)-1:
            stage+=1;passes=0;point['advanced_to_seconds']=BACK_SECONDS[stage]
        result['current_back_seconds']=BACK_SECONDS[stage]
        (directory/'results.json').write_text(json.dumps(result,indent=2));commit()
        print('CURRICULUM',update,'back',back,[(s['mode'],s['final_10s_rate']) for s in point['evaluations']],flush=True)
    checkpoint(0)
    for update in range(1,updates+1):
        back=BACK_SECONDS[stage]
        x=starts(rng,reference,back,count)
        balance_count=count//4
        x[:balance_count]=capture_samples(rng,balance_count,p)
        obsbuf=[];rawbuf=[];logbuf=[];rbuf=[];vbuf=[]
        torch.set_num_threads(1)
        before=time.perf_counter()
        for t in range(900):
            obs=torch.from_numpy(features(x))
            with torch.no_grad():
                dist=Normal(actor(obs),actor.log_std.clamp(-4,-1).exp())
                raw=dist.sample();logp=dist.log_prob(raw);value=critic(obs).squeeze(-1)
            x=advance(x,raw.clamp(-1,1).numpy())
            _,height,upright=measurements(x)
            reward=DT*(height+5*upright-.01*(x[:,2]**2+x[:,3]**2))
            obsbuf.append(obs);rawbuf.append(raw);logbuf.append(logp);vbuf.append(value);rbuf.append(torch.tensor(reward,dtype=torch.float32))
        rollout_time+=time.perf_counter()-before
        # Time-limit truncation is bootstrapped: the control objective continues beyond 45 s.
        with torch.no_grad():bootstrap=critic(torch.from_numpy(features(x))).squeeze(-1)
        adv,targets=advantages(torch.stack(rbuf),torch.stack(vbuf),bootstrap,lam=.995)
        obs=torch.stack(obsbuf).flatten(0,1);raw=torch.stack(rawbuf).flatten();old=torch.stack(logbuf).flatten()
        adv=adv.flatten();adv=(adv-adv.mean())/adv.std().clamp_min(1e-8);targets=targets.flatten()
        torch.set_num_threads(threads)
        before=time.perf_counter();stop=False
        for _ in range(4):
            for idx in torch.randperm(len(raw)).split(4096):
                dist=Normal(actor(obs[idx]),actor.log_std.clamp(-4,-1).exp())
                logratio=dist.log_prob(raw[idx])-old[idx];ratio=logratio.exp()
                if float(((ratio-1)-logratio).mean().detach())>.02:stop=True
                if not stop:
                    anchors=torch.randint(len(bal_obs),(512,))
                    # Train only the currently relevant suffix; teacher labels are never executed.
                    low=max(0,len(teacher_actions)-round(back/DT))
                    ti=rng.integers(low,len(teacher_actions),512)
                    tx=reference[ti]+rng.normal(size=(512,4))*[.003,.003,.01,.01]
                    error=tx-reference[ti];error[:,:2]=(error[:,:2]+np.pi)%(2*np.pi)-np.pi
                    labels=np.clip(teacher_actions[ti]-np.sum(error*teacher_gains[ti],axis=1),-1,1)
                    local_loss=(actor(torch.from_numpy(features(tx)))-torch.tensor(labels,dtype=torch.float32)).square().mean()
                    balance_loss=(actor(bal_obs[anchors])-bal_targets[anchors]).square().mean()
                    loss=-torch.minimum(ratio*adv[idx],ratio.clamp(.8,1.2)*adv[idx]).mean()+2*balance_loss+.5*local_loss
                    opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(actor.parameters(),.5);opt.step()
                loss=(critic(obs[idx]).squeeze(-1)-targets[idx]).square().mean()
                vopt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(critic.parameters(),1);vopt.step()
        optimization_time+=time.perf_counter()-before
        if update%2==0 or update==updates:checkpoint(update)
    result['status']='trained';(directory/'results.json').write_text(json.dumps(result,indent=2));commit()
    return result

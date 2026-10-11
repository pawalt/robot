"""Matched reward ablation with fixed mixed starts; executed only on Modal."""
import json
import time
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from acrobot_comparison.env import measurements
from .policy import Actor,features
from .train import advance,capture_samples,balance_data,advantages,evaluate,DT,SOURCE

from .curriculum import starts,test_suffix


def run(directory,source,previous,commit,updates=40,count=256,threads=4,
        seed=2,start_back=12.,initial_checkpoint='ppo-4',audit_seed=None, reward_kind='baseline', calibration=None):
    if start_back!=12. or count%4 or calibration is None:
        raise ValueError('Reward comparison requires a calibrated fixed 6/8/12-second mixture and count divisible by four')
    torch.set_num_threads(threads);torch.manual_seed(seed);rng=np.random.default_rng(80998+seed)
    actor=Actor();actor.load_state_dict(torch.load(previous/f'{initial_checkpoint}.pt',weights_only=True))
    data=np.load(source/'controller-0.npz');reference=data['states'];p=data['capture_matrix']
    teacher_actions=data['actions'];teacher_gains=data['gains']
    critic=nn.Sequential(nn.Linear(6,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,1))
    opt=torch.optim.Adam(actor.parameters(),lr=3e-5);vopt=torch.optim.Adam(critic.parameters(),lr=1e-3)
    bal_obs,bal_targets=balance_data(rng,p)
    bal_obs=torch.from_numpy(bal_obs);bal_targets=torch.from_numpy(bal_targets)
    history=[];begin=time.perf_counter();rollout_time=0.;optimization_time=0.
    result=dict(id=directory.name,status='running',kind='backward_curriculum',source=SOURCE,
                architecture='6 → 128 tanh → 128 tanh → 128 tanh → 1; clipped torque',
                runtime='Neural actor only',history=history,reward_sweep=True,reward_kind=reward_kind,calibration=calibration,
                config=dict(episode_seconds=45,horizon=900,environments=count,threads=threads,allocated_cpus=4,
                            allocated_memory_mb=8192,gpu=None,updates=updates,gamma=.999,gae_lambda=.995,
                            stage_back_seconds=[12.],advance_success=None,advance_balance=None,
                            consecutive_checks=2,check_every_updates=2,seed=seed,
                            start_mixture={'balance':.25,'6s':.25,'8s':.25,'12s':.25},reward_kind=reward_kind,shaping_beta=5.,warm_start=str(previous/initial_checkpoint),start_back=start_back,audit_seed=audit_seed))
    def checkpoint(update,metrics=None):
        name=f'ppo-{update}';torch.save(actor.state_dict(),directory/f'{name}.pt')
        torch.set_num_threads(1)
        back=start_back
        suffix=test_suffix(actor,reference,back)[0]
        balance=evaluate(actor,p,seed=82000,count=32,mode='balance',seconds=45)[0]
        # Evaluate bottom separately; never infer full swing-up from a suffix success.
        bottom=evaluate(actor,p,seed=82000,count=32,mode='bottom',seconds=45)[0]
        point=dict(stage=name,curriculum_back_seconds=back,elapsed_seconds=time.perf_counter()-begin,
                   training_steps=update*count*900,rollout_seconds=rollout_time,optimization_seconds=optimization_time,
                   evaluations=[balance,bottom,suffix,test_suffix(actor,reference,6.)[0],test_suffix(actor,reference,8.)[0]],**(metrics or {}))
        history.append(point)
        result['current_back_seconds']=start_back
        (directory/'results.json').write_text(json.dumps(result,indent=2));commit()
        print('CURRICULUM',update,'back',back,[(s['mode'],s['final_10s_rate']) for s in point['evaluations']],flush=True)
    checkpoint(0)
    for update in range(1,updates+1):
        back=start_back
        x=starts(rng,reference,back,count)
        balance_count=count//4
        x[:balance_count]=capture_samples(rng,balance_count,p)
        x[balance_count:2*balance_count]=starts(rng,reference,6.,balance_count)
        x[2*balance_count:3*balance_count]=starts(rng,reference,8.,balance_count)
        obsbuf=[];rawbuf=[];logbuf=[];rbuf=[];vbuf=[]
        torch.set_num_threads(1)
        before=time.perf_counter()
        for t in range(900):
            obs=torch.from_numpy(features(x))
            with torch.no_grad():
                dist=Normal(actor(obs),actor.log_std.clamp(-4,-1).exp())
                raw=dist.sample();logp=dist.log_prob(raw);value=critic(obs).squeeze(-1)
            previous_x=x
            x=advance(x,raw.clamp(-1,1).numpy())
            _,height,upright=measurements(x)
            reward=compute_reward(previous_x,x,p,reward_kind,calibration['level'])
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


def potential(x,p,level):
    error=x-np.array([np.pi,0.,0.,0.])
    error[:,:2]=(error[:,:2]+np.pi)%(2*np.pi)-np.pi
    distance=np.einsum('bi,ij,bj->b',error,p,error)
    # Smooth bounded proximity; P is a local heuristic outside the sampled region.
    return 1/(1+distance/level)


def compute_reward(before,after,p,kind,level):
    _,height,upright=measurements(after)
    speed=after[:,2]**2+after[:,3]**2
    if kind=='baseline':return DT*(height+5*upright-.01*speed)
    reward=DT*(height+5*upright-.01*height**4*speed)
    if kind=='capture':reward=reward+5*(.999*potential(after,p,level)-potential(before,p,level))
    elif kind!='braking':raise ValueError(kind)
    return reward


def calibrate(source,parent):
    from acrobot_comparison.env import advance_torque
    torch.set_num_threads(1)
    actor=Actor();actor.load_state_dict(torch.load(parent/'ppo-4.pt',weights_only=True))
    p=np.load(source/'controller-0.npz')['capture_matrix'];rows=[];chosen=None
    for level in [1.,2.,5.,10.,20.,40.]:
        x=capture_samples(np.random.default_rng(211000),256,p,level)
        hold=np.zeros(len(x))
        for _ in range(900):
            u=actor.act(x)
            for _ in range(10):
                x=advance_torque(x,u,.005)
                hold=np.where(measurements(x)[2],hold+.005,0)
        rate=float(np.mean(hold>=10-1e-8));rows.append(dict(level=level,success_rate=rate,count=256))
        if all(r['success_rate']>=.95 for r in rows):chosen=level
    if chosen is None:raise RuntimeError('Neural capture calibration failed even at level 1')
    return dict(level=chosen,rows=rows,seed=211000,description='Empirical sampled neural recovery, not a certified region; P extrapolation is only a shaping heuristic')


def finish(directory,source):
    from .audit import audit
    # Reuse recording and fine-physics audit; selection considers retention jointly.
    result=json.loads((directory/'results.json').read_text())
    def score(point):
        m={e['mode']:e['final_10s_rate'] for e in point['evaluations']}
        retained=m['balance']>=.9 and m['suffix_6s']>=.8 and m['suffix_8s']>=.8
        return (retained,m['bottom'],m['suffix_12s'],sum(m.values()))
    selected=max(result['history'],key=score)['stage']
    # Audit accepts an explicit development-only choice for this experiment.
    result['audit_selected_stage']=selected
    (directory/'results.json').write_text(json.dumps(result,indent=2))
    result=audit(directory,source)
    actor=Actor();reference=np.load(source/'controller-0.npz')['states']
    for point in result['audit']:
        actor.load_state_dict(torch.load(directory/f"{point['stage']}.pt",weights_only=True))
        for back in [6.,8.]:
            point['evaluations'].append(test_suffix(actor,reference,back,seed=result['config']['audit_seed'],count=64)[0])
    result['selection']='Development only: retain balance >=90%, 6s and 8s >=80%; then bottom, 12s, total success. Earliest ties.'
    (directory/'results.json').write_text(json.dumps(result,indent=2))
    return result


def record_final(directory,source):
    """Preserve videos of the final trained behavior even if selection keeps the parent."""
    import imageio.v2 as imageio
    from PIL import Image
    from acrobot_feasibility.record import frame
    torch.set_num_threads(1)
    result=json.loads((directory/'results.json').read_text())
    stage=result['history'][-1]['stage']
    if stage==result['selected_stage']:return result
    actor=Actor();actor.load_state_dict(torch.load(directory/f'{stage}.pt',weights_only=True))
    data=np.load(source/'controller-0.npz')
    for mode in ['suffix_12s','bottom']:
        if mode=='bottom':row,(xs,us)=evaluate(actor,data['capture_matrix'],seed=result['config']['audit_seed'],count=64,mode=mode,seconds=45,capture=True)
        else:row,(xs,us)=test_suffix(actor,data['states'],12.,seed=result['config']['audit_seed'],count=64,capture=True)
        name=f'{stage}-{mode}';label=f'Final neural policy | {mode}'
        Image.fromarray(frame(xs[0],0,us[0],label)).save(directory/f'{name}.jpg')
        with imageio.get_writer(str(directory/f'{name}.mp4'),fps=20,codec='libx264',pixelformat='yuv420p',output_params=['-movflags','+faststart']) as writer:
            for t,x in enumerate(xs):writer.append_data(frame(x,t*.05,us[t],label))
        if not any(c['video']==name+'.mp4' for c in result['recordings']):
            result['recordings'].append(dict(stage=stage,mode=mode,video=name+'.mp4',poster=name+'.jpg',episode_index=0,final_hold_seconds=row['per_episode_final_hold'][0]))
    (directory/'results.json').write_text(json.dumps(result,indent=2));return result

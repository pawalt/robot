"""Larger starting-disturbance evaluation and recovery-demonstration imitation."""
import json
import numpy as np
import torch
from .radial import RadialActor,ExportedActor,Deployment
from .policy import features
from .train import advance,DT
from .imitation import teacher_action
from acrobot_comparison.env import measurements,advance_torque
from acrobot_feasibility.study import errors,design

PARENTS=['20261011T183151-433572cc','20261011T183152-b6856ae9','20261011T183155-78fa3813']
CONDITIONS=[('small',.1,.1),('medium',.25,.5),('large',.5,1.),('very_large',1.,2.),('velocity_2',0.,2.),('velocity_4',0.,4.)]


def initial_states(seed,count):
    rng=np.random.default_rng(seed)
    return np.concatenate([rng.uniform(-1,1,(count,4))*[a,a,v,v] for _,a,v in CONDITIONS])


def evaluate(actor,data,seed,count=128,substeps=10,seconds=120,teacher=False,capture=False):
    x=initial_states(seed,count);initial=x.copy();hold=np.zeros(len(x));caught=np.zeros(len(x),bool);xs=[];us=[]
    for step in range(round(seconds/DT)):
        if teacher:u,caught=teacher_action(x,np.full(len(x),step),data,caught)
        else:u=actor.act(x)
        if capture:xs.append(x.copy());us.append(u.copy())
        for _ in range(substeps):
            x=advance_torque(x,u,DT/substeps)
            hold=np.where(measurements(x)[2],hold+DT/substeps,0)
    rows=[]
    for i,(name,a,v) in enumerate(CONDITIONS):
        sl=slice(i*count,(i+1)*count)
        rows.append(dict(condition=name,angle_bound=a,velocity_bound=v,episodes=count,seed=seed,seconds=seconds,
                         physics_dt=DT/substeps,successes=int(np.sum(hold[sl]>=10-1e-8)),final_hold_seconds=hold[sl].tolist(),initial_states=initial[sl].tolist()))
    return rows,(np.asarray(xs),np.asarray(us))


def collect(directory,source,parent,trajectory,commit):
    torch.set_num_threads(1)
    data=dict(np.load(source/f'controller-{trajectory}.npz'));data['balance_gain']=design(DT)[0]
    neural,_=evaluate(ExportedActor(parent/'deployed.pt'),data,321000)
    teacher,_=evaluate(None,data,321000,teacher=True)
    x=initial_states(331000,64);caught=np.zeros(len(x),bool);hold=np.zeros(len(x));states=[];phases=[];masks=[]
    for step in range(1200):
        u,caught=teacher_action(x,np.full(len(x),step),data,caught)
        if step%3==0:
            states.append(x.copy());phases.append(np.full(len(x),step));masks.append(~caught)
        for _ in range(10):
            x=advance_torque(x,u,.005);hold=np.where(measurements(x)[2],hold+.005,0)
    good=hold>=10-1e-8;mask=np.array(masks)&good[None,:]
    train_x=np.array(states)[mask];train_phase=np.array(phases)[mask]
    np.savez(directory/'demonstrations.npz',states=train_x,phase=train_phase)
    result=dict(id=directory.name,trajectory=trajectory,parent=parent.name,status='collected',baseline=neural,teacher=teacher,
                training_seed=331000,training_episodes=len(good),accepted_episodes=int(good.sum()),examples=len(train_x),conditions=CONDITIONS)
    (directory/'results.json').write_text(json.dumps(result,indent=2));commit()
    print('ROBUST BASELINE',trajectory,[r['successes'] for r in neural],'teacher',[r['successes'] for r in teacher],'accepted',int(good.sum()),flush=True)
    return result


def expand(directory,source,parent,collection,trajectory,units,sigma,commit):
    torch.set_num_threads(1);rng=np.random.default_rng(351000)
    data=dict(np.load(source/f'controller-{trajectory}.npz'));data['balance_gain']=design(DT)[0]
    demo=np.load(collection/'demonstrations.npz');states=demo['states'];phase=demo['phase']
    # Sample a finite candidate pool; select prototypes farthest from current coverage.
    choice=rng.choice(len(states),min(12000,len(states)),replace=False)
    states=states[choice];phase=phase[choice];obs=features(states)
    base=RadialActor.load(parent/'policy.pt');centers=base.centers.detach().numpy()
    distance=np.full(len(obs),np.inf)
    for center in centers:distance=np.minimum(distance,((obs-center)**2).sum(1))
    chosen=[]
    for _ in range(min(units,len(obs))):
        index=int(np.argmax(distance));chosen.append(index)
        distance=np.minimum(distance,((obs-obs[index])**2).sum(1))
    weights=[];bias=[]
    for index in chosen:
        center=states[index];t=min(int(phase[index]),len(data['actions'])-1)
        local=center+rng.normal(size=(256,4))*[.01,.01,.03,.06]
        delta=local-data['states'][t];delta[:,:2]=(delta[:,:2]+np.pi)%(2*np.pi)-np.pi
        raw=data['actions'][t]-(delta*data['gains'][t]).sum(1)
        e=errors(local);value=np.einsum('bi,ij,bj->b',e,data['capture_matrix'],e)
        raw=np.where(value<=float(data['capture_level']),-e@data['balance_gain'],raw)
        f=features(local).astype(np.float64);origin=features(center)[0].astype(np.float64)
        matrix=np.column_stack([f-origin,np.ones(len(f))]);fit=np.linalg.lstsq(matrix,raw,rcond=1e-5)[0]
        weights.append(fit[:-1]);bias.append(fit[-1]-origin@fit[:-1])
    actor=RadialActor(np.vstack([centers,obs[chosen]]),np.vstack([base.weights.detach().numpy(),weights]),np.r_[base.bias.detach().numpy(),bias],sigma)
    actor.save(directory/'policy.pt');torch.jit.trace(Deployment(actor).eval(),torch.zeros(2,4)).save(str(directory/'deployed.pt'))
    deployed=ExportedActor(directory/'deployed.pt')
    rows,_=evaluate(deployed,data,341000,count=32,seconds=60)
    from .radial import evaluate as original_evaluate
    ground,_=original_evaluate(deployed,count=1,noise=0,seconds=60)
    balance,_=original_evaluate(deployed,count=32,mode='balance',p=data['capture_matrix'],seconds=60)
    result=dict(id=directory.name,status='trained',trajectory=trajectory,parent=parent.name,collection=collection.name,
                units=units,sigma=sigma,prototypes=len(actor.centers),development=rows,exact_ground=ground,balance=balance)
    (directory/'results.json').write_text(json.dumps(result,indent=2));commit()
    print('ROBUST TRAINED',trajectory,units,sigma,[r['successes'] for r in rows], 'ground',ground['exact_ground_success'],'balance',balance['final_10s_rate'],flush=True)
    return result


def audit(directory,source,policy_path,trajectory,commit):
    import imageio.v2 as imageio
    from PIL import Image
    from acrobot_feasibility.record import frame
    from .radial import evaluate as original_evaluate
    torch.set_num_threads(1)
    data=dict(np.load(source/f'controller-{trajectory}.npz'));data['balance_gain']=design(DT)[0]
    actor=ExportedActor(policy_path)
    # Independent starts, doubled physics resolution, longer episodes than development.
    rows,(xs,us)=evaluate(actor,data,361000,count=128,substeps=20,seconds=120,capture=True)
    ground,_=original_evaluate(actor,count=1,noise=0,seconds=120,substeps=20)
    balance,_=original_evaluate(actor,count=128,mode='balance',p=data['capture_matrix'],seed=361000,seconds=120,substeps=20)
    result=json.loads((directory/'results.json').read_text())
    result.update(audit=rows,exact_ground_audit=ground,balance_audit=balance,status='complete',recordings=[])
    for condition in ['large','very_large','velocity_4']:
        index=next(i for i,c in enumerate(CONDITIONS) if c[0]==condition);row=rows[index]
        success=np.array(row['final_hold_seconds'])>=10-1e-8
        picks=[('first',0)]
        # Show first failures as well as first successes; mark any outcome-selected clips.
        if not success[0] and success.any():picks.append(('success',int(np.flatnonzero(success)[0])))
        if success[0] and (~success).any():picks.append(('failure',int(np.flatnonzero(~success)[0])))
        for selection,episode in picks:
            j=index*128+episode;name=condition+'-'+selection;label=f'Neural recovery | {condition} | {selection} episode {episode}'
            Image.fromarray(frame(xs[0,j],0,us[0,j],label)).save(directory/f'{name}.jpg')
            with imageio.get_writer(str(directory/f'{name}.mp4'),fps=20,codec='libx264',pixelformat='yuv420p',output_params=['-movflags','+faststart']) as writer:
                for t in range(len(xs)):writer.append_data(frame(xs[t,j],t*.05,us[t,j],label))
            result['recordings'].append(dict(video=name+'.mp4',poster=name+'.jpg',condition=condition,selection=selection,episode=episode,initial_state=row['initial_states'][episode],final_hold_seconds=row['final_hold_seconds'][episode]))
    (directory/'results.json').write_text(json.dumps(result,indent=2));commit()
    print('ROBUST AUDIT',trajectory,result.get('variant','expanded'),[r['successes'] for r in rows],flush=True)
    return result

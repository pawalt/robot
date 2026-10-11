"""State-only radial neural imitation. Training/evaluation must run on Modal."""
import json,time
import numpy as np
import torch
from torch import nn
from .policy import features
from .train import advance,capture_samples,DT
from acrobot_comparison.env import measurements,advance_torque
from acrobot_feasibility.study import design,errors
from .imitation import teacher_action


class RadialActor(nn.Module):
    """Gaussian hidden units gate supervised affine experts; no runtime teacher."""
    def __init__(self,centers,weights,bias,sigma):
        super().__init__()
        self.centers=nn.Parameter(torch.tensor(centers,dtype=torch.float32))
        self.weights=nn.Parameter(torch.tensor(weights,dtype=torch.float32))
        self.bias=nn.Parameter(torch.tensor(bias,dtype=torch.float32))
        self.register_buffer('sigma',torch.tensor(float(sigma)))
    def forward(self,x):
        distance=(x.square().sum(1,keepdim=True)+self.centers.square().sum(1)[None]-2*x@self.centers.T).clamp_min(0)
        gate=torch.softmax(-distance/(2*self.sigma**2),dim=1)
        return (gate*(x@self.weights.T+self.bias)).sum(1)
    def act(self,x):
        with torch.no_grad():return self(torch.from_numpy(features(x))).clamp(-1,1).numpy()
    def save(self,path):torch.save(self.state_dict(),path)
    @classmethod
    def load(cls,path):
        d=torch.load(path,weights_only=True)
        actor=cls(d['centers'].numpy(),d['weights'].numpy(),d['bias'].numpy(),d['sigma'].item())
        actor.load_state_dict(d);actor.eval();return actor


def train(source,trajectory,sigma,noise,seed):
    data=dict(np.load(source/f'controller-{trajectory}.npz'));data['balance_gain']=design(DT)[0]
    # Actual closed-loop demonstration, including early capture and balancing.
    x=np.zeros((1,4));caught=np.zeros(1,bool);states=[];phases=[]
    for t in range(len(data['actions'])+100):
        states.append(x[0].copy());phases.append(t)
        u,caught=teacher_action(x,np.array([t]),data,caught)
        x=advance(x,u,10)
        if caught[0] and measurements(x)[2][0] and t>len(data['actions']):break
    states=np.array(states);phases=np.array(phases)
    # Avoid duplicating many essentially identical balance prototypes.
    keep=np.r_[True,np.linalg.norm(np.diff(features(states),axis=0),axis=1)>1e-4]
    states=states[keep];phases=phases[keep]
    states=np.vstack([states,[np.pi,0,0,0]]);phases=np.r_[phases,len(data['actions'])]
    centers=features(states);weights=[];bias=[];rng=np.random.default_rng(seed)
    for center,phase in zip(states,phases):
        local=center+rng.normal(size=(256,4))*noise*np.array([1.,1.,3.,6.])
        idx=min(phase,len(data['actions'])-1)
        delta=local-data['states'][idx];delta[:,:2]=(delta[:,:2]+np.pi)%(2*np.pi)-np.pi
        raw=data['actions'][idx]-(delta*data['gains'][idx]).sum(1)
        e=errors(local);value=np.einsum('bi,ij,bj->b',e,data['capture_matrix'],e)
        raw=np.where((value<=float(data['capture_level']))|(phase>=len(data['actions'])),-e@data['balance_gain'],raw)
        obs=features(local).astype(np.float64);origin=features(center)[0].astype(np.float64)
        design_matrix=np.column_stack([obs-origin,np.ones(len(obs))])
        # Supervised least squares on unbounded torque labels, actuator clips at inference.
        fit=np.linalg.lstsq(design_matrix,raw,rcond=1e-5)[0]
        weights.append(fit[:-1]);bias.append(fit[-1]-origin@fit[:-1])
    return RadialActor(centers,np.array(weights),np.array(bias),sigma),dict(prototypes=len(centers),examples=len(centers)*256,trajectory=trajectory,noise=noise,sigma=sigma,seed=seed)


def evaluate(actor,count=32,seed=301000,noise=.01,seconds=60,substeps=10,capture=False,mode='bottom',p=None):
    rng=np.random.default_rng(seed)
    x=rng.uniform(-noise,noise,(count,4));x[0]=0
    if mode=='balance':x=capture_samples(rng,count,p)
    hold=np.zeros(count);longest=hold.copy();xs=[];us=[]
    for step in range(round(seconds/DT)):
        u=actor.act(x)
        if capture:xs.append(x[0].copy());us.append(u[0])
        for _ in range(substeps):
            x=advance_torque(x,u,DT/substeps)
            hold=np.where(measurements(x)[2],hold+DT/substeps,0)
            longest=np.maximum(hold,longest)
    return dict(mode=mode,episodes=count,seed=seed,initial_noise=noise,seconds=seconds,physics_dt=DT/substeps,
                final_10s_rate=float(np.mean(hold>=10-1e-8)),exact_ground_success=bool(hold[0]>=10-1e-8) if mode=='bottom' else None,
                final_hold_seconds=hold.tolist(),longest_hold_seconds=longest.tolist()),(xs,us)


def run(directory,source,trajectory,sigma,noise,seed,commit):
    torch.set_num_threads(1);begin=time.perf_counter()
    actor,config=train(source,trajectory,sigma,noise,seed)
    actor.save(directory/'policy.pt')
    row,_=evaluate(actor,count=16,seconds=60)
    result=dict(id=directory.name,status='complete',kind='radial',config=config,development=row,
                elapsed_seconds=time.perf_counter()-begin,runtime='State-only Gaussian neural gates and fitted affine experts; no clock or teacher')
    (directory/'results.json').write_text(json.dumps(result,indent=2));commit()
    print('RADIAL',config,row['final_10s_rate'],row['final_hold_seconds'][0],flush=True)
    return result


class Deployment(nn.Module):
    """Self-contained raw-state-to-torque inference graph."""
    def __init__(self,actor):super().__init__();self.actor=actor
    def forward(self,state):
        obs=torch.stack([torch.sin(state[:,0]),torch.cos(state[:,0]),torch.sin(state[:,1]),torch.cos(state[:,1]),state[:,2]/4,state[:,3]/8],dim=1)
        return self.actor(obs).clamp(-1,1)


class ExportedActor:
    def __init__(self,path):self.graph=torch.jit.load(str(path));self.graph.eval()
    def act(self,x):
        with torch.no_grad():return self.graph(torch.tensor(x,dtype=torch.float32)).numpy()


def audit_export(directory,source,commit):
    import imageio.v2 as imageio
    from PIL import Image
    from acrobot_feasibility.record import frame
    torch.set_num_threads(1)
    result=json.loads((directory/'results.json').read_text())
    trained=RadialActor.load(directory/'policy.pt')
    graph=torch.jit.trace(Deployment(trained).eval(),torch.zeros(2,4))
    graph.save(str(directory/'deployed.pt'))
    actor=ExportedActor(directory/'deployed.pt')
    p=np.load(source/f"controller-{result['config']['trajectory']}.npz")['capture_matrix']
    conditions=[('exact_ground',0.,1,20,120),('fresh_ground',.01,128,10,120),
                ('fine_ground',.01,128,20,120),('wider_ground',.05,128,10,120),
                ('large_ground',.1,128,10,120),('balance',0.,128,20,120)]
    rows=[];recordings=[]
    for mode,noise,count,substeps,seconds in conditions:
        row,(xs,us)=evaluate(actor,count=count,seed=311000,noise=noise,seconds=seconds,substeps=substeps,
                             capture=mode in ['exact_ground','wider_ground'],mode='balance' if mode=='balance' else 'bottom',p=p)
        row['condition']=mode;rows.append(row)
        print('RADIAL AUDIT',result['config']['trajectory'],mode,row['final_10s_rate'],row['final_hold_seconds'][0],flush=True)
        if xs:
            name=mode;label=f'Neural radial policy | {mode}'
            Image.fromarray(frame(xs[0],0,us[0],label)).save(directory/f'{name}.jpg')
            with imageio.get_writer(str(directory/f'{name}.mp4'),fps=20,codec='libx264',pixelformat='yuv420p',output_params=['-movflags','+faststart']) as writer:
                for t,x in enumerate(xs):writer.append_data(frame(x,t*.05,us[t],label))
            recordings.append(dict(video=name+'.mp4',poster=name+'.jpg',condition=mode,episode_index=0,final_hold_seconds=row['final_hold_seconds'][0]))
    result.update(audit=rows,recordings=recordings,export='deployed.pt',verified_runtime='Standalone TorchScript: raw state to neural torque, no teacher, clock, lookup table or LQR invocation')
    (directory/'results.json').write_text(json.dumps(result,indent=2));commit();return result

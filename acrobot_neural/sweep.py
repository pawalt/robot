"""Matched parallel anti-forgetting pilot. All rollout work is remote-only."""
import copy
import json
import time
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from acrobot_comparison.env import measurements
from acrobot_feasibility.study import design,errors
from .policy import Actor,features
from .train import advance,capture_samples,advantages,evaluate,DT,SOURCE
from .curriculum import starts,test_suffix

ARMS=['unprotected','rehearsal','distill','ewc','project','rollback','experts',
      'wide256','wide512','residual256']


class Residual(nn.Module):
    def __init__(self,width):
        super().__init__();self.block=nn.Sequential(nn.Linear(width,width),nn.Tanh(),nn.Linear(width,width))
        nn.init.zeros_(self.block[-1].weight);nn.init.zeros_(self.block[-1].bias)
    def forward(self,x):return x+self.block(x)


class WideActor(Actor):
    def __init__(self,width,residual=False):
        super().__init__()
        layers=[nn.Linear(6,width),nn.Tanh(),nn.Linear(width,width),nn.Tanh(),nn.Linear(width,width),nn.Tanh()]
        if residual:layers.append(Residual(width))
        layers.append(nn.Linear(width,1));self.net=nn.Sequential(*layers)


class Experts(Actor):
    def __init__(self,base):
        nn.Module.__init__(self)
        self.balance=copy.deepcopy(base);self.swing=copy.deepcopy(base)
        for param in self.balance.parameters():param.requires_grad_(False)
        self.gate=nn.Sequential(nn.Linear(6,64),nn.Tanh(),nn.Linear(64,1))
        nn.init.zeros_(self.gate[-1].weight);nn.init.constant_(self.gate[-1].bias,-4.)
        self.log_std=nn.Parameter(base.log_std.detach().clone())
        self.swing.log_std.requires_grad_(False)
    def forward(self,obs):
        gate=self.gate(obs).squeeze(-1).sigmoid()
        return (1-gate)*self.balance(obs)+gate*self.swing(obs)


def make_actor(arm,base):
    if arm=='experts':return Experts(base)
    if arm not in ['wide256','wide512','residual256']:return copy.deepcopy(base)
    width=512 if arm=='wide512' else 256
    actor=WideActor(width,arm=='residual256');repeat=width//128
    old=[m for m in base.net if isinstance(m,nn.Linear)]
    new=[m for m in actor.net if isinstance(m,nn.Linear)]
    with torch.no_grad():
        new[0].weight.copy_(old[0].weight.repeat_interleave(repeat,0));new[0].bias.copy_(old[0].bias.repeat_interleave(repeat))
        for i in [1,2]:
            new[i].weight.copy_(old[i].weight.repeat_interleave(repeat,0).repeat_interleave(repeat,1)/repeat)
            new[i].bias.copy_(old[i].bias.repeat_interleave(repeat))
        new[-1].weight.copy_(old[-1].weight.repeat_interleave(repeat,1)/repeat);new[-1].bias.copy_(old[-1].bias)
        # Zero-sum outgoing perturbations preserve the initial function but break neuron symmetry.
        noise=torch.randn_like(new[-1].weight).reshape(1,128,repeat)*.001
        noise-=noise.mean(-1,keepdim=True);new[-1].weight.add_(noise.reshape(1,width))
        actor.log_std.copy_(base.log_std)
    return actor


def load_actor(arm,path):
    actor=make_actor(arm,Actor());actor.load_state_dict(torch.load(path,weights_only=True));actor.eval();return actor


def memory(base,p,rng):
    states=[capture_samples(rng,8192,p)]
    x=capture_samples(rng,64,p)
    for t in range(300):
        if t%5==0:states.append(x.copy())
        x=advance(x,base.act(x),10)
    x=np.concatenate(states);obs=torch.from_numpy(features(x))
    with torch.no_grad():frozen=base(obs).detach()
    gain,_=design(DT)
    lqr=torch.tensor(np.clip(-errors(x)@gain,-1.2,1.2),dtype=torch.float32)
    return obs,frozen,lqr


def suffix_data(reference,actions,gains,rng,size=16384):
    idx=rng.integers(len(actions)-40,len(actions),size)
    x=reference[idx]+rng.normal(size=(size,4))*[.003,.003,.01,.01]
    e=x-reference[idx];e[:,:2]=(e[:,:2]+np.pi)%(2*np.pi)-np.pi
    labels=np.clip(actions[idx]-np.sum(e*gains[idx],axis=1),-1,1)
    return torch.from_numpy(features(x)),torch.tensor(labels,dtype=torch.float32)


def flat(parameters):return torch.cat([p.detach().flatten() for p in parameters])


def put(parameters,vector):
    with torch.no_grad():
        start=0
        for p in parameters:
            p.copy_(vector[start:start+p.numel()].reshape_as(p));start+=p.numel()


def fisher(actor,obs,parameters):
    values=torch.zeros_like(flat(parameters))
    # Empirical Fisher for the initial Gaussian policy, using per-observation scores.
    for sample in obs[:256]:
        mean=actor(sample[None]);dist=Normal(mean,actor.log_std.exp())
        action=dist.sample();grads=torch.autograd.grad(dist.log_prob(action).sum(),parameters,allow_unused=True)
        g=torch.cat([(torch.zeros_like(p) if v is None else v).flatten() for p,v in zip(parameters,grads)])
        values+=g.detach().square()/256
    return values/values.mean().clamp_min(1e-12)


def metrics(actor,p,reference,seed,count=16,full=False):
    modes=['balance','bottom']+(['kick_0.1','kick_0.5'] if full else [])
    result=[evaluate(actor,p,seed=seed,count=count,mode=m,seconds=45)[0] for m in modes]
    result += [test_suffix(actor,reference,b,seed=seed,count=count)[0] for b in [1.,2.]]
    return result


def run(directory,source,parent,arm,seed,commit,bc_steps=4000,ppo_updates=10):
    torch.set_num_threads(1);torch.manual_seed(1000+seed);rng=np.random.default_rng(91000+seed)
    base=Actor();base.load_state_dict(torch.load(parent/'balance.pt',weights_only=True));base.eval()
    data=np.load(source/'controller-0.npz');reference=data['states'];p=data['capture_matrix']
    mem_obs,frozen,lqr=memory(base,p,rng)
    task_obs,task_targets=suffix_data(reference,data['actions'],data['gains'],rng)
    # Actor construction must not change the subsequent sample/shuffle RNG across arms.
    actor=make_actor(arm,base);torch.manual_seed(2000+seed)
    parameters=[p for p in actor.parameters() if p.requires_grad]
    initial=flat(parameters);importance=fisher(actor,mem_obs,parameters) if arm=='ewc' else None
    torch.manual_seed(3000+seed)
    optimizer=torch.optim.Adam(parameters,lr=1e-4)
    history=[];start=time.perf_counter();rejections=0;projections=0
    accepted=copy.deepcopy(actor.state_dict());accepted_opt=copy.deepcopy(optimizer.state_dict())
    result=dict(arm=arm,seed=seed,status='running',history=history,
                config=dict(bc_steps=bc_steps,bc_batch=512,ppo_updates=ppo_updates,environments=128,episode_seconds=45,
                            allocated_cpus=2,memory_mb=4096,neural_only=True,teacher_source=SOURCE,
                            trainable_parameters=sum(v.numel() for v in parameters),
                            total_parameters=sum(v.numel() for v in actor.parameters()),
                            protection_coefficient=2.,ewc_coefficient=1000.,rollout_noise=.003))
    def persist():
        result.update(elapsed_seconds=time.perf_counter()-start,rejections=rejections,projections=projections)
        (directory/'results.json').write_text(json.dumps(result,indent=2));commit()
    def checkpoint(label):
        nonlocal rejections,accepted,accepted_opt
        evaluation=metrics(actor,p,reference,92000+seed)
        rejected=arm=='rollback' and evaluation[0]['final_10s_rate']<.9
        point=dict(stage=label,evaluations=evaluation,elapsed_seconds=time.perf_counter()-start,rejected=rejected)
        with torch.no_grad():
            point['memory_mse']=float((actor(mem_obs)-frozen).square().mean())
            point['suffix_mse']=float((actor(task_obs)-task_targets).square().mean())
        torch.save(actor.state_dict(),directory/f'{label}.pt')
        if rejected:
            # A rollback across the BC/PPO boundary must retain the phase's LR.
            learning_rates=[group['lr'] for group in optimizer.param_groups]
            actor.load_state_dict(accepted);optimizer.load_state_dict(accepted_opt);rejections+=1
            for group,lr in zip(optimizer.param_groups,learning_rates):group['lr']=lr
            point['restored_accepted_checkpoint']=True
        else:
            accepted=copy.deepcopy(actor.state_dict());accepted_opt=copy.deepcopy(optimizer.state_dict())
        history.append(point);persist()
        print(arm,seed,label,[(m['mode'],m['final_10s_rate']) for m in evaluation],'rejected',rejected,flush=True)
    def update(task_loss,idx):
        nonlocal projections
        old=flat(parameters) if arm=='project' else None
        memory_loss=(actor(mem_obs[idx])-frozen[idx]).square().mean()
        if arm=='project':
            grads=torch.autograd.grad(memory_loss,parameters,allow_unused=True)
            constraint=torch.cat([(torch.zeros_like(p) if g is None else g).flatten() for p,g in zip(parameters,grads)]).detach()
        loss=task_loss
        if arm=='rehearsal':loss=loss+2*(actor(mem_obs[idx])-lqr[idx]).square().mean()
        if arm in ['distill','wide256','wide512','residual256','experts','rollback']:loss=loss+2*memory_loss
        if arm=='ewc':
            current=torch.cat([p.flatten() for p in parameters]);loss=loss+1000*(importance*(current-initial).square()).mean()
        optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(parameters,1.);optimizer.step()
        if arm=='project':
            delta=flat(parameters)-old;dot=delta@constraint
            if dot>0 and constraint.square().sum()>1e-16:
                put(parameters,old+delta-constraint*dot/constraint.square().sum());projections+=1
    checkpoint('initial')
    for step in range(1,bc_steps+1):
        idx=torch.randint(len(task_obs),(512,));mi=torch.randint(len(mem_obs),(512,))
        loss=(actor(task_obs[idx])-task_targets[idx]).square().mean()
        update(loss,mi)
        if step%1000==0 or step==bc_steps:checkpoint(f'bc-{step}')
    # Continue from the retained actor after any rollback, not the rejected checkpoint.
    torch.save(actor.state_dict(),directory/'bc-retained.pt')
    critic=nn.Sequential(nn.Linear(6,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,1))
    value_opt=torch.optim.Adam(critic.parameters(),lr=1e-3)
    for group in optimizer.param_groups:group['lr']=3e-5
    for iteration in range(1,ppo_updates+1):
        x=starts(rng,reference,2.,128);x[:32]=capture_samples(rng,32,p)
        ob=[];ab=[];lb=[];vb=[];rb=[]
        for t in range(900):
            obs=torch.from_numpy(features(x))
            with torch.no_grad():
                dist=Normal(actor(obs),actor.log_std.clamp(-4,-1).exp());raw=dist.sample()
                logp=dist.log_prob(raw);value=critic(obs).squeeze(-1)
            x=advance(x,raw.clamp(-1,1).numpy())
            _,height,balanced=measurements(x)
            reward=DT*(height+5*balanced-.01*(x[:,2]**2+x[:,3]**2))
            ob.append(obs);ab.append(raw);lb.append(logp);vb.append(value);rb.append(torch.tensor(reward,dtype=torch.float32))
        with torch.no_grad():bootstrap=critic(torch.from_numpy(features(x))).squeeze(-1)
        adv,target=advantages(torch.stack(rb),torch.stack(vb),bootstrap,lam=.995)
        obs=torch.stack(ob).flatten(0,1);action=torch.stack(ab).flatten();oldlog=torch.stack(lb).flatten()
        adv=adv.flatten();adv=(adv-adv.mean())/adv.std().clamp_min(1e-8);target=target.flatten()
        stop=False
        for epoch in range(4):
            for idx in torch.randperm(len(action)).split(4096):
                dist=Normal(actor(obs[idx]),actor.log_std.clamp(-4,-1).exp())
                logratio=dist.log_prob(action[idx])-oldlog[idx];ratio=logratio.exp()
                if float(((ratio-1)-logratio).mean().detach())>.02:stop=True
                if not stop:
                    ti=torch.randint(len(task_obs),(512,));mi=torch.randint(len(mem_obs),(512,))
                    task_loss=-torch.minimum(ratio*adv[idx],ratio.clamp(.8,1.2)*adv[idx]).mean()+.5*(actor(task_obs[ti])-task_targets[ti]).square().mean()
                    update(task_loss,mi)
                loss=(critic(obs[idx]).squeeze(-1)-target[idx]).square().mean()
                value_opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(critic.parameters(),1.);value_opt.step()
        if iteration%2==0 or iteration==ppo_updates:checkpoint(f'ppo-{iteration}')
    torch.save(actor.state_dict(),directory/'final-retained.pt')
    # Select only on development states. Rejecting an update cannot make it a candidate.
    candidates=[h for h in history if not h['rejected']]
    def score(h):
        m={v['mode']:v['final_10s_rate'] for v in h['evaluations']}
        return (m['balance']>=.9,m['suffix_2s'],m['bottom'],m['balance'])
    result['final_audit']=metrics(actor,p,reference,96000+seed,count=64,full=True)
    best=max(candidates,key=score)
    actor.load_state_dict(torch.load(directory/f"{best['stage']}.pt",weights_only=True))
    result['selected_stage']=best['stage'];result['audit']=metrics(actor,p,reference,96000+seed,count=64,full=True)
    result['fine_suffix']=test_suffix(actor,reference,2.,seed=96000+seed,count=64,substeps=20)[0]
    result['status']='complete';result['training_transitions']=ppo_updates*128*900
    result['selection']='Require development balance >=90%, then rank 2-second success, bottom success, balance; ties earliest. Rejected checkpoints excluded.'
    persist();print('FINISHED',arm,seed,[(m['mode'],m['final_10s_rate']) for m in result['audit']],flush=True)
    return result

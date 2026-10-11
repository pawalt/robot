"""Remote diagnostics of teacher recovery and neural action fit."""
import json
import numpy as np
import torch
from scipy.linalg import expm
from acrobot_feasibility.study import design
from acrobot_comparison.env import measurements,advance_torque
from .policy import Actor,features
from .imitation import teacher_action
from .curriculum import starts


def diagnose(root,source):
    torch.set_num_threads(1)
    data=dict(np.load(source/'controller-0.npz'));gain,linear=design(.05);data['balance_gain']=gain
    matrix=np.zeros((5,5));matrix[:4,:4]=linear['A'];matrix[:4,4:]=linear['B']
    discrete=expm(.05*matrix);ad,bd=discrete[:4,:4],discrete[:4,4:]
    reference=data['states'][:-1];labels,_=teacher_action(reference,np.arange(len(reference)),data,np.zeros(len(reference),bool))
    rows=[]
    for folder in sorted(root.glob('*-0')):
        result=json.loads((folder/'results.json').read_text());actor=Actor()
        for stage in dict.fromkeys(['initial',result['selected_stage'],'final']):
            actor.load_state_dict(torch.load(folder/f'{stage}.pt',weights_only=True))
            obs=torch.tensor(features([[np.pi,0,0,0]]),requires_grad=True)
            value=actor(obs);g=torch.autograd.grad(value.sum(),obs)[0].numpy()[0]
            slope=g @ np.array([[-1,0,0,0],[0,0,0,0],[0,1,0,0],[0,0,0,0],[0,0,.25,0],[0,0,0,.125]])
            pred=actor.act(reference);error=np.abs(pred-labels)
            rows.append(dict(arm=result['arm'],stage=stage,nominal_upright_torque=float(value.detach()),
                             nominal_linear_spectral_radius=float(np.max(np.abs(np.linalg.eigvals(ad+bd@slope[None,:])))),
                             reference_action_mae={str(b):float(error[-round(b/.05):].mean()) for b in [2,4,6,8,30]}))
    # The teacher must succeed from the same difficult approach region independently of the student.
    backs=[2,4,6,8,12,16,20,25,30];count=32
    x=np.concatenate([starts(np.random.default_rng(147000),data['states'],b,count) for b in backs])
    phase=np.repeat([max(0,len(data['actions'])-round(b/.05)) for b in backs],count)
    caught=np.zeros(len(x),bool);streak=np.zeros(len(x))
    for _ in range(900):
        u,caught=teacher_action(x,phase,data,caught)
        for _ in range(10):
            x=advance_torque(x,u,.005);streak=np.where(measurements(x)[2],streak+.005,0)
        phase+=1
    teacher={str(b):int((streak[i*count:(i+1)*count]>=10-1e-8).sum()) for i,b in enumerate(backs)}
    return dict(rows=rows,teacher_successes=teacher,teacher_episodes_per_start=count,
                caveat='Nominal linear spectral radius is a local diagnostic, not a nonlinear stability proof; nonzero torque means nominal upright is not an equilibrium.')

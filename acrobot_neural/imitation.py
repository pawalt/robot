"""Full swing-up imitation, rehearsal and recovery-validated aggregation; Modal only."""
import json
import time
import numpy as np
import torch
from acrobot_comparison.env import measurements
from acrobot_feasibility.study import design, errors
from .policy import Actor, features
from .train import advance, evaluate, DT
from .curriculum import starts, test_suffix
from .sweep import memory

ARMS = ['full', 'backward', 'corrective']
BACK = [2., 4., 6., 8., 12., 16., 20., 25., 30.]


def action_loss(prediction, target, kind='mse'):
    if kind == 'mse':
        return (prediction-target).square().mean()
    # Equivalent saturated actions need no regression toward an arbitrary raw mean.
    target = target.clamp(-1,1)
    error = prediction-target
    error = torch.where(target >= .999999, error.clamp(max=0), error)
    error = torch.where(target <= -.999999, error.clamp(min=0), error)
    return error.square().mean()


def teacher_action(x, phase, data, caught):
    """Training-only phase-conditioned teacher with the validated capture handoff."""
    e = errors(x)
    value = np.einsum('bi,ij,bj->b', e, data['capture_matrix'], e)
    caught = caught | (value <= float(data['capture_level'])) | (phase >= len(data['actions']))
    idx = np.minimum(phase, len(data['actions']) - 1)
    delta = x - data['states'][idx]
    delta[:, :2] = (delta[:, :2] + np.pi) % (2*np.pi) - np.pi
    tracking = data['actions'][idx] - (delta * data['gains'][idx]).sum(1)
    torque = np.where(caught, -e @ data['balance_gain'], tracking)
    return np.clip(torque, -1, 1), caught


def examples(data, back, rng, count=32768):
    end = len(data['actions'])
    idx = rng.integers(max(0, end-round(back/DT)), end, count)
    x = data['states'][idx] + rng.normal(size=(count,4)) * [.003,.003,.01,.01]
    u, _ = teacher_action(x, idx, data, np.zeros(count, bool))
    return torch.from_numpy(features(x)), torch.tensor(u, dtype=torch.float32)


def corrections(actor, data, back, rng, trim=False):
    """Keep corrected teacher rollouts only after a 45-second recovery test passes."""
    x = starts(rng, data['states'], back, 16)
    initial_phase = max(0, len(data['actions']) - round(back/DT))
    candidates, phases = [], []
    for t in range(min(round(back/DT), 900)):
        if t % 5 == 0:
            candidates.append(x.copy()); phases.extend([initial_phase+t]*len(x))
        x = advance(x, actor.act(x), 10)
    states = np.concatenate(candidates)
    chosen = rng.choice(len(states), min(64,len(states)), replace=False)
    x = states[chosen].copy(); phase = np.asarray(phases)[chosen]
    caught = np.zeros(len(x), bool); streak = np.zeros(len(x))
    xs, us, useful = [], [], []
    for t in range(900):
        u, caught = teacher_action(x, phase, data, caught)
        if t % 5 == 0:
            xs.append(x.copy()); us.append(u.copy()); useful.append((~caught) | (t==0))
        for _ in range(10):
            from acrobot_comparison.env import advance_torque
            x = advance_torque(x, u, .005)
            streak = np.where(measurements(x)[2], streak+.005, 0)
        phase += 1
    success = streak >= 10-1e-8
    keep=np.broadcast_to(success,(len(xs),len(success))).copy()
    if trim:keep &= np.asarray(useful)
    accepted_x = np.asarray(xs)[keep]
    accepted_u = np.asarray(us)[keep]
    info = dict(attempted=len(x), recovered=int(success.sum()), examples=len(accepted_x),
                validation_seconds=45, physics_dt=.005)
    return torch.from_numpy(features(accepted_x)), torch.tensor(accepted_u,dtype=torch.float32), info


def run(directory, source, parent, arm, seed, commit, rounds=24, steps=1000, loss_kind='mse',
        checkpoint_name='final-retained',initial_back=2.,trim_corrections=False):
    torch.set_num_threads(1); torch.manual_seed(4100+seed)
    rng = np.random.default_rng(121000+seed)
    data = dict(np.load(source/'controller-0.npz')); data['balance_gain'] = design(DT)[0]
    actor = Actor(); actor.load_state_dict(torch.load(parent/f'{checkpoint_name}.pt',weights_only=True))
    mem, _, targets = memory(actor, data['capture_matrix'], rng)
    opt = torch.optim.Adam(actor.parameters(), lr=1e-4)
    stage = BACK.index(initial_back); passes = 0; history = []; begin = time.perf_counter()
    pools = {back: examples(data,back,rng) for back in BACK}
    extra_x = torch.empty((0,6)); extra_u = torch.empty(0)
    result = dict(arm=arm, seed=seed, status='running', history=history,
                  config=dict(rounds=rounds,steps_per_round=steps,batch_task=512,batch_balance=512,
                              rehearsal_weight=2,learning_rate=1e-4,episode_seconds=45,
                              warm_start=str(parent),curriculum=BACK,allocated_cpus=2,memory_mb=4096,
                              runtime='State-only neural actor; no teacher, phase or LQR',
                              teacher_source=source.name,rollback=False,loss_kind=loss_kind,
                              warm_checkpoint=checkpoint_name,initial_back=initial_back,trim_corrections=trim_corrections))
    def persist():
        result['elapsed_seconds'] = time.perf_counter()-begin
        (directory/'results.json').write_text(json.dumps(result,indent=2)); commit()
    def check(label,back,correction=None):
        bal = evaluate(actor,data['capture_matrix'],seed=122000+seed,count=16,seconds=45)[0]
        suffix = test_suffix(actor,data['states'],back,seed=122000+seed,count=16)[0]
        bottom = evaluate(actor,data['capture_matrix'],seed=122000+seed,count=16,mode='bottom',seconds=45)[0]
        point = dict(stage=label,back_seconds=back,evaluations=[bal,suffix,bottom],
                     correction=correction,elapsed_seconds=time.perf_counter()-begin)
        history.append(point);torch.save(actor.state_dict(),directory/f'{label}.pt');persist()
        print(arm,seed,label,'back',back,'balance',bal['final_10s_rate'],'approach',suffix['final_10s_rate'],
              'bottom',bottom['final_10s_rate'],'correction',correction,flush=True)
        return bal['final_10s_rate']>=.9 and suffix['final_10s_rate']>=.8
    check('initial',30. if arm=='full' else BACK[stage])
    for iteration in range(1,rounds+1):
        back = 30. if arm=='full' else BACK[stage]
        info = None
        if arm=='corrective' and iteration % 2 == 0:
            nx,nu,info = corrections(actor,data,back,rng,trim_corrections)
            extra_x=torch.cat([extra_x,nx])[-65536:];extra_u=torch.cat([extra_u,nu])[-65536:]
        obs,labels = pools[back]
        for _ in range(steps):
            idx=torch.randint(len(obs),(512,)); mi=torch.randint(len(mem),(512,))
            tx,ty=obs[idx],labels[idx]
            if len(extra_x):
                ei=torch.randint(len(extra_x),(256,))
                tx=torch.cat([tx[:256],extra_x[ei]]);ty=torch.cat([ty[:256],extra_u[ei]])
            loss=action_loss(actor(tx),ty,loss_kind)+2*action_loss(actor(mem[mi]),targets[mi],loss_kind)
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(actor.parameters(),1.);opt.step()
        passed=check(f'round-{iteration}',back,info)
        passes=passes+1 if passed else 0
        if arm!='full' and passes>=2 and stage<len(BACK)-1:
            stage+=1;passes=0
    torch.save(actor.state_dict(),directory/'final.pt')
    def score(point):
        bal,suffix,bottom=point['evaluations']
        passed=bal['final_10s_rate']>=.9 and suffix['final_10s_rate']>=.8
        return (bottom['final_10s_rate'] if bal['final_10s_rate']>=.9 else -1,
                point['back_seconds'] if passed else 0,bal['final_10s_rate'],suffix['final_10s_rate'])
    best=max(history,key=score);result['selected_stage']=best['stage'];result['selected_back_seconds']=best['back_seconds']
    def audit():
        rows=[evaluate(actor,data['capture_matrix'],seed=126000+seed,count=64,mode=m,seconds=45)[0]
              for m in ['balance','bottom','kick_0.1','kick_0.5']]
        rows += [test_suffix(actor,data['states'],b,seed=126000+seed,count=64)[0] for b in BACK]
        return rows
    result['final_audit']=audit()
    actor.load_state_dict(torch.load(directory/f"{best['stage']}.pt",weights_only=True))
    result['audit']=audit()
    result['fine_bottom']=evaluate(actor,data['capture_matrix'],seed=126000+seed,count=64,mode='bottom',seconds=45,physics_steps=20)[0]
    result['status']='complete';persist()
    print('IMITATION COMPLETE',arm,seed,'selected',best['stage'],'back',best['back_seconds'],flush=True)
    return result

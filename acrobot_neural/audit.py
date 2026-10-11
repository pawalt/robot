"""Held-out evaluation and preselected videos of neural-only control."""
import json
import numpy as np
import torch
import imageio.v2 as imageio
from PIL import Image
from acrobot_feasibility.record import frame
from .policy import Actor
from .train import evaluate,SOURCE


def audit(directory,source):
    torch.set_num_threads(1)
    result=json.loads((directory/'results.json').read_text())
    p=np.load(source/'controller-0.npz')['capture_matrix']
    def score(point):return sum(e['final_10s_rate'] for e in point['evaluations'])
    curriculum=result.get('kind')=='backward_curriculum'
    if curriculum:
        def curriculum_score(point):
            metrics={m['mode']:m['final_10s_rate'] for m in point['evaluations']}
            suffix=next(v for k,v in metrics.items() if k.startswith('suffix_'))
            passed=suffix>=.8 and metrics['balance']>=.9
            return (point['curriculum_back_seconds'] if passed else 0,score(point))
        selected=(next(point for point in result['history'] if point['stage']==result['audit_selected_stage'])
                  if result.get('audit_selected_stage') else max(result['history'],key=curriculum_score))
        chosen=selected['stage'];back=selected['curriculum_back_seconds']
        stages=list(dict.fromkeys(['ppo-0',chosen,result['history'][-1]['stage']]))
    else:
        chosen=max(result['history'],key=score)['stage']
        stages=list(dict.fromkeys(['balance','imitation',chosen,result['history'][-1]['stage']]))
    audit_seed=(result.get('config',{}).get('audit_seed') or 84000) if curriculum else 74000
    actor=Actor();rows=[];recordings=[]
    for stage in stages:
        actor.load_state_dict(torch.load(directory/f'{stage}.pt',weights_only=True));actor.eval()
        metrics=[]
        modes=['balance','bottom','kick_0.1','kick_0.5','kick_1.0']
        if curriculum:modes.extend(f'suffix_{seconds:g}s' for seconds in sorted({back,result['current_back_seconds']}))
        for mode in modes:
            if mode.startswith('suffix_'):
                from .curriculum import test_suffix
                reference=np.load(source/'controller-0.npz')['states']
                row,trace=test_suffix(actor,reference,float(mode.removeprefix('suffix_').removesuffix('s')),seed=audit_seed,count=64,capture=True)
            else:
                row,trace=evaluate(actor,p,seed=audit_seed,count=64,mode=mode,seconds=45,capture=True)
            metrics.append(row)
            if stage==chosen:
                name=f'{stage}-{mode}'
                xs,us=trace
                Image.fromarray(frame(xs[0],0,us[0],f'Neural only | {mode}')).save(directory/f'{name}.jpg')
                with imageio.get_writer(str(directory/f'{name}.mp4'),fps=20,codec='libx264',pixelformat='yuv420p',output_params=['-movflags','+faststart']) as writer:
                    for t,x in enumerate(xs):writer.append_data(frame(x,t*.05,us[t],f'Neural only | {mode}'))
                recordings.append(dict(stage=stage,mode=mode,video=name+'.mp4',poster=name+'.jpg',episode_index=0,
                                       final_hold_seconds=row['per_episode_final_hold'][0]))
                successful=np.flatnonzero(np.asarray(row['per_episode_final_hold'])>=10-1e-8)
                if mode=='bottom' and len(successful) and successful[0]!=0:
                    index=int(successful[0]);name=f'{stage}-bottom-success'
                    _,(xs,us)=evaluate(actor,p,seed=audit_seed,count=64,mode=mode,seconds=45,capture=True,capture_index=index)
                    label=f'Neural only | selected successful episode {index}'
                    Image.fromarray(frame(xs[0],0,us[0],label)).save(directory/f'{name}.jpg')
                    with imageio.get_writer(str(directory/f'{name}.mp4'),fps=20,codec='libx264',pixelformat='yuv420p',output_params=['-movflags','+faststart']) as writer:
                        for t,x in enumerate(xs):writer.append_data(frame(x,t*.05,us[t],label))
                    recordings.append(dict(stage=stage,mode='bottom',selection='successful',video=name+'.mp4',poster=name+'.jpg',episode_index=index,
                                           final_hold_seconds=row['per_episode_final_hold'][index]))
        rows.append(dict(stage=stage,evaluations=metrics))
        print('AUDIT',stage,[(m['mode'],m['final_10s_rate']) for m in metrics],flush=True)
    # Fine-step confirmation of the selected checkpoint, same held-out starts.
    actor.load_state_dict(torch.load(directory/f'{chosen}.pt',weights_only=True))
    fine=[evaluate(actor,p,seed=audit_seed,count=64,mode=mode,seconds=45,physics_steps=20)[0] for mode in ['balance','bottom']]
    if curriculum:
        fine.append(test_suffix(actor,reference,back,seed=audit_seed,count=64,substeps=20)[0])
        result['selected_back_seconds']=back
    result.update(status='complete',selected_stage=chosen,selection='Highest summed development success across balance, bottom and 0.1 rad/s kicks; ties choose earliest checkpoint',
                  audit=rows,fine_evaluations=fine,recordings=recordings)
    if curriculum:result['selection']='Farthest development suffix passing 80% success with at least 90% balance retention; tie by summed development success'
    (directory/'results.json').write_text(json.dumps(result,indent=2))
    return result

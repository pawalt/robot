"""Audit earlier approach skills after PPO; run on Modal after the main audit."""
import json
import numpy as np
import torch
import imageio.v2 as imageio
from PIL import Image
from acrobot_feasibility.record import frame
from .policy import Actor
from .curriculum import test_suffix


def audit_earlier(directory,source):
    torch.set_num_threads(1)
    result=json.loads((directory/'results.json').read_text());assert result['status']=='complete'
    actor=Actor();reference=np.load(source/'controller-0.npz')['states'];seed=result['config']['audit_seed']
    for point in result['audit']:
        actor.load_state_dict(torch.load(directory/f"{point['stage']}.pt",weights_only=True))
        for back in [2.,4.,6.]:
            mode=f'suffix_{back:g}s'
            if any(m['mode']==mode for m in point['evaluations']):continue
            row,(xs,us)=test_suffix(actor,reference,back,seed=seed,count=64,capture=True)
            point['evaluations'].append(row)
            if back==6 and point['stage']==result['selected_stage']:
                name=f"{point['stage']}-{mode}";label='Neural PPO | six-second retention test'
                Image.fromarray(frame(xs[0],0,us[0],label)).save(directory/f'{name}.jpg')
                with imageio.get_writer(str(directory/f'{name}.mp4'),fps=20,codec='libx264',pixelformat='yuv420p',output_params=['-movflags','+faststart']) as writer:
                    for t,x in enumerate(xs):writer.append_data(frame(x,t*.05,us[t],label))
                result['recordings'].append(dict(stage=point['stage'],mode=mode,video=name+'.mp4',poster=name+'.jpg',episode_index=0,final_hold_seconds=row['per_episode_final_hold'][0]))
    result['earlier_start_audit_complete']=True
    (directory/'results.json').write_text(json.dumps(result,indent=2));return result

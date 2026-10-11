"""Preselected neural-only held-out recordings, rendered on Modal."""
import json
import numpy as np
import torch
import imageio.v2 as imageio
from PIL import Image
from acrobot_feasibility.record import frame
from .policy import Actor
from .train import evaluate
from .curriculum import test_suffix
from .imitation import BACK


def record(directory, source, confirm=False):
    torch.set_num_threads(1)
    result=json.loads((directory/'results.json').read_text())
    assert result['status']=='complete'
    data=np.load(source/'controller-0.npz');actor=Actor();movies=[]
    for checkpoint,stage,modes in [('selected',result['selected_stage'],['balance','approach','bottom']),('final','final',['bottom'])]:
        actor.load_state_dict(torch.load(directory/f'{stage}.pt',weights_only=True));actor.eval()
        for mode in modes:
            if mode=='approach':
                row,(xs,us)=test_suffix(actor,data['states'],result['selected_back_seconds'],seed=126000+result['seed'],count=64,capture=True)
            else:
                row,(xs,us)=evaluate(actor,data['capture_matrix'],seed=126000+result['seed'],count=64,mode=mode,seconds=45,capture=True)
            expected=next(m for m in result['audit' if checkpoint=='selected' else 'final_audit'] if m['mode']==row['mode'])
            assert np.allclose(expected['per_episode_final_hold'],row['per_episode_final_hold'],atol=.011)
            name=f'{checkpoint}-{mode}'
            label=f"{result['arm']} | {checkpoint} neural | {row['mode']}"
            Image.fromarray(frame(xs[0],0,us[0],label)).save(directory/f'{name}.jpg')
            with imageio.get_writer(str(directory/f'{name}.mp4'),fps=20,codec='libx264',pixelformat='yuv420p',output_params=['-movflags','+faststart']) as writer:
                for t,x in enumerate(xs):writer.append_data(frame(x,t*.05,us[t],label))
            movies.append(dict(checkpoint=checkpoint,stage=stage,mode=row['mode'],episode_index=0,
                               final_hold_seconds=row['per_episode_final_hold'][0],video=name+'.mp4',poster=name+'.jpg'))
    result['recordings']=movies
    if confirm:
        actor.load_state_dict(torch.load(directory/f"{result['selected_stage']}.pt",weights_only=True))
        confirmation_seed=156000+result['seed']
        result['confirmation_audit']=[evaluate(actor,data['capture_matrix'],seed=confirmation_seed,count=128,
                                              mode=mode,seconds=45)[0] for mode in ['balance','bottom','kick_0.1','kick_0.5']]
        result['confirmation_audit'] += [test_suffix(actor,data['states'],back,seed=confirmation_seed,count=128)[0] for back in BACK]
        result['confirmation_fine']=test_suffix(actor,data['states'],result['selected_back_seconds'],seed=confirmation_seed,count=128,substeps=20)[0]
    (directory/'results.json').write_text(json.dumps(result,indent=2))
    return result

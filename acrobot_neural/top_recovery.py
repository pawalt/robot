"""Top releases, uncontrolled falls, and actual mid-episode kicks; Modal only."""
import json
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
import imageio.v2 as imageio
from .radial import ExportedActor
from .radial_robust import PARENTS
from acrobot_comparison.env import advance_torque,measurements
from acrobot_feasibility.record import frame

CONDITIONS=[(f'release_{deg:+d}', 'release',deg) for deg in [15,-15,30,-30,60,-60,90,-90]]+[
    ('free_fall','drop',1),('kick_2','kick',2),('kick_4','kick',4)]


def run(directory,parent,trajectory,commit):
    torch.set_num_threads(1);actor=ExportedActor(parent/'deployed.pt');rng=np.random.default_rng(381000)
    count=32;n=len(CONDITIONS);x=np.zeros((n*count,4));initial=[]
    for i,(name,kind,value) in enumerate(CONDITIONS):
        sl=slice(i*count,(i+1)*count)
        if kind=='kick':x[sl]=0
        else:
            x[sl]=np.array([np.pi+np.deg2rad(value),0.,0.,0.])+rng.uniform(-1,1,(count,4))*[np.deg2rad(1),np.deg2rad(1),.02,.02]
            x[i*count]=[np.pi+np.deg2rad(value),0.,0.,0.]
    initial=x.copy();hold=np.zeros(len(x));peak=np.zeros(len(x));pre_kick=np.zeros(len(x),bool)
    traces=[];torques=[]
    for step in range(2400):
        t=step*.05
        for i,(_,kind,value) in enumerate(CONDITIONS):
            if kind=='kick' and step==800:
                sl=slice(i*count,(i+1)*count);pre_kick[sl]=hold[sl]>=10
                x[sl,2]+=value;x[sl,3]-=value;hold[sl]=0
        u=actor.act(x)
        for i,(_,kind,_) in enumerate(CONDITIONS):
            if kind=='drop' and t<2:u[i*count:(i+1)*count]=0
        traces.append(x.copy());torques.append(u.copy())
        for _ in range(20):
            x=advance_torque(x,u,.0025);_,height,balanced=measurements(x)
            hold=np.where(balanced,hold+.0025,0);peak=np.maximum(peak,height)
    traces=np.array(traces);torques=np.array(torques);rows=[];recordings=[]
    for i,(name,kind,value) in enumerate(CONDITIONS):
        sl=slice(i*count,(i+1)*count);success=hold[sl]>=10-1e-8
        row=dict(condition=name,kind=kind,value=value,episodes=count,successes=int(success.sum()),exact_success=bool(success[0]),
                 final_hold_seconds=hold[sl].tolist(),initial_states=initial[sl].tolist(),pre_kick_balanced=int(pre_kick[sl].sum()) if kind=='kick' else None)
        rows.append(row)
        if name not in ['release_+15','release_-15','release_+60','free_fall','kick_4']:continue
        picks=[('exact',0)]
        if success[0] and (~success).any():picks.append(('failure',int(np.flatnonzero(~success)[0])))
        if not success[0] and success.any():picks.append(('success',int(np.flatnonzero(success)[0])))
        for selection,episode in picks:
            j=i*count+episode;filename=name.replace('+','p').replace('-','m')+'-'+selection
            with imageio.get_writer(str(directory/f'{filename}.mp4'),fps=20,codec='libx264',pixelformat='yuv420p',output_params=['-movflags','+faststart']) as writer:
                # Kick video includes a visible five-second balanced lead-in before impact.
                start=700 if kind=='kick' else 0
                for step in range(start,2400):
                    t=step*.05
                    im=Image.fromarray(frame(traces[step,j],t,torques[step,j],f'Top recovery | {name} | {selection}'))
                    draw=ImageDraw.Draw(im)
                    if kind=='kick':text='BALANCED BEFORE KICK' if t<40 else f'KICK at 40s: velocities +{value}, -{value} rad/s'
                    elif kind=='drop':text='RELEASED: MOTOR OFF' if t<2 else 'NEURAL CONTROL ON since 2s'
                    else:text=f'Released from {value:+d} degrees off upright'
                    draw.rectangle((0,600,640,640),fill='#6e2634' if (kind=='kick' and 40<=t<43) else '#142c38')
                    draw.text((12,611),text,font=ImageFont.load_default(size=19),fill='white')
                    if step==start:im.save(directory/f'{filename}.jpg')
                    writer.append_data(np.asarray(im))
            recordings.append(dict(condition=name,video=filename+'.mp4',poster=filename+'.jpg',selection=selection,episode=episode,success=bool(success[episode]),final_hold_seconds=float(hold[j]),start_time=35 if kind=='kick' else 0))
    result=dict(id=directory.name,trajectory=trajectory,parent=parent.name,status='complete',seed=381000,physics_dt=.0025,seconds=120,episodes_per_condition=count,
                description='Top releases with zero nominal velocity; both links tilted together. Drop: near-upright, motor off first 2s. Kick: velocity impulse at40s after swing-up. No state reset at kick.',evaluations=rows,recordings=recordings)
    (directory/'results.json').write_text(json.dumps(result,indent=2));commit()
    print('TOP RECOVERY',trajectory,[(r['condition'],r['successes'],r['exact_success']) for r in rows],flush=True)
    return result

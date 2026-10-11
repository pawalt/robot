"""Sparse constrained trajectory optimization using CasADi/IPOPT."""
import time
import casadi as ca
import numpy as np
from .trajectory import seed_path, feedback_gains, replay, DT
from acrobot_feasibility.study import errors


def symbolic_step():
    x=ca.MX.sym('x',4); u=ca.MX.sym('u')
    t1,t2,v1,v2=ca.vertsplit(x)
    d1=3.5+ca.cos(t2); d2=1.25+.5*ca.cos(t2)
    phi2=4.9*ca.sin(t1+t2)
    phi1=-.5*v2**2*ca.sin(t2)-v2*v1*ca.sin(t2)+14.7*ca.sin(t1)+phi2
    a2=(u+d2/d1*phi1-.5*v1**2*ca.sin(t2)-phi2)/(1.25-d2*d2/d1)
    a1=-(d2*a2+phi1)/d1
    f=ca.Function('dynamics',[x,u],[ca.vertcat(v1,v2,a1,a2)])
    y=x; h=DT/5
    for _ in range(5):
        k1=f(y,u); k2=f(y+h*k1/2,u); k3=f(y+h*k2/2,u); k4=f(y+h*k3,u)
        y=y+h/6*(k1+2*k2+2*k3+k4)
    return ca.Function('step',[x,u],[y])


def solve(states,actions,limit=1.):
    n=len(actions); opt=ca.Opti()
    x=opt.variable(4,n+1); u=opt.variable(1,n)
    target=np.array([np.pi+2*np.pi*round((states[-1,0]-np.pi)/(2*np.pi)),
                     2*np.pi*round(states[-1,1]/(2*np.pi)),0,0])
    opt.subject_to(x[:,0]==np.zeros(4)); opt.subject_to(x[:,-1]==target)
    opt.subject_to(x[:,1:]==symbolic_step().map(n)(x[:,:-1],u))
    opt.subject_to(opt.bounded(-limit,u,limit))
    opt.subject_to(opt.bounded(-4*np.pi,x[2,:],4*np.pi))
    opt.subject_to(opt.bounded(-9*np.pi,x[3,:],9*np.pi))
    opt.minimize(DT*ca.sumsqr(u)*.001)
    opt.set_initial(x,states.T); opt.set_initial(u,np.clip(actions,-limit,limit))
    opt.solver('ipopt',{'print_time':False},{'print_level':0,'max_iter':2500,
               'tol':1e-7,'constr_viol_tol':1e-8,'max_cpu_time':240.})
    start=time.monotonic()
    try:
        sol=opt.solve(); success=True
    except RuntimeError:
        sol=opt.debug; success=False
    fitted=np.asarray(sol.value(x)).T; torques=np.asarray(sol.value(u)).reshape(-1)
    report=dict(solver_success=success,status=opt.stats()['return_status'],
                iterations=opt.stats()['iter_count'],seconds=time.monotonic()-start)
    return fitted,torques,report


def search(seed,directory,limit=1.):
    states,actions=seed_path(seed,limit=limit)
    e=errors(states); score=10*np.sum(e[:,:2]**2,axis=1)+np.sum(e[:,2:]**2,axis=1)
    score[:200]=np.inf
    ends=[]
    for i in np.argsort(score):
        if all(abs(i-j)>100 for j in ends):ends.append(int(i))
        if len(ends)==2:break
    attempts=[]
    for attempt,end in enumerate(ends):
        print('IPOPT',seed,'duration',end*DT,flush=True)
        x,u,fit=solve(states[:end+1],actions[:end],limit)
        gains=feedback_gains(x,u)
        result,_=replay(x,u,gains,limit)
        filename=f'optimal-{seed}-{attempt}.npz'
        np.savez(directory/filename,states=x,actions=u,gains=gains,limit=limit)
        attempts.append(dict(attempt=attempt,fit=fit,validation=result))
        print('IPOPT replay',seed,fit,result,flush=True)
        if fit['solver_success'] and result['final_10s_rate']==1:
            return dict(seed=seed,success=True,attempts=attempts,file=filename)
    return dict(seed=seed,success=False,attempts=attempts)

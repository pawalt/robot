"""Empirical LQR capture region, not a formal region-of-attraction certificate."""
import numpy as np
from scipy.linalg import solve_discrete_are
from acrobot_comparison.env import advance_torque, measurements
from acrobot_feasibility.study import design, errors
from .trajectory import step_raw, DT


def evaluate(initials, limit=1., seconds=12.):
    x=initials.copy(); gain,_=design(DT); streak=np.zeros(len(x))
    for _ in range(round(seconds/DT)):
        u=np.clip(-errors(x)@gain,-limit,limit)
        for _ in range(10):
            x=advance_torque(x,u,.005)
            streak=np.where(measurements(x)[2],streak+.005,0)
    return streak>=5-1e-8


def map_capture(limit=1.):
    equilibrium=np.array([[np.pi,0.,0.,0.]])
    eps=1e-5
    a=np.column_stack([(step_raw(equilibrium+np.eye(4)[i]*eps,np.zeros(1),substeps=10)-
                       step_raw(equilibrium-np.eye(4)[i]*eps,np.zeros(1),substeps=10))[0]/(2*eps) for i in range(4)])
    b=((step_raw(equilibrium,np.array([eps]),substeps=10)-step_raw(equilibrium,np.array([-eps]),substeps=10))/(2*eps)).T
    p=solve_discrete_are(a,b,np.diag([10.,10.,1.,1.]),np.array([[1.]]))
    angles=np.linspace(-.2,.2,41); speeds=np.linspace(-1.,1.,41)
    aa,vv=np.meshgrid(angles,speeds)
    slices=[]
    for name,axes in [('Whole-robot lean and rotation',(0,2)),('Elbow bend and elbow velocity',(1,3))]:
        x=np.tile(equilibrium,(aa.size,1))
        x[:,axes[0]]+=aa.ravel(); x[:,axes[1]]+=vv.ravel()
        passed=evaluate(x,limit).reshape(aa.shape)
        slices.append(dict(name=name,angle_axis=axes[0],velocity_axis=axes[1],angles=angles.tolist(),
                           velocities=speeds.tolist(),success=passed.astype(int).tolist()))
    rng=np.random.default_rng(61000)
    directions=rng.normal(size=(256,4)); directions/=np.linalg.norm(directions,axis=1)[:,None]
    # e^T P e = level, with uniformly distributed directions in transformed coordinates.
    transform=np.linalg.inv(np.linalg.cholesky(p).T)
    levels=np.geomspace(.001,100.,21)
    offsets=np.concatenate([np.sqrt(level)*(directions@transform.T) for level in levels])
    success=evaluate(equilibrium+offsets,limit).reshape(len(levels),-1)
    rates=success.mean(axis=1)
    safe=0.
    for level,rate in zip(levels,rates):
        if rate<1: break
        safe=float(level)
    # Independently check filled ellipsoid, not only the searched shell directions.
    directions=rng.normal(size=(1024,4)); directions/=np.linalg.norm(directions,axis=1)[:,None]
    offsets=np.sqrt(safe)*(directions@transform.T)*rng.random((1024,1))**.25
    check=evaluate(equilibrium+offsets,limit)
    return dict(limit=limit,dt=DT,physics_dt=.005,seconds=12.,criterion='Balanced throughout final 5 seconds',
                slices=slices,lyapunov_matrix=p.tolist(),levels=levels.tolist(),success_rates=rates.tolist(),
                directions_per_level=256,empirical_level=safe,independent_samples=1024,
                independent_successes=int(check.sum()),
                coordinate_intercepts=np.sqrt(safe/np.diag(p)).tolist(),
                caveat='Sampled recovery region for this controller; not a proof and not independent coordinate bounds.')

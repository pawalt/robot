"""Bounded multiple shooting and deterministic trajectory feedback control."""
import time
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.linalg import solve_discrete_are
from acrobot_comparison.env import derivatives, advance_torque, measurements
from acrobot_feasibility.study import design, errors
from .controller import DT, swing_torque


def step_raw(x, u, dt=DT, substeps=2):
    h = dt / substeps
    for _ in range(substeps):
        k1 = derivatives(x, u)
        k2 = derivatives(x + h*k1/2, u)
        k3 = derivatives(x + h*k2/2, u)
        k4 = derivatives(x + h*k3, u)
        x = x + h/6*(k1+2*k2+2*k3+k4)
    return x


def seed_path(seed, seconds=60, limit=1.):
    config = dict(family='power', ke=[3., 1., .3][seed % 3], kp=0, kd=.05, limit=limit)
    x = np.zeros((1, 4))
    states, actions = [x[0].copy()], []
    for t in range(round(seconds / DT)):
        u = np.array([limit * (1 if seed % 2 == 0 else -1)]) if t < 5 + seed else swing_torque(x, config)
        x = step_raw(x, u, substeps=10)
        states.append(x[0].copy()); actions.append(u[0])
    return np.array(states), np.array(actions)


def fit_segment(initial, guess_states, guess_actions, limit, max_nfev=700):
    n = len(guess_actions)
    # Fix initial and terminal states exactly. Optimize intervening states and torques.
    target = np.array([np.pi + 2*np.pi*round((guess_states[-1, 0]-np.pi)/(2*np.pi)),
                       2*np.pi*round(guess_states[-1, 1]/(2*np.pi)), 0., 0.])
    def unpack(z):
        return np.vstack([initial, z[:4*(n-1)].reshape(-1, 4), target]), z[4*(n-1):]
    def residual(z):
        x, u = unpack(z)
        return (x[1:] - step_raw(x[:-1], u)).ravel()
    sparsity = lil_matrix((4*n, 5*n-4), dtype=int)
    for i in range(n):
        if i > 0: sparsity[4*i:4*i+4, 4*(i-1):4*i] = 1
        if i < n-1: sparsity[4*i:4*i+4, 4*i:4*i+4] = 1
        sparsity[4*i:4*i+4, 4*(n-1)+i] = 1
    z = np.r_[guess_states[1:-1].ravel(), np.clip(guess_actions, -limit+1e-9, limit-1e-9)]
    lower = np.r_[np.tile([-np.inf,-np.inf,-4*np.pi,-9*np.pi], n-1), np.full(n,-limit)]
    upper = np.r_[np.tile([np.inf,np.inf,4*np.pi,9*np.pi], n-1), np.full(n,limit)]
    start=time.monotonic()
    fit = least_squares(residual, z, bounds=(lower,upper), jac_sparsity=sparsity.tocsr(),
                        max_nfev=max_nfev, ftol=1e-10, xtol=1e-10, gtol=1e-10,
                        tr_options={'atol':1e-10,'btol':1e-10,'maxiter':1000})
    x,u=unpack(fit.x)
    report=dict(nfev=fit.nfev, max_defect=float(np.max(np.abs(fit.fun))),
                residual_norm=float(np.linalg.norm(fit.fun)), seconds=time.monotonic()-start)
    print('Fit', report, flush=True)
    return x,u,report


def feedback_gains(states, actions):
    n=len(actions); eps=1e-5
    a=np.stack([(step_raw(states[:-1]+np.eye(4)[j]*eps,actions,substeps=10)-
                 step_raw(states[:-1]-np.eye(4)[j]*eps,actions,substeps=10))/(2*eps) for j in range(4)],axis=2)
    b=((step_raw(states[:-1],actions+eps,substeps=10)-step_raw(states[:-1],actions-eps,substeps=10))/(2*eps))[:,:,None]
    q=np.diag([10.,10.,1.,1.]); r=np.array([[1.]])
    p=solve_discrete_are(a[-1],b[-1],q,r)
    gains=np.empty((n,4))
    for i in range(n-1,-1,-1):
        k=np.linalg.solve(r+b[i].T@p@b[i],b[i].T@p@a[i])
        gains[i]=k[0]
        p=q+a[i].T@p@(a[i]-b[i]@k)
        p=(p+p.T)/2
    return gains


def replay(states, actions, gains, limit=1., physics_dt=.005, initials=None, capture=False, capture_region=None):
    x=np.zeros((1,4)) if initials is None else np.asarray(initials).copy()
    controller=TrajectoryController(states,actions,gains,limit,capture_region)
    streak=np.zeros(len(x)); longest=streak.copy(); peak=streak.copy()
    traces=[]; torques=[]; entered=np.full(len(x),np.nan)
    for t in range(len(actions)+round(12/DT)):
        u=controller(x)
        entered=np.where(np.isnan(entered) & controller.balancing,t*DT,entered)
        if capture: traces.append(x.copy()); torques.append(u.copy())
        for _ in range(round(DT/physics_dt)):
            x=advance_torque(x,u,physics_dt)
            _,height,balanced=measurements(x)
            streak=np.where(balanced,streak+physics_dt,0)
            longest=np.maximum(longest,streak); peak=np.maximum(peak,height)
    report=dict(episodes=len(x), final_10s_rate=float(np.mean(streak>=10-1e-8)),
                final_hold_seconds=streak.tolist(), longest_hold_seconds=longest.tolist(),
                final_states=x.tolist(), physics_dt=physics_dt, max_height=peak.tolist(),
                handoff_seconds=[None if np.isnan(v) else float(v) for v in entered])
    return report,(np.asarray(traces),np.asarray(torques))


def search(seed, directory, limit=1.):
    states,actions=seed_path(seed,limit=limit)
    e=errors(states)
    score=10*(e[:,0]**2+e[:,1]**2)+e[:,2]**2+e[:,3]**2
    score[:160]=np.inf
    indices=[]
    for idx in np.argsort(score):
        if all(abs(idx-other)>80 for other in indices): indices.append(int(idx))
        if len(indices)==3: break
    attempts=[]
    for attempt,end in enumerate(indices):
        start=max(0,end-160)
        print('Seed',seed,'candidate',attempt,'end',end*DT,'score',score[end],flush=True)
        x,u,fit=fit_segment(states[start], states[start:end+1], actions[start:end], limit)
        fullx=np.vstack([states[:start],x]); fullu=np.r_[actions[:start],u]
        gains=feedback_gains(fullx,fullu)
        result,_=replay(fullx,fullu,gains,limit)
        attempts.append(dict(attempt=attempt,fit=fit,validation=result))
        np.savez(directory/f'attempt-{seed}-{attempt}.npz',states=fullx,actions=fullu,gains=gains,limit=limit)
        print('REPLAY',seed,attempt,result,flush=True)
        if result['final_10s_rate']==1:
            return dict(seed=seed,success=True,attempts=attempts,file=f'attempt-{seed}-{attempt}.npz')
    return dict(seed=seed,success=False,attempts=attempts)


class TrajectoryController:
    """Deterministic state-feedback tracker; reset the clock for each episode.

    The reference starts hanging down. After the reference finishes, local LQR
    holds upright indefinitely. This is not a global disturbance-recovery policy.
    """
    def __init__(self, states, actions, gains, limit=1., capture_region=None):
        self.states, self.actions, self.gains = states, actions, gains
        self.limit = limit
        self.capture_region = capture_region
        self.balance_gain, _ = design(DT)
        self.reset()

    @classmethod
    def load(cls, path):
        with np.load(path) as data:
            region=({'lyapunov_matrix':data['capture_matrix'], 'empirical_level':float(data['capture_level'])}
                    if 'capture_matrix' in data else None)
            return cls(data['states'], data['actions'], data['gains'], float(data['limit']),region)

    def reset(self):
        self.index = 0
        self.balancing = False

    def __call__(self, state):
        state = np.asarray(state)
        if self.index < len(self.actions):
            error = state - self.states[self.index]
            error[..., :2] = (error[..., :2]+np.pi)%(2*np.pi)-np.pi
            torque = self.actions[self.index] - error @ self.gains[self.index]
        else:
            torque = -errors(np.atleast_2d(state)) @ self.balance_gain
            if state.ndim == 1: torque = torque[0]
        balance_error=errors(np.atleast_2d(state))
        if self.capture_region is not None:
            p=np.asarray(self.capture_region['lyapunov_matrix'])
            value=np.einsum('bi,ij,bj->b',balance_error,p,balance_error)
            caught=value<=self.capture_region['empirical_level']
            if state.ndim==1: caught=caught[0]
            self.balancing=self.balancing | caught
        self.balancing=self.balancing | (self.index>=len(self.actions))
        balance_torque=-balance_error@self.balance_gain
        if state.ndim==1: balance_torque=balance_torque[0]
        torque=np.where(self.balancing,balance_torque,torque)
        self.index += 1
        return np.clip(torque, -self.limit, self.limit)

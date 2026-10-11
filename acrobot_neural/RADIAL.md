# Verified neural ground-to-balance policies

Three independently audited, state-only radial neural policies swing Acrobot from
exactly motionless at the bottom and balance through a two-minute episode.
All training, simulation, audits and recording ran on Modal in `peyton-agents`.
The dashboard is `/experiments/acrobot-radial`.

## What is learned

This is a Gaussian radial-basis neural network with local affine output experts,
not PPO or the previous tanh MLP. Each hidden unit has a center taken from a
successful closed-loop training demonstration. Each local expert's six weights
and bias are fitted by least squares to 256 nearby state/torque examples. Teacher
labels are unsaturated torques; the deployed output is clipped to +/-1 N m.
Training uses the existing trajectory feedback controller and LQR to label data.
Neither is invoked by the deployed policy.

The inference graph performs:

1. Four raw states (shoulder angle, relative elbow angle, and their velocities)
   become six features: sin/cos of each angle and velocities divided by4 and8.
2. Squared feature distances to fixed, training-derived radial centers produce
   Gaussian softmax gates with width sigma.
3. The gates blend the fitted affine torque predictions, then clip to +/-1.

There is no clock, recurrent state, phase index, trajectory advancement, external
reference-file access, deterministic fallback or LQR call during inference.
The centers retain demonstration structure as neural parameters. This is a
specialized local imitation architecture, not evidence that generic PPO now works
or that the earlier MLP could not learn the task. Both architecture and training
procedure changed, so the experiment does not isolate a single cause of success.

## Training and selection

Twenty-four variants: three demonstrated trajectories, widths .003/.01/.03/.1,
and local training noise .001/.01. The four-dimensional noise multipliers are
[1,1,3,6]. Data seed0. Supervised fit uses float64 least squares (rcond1e-5), with
float32 deployment weights. Prototype counts are603,467,524, respectively.
Development: 16 starts, seed301000, uniform +/-0.01 perturbations, first start
exactly zero, 60seconds, .005second physics. 22/24 variants passed16/16;
the two width.1 variants for trajectory1 passed12/16 and13/16.
Each job (training plus development) took2.6–4.7seconds on a two-CPU Modal worker.

Before independent audits, select width.03/noise.01 for each of the three
trajectories. Export to standalone TorchScript (`deployed.pt`), reload the export,
and use that export for **all** independent tests. A test-only serialization check
also verifies different batch sizes and output clipping.

## Independent results

Every episode runs120seconds. Control updates every.05seconds. Success means
both links within10degrees of vertical and both absolute link velocities below
.5rad/s continuously during the final10seconds. The held-out seed is311000.
The randomized bottom tests include exactly zero as episode0; all other episodes
sample each angle and angular velocity uniformly in the stated interval.

| Training trajectory | Exact-ground final balance hold | +/- .01,128 starts | Same starts, finer physics | +/- .05,128 starts | +/- .1,128 starts | Local balance,128 starts |
|---|---:|---:|---:|---:|---:|---:|
|0|90.85s|128/128|128/128|128/128|128/128|128/128|
|1|98.50s|128/128|128/128|128/128|127/128|128/128|
|2|95.7625s|128/128|128/128|128/128|128/128|128/128|

Exact-ground and fine tests use.0025second physics steps; other ground tests
use.005seconds. Local balance starts use the existing level10 capture ellipsoid
and.0025second physics. The fastest verified policy starts sustained balance at
about21.5seconds. No broad disturbance-recovery or real-robot claim is made.
These dynamics use the project's continuous-torque Acrobot model, not the stock
discrete-action Acrobot-v1 task.

## Artifacts

On Modal Volume `robot-grid-recordings`, under `/neural-radial/RUN/`:
`policy.pt` (radial weights), `deployed.pt` (standalone inference), `results.json`,
and120secondMP4 recordings. All24 results and the three audited policies remain
available, including weaker variants. JSON is mirrored in `artifacts/`.

- Trajectory0: `20261011T183151-433572cc`.
- Trajectory1: `20261011T183152-b6856ae9`.
- Trajectory2: `20261011T183155-78fa3813`.

The exported graph accepts a float32 tensor shaped(batch,4), in radians and
radians/second, and returns torque shaped(batch,). Example inference:

```python
policy = torch.jit.load("deployed.pt")
with torch.no_grad():
    torque = policy(state_tensor)
```

Reproduce remotely with the `radial_sweep` then `radial_audit` entrypoints in
`acrobot_neural.modal_app`, using the locked uv environment and Modal credentials.

Larger starting-disturbance tests and expanded imitation: [DISTURBANCE.md](DISTURBANCE.md).

# Recovery from top releases and mid-episode kicks

This addresses top disturbances, separately from the hanging-start tests in
DISTURBANCE.md. All evaluations and video generation run on Modal, peyton-agents.
The dashboard is `/experiments/acrobot-top`. The three original radial neural
exports are tested without additional training.

## Cases

- Both links aligned and tilted+/-15,+/-30,+/-60,+/-90degrees from upright,
  released with zero nominal angular velocity. Each case contains the exact
  stated state and31 nearby perturbations (up to1degree per joint coordinate,
  .02rad/s per velocity), seed381000.
- Uncontrolled fall: initially1degree off upright, motor torque forced tozero
  for the first2seconds, then neural control enabled. Same nearby perturbations.
- Actual kicks: start at the bottom, let the policy swing up and balance, then
  at40seconds add+2/-2 or+4/-4rad/s to shoulder/relative elbow velocities.
  Positions are unchanged. No reset or teleport to a different configuration.
  Require/check at least10seconds of preceding balance. These are fixed
  deterministic kicks, simulated in32 identical batch members; they constitute
  **one distinct test each**, not32 independent robustness trials.

All tests last120seconds, with.0025second physics steps,.05second control, and
+/-1N m actuator limits. Success requires both links within10degrees of upright
and both absolute link speeds below.5rad/s for the final10seconds continuously.
A fall and subsequent new swing-up is allowed; staying near the top is not
required. The policy receives only the current state, with no reset of any
controller clock (there is none), no LQR fallback, and no reference trajectory
access during inference.

## Recordings

Exact+15,-15,+60degree releases, free fall, and the4rad/s kick are recorded for
each policy. Where nearby perturbations give the opposite outcome, include the
first such episode and label it success/failure-selected. The kick video begins
at35seconds, showing five seconds of balancing before the highlighted40second
kick. Other videos begin at release. The motor-off interval and switch to neural
control are explicitly labeled in the drop video.

Artifacts are under `/data/neural-top/RUN` on the existing Modal Volume, with JSON
mirrored locally and all outcomes preserved. Reproduce with the `top_recovery`
entrypoint in `acrobot_neural.modal_app` using the locked uv environment.

## Results

| Policy | Successful release/drop episodes /288 | Exact +15° | Fixed2rad/s kick | Fixed4rad/s kick |
|---|---:|---|---|---|
|0|276|True|False|True|
|1|288|True|True|True|
|2|282|False|True|True|

Policy1 passed all288 release/drop episodes and both fixed kicks. Exact+15degree release reached sustained balance at28.355seconds. The motor-off drop reached sustained balance at10.8875seconds. The larger kick occurred at40seconds after verified balancing; sustained balance returned at76.24seconds (36.24seconds after impact). Every clip continues to120seconds.

Other policies have failures. Policy2's exact+15degree release held balance for9.785seconds at the end, narrowly missing the fixed10second success threshold; it is recorded as a failure rather than relaxing the criterion. Policy0 failed the fixed2rad/s kick despite passing the4rad/s kick: recovery is not guaranteed monotonic in disturbance size.

Run IDs:
- Policy0: `20261011T190938-87cfb05f`.
- Policy1: `20261011T190939-d3480af8`.
- Policy2: `20261011T190939-f88ac94d`.

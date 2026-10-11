# Acrobot swing-up and capture

[Live results and recordings](https://modal-labs-peyton-agents--robot-grid-learning-dashboard.modal.run/experiments/acrobot-swingup)

Three independently optimized trajectories now swing up from **exactly motionless,
fully hanging down**, under the original **±1 N m** torque bound. A deterministic
controller tracks each reference with time-varying LQR feedback, then switches to
upright LQR on entering a sampled capture region. Control updates remain 50 ms.
No neural policy was trained in this experiment.

## Verified run

Run `20261011T160551-2c0f97e7`:

| Path | Planned approach | Capture handoff | Final uninterrupted balance | Perturbed starts passed |
| --- | --- | --- | --- | --- |
| 1 | 30.00 s | 29.15 s | 12.85 s | 64/64 |
| 2 | 23.20 s | 21.40 s | 13.69 s | 57/64 |
| 3 | 26.05 s | 23.95 s | 13.83 s | 64/64 |

Each rollout continues 12 seconds beyond the planned approach. Independent replay
uses the original NumPy simulator at 5 ms and 2.5 ms physics steps. All three
nominal rollouts hold for more than ten seconds at both resolutions. The table's
perturbations are independent uniform ±0.05 on all four initial coordinates
(angles in radians, joint velocities in rad/s), seed 51000. All three paths pass
64/64 tests at ±0.001 and ±0.01. These are near-bottom starting perturbations,
not arbitrary mid-flight disturbances. Peak recorded torques are below 1 N m;
recorded velocities are comfortably below Gymnasium's velocity clipping limits.

The demonstrations are selected successes, explicitly labeled as such. Unsuccessful
searches and the original energy-controller failures are also retained.

## How the controller works

`optimal.py` uses CasADi/IPOPT multiple shooting. The initial state is fixed to
zero, the terminal state to upright at rest, and every interval must satisfy the
Acrobot dynamics with bounded torque and joint velocities. The RK4 model uses five
10 ms substeps per control interval. Initial guesses come from kicked energy
control trajectories; the optimizer can change the entire approach. It does not
teleport states or accept approximate dynamics as proof of success.

`trajectory.py` constructs time-varying feedback gains by a backward Riccati pass.
Actual control is `clip(reference_torque - gain @ state_error, -1, 1)`, with wrapped
angle errors. After capture it uses upright LQR indefinitely. The policy is
**deterministic and state-dependent, with an episode clock**; it is not simply
open-loop playback and is not a global recovery controller.

`TrajectoryController.load(path)` loads a downloadable `controller-N.npz` with
reference states, torques, gains, and capture parameters. Call `reset()` at the
start of an episode, then call the controller once every 50 ms with
`[shoulder_angle, relative_elbow_angle, shoulder_speed, relative_elbow_speed]`.
All rollout execution belongs on Modal, per the repository instructions.

## Capture window

`capture.py` maps two 41×41 slices of upright recovery: shoulder tilt versus
shoulder speed with elbow coordinates zero, and elbow bend versus elbow speed
with shoulder coordinates fixed upright. A state passes if upright LQR balances
throughout the final five seconds of a 12-second simulation.

A coupled ellipsoid `e.T @ P @ e <= 10` supplies the handoff condition. `P` comes
from the discrete LQR Riccati equation. Searching 21 sizes with 256 directions per
size selected the largest consecutive all-success size. Another 1024 independent
random interior samples all recovered. This is an **empirical region**, not a
formal stability certificate or a maximal recoverable set.

With other errors zero its coordinate intercepts are approximately ±0.396°
shoulder tilt, ±1.322° elbow bend, ±0.0144 rad/s shoulder speed, and ±0.0394 rad/s
relative elbow speed. These are **not independent bounds**: correlated position
and velocity errors can recover outside these one-axis intercepts. The plots show
why velocity direction matters. The optimizer currently aims for the center of
this region; online handoff occurs earlier when the state enters it.

## Reproduce on Modal

After sourcing `~/.codex/modal.env`:

```sh
uv run --extra acrobot --extra feasibility --locked modal run --env peyton-agents -m acrobot_swingup.modal_app::optimize
uv run --extra acrobot --extra feasibility --locked modal run --env peyton-agents -m acrobot_swingup.modal_app::capture --run-id RUN_ID
uv run --extra acrobot --extra feasibility --locked modal run --env peyton-agents -m acrobot_swingup.modal_app::publish_trajectories --run-id RUN_ID
```

Use the new run ID printed by `optimize`. Results, checkpoints, and MP4s persist
under `/data/swingup/RUN_ID` on the shared Modal volume. The root app only composes
experiment apps; all experiment logic remains in this package.

## Retained failures

- `20261011T155033-e22343c2`: 60 energy-controller gain settings, 8 search starts,
  then 64 held-out starts. Zero five-second balances; four unselected recordings.
- `20261011T160247-dab369d0`: nine short-segment least-squares attempts. None passed
  replay. Approximate collocation residuals were not accepted as success.

The successful full-trajectory optimizer resolves nominal simulated feasibility
under our settings. It does not establish physical hardware performance or recovery
from arbitrary large disturbances.

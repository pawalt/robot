# Acrobot controller feasibility

[Live experiment](https://modal-labs-peyton-agents--robot-grid-learning-dashboard.modal.run/experiments/acrobot-feasibility)
· [Experiment directory](https://modal-labs-peyton-agents--robot-grid-learning-dashboard.modal.run/)

## Result

**Continuous-valued torque is not required to balance this simulated Acrobot.**
At the RL experiment's 0.05s control interval and ±1 N m limit, both continuous
LQR and its three-level quantized version successfully balanced all 128 very
small perturbations. Both recovered 20/128 of the original near-upright starts.

Success means remaining balanced throughout the final five seconds of a
20-second rollout. All starts and controller settings were fixed before the
sweep; no training or reward optimization was involved.

| Conditions (±1 N m) | Continuous-valued | Three torque levels | Nine levels |
| --- | --- | --- | --- |
| 0.05s interval, 0.05× disturbances | 128/128 | 128/128 | 128/128 |
| 0.05s interval, 1× disturbances | 20/128 | 20/128 | 20/128 |
| 0.2s interval, 0.05× disturbances | 128/128 | 50/128 | 128/128 |
| 0.2s interval, 1× disturbances | 21/128 | 8/128 | 20/128 |

At 0.05s with the original disturbances, increasing the torque limit from ±1 to
±2 raised success to 36/128 for continuous LQR and 38/128 for three-level LQR.
Zero-torque control succeeded in none of the sampled conditions.

The evidence points toward a limited recovery region under bounded torque for
these controllers, rather than a fundamental need for continuous action values.
This is not a proof that other controllers cannot recover the failed starts.
It also does not establish swing-up from the downward position. A sensible next
RL test would first use the smaller disturbance distribution with a verified
successful controller baseline, then broaden the starts gradually.

## Method

`study.py` linearizes the same Gymnasium-derived nonlinear Acrobot equations used
by `acrobot_comparison/env.py` at `[pi, 0, 0, 0]`. The linearized system has rank-4
controllability. For each control interval, it uses exact zero-order-hold
discretization and solves the discrete algebraic Riccati equation with
`Q=diag(10,10,1,1), R=1`. All computed linear closed loops are stable.

The actual rollout always uses the nonlinear dynamics, not the linear model.
The requested torque is clipped, then optionally rounded to three or nine levels.
Continuous-valued torque is still held constant between controller updates;
discrete actions also apply torque for the entire control interval.

- Control intervals: 0.01, 0.02, 0.05, 0.1, 0.2 seconds.
- Torque limits: ±0.5, ±1, ±2 N m.
- Disturbance scales: 0.05×, 0.25×, 1×, 2×.
- At 1×: independent uniform angle errors ±0.08 rad and joint speeds ±0.1 rad/s.
- All conditions reuse 128 random initial vectors, seed 30000.
- Controllers: bounded continuous LQR, 3-level LQR, 9-level LQR, passive zero torque.
- 200 conditions, 25,600 main rollouts, each lasting 20 seconds.
- Physics RK4 step fixed at 0.005s; Gymnasium velocity limits retained.
- Balance checked every physics step: both links within 10° of vertical and
  both absolute link speeds below 0.5 rad/s.
- Additional validation halved the integration step to 0.0025s for continuous
  and 3-level controllers at 0.05s, ±1 torque, and 0.05×/1× disturbances.
  No success-rate changes were observed in these four checks.

All simulation, numerical sweeps, validation rollouts, and video rendering ran
on Modal. The eight videos show the preselected first start for every controller
at 0.05× and 1× disturbances, ±1 torque, 0.05s control interval. They are not
selected to show the most successful outcome.

## Run

```sh
set -a
source ~/.codex/modal.env
set +a
uv run --extra feasibility modal run --env peyton-agents -m acrobot_feasibility.modal_app::study
```

The subfolder owns the experiment logic, Modal jobs, and command. The root
`modal_app.py` only includes its app alongside the other experiments and shared UI.
Results and videos are stored under `/data/feasibility/<id>/` on the shared Modal
Volume, with a local downloaded JSON copy under `artifacts/<id>/results.json`.

Small correctness tests (not simulation sweeps) can run locally:

```sh
uv run --extra feasibility python -m unittest acrobot_feasibility.test_study
```

References: [SciPy DARE](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_discrete_are.html)
and [Acrobot controllability and LQR](https://underactuated.mit.edu/acrobot.html).

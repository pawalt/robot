# Parallel forgetting and architecture comparison

[Live comparison](https://modal-labs-peyton-agents--robot-grid-learning-dashboard.modal.run/experiments/acrobot-neural-sweep)

Run `20261011T165814-20ab07d2` compares ten arms with three training seeds each.
All simulation, training, evaluation and video recording run on Modal in
`peyton-agents`. Each worker requests two CPUs and 4 GB RAM; the training pool
is capped at ten concurrent workers (20 requested CPUs, 40 GB RAM), with no GPU.

Every arm begins with the same saved neural balancing policy. The wider actors
duplicate neurons and divide their outgoing weights to preserve the initial
function. Zero-sum output-weight noise breaks symmetry without changing that
function. The residual block initially produces zero. The expert model begins
with two copies of the original actor. Its balancing copy is frozen, while its
other expert and gate learn. All deployed actors use only the six observed state
features and output a continuous torque clipped to ±1 N m. No deployed actor
calls LQR, follows a clock or looks up a trajectory.

## Arms

| Arm | Protection / architecture |
| --- | --- |
| unprotected | Three 128-unit tanh layers; no balancing auxiliary loss |
| rehearsal | Same actor; 2× MSE against LQR labels on remembered balancing states |
| distill | Same actor; 2× MSE against frozen original neural-policy outputs |
| ewc | Same actor; empirical-Fisher weighted parameter penalty, coefficient 1000 |
| project | Same actor; project proposed Adam parameter step against remembered-loss gradient |
| rollback | Distillation plus checkpoint rollback below 90% development balancing success |
| experts | Frozen neural balancing expert, trainable neural expert and learned 64-unit gate; distillation |
| wide256 | Three 256-unit tanh layers; distillation |
| wide512 | Three 512-unit tanh layers; distillation |
| residual256 | Three 256-unit layers plus a two-linear-layer residual block; distillation |

The projected-step arm is GEM-inspired, **not a reproduction of GEM/A-GEM**:
it projects the actual Adam parameter delta; optimizer moments are not projected.
EWC estimates a per-observation Gaussian score Fisher from 256 remembered
observations, normalizes its mean to one, and tests only one penalty coefficient.
These are implementation/setting comparisons, not universal method rankings.

## Matched budget and data

Each job has 4,000 supervised updates of batch size 512 followed by ten PPO
updates. Each PPO update collects 128 full 900-step / 45-second episodes:
1,152,000 transitions per job, 34,560,000 across all 30 jobs. This is a matched
data/update budget, not matched wall time or parameter count.

The supervised task is the last two seconds of deterministic trajectory 1, with
feedback torque labels on 16,384 perturbed reference states. Remembered balancing
states include 8,192 capture-region samples and samples from 64 original-policy
rollouts. Those memory states are common across arms within a seed. Rehearsal
uses LQR targets; distillation uses the original actor's raw output targets.

PPO resets 25% of environments into the capture region and 75% near the
two-second approach start. Falls do not terminate episodes. Actor learning rates
are 1e-4 for imitation and 3e-5 for PPO; the separate 2×128 critic uses 1e-3.
PPO uses gamma .999, GAE lambda .995, four epochs, minibatches of 4096, ratio
clipping .2 and actor KL stop .02. Every arm also receives 0.5× suffix imitation
MSE during PPO, plus its specified protection loss. Thus these are
**PPO-with-imitation** comparisons, not pure PPO runs.

The control interval is .05 seconds. Training uses five RK4 substeps per control
interval; audits use ten. Reward is `dt*(height + 5*balanced - .01*(v1²+v2²))`.

## Evaluation and selection

Development tests use 16 starts per condition, seed `92000 + training_seed`.
Checkpoints are saved every 1,000 imitation updates and every two PPO updates.
The rollback arm rejects checkpoints below 90% development balance success,
restoring the last accepted actor and optimizer. Rejected checkpoint weights and
metrics remain available.

Independent audit tests use 64 starts per condition, seed `96000 + training_seed`.
Success means being balanced for the **final ten seconds** of a 45-second test.
Both links must lie within 10 degrees of upright, with absolute link angular
velocities below .5 rad/s. Audits separately test balancing, one- and two-second
approach starts, bottom starts, and velocity kicks of up to ±.1 / ±.5 rad/s at
five seconds. Bottom success is never inferred from approach success.

Both the final retained policy and a development-selected checkpoint are
audited. Selection prefers balance success ≥90%, then two-second approach
success, bottom success and balance success; ties choose the earliest checkpoint.
Rejected checkpoints cannot be selected. The selected checkpoint's approach is
retested with finer 2.5 ms physics. Selecting the original checkpoint can preserve
a useful policy but does not mean training avoided forgetting; the default UI
therefore compares **final retained policies**.

The three training seeds share the same original balancing checkpoint, and
approach starts are small perturbations of one reference state. They do not
establish broad state-space robustness or independent from-scratch learning.
Seed-zero videos show the first held-out episode for each condition using the
final retained policy, including failures. They are not selected for success.

## Reproduce

After sourcing `~/.codex/modal.env`:

```sh
uv run --extra acrobot --extra feasibility --locked modal run --env peyton-agents -m acrobot_neural.modal_app::sweep
uv run --extra acrobot --extra feasibility --locked modal run --env peyton-agents -m acrobot_neural.modal_app::sweep_record --run-id RUN_ID
```

Artifacts live in `/data/neural-sweeps/RUN_ID/ARM-SEED` on the shared volume,
with result JSON mirrored in `acrobot_neural/artifacts/RUN_ID`. Load checkpoints
using `acrobot_neural.sweep.load_actor(arm, path)` to reconstruct the correct
architecture. All subsequent experimental rollouts must also run on Modal.

## Completed pilot results

Each fraction pools 64 held-out starts from each of three training seeds. These
are **final retained policy** results, not the best checkpoint found during training.

| Approach | Balance | Two-second approach | ±0.1 kick | ±0.5 kick | Mean job seconds |
| --- | --- | --- | --- | --- | --- |
| No protection | 0/192 | 0/192 | 0/192 | 0/192 | 193 |
| LQR rehearsal | 192/192 | 192/192 | 56/192 | 7/192 | 209 |
| Frozen neural distillation | 192/192 | 190/192 | 38/192 | 5/192 | 189 |
| EWC (tested setting) | 0/192 | 0/192 | 0/192 | 0/192 | 226 |
| Projected Adam step | 128/192 | 128/192 | 38/192 | 7/192 | 210 |
| Distillation + rollback | 188/192 | 0/192 | 35/192 | 5/192 | 210 |
| Neural experts + gate | 192/192 | 192/192 | 36/192 | 5/192 | 276 |
| 3×256 + distillation | 185/192 | 181/192 | 36/192 | 5/192 | 273 |
| 3×512 + distillation | 136/192 | 118/192 | 24/192 | 2/192 | 494 |
| Residual 256 + distillation | 192/192 | 192/192 | 40/192 | 5/192 | 348 |

All ten arms scored **0/192 on bottom-start swing-up**. This pilot solves the
narrow final-two-second approach for several methods, not full swing-up or
general disturbance recovery. The mean job time includes training and evaluations,
but excludes queueing, initial data preparation, and separate video rendering.

Rehearsal, neural experts and residual-256 retained balance and learned the approach
in every final audited episode. The 34,050-parameter rehearsal actor is the
simplest of these; the residual actor has 265,218 parameters. Increasing plain MLP
width to 256 or 512 did not outperform the small protected network under this budget.

Distillation was also strong (190/192 approach successes), but several arms
temporarily lost balance during imitation before recovering. EWC at the tested
coefficient failed in all three seeds. Projected updates learned the task but one
seed collapsed late in PPO. Rollback preserved most balancing success while
rejecting every supervised block and preventing approach learning. These observations
motivate separating an exploratory training candidate from a trusted deployable
checkpoint, rather than requiring every intermediate update to remain successful.

The selected checkpoints generally score better than final policies, emphasizing
the need to preserve and evaluate earlier checkpoints. Fine-physics confirmation
applies to the selected checkpoints, not every final checkpoint. The dashboard
lets readers switch between both audits; `summary.json` contains both sets of totals.

Total PPO budget: 34,560,000 transitions. Thirty seed-zero scenario videos show
the first held-out episode from each final actor, including failures. All videos
were checked against the corresponding audit results and verified to support HTTP
range requests. The architecture-preserving initialization and parameter restoration
unit checks passed, as did the dashboard route/media test and JavaScript syntax check.

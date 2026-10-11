# Neural-only Acrobot controller

**Verified ground-to-balance:** the [radial neural imitation experiment](RADIAL.md) now provides three standalone neural policies that swing up from rest and balance through two-minute independent tests. The older MLP experiments below remain preserved.

The [full swing-up imitation experiments](IMITATION.md) compare full demonstrations,
backward curricula and recovery-validated corrections while rehearsing balancing.

The latest [parallel forgetting and architecture comparison](SWEEP.md) tests ten
approaches over three training seeds, with separate final-policy and selected-checkpoint audits.

[Live experiment](https://modal-labs-peyton-agents--robot-grid-learning-dashboard.modal.run/experiments/acrobot-neural)

The deployed actor is a six-input MLP with three 128-unit tanh hidden layers and
one continuous output clipped to ±1 N m. Inputs are sine/cosine pairs for the two
joint angles and normalized joint velocities. There is no clock, trajectory
lookup, LQR call, capture-region test, or teacher in `policy.py`.

Teachers are used only to generate supervised training examples. The separate
critic is used only by PPO during training.

## Measured outcome

Two pilot runs are retained: `20261011T162331-f3d100d1` and
`20261011T162853-aa740ed3`. The first learned balancing, then lost it during
swing-up imitation; 60 PPO updates did not restore it. Stronger balance rehearsal
in the second run preserved that skill through imitation and PPO.

On 64 held-out starts, the second run's development-selected `ppo-20` checkpoint
achieved:

| Condition | Final ten seconds balanced |
| --- | --- |
| Inside the measured capture region | 64/64 |
| Near hanging down, including one exactly motionless start | 0/64 |
| Joint-velocity kick up to ±0.1 rad/s | 15/64 |
| Joint-velocity kick up to ±0.5 rad/s | 2/64 |
| Joint-velocity kick up to ±1.0 rad/s | 0/64 |

The policy was balanced before every held-out kick. One development swing-up
success at `ppo-20` did not generalize to the held-out set. This experiment
establishes learned neural balancing, **not reliable neural swing-up or broad
recovery**. No LQR or trajectory controller acts during these evaluations.
The next training investigation should isolate swing-up learning with a backward
curriculum along the demonstrations while preserving the learned balance skill.

## Procedure

1. Regress on bounded LQR torque labels around the previously tested capture
   region. Keep a separate balancing checkpoint and test it in closed loop.
2. Generate successful trajectory-controller rollouts from perturbed bottom
   starts, plus feedback labels near the reference. Start with trajectory 1 to
   avoid conflicting time-conditioned teachers at nearly identical states.
3. Mix swing-up and balancing labels for behavioral cloning.
4. Fine-tune with PPO, random resets near trajectory states, capture-region
   states, and hanging-down states. Add joint-velocity kicks during rollouts.
   Supervised rehearsal remains in the loss to discourage forgetting.
5. Select a checkpoint using development starts, then independently audit it
   on held-out balance, bottom, and three kick scenarios. Save failed stages too.

Training and recording run exclusively on Modal in `peyton-agents`. The initial
pilot uses one network seed, not a multi-seed algorithm comparison. A later run
can fail even if an earlier checkpoint succeeds; use the per-stage metrics.

## PPO details

Each of 60 updates collects 64 continuing-task chunks of 256 control steps:
983,040 transitions total. Each control interval is 50 ms, integrated with five
10 ms RK4 substeps. GAE uses gamma 0.999, lambda 0.98 and a value bootstrap at
chunk boundaries. Falling is not terminal. The critic sees the same six features.

Actions are sampled in raw torque space from a Normal distribution and clipped
before application. PPO stores the *unclipped* sampled action and its likelihood;
clipping is the fixed environment action transform. Deterministic evaluation
clips the mean. Four update epochs use minibatches of 1024, ratio clipping 0.2,
actor learning rate 3e-5, critic learning rate 1e-3 and policy KL stop 0.02.

Reward per control interval is `dt * (height + 5*balanced - .01*(v1²+v2²))`.
Training kicks affect the capture-region group after five seconds, with joint
velocity magnitude growing from ±0.06 toward ±0.5 rad/s across updates.

The actor loss also includes 0.5 times supervised rehearsal MSE. This is
PPO with an imitation auxiliary loss, not a pure PPO baseline. The second pilot
changes supervised sampling to 95% balancing examples to test whether stronger
rehearsal prevents the forgetting observed in the first pilot.

## Evaluation

Development starts use seed 71000, 16 episodes per condition and 40-second
rollouts. Checkpoint selection maximizes summed final-ten-second success across
balance, bottom, and ±0.1 rad/s kick scenarios; ties select the earliest stage.

The independent audit uses seed 74000, 64 starts per condition, 45-second rollouts,
and 5 ms physics steps. Balance starts are sampled inside the empirical capture
ellipsoid; bottom starts have independent uniform ±0.01 initial coordinate
perturbations (episode zero is exactly motionless at the bottom). Kick tests start
exactly upright and inject independent uniform joint-velocity changes after five
seconds, using magnitudes 0.1, 0.5 and 1.0 rad/s. Pre-kick balance is reported, so a
policy that already fell cannot masquerade as a successful disturbance test.
The selected checkpoint's balance/bottom evaluation is repeated at 2.5 ms.

Videos show the first held-out episode for each scenario, without selection for
success. The page distinguishes development from held-out results.

## Run

After sourcing `~/.codex/modal.env`:

```sh
uv run --extra acrobot --extra feasibility --locked modal run --env peyton-agents -m acrobot_neural.modal_app::train
uv run --extra acrobot --extra feasibility --locked modal run --env peyton-agents -m acrobot_neural.modal_app::evaluate --run-id RUN_ID
uv run --extra acrobot --extra feasibility --locked modal run --env peyton-agents -m acrobot_neural.modal_app::consolidate --parent RUN_ID
```

Checkpoints and results persist at `/data/neural/RUN_ID`. Load downloaded weights
with `Actor.load_state_dict(torch.load(path, weights_only=True))`, then call
`actor.act(state)` once per control interval. Run rollout experiments on Modal.

## Longer episodes and backward curriculum

The `curriculum` entrypoint starts from the saved neural balancing actor. Each
rollout now lasts **900 steps / 45 seconds**, compared with the earlier
256-step / 12.8-second chunks. Time-limit truncations bootstrap the critic; falls
do not terminate rollouts. The actor still observes no clock or teacher state.

The starting point moves backward along the 30-second demonstration through
1, 2, 4, 6, 8, 12, 16, 20, 25 and 30 seconds remaining. It advances only after two
consecutive development checks achieve at least 80% final-ten-second success at
the current starting point and retain at least 90% balancing success. Full
bottom-start evaluation remains separate. Each stage uses small state
perturbations and includes 25% capture-region starts to retain balancing.

The current suffix supplies training-only feedback labels. PPO includes 2×
balance rehearsal MSE and 0.5× suffix imitation MSE, gamma 0.999, lambda 0.995,
four update epochs and 4096-sample minibatches. Thus this experiment changes the
curriculum, rollout length, batching and credit horizon together; it is not a
controlled ablation of episode length alone.

```sh
uv run --extra acrobot --extra feasibility --locked modal run --env peyton-agents -m acrobot_neural.modal_app::benchmark
uv run --extra acrobot --extra feasibility --locked modal run --env peyton-agents -m acrobot_neural.modal_app::curriculum --updates 40 --count 256 --threads 4
```

This requests 4 CPUs and 8 GB RAM with no GPU. Simulation/inference uses one
PyTorch thread; gradient updates use four. Forty updates collect 9,216,000
transitions. The dashboard displays collection and optimization wall times.
Independent curriculum audits use fresh seed 84000, while development uses 82000.

A short benchmark on an 8-CPU Modal worker measured approximately 47,922 rollout
transitions/s for 64 environments and 109,229/s for 256, both with one thread.
Four-thread updates achieved about 165,499 supervised samples/s versus 125,197/s
in the corresponding one-thread measurement. These are short component timings,
not a promised end-to-end speedup. No GPU benchmark has been run. The earlier
PPO phases collected 983,040 transitions in approximately 100 seconds, including
periodic development checks, on jobs requesting 2 CPUs and 4 GB RAM.

The first longer-episode run (`20261011T164242-9c48fced`) completed all
9,216,000 transitions in 313.9 seconds. Measured collection time was 82.1 seconds
and optimization time was 118.8 seconds; the remainder includes evaluation,
advantage computation and checkpoint work. This is approximately 29,360
transitions/s including those overheads, but is not a controlled speed comparison
with the earlier runs because the rollout and minibatch settings also changed.

It passed the final one-second approach using the existing balancing actor, but
never passed the two-second approach. Later checkpoints lost balancing as well.
Longer episodes and more compute did not fix that learning failure. The initial
checkpoint is preserved and the audit compares it with the final checkpoint,
including videos from both one- and two-second starting points.

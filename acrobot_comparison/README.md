# Acrobot: PPO vs grouped, critic-free learning

[Experiment page](https://modal-labs-peyton-agents--robot-grid-learning-dashboard.modal.run/experiments/acrobot)
· [All experiments](https://modal-labs-peyton-agents--robot-grid-learning-dashboard.modal.run/)

This experiment compares a learned value baseline against a group-relative
outcome baseline. It is a small matched-budget study, not an algorithm leaderboard.
All training, evaluation sweeps, and MP4 recording run on Modal in `peyton-agents`.

## Observed results

Two experiments are retained: 2.048M steps per job (`20261011T151557-1a40af2d`)
and 10.24M steps per job (`20261011T151911-eb5e88ac`). Each has three seeds per
method. **Neither method learned a five-second hold at either budget.**

The longer experiment's independent evaluation (128 fresh starts per condition,
per seed) gave the following near-upright results using most-likely actions:

| Method | Mean longest hold | Episodes holding ≥5 seconds | Mean training time per seed |
| --- | --- | --- | --- |
| PPO + critic | 0.708 seconds | 0% | 303.0 seconds |
| Grouped, critic-free | 0.745 seconds | 0% | 217.7 seconds |
| Passive, zero torque | 0.770 seconds | 0% | No training |

Sampled-action evaluation also failed: mean holds were 0.699s for PPO and 0.668s
for grouped. Neither method achieved balanced samples from hanging-down starts.
Grouped used less measured training time, but **there is no learning-speed winner**:
both failed the behavioral criterion and neither clearly improved on passive motion.
The small hold differences are not evidence of algorithm superiority.

Observed failures include maintaining poor near-upright behavior, failing to
discover swing-up, and fluctuations in deterministic behavior across grouped seeds.
PPO's high explained variance on its own value targets did not imply successful
control. Possible causes include weak exploration at the fast control interval,
reward local optima, coarse trajectory-level credit in the grouped method, and
short effective GAE credit horizons. These causes have not been isolated by
ablations. A useful next experiment would first establish learnable near-upright
balancing, then test swing-up separately with the same controlled comparison.

The page includes all seeds and 72 MP4s across the two experiments, plus raw JSON
downloads. `artifacts/<experiment-id>/audit.json` retains the measured results.

## Environment and reward

`env.py` implements a Gymnasium environment and a NumPy batched version of the
same dynamics for faster collection. Tests compare 30 steps over 12 states
against Gymnasium's original Acrobot RK4 dynamics, to 1e-10 absolute tolerance.
The algebra derives from Farama's BSD-licensed `AcrobotEnv` book dynamics.

- Original three elbow torques: −1, 0, +1. No motor at the shoulder.
- Control interval reduced from 0.2 to 0.05 seconds.
- Each task lasts exactly 400 steps / 20 seconds; reaching the top does not end it.
- The task's horizon is a true terminal boundary, not an external truncation.
- Seven inputs: angle sine/cosine pairs, normalized joint speeds, and time left.
  Time left makes the finite-horizon value function well-defined. This adds one
  input to the six-input architecture originally discussed.
- Training starts are drawn from a fixed mixture: 75% near upright (angle
  perturbations ±0.08 rad, joint speeds ±0.1 rad/s), 25% near hanging down
  (angles and joint speeds ±0.1). No adaptive curriculum or demonstration data.

With normalized tip height `H` in [0,1], reward is:

```text
dt * (H + 2 * balanced - 0.05 * H**4 * (v1**2 + (v1+v2)**2))
```

`balanced` requires both links within 10° of upward vertical and both absolute
link angular speeds below 0.5 rad/s. This is a proposed shaping reward: it may
favor undesirable local behaviors and does not guarantee successful learning.

## Matched protocol

Both methods use the same policy: 7 → 64 Tanh → 64 Tanh → 3 categorical logits.
PPO has a separate 7 → 64 Tanh → 64 Tanh → 1 critic. For a training seed, actor
initialization and the sampled start-state sequence match. Action sampling streams
need not match. Floating-point details can also vary between Modal CPU hosts.

Each update samples eight initial states and generates eight trajectories per
state: 64 × 400 = 25,600 simulator steps. Both methods use these same groups.
This deliberately controls collection, although ordinary PPO need not duplicate
start states and could spend that budget on more distinct starts.

| Setting | Both methods |
| --- | --- |
| Seeds | 0, 1, 2 |
| Policy learning rate | 0.0003, Adam epsilon 1e-5 |
| Update epochs / minibatch | 4 / 1024 |
| Policy clipping / entropy bonus | 0.2 / 0.01 |
| Gradient norm limit | 0.5 |
| Discount | 1.0, undiscounted finite task |
| Policy KL early stop | 0.03 |

PPO uses GAE (lambda 0.95) and trains a critic by squared-error regression to
lambda-return targets, with critic learning rate 0.001. The grouped method
subtracts each group's mean episode return and divides by its population standard
deviation. Every action in a trajectory receives that same normalized score.
Identical-return groups have zero advantage. Both methods then normalize the
batch advantages and use the same clipped policy objective.

This is **GRPO-inspired**, not a reproduction of an LLM training recipe: there is
no frozen reference-policy KL term. It compares whole-trajectory group advantages
with stepwise GAE, so any difference is not attributable to critic removal alone.

## Evaluation

At initialization and every 10 updates, evaluate 32 fixed states per condition
with seed 10000, using both most-likely and sampled actions. Separate near-upright
and hanging-down results. After training, `audit` evaluates 128 fresh starts per
condition with seed 20000 and compares a zero-torque passive baseline.

Report mean return, normalized height, time balanced, longest uninterrupted hold,
and fraction of episodes holding for at least five seconds. Evaluation steps do
not count toward training steps. Timed training includes collection and updates,
but excludes evaluation, persistence, container startup/build, and video encoding.
Wall times are from equally allocated 2-CPU jobs, not controlled identical physical
hosts, so they should be interpreted cautiously.

Videos show the first fixed evaluation episode at initialization, halfway through,
and at the final checkpoint, for every seed and both starting conditions. They use
most-likely actions and are not cherry-picked successful episodes. Check sampled
evaluation separately: a stochastic policy's argmax behavior can be quite different.

## Run on Modal

```sh
set -a
source ~/.codex/modal.env
set +a

# Six jobs: 3 seeds × 2 methods, 2.048M steps per job.
uv run --extra acrobot modal run --env peyton-agents -m acrobot_comparison.modal_app::compare --updates 80

# Longer comparison, unchanged settings: 10.24M steps per job.
uv run --extra acrobot modal run --env peyton-agents -m acrobot_comparison.modal_app::compare --updates 400

# Independent final evaluation; replace RUN_ID with the printed experiment ID.
uv run --extra acrobot modal run --env peyton-agents -m acrobot_comparison.modal_app::audit --experiment RUN_ID

# Publish the shared directory and all experiment subpages.
uv run modal deploy --env peyton-agents -m modal_app
```

Each experiment lives under `/data/acrobot/<experiment-id>/<method>-<seed>/` on
the `robot-grid-recordings` Modal Volume. Artifacts include metrics, actor
checkpoints, MP4s, and posters. Local `artifacts/` contains downloaded JSON results.
Prior experiments remain selectable on the dashboard.

Small correctness tests may run locally:

```sh
uv run --extra acrobot python -m unittest acrobot_comparison.test_comparison
```

References: [Gymnasium Acrobot](https://gymnasium.farama.org/environments/classic_control/acrobot/),
[GRPO / DeepSeekMath](https://arxiv.org/abs/2402.03300),
[PPO implementation details](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/).

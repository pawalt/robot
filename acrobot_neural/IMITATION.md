# Full swing-up imitation with balancing rehearsal

[Live experiment](https://modal-labs-peyton-agents--robot-grid-learning-dashboard.modal.run/experiments/acrobot-imitation)

All training, recovery validation, evaluation and recording run on Modal in
`peyton-agents`. The shared root app only composes the experiment apps. This
experiment uses the existing 6→128→128→128→1 tanh actor, continuous torque clipped
to ±1 N m, and 45-second test episodes. The deployed network observes only the
current joint angles and velocities. It has no clock, LQR, reference lookup or
external controller. Teachers are training-only.

## First two batches

- `20261011T172359-d6f517e1`: ordinary raw-output mean squared error.
- `20261011T172722-8b791d5f`: actuator-aware loss. At a saturated teacher torque,
  any raw output producing the same clipped action has zero error. Incorrectly
  signed saturation still receives a gradient toward the target. Unsaturated
  targets use ordinary squared error.

Each batch runs nine jobs concurrently: three methods × three seeds. Every job
requests two CPUs and 4 GB RAM, with no GPU. Both batches can overlap, requesting
36 CPUs and 72 GB RAM for eighteen active training workers. This is not a
measurement of CPU utilization or billed cost. Recording uses additional workers.

All jobs start from the same successful rehearsal actor, `final-retained.pt` in
sweep `20261011T165814-20ab07d2/rehearsal-0`. Seeds change training samples and batch
sampling, not the warm-start weights. Each job trains for 24 rounds of 1,000 Adam
updates at learning rate 1e-4, using 512 task examples plus 512 balancing examples
per update. The balance loss has weight two; gradients are clipped to norm one.
**No PPO is used in these batches.**

## Methods

1. **Full:** imitate the whole 30-second demonstration immediately.
2. **Backward:** imitate progressively longer suffixes: 2, 4, 6, 8, 12, 16, 20,
   25, and 30 seconds before the end. Advance only after two consecutive checks
   have ≥90% balance retention and ≥80% current-approach success.
3. **Corrective:** use the same curriculum, and every second round collect states
   reached by the current student. The deterministic teacher attempts recovery
   from 64 sampled states, starting at their corresponding reference phases. It
   must balance for the final ten seconds of a 45-second rollout before any of
   that rollout's examples enter the correction buffer. Reject failed recoveries.
   Half the task minibatch then comes from that buffer, capped at 65,536 examples.

Each suffix has 32,768 perturbed reference-state examples, with standard deviations
(.003, .003, .01, .01) for the two angles and velocities. The feedback teacher uses
its previously measured capture-region handoff. Balance rehearsal combines 8,192
capture samples with states visited by the original learned actor, labeled by LQR.
The ordinary loss keeps raw balancing targets clipped at ±1.2, matching the prior
sweep. The actuator-aware loss uses the equivalent applied targets clipped at ±1.

Corrective labels are recovery-validated, but validation does not establish
optimality, a global recovery guarantee, or that a state-only network can fit all
phase-conditioned labels. The correction rollouts include post-capture balancing
samples; this can dilute approach examples. The two methods therefore differ in
data distribution and additional simulation cost, not just their loss.

## Checkpoints and evaluation

A candidate continues training through regressions; there is no rollback. Every
round saves weights and development metrics. Development uses 16 starts per
condition, seed `122000 + seed`; independent audits use 64 starts, seed
`126000 + seed`. All successes require the final ten seconds balanced, with both
links within ten degrees upright and absolute link velocities below .5 rad/s.
Physics uses ten RK4 substeps per 50 ms control interval. Selected-checkpoint
bottom audits also repeat with twenty substeps.

Selection first favors bottom success while retaining ≥90% balancing, then the
farthest tested approach passing ≥80% with retained balance, then balance and
approach success. It uses development data only. Both selected and final policies
are audited across every curriculum start, balancing, bottom, and ±.1/±.5 rad/s
velocity kicks. A selected checkpoint's recorded `back_seconds` is the start it
was tested from, **not a claim it passed**—especially for the full arm when the
initial checkpoint is selected. Read the actual audit fractions.

Bottom audit episode zero starts exactly motionless; other starts use independent
±.01 perturbations. Approach starts use ±.003 perturbations, with episode zero on
the reference. Videos show episode zero without selecting for success. Selected
balance, selected approach, selected bottom and final bottom are all recorded.

## Reproduce

After sourcing `~/.codex/modal.env`:

```sh
uv run --extra acrobot --extra feasibility --locked modal run --env peyton-agents -m acrobot_neural.modal_app::imitation
uv run --extra acrobot --extra feasibility --locked modal run --env peyton-agents -m acrobot_neural.modal_app::imitation --loss-kind actuator
uv run --extra acrobot --extra feasibility --locked modal run --env peyton-agents -m acrobot_neural.modal_app::imitation_record --run-id RUN_ID
```

Artifacts persist under `/data/neural-imitation/RUN_ID/ARM-SEED` on the shared
volume. Returned JSON is also mirrored in `acrobot_neural/artifacts/RUN_ID`.
Load a checkpoint into `acrobot_neural.policy.Actor` and run rollouts on Modal.

## Longer continuations

The original corrective seed-zero selected `round-23` passed 53/64 held-out
six-second approaches and 64/64 balancing starts. Two further batches start
from that same checkpoint, each with three sampling seeds and 48,000 additional
updates:

- `20261011T173237-304bb72f`: retains entire successful correction rollouts.
- `20261011T173237-8df2d72c`: retains only pre-capture correction samples, plus
  the first sample of each recovered rollout. Teacher validation still runs the
  full 45 seconds; only the training buffer changes.

These are follow-up experiments from a selected parent, not independent reruns
of the original comparisons. Their curriculum begins six seconds before the
reference end. They keep ordinary MSE and the same architecture and learning rate.
The training candidate still continues through regressions and every checkpoint
is preserved. Selection and held-out evaluations follow the same rules above.
The continuation audits reuse the pilot's held-out starts for comparability, so
these states are now a reused benchmark, not an untouched final test set.

```sh
uv run --extra acrobot --extra feasibility --locked modal run --env peyton-agents -m acrobot_neural.modal_app::imitation_extend
uv run --extra acrobot --extra feasibility --locked modal run --env peyton-agents -m acrobot_neural.modal_app::imitation_record --run-id RUN_ID --arms corrective
```

## Diagnostic evidence

A separate Modal diagnostic on the ordinary-MSE batch replays the deterministic
teacher from 32 perturbed starts at every curriculum point. It passed **32/32 at
all nine starts**, including the bottom. This verifies teacher feasibility for
these starts, not global recoverability from arbitrary student deviations.

Seed-zero full imitation's final network had mean absolute applied-torque error
0.064 N m along the nominal full reference, yet failed bottom-start and balancing
evaluations. The ordinary backward final policy had lower six-second reference
error (0.054 N m) than the corrective checkpoint (0.065 N m), despite the latter
recovering in 53/64 held-out six-second tests and the former in 0/64. These are
examples, not a population-level comparison, but show why average reference-action
error is insufficient and support collecting feedback-recovery data off reference.
Local linear diagnostics are also retained in `diagnostics.json`; nonzero nominal
torque means upright is not necessarily an equilibrium, so those eigenvalues must
not be interpreted as a nonlinear stability certificate.

Selected continuation policies receive an additional fresh confirmation with
128 starts per condition and seed `156000 + seed`, including a finer-physics
repeat of their selected approach. Those confirmations are separate from the
reused 64-start benchmark and are not used for checkpoint selection.

## Initial batch results

Selected checkpoints, 64 audit starts per training seed (192 total). These
checkpoints are chosen on development states. The full-trajectory arms selected
their original warm-start policy; their final trained policies lost balancing.

| Loss | Method | Balance | Four-second approach | Six-second approach | Bottom |
| --- | --- | --- | --- | --- | --- |
| mse | full | 192/192 | 0/192 | 0/192 | 0/192 |
| mse | backward | 192/192 | 188/192 | 0/192 | 0/192 |
| mse | corrective | 192/192 | 173/192 | 53/192 | 0/192 |
| actuator | full | 192/192 | 0/192 | 0/192 | 0/192 |
| actuator | backward | 192/192 | 183/192 | 1/192 | 0/192 |
| actuator | corrective | 192/192 | 187/192 | 45/192 | 0/192 |

The actuator-aware loss did not resolve full imitation or clearly outperform
ordinary MSE in this pilot. Corrective imitation extended the learned approach
beyond ordinary curriculum training. Full bottom-start swing-up remained unsolved.

## PPO follow-up

Three PPO jobs start from trimmed continuation seed zero's `round-18`, which
passed two consecutive development checks at the eight-second start. This
checkpoint is selected by development performance, before the continuation's
final audit. They run 40 updates of 256 × 900 transitions (9,216,000 per job),
retaining balance and suffix rehearsal, and beginning the curriculum at eight
seconds. All three use the same parent weights but different training seeds.
They request four CPUs and 8 GB RAM each and use the previously benchmarked
single-thread rollout/four-thread update configuration. Audit seeds are fresh
`176000 + seed`; selection continues to use development evaluations only.

These follow-ups appear on the existing [neural/PPO page](https://modal-labs-peyton-agents--robot-grid-learning-dashboard.modal.run/experiments/acrobot-neural),
linked from the imitation page. Their artifacts are stored under `/data/neural`.
PPO uses the existing curriculum implementation and balancing-data recipe, so
this is a follow-up learning test, not an isolated ablation of RL versus imitation.

```sh
uv run --extra acrobot --extra feasibility --locked modal run --env peyton-agents -m acrobot_neural.modal_app::imitation_ppo
```

## Continuation confirmation results

Fresh confirmation uses **128 new starts per condition**, after development-based
checkpoint selection. These are selected policies, not the final training candidates.

| Correction buffer | Seed | Balance | Four-second | Six-second | Eight-second | Twelve-second | Bottom |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Full | 0 | 128/128 | 128/128 | 128/128 | 61/128 | 0/128 | 0/128 |
| Full | 1 | 128/128 | 128/128 | 19/128 | 95/128 | 1/128 | 0/128 |
| Full | 2 | 128/128 | 128/128 | 127/128 | 94/128 | 0/128 | 0/128 |
| Trimmed | 0 | 128/128 | 74/128 | 0/128 | 74/128 | 0/128 | 0/128 |
| Trimmed | 1 | 128/128 | 128/128 | 0/128 | 117/128 | 0/128 | 0/128 |
| Trimmed | 2 | 128/128 | 128/128 | 0/128 | 118/128 | 0/128 | 0/128 |

The strongest trimmed policies passed 117/128 and 118/128 eight-second starts,
while failing all 128 six-second starts. This is **specialization to particular
approach states**, not mastery of every later point or a global swing-up controller.
Trimming increased eight-second success for two seeds, but did not provide broad
retention of earlier approach skills. Every selected continuation retained balance;
none passed a bottom-start episode. A single twelve-second success in one full-buffer
policy is not evidence of reliable twelve-second recovery.

The next retention target must include previously successful approach behavior,
in addition to balancing. Checkpoint selection should assess that collection of
skills rather than rank only the farthest current start. Longer training can otherwise
produce a controller specialized to the newest start while losing earlier maneuvers.

## PPO outcomes

Each PPO seed collected 9,216,000 transitions, or 27,648,000 across the three.
The table uses fresh audit seeds and 64 starts per condition. Earlier-start
audits were added after the main audit to check retention explicitly.

| Seed | Checkpoint | Balance | Four-second | Six-second | Eight-second | Bottom |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | ppo-0 | 64/64 | 64/64 | 61/64 | 47/64 | 0/64 |
| 0 | ppo-4 (selected) | 64/64 | 64/64 | 64/64 | 59/64 | 0/64 |
| 0 | ppo-40 | 64/64 | 0/64 | 0/64 | 0/64 | 0/64 |
| 1 | ppo-0 (selected) | 64/64 | 64/64 | 62/64 | 40/64 | 0/64 |
| 1 | ppo-40 | 64/64 | 61/64 | 8/64 | 2/64 | 0/64 |
| 2 | ppo-0 | 64/64 | 64/64 | 62/64 | 52/64 | 0/64 |
| 2 | ppo-2 (selected) | 64/64 | 64/64 | 64/64 | 56/64 | 0/64 |
| 2 | ppo-40 | 64/64 | 64/64 | 0/64 | 12/64 | 0/64 |

Early PPO improved the eight-second audit from 47/64 to 59/64 in seed zero,
and from 52/64 to 56/64 in seed two. Seed one retained its original checkpoint.
Final eight-second success was 0/64, 2/64 and 12/64 across the three seeds, even
though balancing remained 64/64. No selected or final PPO policy succeeded from
the bottom. One run advanced its training curriculum to twelve seconds but never
passed that approach. PPO therefore did not resolve full swing-up in this budget,
and reinforced the need to retain earlier approach behaviors—not only balancing.

All 27 training jobs completed on Modal. The 24 imitation jobs consumed 720,000
supervised updates in total. There are 96 imitation videos plus 22 PPO/retention
videos. Teacher-action equivalence and actuator-loss gradient tests passed; the
dashboard route/media test and frontend syntax checks passed. All videos are
stored on the shared Modal volume and exposed through seekable HTTP responses.

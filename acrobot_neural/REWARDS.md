# Matched PPO reward comparison

All calibration, training, evaluation and recording run on Modal in peyton-agents.
The shared UI is `/experiments/acrobot-rewards`.

Nine jobs: baseline, height-gated braking, and capture-progress shaping, each with
seeds 0, 1, 2. All start from `20261011T174348-20a2d60f/ppo-4.pt` and use
40 PPO updates, 256 environments, 900 steps (45 seconds), four CPUs and 8 GB.
Total training transitions: 82,944,000. Actor, critic, optimizer, rehearsal,
starting-state mixture, seeds and evaluation protocol match across reward arms.
This mixture differs from earlier adaptive-curriculum experiments; comparisons
between these nine jobs isolate reward, comparisons with older jobs do not.

Each rollout batch has 25% capture-region starts, 25% six-second demonstration
suffix starts, 25% eight-second starts, and 25% twelve-second starts. The starting
mixture stays fixed. Twelve seconds denotes a reference state twelve seconds
before the end of a demonstration, not twelve seconds from hanging motionless.
The same existing balance rehearsal and noisy reference-feedback imitation loss
are retained in every arm, with coefficients 2 and 0.5. PPO settings remain
actor LR 3e-5, critic LR 1e-3, gamma .999, lambda .995, four epochs, minibatch4096,
clip .2 and KL stopping .02. No critic warmup is added.

Let h be normalized tip height, b the tight balanced-state indicator, and
v² = shoulder angular velocity squared + relative elbow velocity squared.
The timestep is .05 seconds.

- Baseline: .05 * (h + 5b - .01v²).
- Braking: .05 * (h + 5b - .01h^4 v²).
- Capture: braking + 5 * (.999 Phi(next) - Phi(current)).

Phi(s) = 1 / (1 + e'Pe / level), with wrapped angle errors around upright.
P is the existing local LQR matrix, but the scale is calibrated using the
initial **neural** policy. Each level 1, 2, 5, 10, 20, 40 has 256 sampled
starts evaluated for 45 seconds with .005 physics steps, seed211000.
Final ten seconds must be balanced. Results: 256,256,256,256,240,150 successes.
Choose level10, the largest level with >=95% success at every tested level up
to it. This is empirical sampling, not certification or a maximal region.
Outside this local region, P is only a heuristic and may poorly represent
useful swing-up progress. No deterministic controller runs at inference.
Potential-based shaping is bootstrapped at time limits as a continuing task.
No claim that this particular potential or coefficient is optimal is intended.

Every two updates, evaluate development balance, bottom, and 6/8/12-second
starts. Choose checkpoints using only development results: prefer those keeping
balance >=90% and both earlier starts >=80%, then prioritize bottom success,
12-second success, summed success; earliest ties. No rollback is performed.
Initial, selected and final checkpoints receive independent 64-start audits
(seed221000+trainingseed). Selected balance, bottom and12s are repeated with
.0025 physics steps. Selected-checkpoint videos use episode0, including failures.
Kick tests are also reported, though not used for checkpoint selection.

Artifacts live in `/data/neural/RUN/`, with results mirrored locally. Previous
experiments are preserved. Run with the `reward_sweep` entrypoint in
`acrobot_neural.modal_app` after sourcing the Modal credentials.

## Results

All nine runs completed 40 updates. Training plus development checks took
367–670 seconds per worker (calibration, final audits and recordings additional).
Each row below aggregates three paired seeds, 64 held-out starts each.

| Final policy | Balance | 6-second approach | 8-second approach | 12-second approach | Bottom |
|---|---:|---:|---:|---:|---:|
| Starting policy, same for every arm | 192/192 | 191/192 | 164/192 | 0/192 | 0/192 |
| Current reward | 192/192 | 21/192 | 0/192 | 1/192 | 0/192 |
| Height-gated braking | 192/192 | 25/192 | 0/192 | 2/192 | 0/192 |
| Capture-progress shaping | 192/192 | 113/192 | 0/192 | 1/192 | 0/192 |

Capture shaping retained/recovered more six-second success in aggregate, but
all three rewards lost the eight-second approach and none reliably learned the
longer approach. This particular shaping function and coefficient did not solve
forgetting or full swing-up. It does not establish that reward design is irrelevant,
or that critic error is the cause. All nine runs selected the original checkpoint
under the predeclared retention-first rule. Initial and final metrics are exposed
separately so selection does not conceal regression.

The useful next diagnostic is to separate the PPO update from the supervised
rehearsal update, and test preservation against actual successful neural approach
rollouts. Existing rehearsal uses deterministic reference-controller targets,
which are not identical to the successful neural policy's behavior. Changing
reward does not alter that supervised loss. This is a hypothesis for a follow-up,
not a causal result of this comparison.

Run IDs:

- Baseline seeds0–2: `20261011T180512-8dbddb4d`, `20261011T180510-867eaf9a`, `20261011T180511-253e13d2`.
- Braking seeds0–2: `20261011T180511-4d49d33f`, `20261011T180512-ad2dc11d`, `20261011T180511-8c663f42`.
- Capture seeds0–2: `20261011T180513-95e668a4`, `20261011T180513-c527477d`, `20261011T180515-5b6deaba`.

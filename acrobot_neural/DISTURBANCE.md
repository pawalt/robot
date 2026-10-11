# Larger starting disturbances

All simulation, demonstrations, fitting, audits and recordings run on Modal in
peyton-agents. Results and videos are under `/experiments/acrobot-disturbance`.
This tests disturbances at the start around the hanging configuration, not kicks
applied after balancing at the top.

## Protocol

Compare the three exported radial neural policies from RADIAL.md with their
clock-based trajectory teachers on identical initial states. Every angle and
velocity is independently uniformly sampled; none is replaced by a zero start.
Bounds apply to shoulder and relative elbow coordinates separately.

| Condition | Each angle bound (radians) | Each angular velocity bound (rad/s) |
|---|---:|---:|
| Small | 0.1 | 0.1 |
| Medium | 0.25 | 0.5 |
| Large | 0.5 | 1 |
| Very large | 1 | 2 |
| Velocity 2 | 0 | 2 |
| Velocity 4 | 0 | 4 |

Initial comparison:128 starts per condition, seed321000,120seconds, physics.005s.
Each episode must balance continuously for the final10seconds: both links within
10degrees and both absolute link speeds below.5rad/s. Control interval.05s,
continuous torque limited to+/-1N m, same project dynamics as previous tests.

Generate64 separate training starts per condition (seed331000), roll out the
teacher for60seconds, and retain only successful recovery episodes. Keep every
third pre-capture state and its teacher phase. Select512 or2048 additional neural
prototypes by farthest-point coverage from up to12,000 candidate states, retaining
all original radial units. Fit each additional affine expert on256 nearby examples
with noise[.01,.01,.03,.06], using the same supervised least-squares method.
Test widths.01 and.03:12 training variants total. These changes expand coverage,
but do not mathematically guarantee that the old policy's behavior is preserved:
adding radial units also changes the normalized mixture weights.

Development:32 starts per condition, seed341000,60seconds, plus exact-ground and
32 local-balance starts. Select one expanded policy per trajectory, prioritizing
exact-ground success and >=90% local balance, then total development successes.
No confirmation metric selects the candidates.

Independent confirmation compares the three originals and three selected expanded
policies:128 entirely randomized starts per condition, seed361000,120seconds,
physics.0025s. Also test exact ground and128 local-balancing starts. All neural
evaluations use reloaded standalone TorchScript policies. No runtime teacher,
clock, trajectory file or LQR fallback is used.

For each audited policy, record episode0 for large, very-large andvelocity4;
also record the first opposite-outcome episode if one exists. Videos explicitly
label outcome-selected successes/failures, episode indices and initial states.
This preserves visible failures rather than displaying only chosen successes.

## Initial comparison

Successes out of128, in condition order above:

| Trajectory | Original neural | Clock-based teacher |
|---|---|---|
|0|128,128,126,125,126,125|128,88,20,3,20,15|
|1|128,128,128,128,128,128|62,9,2,0,2,1|
|2|128,128,124,125,127,123|108,86,39,9,52,19|

The original state-based imitation generalizes substantially beyond the small
starting perturbations previously tested. The teacher comparison does not establish
an upper bound on neural success: the teacher advances a clock, while the radial
policy selects local predictions from the current state. Rejoining the maneuver
after a delay is a plausible explanation for the difference, not an isolated
causal finding.

Additional recovery data generally reduced development success. The added units
change interpolation and may introduce competing phase-conditioned labels; this
experiment does not separately isolate those mechanisms. The original policies
are preserved, not replaced by the expanded variants.

## Independent confirmation results

Out of128 per condition, seed361000, .0025second physics,120second episodes:

| Policy | Small | Medium | Large | Very large | Velocity2 | Velocity4 | Total /768 |
|---|---:|---:|---:|---:|---:|---:|---:|
|Original0|128|126|127|128|125|125|759|
|Expanded0|127|128|126|119|125|123|748|
|Original1|127|128|128|128|128|128|767|
|Expanded1|127|128|127|128|128|126|764|
|Original2|128|126|127|127|128|127|763|
|Expanded2|127|127|126|124|126|125|755|

All six also passed exact ground and128/128 local-balance starts. Original1's
median sustained-balance onset was about27.2seconds for very-large starts and
29.6seconds for velocity4. Success is therefore not generally delayed to the
end of the allowed120seconds. The one failure of original1 occurred in the
smallest-disturbance group; sampled success need not vary monotonically with
initial perturbation bounds. These are finite empirical tests, not guarantees.

The original imitation policies already handle substantially larger starting
disturbances. This particular expansion of teacher demonstrations did not improve
any policy's aggregate confirmation success. All candidates and failures remain
available; none replaces the original policies.

Audit artifacts in `/data/neural-robust/` (each has results and recordings):

- Original0 `20261011T185158-37928d61`; expanded0 `20261011T185259-dc92eddd` (512 additional units,width.03).
- Original1 `20261011T185149-9d5a0577`; expanded1 `20261011T185258-72499ec5` (2048 additional units,width.01).
- Original2 `20261011T185149-a378865f`; expanded2 `20261011T185255-852af198` (512 additional units,width.03).

There are31 recorded videos, including outcome-labeled failures. Original1's
standalone policy remains the artifact linked in RADIAL.md:
`neural-radial/20261011T183152-b6856ae9/deployed.pt`.

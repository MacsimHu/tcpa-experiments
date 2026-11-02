# Company-side experiment handoff

## What to do first

The priority is scientific execution, not repeated environment reports.
Use an existing Python 3.10/PyTorch environment first. Do not upgrade or
reinstall a working environment without an actual dependency failure.
Use an authorized writable data-volume subdirectory for inputs, caches,
temporary files and checkpoints, not a nearly full system volume.

The planned order is:

1. **A / IEEE118:** four methods, five adaptation seeds, 1280 updates each.
   The L2 method additionally has three independent-seed tuning trajectories.
2. **A / GOC500 and IEEE30:** repeat the same matched comparisons.
3. **B:** genuinely trained budget settings, reusing only matching A100 Full.
4. **C:** controlled surrogate comparisons; then D new-identity multi-round.
5. **C2/E/F:** weight sensitivity, new pretrained models and label-assisted
   references. These are extensions, not prerequisites for every paper claim.

`jobs.jsonl` enumerates the requested work. Regenerate it with
`python make_jobs.py`. This command writes a schedule; it does not train models.
There are 450 formal and 36 tuning trajectories, plus six pretraining tasks
whose update budgets remain unresolved. Three eight-update qualifications
cover the new backend/system paths; do not repeat them for every seed.
The first delivery A/B/C has 165 formal and nine tuning trajectories.
Do not wait for all optional extensions before reporting the main evidence.

## Required research inputs: not bundled yet

**This repository is not yet a complete training package.** Its generic checks
and schedule do not replace the original numerical implementation.
Do not reimplement a different method from these notes and claim that the
original experiments ran. Missing assets must be reported explicitly.

| Asset role | Required content |
|---|---|
| Numerical bundle | Model, objectives, output reconstruction, data-role masks, sampling, training, complete-state saving, normal monitoring, interpolation and readout |
| Pretrained N0 | One identified checkpoint per grid and its configuration |
| Historical normal | Original train/monitoring split and fixed calibration subset |
| Added normal B | Prepared labeled replay population and provenance |
| Outage learning | Disjoint adaptation/development identity lists and inputs; no solution labels in the primary learning path |
| Report-only data | Fixed contingency test and independent normal C; never used to tune or select |
| D/E/F extensions | New identity split, actual pretraining recipe/budget and separate label-budget lists |

Known IEEE118 fingerprints, for identifying matching assets rather than
substituting any similarly named model:

```text
N0 checkpoint SHA256:
128d348d28fe28539b7b4846a0fdcc70cd75f61a03eb26dd420a838e46abcc40
Prepared normal B SHA256:
37de6c9dde845d82bc94416e6bf434fcc85c2b1d786de9be76cb4f6effbdf563
Prepared normal B receipt SHA256:
370cd8a8c151c7458a6c04b8849f43ec9e8db455395da4296b715c00c6c0df0c
```

Its historical training population has 48600 conditions; calibration uses
256. Outage adaptation/development have 920/308 conditions. Added normal B
and report-only normal C each have 1024, but are different roles. The fixed
contingency test has 551 conditions.

No private source, checkpoint, reference labels, company configuration or raw
logs should be copied wholesale into public Git. Obtain release-cleared assets
or use an approved research-input/result transfer channel.

## Shared scientific rules

Within each grid and comparison block, match N0, actual A100 backend,
precision, data identities, sampling streams, update budget and normal-gated
checkpoint selection. Use Adam at 2.5e-6; batches contain 32 outage inputs and
16 normal inputs from each cohort; monitor every 64 updates.
Keep TF32/AMP settings the same across methods. Do not splice CPU reference
scores into A100 paired comparisons.

Full uses the outage physics objective, normal replay and cohort-separated
quality penalty. Physics-only removes normal objective terms, not the shared
normal diagnostics/gate. Physics+Replay removes the quality penalty.
L2 adds lambda/2 times the sum of squared parameter changes to physics;
choose lambda from .001/.01/.1 using tuning seed 20261102 and normal-gated
unlabeled development PBL, tie-breaking toward smaller lambda.
Formal seeds are 1,11,42,123,2026. Preserve all fallback and adverse outcomes.
Formal test labels and normal C enter only after the complete block is locked.

B uses allowances d=0,.05,.10,.20: state-MSE factors (1+d)^2 and PBL factor
1+d. Change the training proxy and full gate together. Keep the supervised
budget at twice N0 plus 1e-4. The d=.10 runs are A's matching Full, not new jobs.

C crosses linear/quadratic positive-part penalties with paired/fixed
training-population references, preserving scales and all other coefficients.
The original quadratic-paired case is A Full. Its fourth new recipe replaces
supervised replay with a supervised one-sided penalty, retaining the actual
supervised budget and offset. Fixed references are over the actual normal
training populations, not silently equated with monitoring-population risks.
These are proxy designs, not exact constrained reformulations.

D has five genuinely disjoint-identity rounds on IEEE118/GOC500. Do not reuse
A's mixed-distribution first round: the new split must be bound first.
Round1 is shared by all arms; later Continue and current-only Restart use
1280 updates; cumulative-data Restart starts from N0 and uses 1280*t at
round t. Keep normal budgets anchored to N0. Lock the complete matrix before
scoring past/future test tasks. More rounds alone do not prove lifelong learning.

E uses three new pretrained models per larger grid, seeds7/17/29, with Full
and Replay under five adaptation seeds. Freeze the actual pretraining budget
before execution. F uses .10/.25/1.00 nested adaptation-label fractions with
separate tuning at each fraction, so smaller fractions cannot borrow labels
or tuned information from larger fractions. F is label-assisted, not an
accuracy upper bound.

## Execution and return

After the numerical/input bundles are present, start with one real trajectory
and increase concurrency from its measured GPU/host-memory peaks. Do not keep
repeating synthetic canaries or infer training capacity solely from free VRAM.
Keep each run separate and save model/optimizer/RNG/sampler/current step
atomically. Resume the actual latest state, never best weights as if complete.
Do not run the old lab-server launcher: its paths, CPU masks and environment
authority refer to a different machine.

Return per-job status, actual updates, selected tensor identity, source/input
hashes, timing and full failures through the approved channel. Public release
of real results requires its own non-sensitive content review.

# Current company runbook

## The scope is deliberately smaller

The author's 2026-11-02 deadline revision supersedes the earlier expansive
450-formal / 36-tuning proposal. Best-paper ambitions do not justify
redundant experiments or unsupported claims.

1. Finish the existing IEEE118 CPU comparison and its queued trained-budget
   study. Do not create another company version of either block.
2. Run GOC500 Full / Physics-only / Physics+Replay / selected L2 on the company
   A100: twenty formal trajectories plus three independent-seed L2 tunes.
3. If runtime permits a complete five-seed block before the result freeze,
   add only three GOC500 proxy contrasts: linear paired quality penalty,
   quadratic fixed training-population reference, and quadratic paired
   quality with supervised one-sided penalty instead of linear replay.
   The five matching company Full returns are reused, not retrained.

The third item requires its own implemented and frozen objective/selection
authority. `start_company.py` currently starts item 2 only. The three
conditional contrasts do not estimate all interactions or establish globally
optimal proxy design. Do not advertise them as a completed factorial study.

Extra case30 controls, company IEEE118 replication, company budget sweeps,
five-round new-identity learning, new pretrained models, label-assisted
comparators and broad weight sweeps are deferred. They are not hidden jobs.
Existing main results, three-round returns and negative evidence remain intact.

## Input delivery

`additional_inputs.json` gives the complete compressed 30/500 source/cache
payloads, and IEEE118 report-only references. Download only this manifest to
start the GOC500 company block; the IEEE118 training release is not required
and is not an instruction to rerun it.

`scientific_inputs.json` separately preserves the IEEE118 numerical source,
original train/val/N0, 1228 learning graphs and normal B for reproducibility.
`report_inputs.json` contains the original GOC500 test388 / normal-C1019 and
IEEE30 overall five-family test308 / normal-C1024. It does not generate a
new normal test population.

The downloader hashes all parts and assembled payloads, admits no traversal
paths/symlinks, preserves failed partials, and never overwrites different
scientific files. Report-only archives are downloaded but not extracted by
the installer. Native `.pt` graph caches are trusted only under their declared
research-input hashes. The explicit `weights_only=False` compatibility change
is limited to the identified legacy graph loader, not a global override.

GOC500 N0:
`3ebf69781a5d5b34195eae000bfd556fd6d713e5dd7fd5738e0f3aa282f6be9e`.
Its tensor identity:
`69dd408759d3c3c53b3ba3d5bde001395647b4d4f28041b2a1471fb8afc47cb0`.
Its original source train/val have 48600/5400 graphs; calibration256;
outage learning1156/development388; normal B1022.
Normal B is not silently rounded to1024. Rebuilding the same labeled inputs
under a new Torch serialization is declared as a new cache-byte authority,
not a match to the original CPU serialized cache hash.

## Actual first-block execution

Use an existing working environment before considering dependency changes.
The only necessary changed-backend qualification is one actual Full8 path;
do not repeat generic synthetic tests for each method/seed. The launcher binds
the actual versions, permitted CPUs, writable paths and source hashes into a
new plan. It never uses the old lab-server masks or rewrites frozen originals.

Within the comparison, match N0, actual A100 backend/precision, identities,
sampling streams, updates and normal-gated checkpoint/interpolation selection.
Adam is 2.5e-6; each update uses32 outage samples and16 historical +16 added
normal samples. Monitor every64 updates. TF32 and AMP are not introduced.
The company Full must be newly trained within this block; old CPU/A5000 scores
cannot substitute for it.

Full uses physics, replay and cohort-separated quality. Physics-only drops
normal objective terms, not the shared normal diagnostics/selection gate.
Replay drops the quality penalty. L2 adds lambda/2 times the sum of squared
parameter changes to physics, without replay. Tune .001/.01/.1 using separate
seed20261102 and normal-gated unlabeled development PBL, tie lower lambda.
Formal seeds are1/11/42/123/2026. Preserve all fallback and adverse outcomes.
Extra L2 tuning cost is reported, not called matched total compute.

Current-state checkpoints contain model, Adam, RNG and sampler state.
Their presence alone does not certify numerical resumption. The launcher
refuses duplicate runs rather than restarting science from N0 after a fault.
Keep the original disk write-stop margin and all raw native states; no
automatic cleanup or archiving/deletion loop is part of this company entry.

## Readout, paper and stopping

Lock the entire formal block and all three tuning returns before scoring
test/C. Use a fresh N0 on the same readout backend and the same fixed cohorts.
Report overall target MSE/PBL, C risks, paired seed differences, conditional
uncertainty, fallback frequency and actual training/selection/inference costs.
Forward-only timing is not end-to-end solver speedup. More readouts do not
require new adaptation matrices. Residual reduction is not solution accuracy.

Keep the existing three-round conclusion bounded: retained parameters improve
later adaptation on related conditions with the same learning identity support.
It is not proof of unseen-identity lifelong learning or matched cumulative
compute. Do not add rounds merely to strengthen that unsupported wording.

The author's target is November10, not an independently verified venue rule.
Admit no new block after November6; freeze usable evidence November7; reserve
November8-9 for figures, tables, references, page count and author review.
Use actual observed runtime, not favorable scores, to decide whether a whole
five-seed block fits. Do not cherry-pick completed good seeds or silently
extend a frozen deadline. Existing jobs and failed evidence are not discarded.

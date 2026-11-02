# TCPA: deadline-prioritized PESGM experiments

**Follow [CURRENT_SCOPE.json](CURRENT_SCOPE.json), not the historical 495-entry
proposal.** The author has reduced scope to necessary, nonduplicate evidence.

| Work | Where | Status |
|---|---|---|
| IEEE118 four-method comparison, 20 formal + 3 L2 tunes | Existing CPU study | Do not rerun on company GPU |
| IEEE118 trained budgets, 15 additional formal | Existing CPU successor queue | Reuses its matching 10% Full; do not duplicate |
| GOC500 four-method comparison, 20 formal + 3 L2 tunes | Company A100 | First company block |
| Three targeted proxy contrasts, 15 additional formal | Company A100 / GOC500 | Separate implementation/authority pending; reuse company Full |
| Extra case30, five-round continual, new pretraining, label-assisted and broad weight sweeps | Deferred | Do not dispatch |

The maximum new company scope is **35 formal + 3 tuning trajectories**, plus
one eight-update changed-backend qualification. The original source, data and
N0 are transported with SHA256 bindings. Publication is not evidence that
company training has started or that the new backend is qualified.

## Start the first company block

Use the existing Python 3.10 / PyTorch 2.6 CUDA environment if working.
Select an actual writable data-volume directory, with space for inputs and
retained checkpoints. Do not use the almost-full system volume.

```text
git clone https://github.com/MacsimHu/tcpa-experiments.git
cd tcpa-experiments
python fetch_inputs.py --work-dir "$WORK_DIR" --manifest additional_inputs.json
python start_company.py --work-dir "$WORK_DIR" --case case500 --run-id case500_matched_01 --gpu 0
```

Set `WORK_DIR` in the company's Linux shell first; it is not a literal default.
Run in an existing persistent terminal/session. If `additional_inputs.json`
is absent, the full transfer is not yet committed: pull the completed handoff,
not the old generic smoke. Nothing installs or upgrades packages automatically.
The numerical path imports PyTorch, NumPy, PyG, PyYAML and tqdm.

The launcher refuses other occupied GPU workloads and rejects repeated run
directories. It reconstructs the original 1022-member GOC500 normal B from
fixed references, with no new NN forwards or PF solves, before freezing its
new serialization/backend authority. It then executes Full8 once, three
independent-seed L2 tunes and 20 actual 1280-update formal trajectories.
All native states remain on the data volume. A failed qualification is not
silently skipped or converted into a completed formal experiment.

`jobs.jsonl` has 39 entries: one qualification, three tunes, twenty first-block
formal cells and fifteen pending proxy cells. It is an inventory, **not an
automatic dispatcher**. `make_jobs_full_draft.py` and `jobs_full_draft.jsonl`
retain the superseded large proposal for provenance only.

## Reporting

Use overall MSE/PBL and fixed independent normal C. Keep every seed, N0
fallback, adverse direction and actual cost. No new test/C labels may enter
training, L2 tuning or checkpoint selection. Report-only archives stay
unextracted until the complete block is locked.

The public benchmark transport does not include credentials or company/site
configuration. Full readout and targeted proxy integration remain separately
identified work; do not call the first training block the entire study.
See [RUNBOOK.md](RUNBOOK.md) for the controls and deadline rules.

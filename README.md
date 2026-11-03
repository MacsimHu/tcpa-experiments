# TCPA company A100 handoff — 2026-11-03

Start at **[CURRENT_TASKS.md](CURRENT_TASKS.md)** and [RUNBOOK.md](RUNBOOK.md).
`CURRENT_SCOPE.json` is the current finite scope; older expansive draft queues
are historical proposals, not launch instructions.

| Work | Required trajectories | Status / action |
|---|---:|---|
| Existing company500 | 20 formal + 3 independent L2 tunes | User reports started; actual run identity/results not received. **Finish/reuse it; never restart or duplicate it.** |
| New company118 matched block | 20 formal + 3 independent L2 tunes | Implemented handoff, **not launched/qualified here**. Serialize after existing500. |
| New trained118 budget block | 15 formal | Implemented handoff, **not launched**. Requires the complete new118 block and its fixed readout; reuse its five Full returns at10%. |
| Old optional500 proxy block | 15 | **Deferred/unimplemented; not a current prerequisite.** |

Essential total: **55 formal + 6 tunes**, including existing500.
The new transfer is **38 trajectories = 35 formal + 3 tunes**, not a500 rerun.
Each trajectory has1280 Adam updates; new118 has one eight-update backend
qualification, which is not a formal result. No additional30/pretraining,
new-identity/oracle grids or proxy expansion.

The local CPU118 block is incomplete (last known3/20 formal,3/3 tunes);
continuation_v2 actually exited1 at the disk guard (free6,364,659,712 bytes).
Budgets never started locally. Preserve those failures; local no longer
dispatches118/budget. **No pooling of CPU partials with A100 results and no old
CPU Full substitution.**

## What is executable

* `fetch_inputs.py`: hash-verified published inputs. The narrow
  `company_reports_case118.json` / `company_reports_case500.json` reuse existing
  release bytes without downloading unrelated30/500 inputs.
* `start_company.py --case case118`: freezes a new independent source-bound
  A100118 authority; requires the completed existing500 directory.
* `company_readout.py`: external adapter accepting an explicit **unchanged**
  completed old500 or new118 directory. All23 locks and actual exits must
  pass before test/C is opened. No editing a running/frozen500 authority.
* `start_budget.py`:15 actual trained budgets at0/5/20%, changing **both**
  training penalty and final gate; parent five Full at10% are reused exactly.
* `export_results.py`: small manual JSON/CSV/evidence archive, including
  complete percondition identity/error/residual records, allseed fallbacks,
  costs, source/input/model hashes and failures. No automatic upload;
  complete native model/optimizer/RNG states stay on the data volume.

Fixed readouts report raw identity-/family-weighted MSE, PBL and four C risks,
sampleSD and paired approximate nonsimultaneous t4 95% intervals across allfive
seeds. They are conditional on one pretrained checkpoint and fixed research
cohorts, not a new blind evaluation. **No rPBL imputation, pure-gate timing,
forward-latency measurement or AC-solver speedup claim.**

The adapters/protocol guards are IO-only tested; an actual company CUDA
environment, storage quota and numerical end-to-end execution are **not**
qualified by those tests. No scientific job, NN/PF evaluation, server command,
new canary or cleanup was performed for this publication. Author-requested
last new block: November6; result freeze: November7 (not venue policy).

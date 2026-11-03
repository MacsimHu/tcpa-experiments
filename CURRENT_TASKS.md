# Current company tasks — 2026-11-03

1. **Existing500 finishes first.** Do not kill, restart, duplicate, modify its
   frozen run, or infer success from a process name. Its actual run identity
   and results have not been returned. Record its existing absolute directory
   as `RUN500`; the new launcher refuses case500.
2. Read out/export the **complete** existing500 using the external adapter.
3. New118: **23 trajectories** (20 formal +3 independent L2 tunes), then its
   fixed readout/export. New authority; never splice in the old CPU partials.
4. Trained118 budgets: **15 additional trajectories**, only after the new118
   block/readout completes. Reuse exactly its five Full at10%; no new Full.
5. Budget fixed readout/export. Keep every seed, negative result and unchanged
   N0 fallback. **Old15 proxy trajectories are deferred, not a prerequisite.**

**55 formal +6 tunes total;38 new transferred trajectories. None launched by
this handoff.** Existing500 start is user-reported, not independently observed.
Detailed commands, failure policy and metric/timing definitions:
[RUNBOOK.md](RUNBOOK.md). Source lineage: `PROTOCOL_LINEAGE.json`.

Commands below run **only on the company Ubuntu data volume in the existing
qualified environment**, not on the lab/Windows host. Set `DATA` to the actual
authorized input/run volume, and `RUN500` to the actual existing run; do not
copy literal placeholders or create a replacement500 directory.

```bash
git pull --ff-only
python -B fetch_inputs.py --work-dir "$DATA" --manifest company_reports_case500.json
python -B company_readout.py --run-dir "$RUN500" --work-dir "$DATA" --output-dir "$DATA/readout500_v1"
python -B export_results.py --run-dir "$RUN500" --readout-dir "$DATA/readout500_v1" --archive "$DATA/company500_return_v1.tar.gz"

python -B fetch_inputs.py --work-dir "$DATA" --manifest scientific_inputs.json
python -B fetch_inputs.py --work-dir "$DATA" --manifest company_reports_case118.json
python -B start_company.py --case case118 --work-dir "$DATA" --run-id company118_matched_v1 --gpu 0 --after-run-dir "$RUN500"
python -B company_readout.py --run-dir "$DATA/runs/company118_matched_v1" --work-dir "$DATA" --output-dir "$DATA/readout118_v1"
python -B export_results.py --run-dir "$DATA/runs/company118_matched_v1" --readout-dir "$DATA/readout118_v1" --archive "$DATA/company118_return_v1.tar.gz"

python -B start_budget.py --parent-run "$DATA/runs/company118_matched_v1" --parent-readout "$DATA/readout118_v1" --work-dir "$DATA" --run-id company118_budgets_v1
python -B company_readout.py --budget --run-dir "$DATA/runs/company118_budgets_v1" --work-dir "$DATA" --output-dir "$DATA/readout118_budget_v1"
python -B export_results.py --budget --run-dir "$DATA/runs/company118_budgets_v1" --readout-dir "$DATA/readout118_budget_v1" --archive "$DATA/company118_budget_return_v1.tar.gz"
```

Run commands sequentially after inspecting the previous actual success; this
is **not** an unattended shell queue. On any failure stop, preserve everything,
return the failure archive and diagnose—never reset/reseed/restart from N0.
Result archives are manually returned by the operator; no credentials, site
configuration or unpublished manuscript belongs in this public repository.

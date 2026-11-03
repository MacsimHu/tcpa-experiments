"""Freeze exactly 15 trained118 budget trajectories after the matched A100 parent readout."""

import argparse
import copy
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import shutil
import sys

from company_protocol import (BUDGETS, SEEDS, completed_readout, read, sha, validate_run, write)
from start_company import native_admission_bytes, require_idle_gpu, validate_run_id


HERE = Path(__file__).resolve().parent


def budget_plan(parent, parent_run, parent_readout, destination):
    scope = completed_readout(parent_readout, parent_run)
    if parent["company_case"] != "case118" or parent["profiles"]["case118"]["device"] != "cuda":
        raise ValueError("Only the complete matched company118 block is a valid budget parent")
    plan = copy.deepcopy(parent)
    plan.update(
        durable_output_root=str(destination), utc_frozen=datetime.now(timezone.utc).isoformat(),
        matched_parent_run=str(parent_run), matched_parent_plan_sha256=sha(parent_run / "plan.json"),
        matched_parent_readout=str(parent_readout),
        matched_parent_readout_plan_sha256=sha(parent_readout / "READOUT_PLAN.json"),
        matched_parent_analysis_sha256=sha(parent_readout / "analysis.json"),
        budget_readout_population_bindings=scope["report_payload_bindings"],
        qualification_schedule=[], tuning_schedule=[], formal_methods=list(BUDGETS),
        methods=list(BUDGETS), maximum_common_readout_graphs=26000,
        logical_readout_models=20, budget_parent_Full_retrained=False,
        schedule=[{"case": "case118", "seed": seed, "method": method,
                   "name": f"case118_seed{seed}_{method}",
                   "cpus": parent["profiles"]["case118"]["cpu_affinity"]}
                  for seed in SEEDS for method in BUDGETS])
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-run", type=Path, required=True)
    parser.add_argument("--parent-readout", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if os.name != "posix":
        raise OSError("Ubuntu company entry point only; not a local launcher")
    validate_run_id(args.run_id)
    if datetime.now(timezone(timedelta(hours=8))).date().isoformat() > "2026-11-06":
        raise ValueError("The author's last-new-block date passed; do not extend the experiment")
    parent_run, parent_readout = args.parent_run.resolve(strict=True), args.parent_readout.resolve(strict=True)
    parent, _ = validate_run(parent_run)
    completed_readout(parent_readout, parent_run)
    root = args.work_dir.resolve(strict=True)
    anchor = Path(parent["profiles"]["case118"]["workspace"]) / "runs" / "tcpa_strong_anchor_v1" / "anchor_assets" / "case118" / "s1" / "model.pt"
    admission_bytes = native_admission_bytes(anchor.stat().st_size, 15)
    if shutil.disk_usage(root).free < admission_bytes:
        raise OSError(f"Protect complete raw-native budgets and5GiB write-stop margin; require {admission_bytes} free bytes")
    require_idle_gpu(parent["profiles"]["case118"]["cuda_visible_devices"])
    destination = root / "runs" / args.run_id
    destination.mkdir(parents=True, exist_ok=False)
    for relative in parent["source_bindings"]:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(parent_run / relative, target)
    for name in ("company_budget_supervisor.py", "company_protocol.py", "start_budget.py", "start_company.py"):
        shutil.copyfile(HERE / name, destination / name)
    plan = budget_plan(parent, parent_run, parent_readout, destination)
    plan["native_storage_admission"] = {"required_free_bytes": admission_bytes,
                                        "conservative_raw_N0_multiples_per_cell": 48,
                                        "whole_volume_quota_verified": False}
    for name in ("company_budget_supervisor.py", "company_protocol.py", "start_budget.py", "start_company.py"):
        plan["source_bindings"][name] = sha(destination / name)
    write(destination / "plan.json", plan)
    profile = plan["profiles"]["case118"]
    os.environ["CUDA_VISIBLE_DEVICES"] = profile["cuda_visible_devices"]
    for name in ("TMPDIR", "TMP", "TEMP"):
        os.environ[name] = str(root / "scratch")
    (root / "scratch").mkdir(exist_ok=True)
    print("FROZEN_NOT_YET_TRAINED", destination, "15x1280 updates; zero new Full", flush=True)
    os.execv(sys.executable, [sys.executable, "-B", str(destination / "company_budget_supervisor.py"),
                            "--plan-sha256", sha(destination / "plan.json")])


if __name__ == "__main__":
    main()

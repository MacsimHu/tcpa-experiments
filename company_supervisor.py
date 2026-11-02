"""Execute a finite matched study, retaining every native state on the data disk."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from common import read, sha, verify, write
import review_supervisor as inherited
from study_methods import choose_anchor


HERE = Path(__file__).resolve().parent


def run(plan):
    case = plan["company_case"]
    profile = plan["profiles"][case]
    os.sched_setaffinity(0, profile["cpu_affinity"])
    verify(Path(profile["workspace"]), plan["workspace_input_bindings"])
    deadline = time.monotonic() + plan["maximum_batch_seconds"]
    phases = (("qualify", plan["qualification_schedule"]),
              ("tune", plan["tuning_schedule"]), ("formal", plan["schedule"]))
    for phase, schedule in phases:
        for row in schedule:
            inherited.run_child(plan, row, phase, deadline)
            if phase != "qualify":
                inherited.run_child(plan, row, phase, deadline, selector=True)
            directory = HERE / phase / row["name"]
            members = {
                str(path.relative_to(HERE)): {"bytes": path.stat().st_size, "sha256": sha(path)}
                for path in directory.glob("*.pt")
            }
            write(HERE / "retention" / phase / row["name"] / "NATIVE_RETAINED.json", {
                "members": members, "deletions": 0, "numerical_resume_verified": False,
                "plan_sha256": sha(HERE / "plan.json"),
            })
        if phase == "tune":
            paths = [HERE / "tune" / (row["name"] + ".return.json") for row in schedule]
            write(HERE / "ANCHOR_SELECTION.json", {
                "plan_sha256": sha(HERE / "plan.json"),
                "selected_method": choose_anchor([read(path) for path in paths]),
                "tuning_return_bindings": {str(path.relative_to(HERE)): sha(path) for path in paths},
                "selection_uses_only_unlabeled_development_PBL_after_normal_gate": True,
            })
    inherited.paired_lock(plan)
    write(HERE / "READOUT_REQUEST.json", {
        "status": "TRAINING_COMPLETE_COMMON21_MODEL_READOUT_REQUIRED",
        "global_return_lock_sha256": sha(HERE / "GLOBAL_RETURN_LOCK.json"),
        "case": case, "target_count": plan["cases"][case]["target_conditions"],
        "normal_C_count": 1019 if case == "case500" else 1024,
        "target_or_C_readout_started": False,
        "response_data_and_readout_adapter_pending": True,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    plan = read(HERE / "plan.json")
    if sha(HERE / "plan.json") != args.plan_sha256 or str(HERE) != plan["durable_output_root"]:
        raise ValueError("The company authority or output directory differs")
    verify(HERE, plan["source_bindings"])
    write(HERE / "EXCLUSIVE_RUN_CLAIM.json", {
        "pid": os.getpid(), "utc": datetime.now(timezone.utc).isoformat(),
        "plan_sha256": args.plan_sha256,
    })
    try:
        run(plan)
    except Exception as error:
        write(HERE / "BATCH_EXIT.json", {"exit_code": 1, "error": repr(error), "automatic_retry": False})
        raise
    else:
        write(HERE / "BATCH_EXIT.json", {"exit_code": 0, "common_readout_pending": True})

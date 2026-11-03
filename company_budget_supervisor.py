"""Execute fifteen actual budget trainings; retain native states, never retrain parent Full."""

import argparse
import os
from pathlib import Path
import time

from company_protocol import BUDGETS, SEEDS, read, sha, validate_parent, verify, write


HERE = Path(__file__).resolve().parent


def lock(plan):
    from study_methods import specification
    parent, full = validate_parent(plan)
    returns, bindings, pairs = [], {}, []
    for seed in SEEDS:
        previous = read(parent / "formal" / (full[seed]["name"] + ".locked.json"))
        sampling = {sha(parent / "formal" / (full[seed]["name"] + ".sampling.jsonl"))}
        calibration = {previous["calibration_sha256"]}
        for row in (item for item in plan["schedule"] if item["seed"] == seed):
            prefix = HERE / "formal" / row["name"]
            returned = read(prefix.with_suffix(".return.json"))
            trained = read(prefix.with_suffix(".locked.json"))
            expected = specification(row["method"])["quality_factors"]
            if (trained["optimizer_updates"] != 1280 or trained["quality_factors"] != expected
                    or returned["plan_sha256"] != sha(HERE / "plan.json")
                    or sha(HERE / returned["file"]) != returned["sha256"]
                    or returned["target_response_or_C_selection"]):
                raise ValueError("A real trained budget return or its training penalty differs")
            selection_input = read(HERE / "operations" / row["name"] / "selection_input.json")
            if selection_input["quality_factors"] != expected:
                raise ValueError("Both the training penalty and final gate must change with delta")
            sampling.add(sha(prefix.with_suffix(".sampling.jsonl")))
            calibration.add(trained["calibration_sha256"])
            returns.append(returned)
            relative = str(prefix.with_suffix(".return.json").relative_to(HERE))
            bindings[relative] = sha(prefix.with_suffix(".return.json"))
        if len(sampling) != 1 or len(calibration) != 1:
            raise ValueError("Budgets must exactly match their own A100 Full streams and calibration")
        pairs.append({"case": "case118", "seed": seed, "sampling_sha256": sampling.pop(),
                      "calibration_sha256": calibration.pop()})
    write(HERE / "GLOBAL_RETURN_LOCK.json", {
        "status": "ALL15_TRAINED_BUDGET_RETURNS_AND5_PARENT_FULL_RETURNS_CLOSED",
        "plan_sha256": sha(HERE / "plan.json"), "returns": returns, "return_bindings": bindings,
        "paired_sampling": pairs, "target_and_C_response_values_read": 0,
        "matched_parent_plan_sha256": plan["matched_parent_plan_sha256"]})


def run(plan):
    import review_supervisor as inherited
    validate_parent(plan)
    if (len(plan["schedule"]) != 15 or
            {(row["seed"], row["method"]) for row in plan["schedule"]} !=
            {(seed, method) for seed in SEEDS for method in BUDGETS}
            or plan["qualification_schedule"] or plan["tuning_schedule"]):
        raise ValueError("Exactly fifteen budget trajectories, no extra Full or tuning")
    os.sched_setaffinity(0, plan["profiles"]["case118"]["cpu_affinity"])
    deadline = time.monotonic() + plan["maximum_batch_seconds"]
    for row in plan["schedule"]:
        inherited.run_child(plan, row, "formal", deadline)
        inherited.run_child(plan, row, "formal", deadline, selector=True)
        members = {str(path.relative_to(HERE)): {"bytes": path.stat().st_size, "sha256": sha(path)}
                   for path in (HERE / "formal" / row["name"]).glob("*.pt")}
        write(HERE / "retention" / "formal" / row["name"] / "NATIVE_RETAINED.json", {
            "members": members, "deletions": 0, "numerical_resume_verified": False,
            "plan_sha256": sha(HERE / "plan.json")})
    lock(plan)
    write(HERE / "READOUT_REQUEST.json", {"status": "ALL15_ACTUAL_BUDGET_RETURNS_READY",
          "global_return_lock_sha256": sha(HERE / "GLOBAL_RETURN_LOCK.json"),
          "target_or_C_readout_started": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    plan = read(HERE / "plan.json")
    if sha(HERE / "plan.json") != args.plan_sha256 or str(HERE) != plan["durable_output_root"]:
        raise ValueError("Wrong independent budget authority")
    verify(HERE, plan["source_bindings"])
    write(HERE / "EXCLUSIVE_RUN_CLAIM.json", {"pid": os.getpid(), "plan_sha256": args.plan_sha256})
    try:
        run(plan)
    except Exception as error:
        write(HERE / "BATCH_EXIT.json", {"exit_code": 1, "error": repr(error), "automatic_retry": False})
        raise
    write(HERE / "BATCH_EXIT.json", {"exit_code": 0, "common_readout_pending": True})

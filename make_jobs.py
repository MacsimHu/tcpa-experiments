"""Export the finite essential inventory; existing500 is reuse, not a launch queue."""

import json
from pathlib import Path

from company_protocol import BUDGETS, METHODS, SEEDS


HERE = Path(__file__).resolve().parent


def build_jobs():
    scope = json.loads((HERE / "CURRENT_SCOPE.json").read_bytes())
    if scope["essential_total_including_existing500"] != {"formal": 55, "tuning": 6}:
        raise ValueError("The author's finite essential scope changed")
    jobs = []
    for case in ("case500", "case118"):
        block = "existing500" if case == "case500" else "new118"
        qualification = f"{block}.qualification"
        jobs.append({"job_id": qualification, "grid": case, "block": block, "phase": "qualification",
                     "method": "full", "seed": 1, "updates": 8, "depends_on": [],
                     "implementation_available": True, "execution_ready": False,
                     "action": "reuse_existing_no_dispatch" if case == "case500" else "new_after_existing500"})
        tunes = []
        for method in ("anchor_001", "anchor_01", "anchor_1"):
            identifier = f"{block}.{method}.s20261102"
            tunes.append(identifier)
            jobs.append({"job_id": identifier, "grid": case, "block": block, "phase": "tune",
                         "method": method, "seed": 20261102, "updates": 1280,
                         "depends_on": [qualification], "implementation_available": True,
                         "execution_ready": False,
                         "action": "reuse_existing_no_dispatch" if case == "case500" else "new"})
        for seed in SEEDS:
            for method in METHODS:
                jobs.append({"job_id": f"{block}.{method}.s{seed}", "grid": case, "block": block,
                             "phase": "formal", "method": method, "seed": seed, "updates": 1280,
                             "depends_on": [qualification, *tunes] if method == "selected_anchor" else [qualification],
                             "implementation_available": True, "execution_ready": False,
                             "action": "reuse_existing_no_dispatch" if case == "case500" else "new"})
    old500 = [row["job_id"] for row in jobs if row["block"] == "existing500"]
    next(row for row in jobs if row["job_id"] == "new118.qualification")["depends_on"] = old500
    parent = [row["job_id"] for row in jobs if row["block"] == "new118"]
    for seed in SEEDS:
        for method in BUDGETS:
            jobs.append({"job_id": f"budget118.{method}.s{seed}", "grid": "case118", "block": "budget118",
                         "phase": "formal", "method": method, "seed": seed, "updates": 1280,
                         "depends_on": parent, "additional_prerequisite": "Complete matching A100118 fixed readout",
                         "implementation_available": True, "execution_ready": False, "action": "new"})
    return jobs


def main():
    jobs = build_jobs()
    identifiers = {row["job_id"] for row in jobs}
    if (len(identifiers) != 63 or sum(row["phase"] == "formal" for row in jobs) != 55
            or sum(row["phase"] == "tune" for row in jobs) != 6
            or any(set(row["depends_on"]) - identifiers for row in jobs)):
        raise ValueError("The fixed essential matrix or dependency graph changed")
    (HERE / "jobs.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in jobs), encoding="utf-8")
    print(json.dumps({"essential_formal": 55, "essential_tuning": 6,
                      "new_trajectories_excluding_existing500": 38, "automatic_dispatch": False,
                      "old_optional15_proxy": "deferred_not_a_prerequisite"}))


if __name__ == "__main__":
    main()

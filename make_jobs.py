"""Export only the author's deadline-prioritized, nonduplicate company scope."""

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SEEDS = (1, 11, 42, 123, 2026)


def build_jobs():
    scope = json.loads((HERE / "CURRENT_SCOPE.json").read_bytes())
    if scope["company_dispatch_cases"] != ["case500"]:
        raise ValueError("Review any expansion before regenerating the deadline queue")
    jobs = []

    def add(block, method, seed, phase, updates=1280, needs=(), implemented=True):
        identifier = f"{block}.case500.{method}.s{seed}"
        jobs.append({"job_id": identifier, "block": block, "grid": "case500",
                     "method": method, "seed": seed, "phase": phase, "updates": updates,
                     "depends_on": list(needs), "implementation_available": implemented,
                     "execution_ready": False,
                     "admission": "Actual company GPU/inputs/output ownership required; no automatic dispatch"})
        return identifier

    qualification = add("Q", "full", 1, "qualification", updates=8)
    tuning = [add("A", method, 20261102, "tune", needs=[qualification])
              for method in ("anchor_001", "anchor_01", "anchor_1")]
    first_block = list(tuning)
    for seed in SEEDS:
        for method in ("full", "physics_only", "physics_replay", "selected_anchor"):
            first_block.append(add(
                "A", method, seed, "formal",
                needs=[qualification, *tuning] if method == "selected_anchor" else [qualification]))
    for method in scope["company_second_block"]["methods"]:
        for seed in SEEDS:
            add("C", method, seed, "formal", needs=first_block, implemented=False)
    return jobs


def main():
    jobs = build_jobs()
    identifiers = {row["job_id"] for row in jobs}
    if (len(identifiers) != 39 or sum(row["phase"] == "formal" for row in jobs) != 35
            or sum(row["phase"] == "tune" for row in jobs) != 3
            or any(set(row["depends_on"]) - identifiers for row in jobs)):
        raise ValueError("The fixed deadline scope or dependency graph changed")
    target = HERE / "jobs.jsonl"
    if target.exists() and len(target.read_text(encoding="utf-8").splitlines()) == 495:
        historical = HERE / "jobs_full_draft.jsonl"
        if historical.exists():
            raise FileExistsError("Preserve both historical proposals; inspect rather than overwrite")
        target.rename(historical)
    target.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in jobs), encoding="utf-8")
    print(json.dumps({"company_formal_maximum": 35, "company_tuning": 3,
                      "company_qualification_updates": 8,
                      "existing_CPU_formal": 35, "existing_CPU_tuning": 3,
                      "automatic_dispatch": False}))


if __name__ == "__main__":
    main()

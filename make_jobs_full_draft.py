"""Historical expansive proposal, superseded by CURRENT_SCOPE.json; do not dispatch."""

import json
from pathlib import Path


SEEDS = (1, 11, 42, 123, 2026)
GRIDS = ("case118", "case500", "case30")


def build_jobs():
    jobs = []

    def add(block, grid, method, seed, updates=1280, phase="formal", needs=(), **extra):
        identifier = f"{block}.{grid}.{method}.s{seed}"
        jobs.append({
            "job_id": identifier, "block": block, "grid": grid, "method": method,
            "seed": seed, "phase": phase, "updates": updates,
            "depends_on": list(needs), "execution_ready": False,
            "requires": ["release-cleared numerical bundle", "frozen input-role manifest",
                         "authorized writable workspace", "admitted actual runtime"],
            **extra,
        })
        return identifier

    for grid in GRIDS:
        qualification = add("Q", grid, "full", 1, updates=8, phase="qualification")
        tuning = [
            add("A", grid, f"anchor_{coefficient}", 20261102, phase="tune",
                needs=[qualification], anchor_coefficient=coefficient)
            for coefficient in (.001, .01, .1)
        ]
        for method in ("full", "physics_only", "physics_replay", "selected_anchor"):
            for seed in SEEDS:
                add("A", grid, method, seed,
                    needs=[qualification, *tuning] if method == "selected_anchor" else [qualification])
        for allowance in (0., .05, .2):
            for seed in SEEDS:
                add("B", grid, f"budget_{allowance}", seed,
                    needs=[f"A.{grid}.full.s{seed}"], state_rmse_allowance=allowance,
                    pbl_allowance=allowance, supervised_factor=2., supervised_offset=1e-4)
        for method in ("linear_paired", "quadratic_fixed_source", "linear_fixed_source",
                       "quadratic_paired_supervised_hinge"):
            for seed in SEEDS:
                add("C", grid, method, seed, needs=[f"A.{grid}.full.s{seed}"])
    for method in ("quadratic_paired", "linear_paired", "quadratic_fixed_source",
                   "linear_fixed_source", "quadratic_paired_supervised_hinge"):
        for multiplier in (.1, 10.):
            for seed in SEEDS:
                add("C2", "case118", f"{method}_weight_{multiplier}", seed,
                    quality_weight_multiplier=multiplier)
    for grid in ("case118", "case500"):
        for seed in SEEDS:
            first = add("D", grid, "round1_shared", seed, identity_split="new_five_round_split_required")
            previous = first
            for round_number in range(2, 6):
                previous = add("D", grid, f"round{round_number}_continue", seed,
                               needs=[previous], round=round_number)
                add("D", grid, f"round{round_number}_restart_current", seed, round=round_number)
                add("D", grid, f"round{round_number}_restart_cumulative", seed,
                    updates=1280 * round_number, round=round_number)
        for pretrained_seed in (7, 17, 29):
            parent = add("E", grid, f"pretrain_{pretrained_seed}", pretrained_seed,
                         updates=None, phase="pretrain", budget_status="UNFROZEN_DO_NOT_START")
            for method in ("full", "physics_replay"):
                for seed in SEEDS:
                    add("E", grid, f"pretrained_{pretrained_seed}_{method}", seed, needs=[parent])
    for grid in GRIDS:
        for fraction in (.1, .25, 1.):
            tuning = [
                add("F", grid, f"labels_{fraction}_coefficient_{coefficient}", 20261102,
                    phase="tune", label_fraction=fraction, supervised_coefficient=coefficient)
                for coefficient in (.1, 1., 10.)
            ]
            for seed in SEEDS:
                add("F", grid, f"labels_{fraction}_selected", seed,
                    needs=tuning, label_fraction=fraction)
    return jobs


def main():
    jobs = build_jobs()
    identifiers = {job["job_id"] for job in jobs}
    if len(identifiers) != len(jobs):
        raise ValueError("Duplicate planned job identities")
    if any(set(job["depends_on"]) - identifiers for job in jobs):
        raise ValueError("A planned dependency is absent")
    summary = {
        "status": "PLANNED_NOT_EXECUTION_READY",
        "formal": sum(job["phase"] == "formal" for job in jobs),
        "tuning": sum(job["phase"] == "tune" for job in jobs),
        "qualification": sum(job["phase"] == "qualification" for job in jobs),
        "pretraining_unfrozen": sum(job["phase"] == "pretrain" for job in jobs),
        "formal_and_tuning_updates": sum(job["updates"] for job in jobs
                                       if job["phase"] in ("formal", "tune")),
        "qualification_updates": sum(job["updates"] for job in jobs if job["phase"] == "qualification"),
    }
    if summary != {
        "status": "PLANNED_NOT_EXECUTION_READY", "formal": 450, "tuning": 36,
        "qualification": 3, "pretraining_unfrozen": 6,
        "formal_and_tuning_updates": 750080, "qualification_updates": 24,
    }:
        raise ValueError("The exported queue differs from the declared finite matrix")
    destination = Path(__file__).with_name("jobs_full_draft.jsonl")
    if destination.exists():
        raise FileExistsError("Preserve the existing schedule; generate a revision explicitly")
    with destination.open("x", encoding="utf-8", newline="\n") as handle:
        for job in jobs:
            handle.write(json.dumps(job, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

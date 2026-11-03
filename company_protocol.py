"""IO-only authorities shared by training, fixed readout and manual export."""

import hashlib
import json
import math
import os
from pathlib import Path
import statistics


SEEDS = (1, 11, 42, 123, 2026)
METHODS = ("full", "physics_only", "physics_replay", "selected_anchor")
BUDGETS = {"budget_00": 0.0, "budget_05": 0.05, "budget_20": 0.20}
METRICS = ("mse_entry", "cmse", "pbl_mean", "theta_mse_rad2", "pq_vm_mse_pu2")
WEIGHTS = {"generator": .25, "generator-generator": .25, "line": .25,
           "line-generator": .125, "line-line": .125}
COUNTS = {"case118": (551, 1024), "case500": (388, 1019)}


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def member(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if path == root or root not in path.parents or path.is_symlink():
        raise ValueError("An authority member escapes its run")
    return path


def verify(root, bindings):
    for relative, expected in bindings.items():
        if sha(member(root, relative)) != expected:
            raise ValueError("A frozen member changed: " + relative)


def interval(values):
    if len(values) != 5 or any(not math.isfinite(x) for x in values):
        raise ValueError("Require all five finite actual returns, including fallbacks")
    mean, sd = statistics.mean(values), statistics.stdev(values)
    half = 2.7764451051977987 * sd / math.sqrt(5)
    return {"n": 5, "mean": mean, "sample_sd": sd,
            "approximate_t4_95_percent_interval": [mean - half, mean + half]}


def risk_ratios(values, anchor, delta=.10):
    denominators = {"angle": (1 + delta)**2 * anchor["theta_mse_rad2"],
                    "pq_voltage": (1 + delta)**2 * anchor["pq_vm_mse_pu2"],
                    "pbl": (1 + delta) * anchor["pbl_mean"],
                    "supervised": 2 * anchor["cmse"] + 1e-4}
    numerators = {"angle": values["theta_mse_rad2"], "pq_voltage": values["pq_vm_mse_pu2"],
                  "pbl": values["pbl_mean"], "supervised": values["cmse"]}
    if any(x <= 0 or not math.isfinite(x) for x in denominators.values()):
        raise ValueError("A zero/nonfinite N0 risk cannot be silently normalized")
    return {key: numerators[key] / denominator for key, denominator in denominators.items()}


def validate_run(root, budget=False):
    """Validate all 23 locks (or all 15 plus their complete A100 parent) before test/C IO."""
    root = Path(root).resolve(strict=True)
    plan = read(root / "plan.json")
    digest = sha(root / "plan.json")
    case = plan["company_case"]
    if (case not in COUNTS or plan["profiles"][case]["device"] != "cuda"
            or str(root) != plan["durable_output_root"]
            or set(plan["seeds"]) != set(SEEDS) or len(plan["seeds"]) != 5
            or plan["tuning_seed"] != 20261102
            or read(root / "BATCH_EXIT.json")["exit_code"] != 0):
        raise ValueError("Require a completed, identified company A100 block, never CPU partials")
    verify(root, plan["source_bindings"])
    profile = plan["profiles"][case]
    verify(Path(profile["workspace"]), plan["workspace_source_bindings"])
    verify(Path(profile["workspace"]), plan["workspace_input_bindings"])
    locked = read(root / "GLOBAL_RETURN_LOCK.json")
    methods = tuple(BUDGETS) if budget else METHODS
    expected = {(seed, method) for seed in SEEDS for method in methods}
    rows = locked["returns"]
    if (locked["plan_sha256"] != digest or locked["target_and_C_response_values_read"] != 0
            or len(rows) != len(expected)
            or {(row["seed"], row["method"]) for row in rows} != expected
            or len(plan["schedule"]) != len(expected)
            or {(row["seed"], row["method"]) for row in plan["schedule"]} != expected
            or set(plan["formal_methods"]) != set(methods)):
        raise ValueError("The complete prespecified matrix is not locked")
    if budget:
        parent_root, parent_full = validate_parent(plan)
    else:
        tuning = plan["tuning_schedule"]
        if (len(tuning) != 3 or {row["method"] for row in tuning} !=
                {"anchor_001", "anchor_01", "anchor_1"}
                or any(row["seed"] != 20261102 for row in tuning)):
            raise ValueError("Require three independent-seed L2 coefficients")
        selected = read(root / "ANCHOR_SELECTION.json")
        tuning_bindings = {"tune/" + row["name"] + ".return.json" for row in tuning}
        if (sha(root / "ANCHOR_SELECTION.json") != locked["anchor_selection_sha256"]
                or selected["plan_sha256"] != digest
                or set(selected["tuning_return_bindings"]) != tuning_bindings
                or not selected["selection_uses_only_unlabeled_development_PBL_after_normal_gate"]):
            raise ValueError("The development-only coefficient selection is incomplete")
        verify(root, selected["tuning_return_bindings"])
        for row in tuning:
            validate_cell(root, plan, read(root / "tune" / (row["name"] + ".return.json")), "tune")
        scores = [read(root / relative) for relative in tuning_bindings]
        coefficient = {"anchor_001": .001, "anchor_01": .01, "anchor_1": .1}
        if any(not math.isfinite(row["point"]["development_pbl"]) or row["point"]["development_pbl"] < 0
               for row in scores):
            raise ValueError("Invalid tuning observations")
        choice = min(scores, key=lambda row: (row["point"]["development_pbl"], coefficient[row["method"]]))
        if selected["selected_method"] != choice["method"]:
            raise ValueError("The selected L2 coefficient violates the fixed rule")
        for row in plan["qualification_schedule"]:
            if read(root / "operations" / ("qualify_" + row["name"]) / "train_EXIT.json")["exit_code"]:
                raise ValueError("The original qualification did not complete")
    verify(root, locked["return_bindings"])
    if set(locked["return_bindings"]) != {"formal/" + row["name"] + ".return.json" for row in rows}:
        raise ValueError("The actual return files are not completely bound")
    pairs = {row["seed"]: row for row in locked["paired_sampling"]}
    if set(pairs) != set(SEEDS) or len(locked["paired_sampling"]) != 5:
        raise ValueError("Require five paired streams")
    if budget:
        for seed in SEEDS:
            name = parent_full[seed]["name"]
            original = read(parent_root / "formal" / (name + ".locked.json"))
            if (pairs[seed]["sampling_sha256"] != sha(parent_root / "formal" / (name + ".sampling.jsonl"))
                    or pairs[seed]["calibration_sha256"] != original["calibration_sha256"]):
                raise ValueError("Budget streams/calibration differ from their matching A100 Full")
    for row in rows:
        validate_cell(root, plan, row, "formal")
        returned_path = root / "formal" / (row["name"] + ".return.json")
        if read(returned_path) != row:
            raise ValueError("The global lock does not contain the actual return")
        trained = read(root / "formal" / (row["name"] + ".locked.json"))
        if (sha(root / "formal" / (row["name"] + ".sampling.jsonl")) != pairs[row["seed"]]["sampling_sha256"]
                or trained["calibration_sha256"] != pairs[row["seed"]]["calibration_sha256"]):
            raise ValueError("All methods must share source/B/outage sampling and calibration per seed")
    return plan, locked


def validate_cell(root, plan, returned, phase):
    name = returned["name"]
    trained_path = root / phase / (name + ".locked.json")
    trained = read(trained_path)
    if (returned["plan_sha256"] != sha(root / "plan.json")
            or returned["target_response_or_C_selection"]
            or sha(member(root, returned["file"])) != returned["sha256"]
            or returned["training_lock_sha256"] != sha(trained_path)
            or trained["plan_sha256"] != returned["plan_sha256"]
            or trained["optimizer_updates"] != 1280
            or trained["seed"] != returned["seed"] or trained["arm"] != returned["method"]
            or returned["is_N0"] != (returned["tensor_sha256"] ==
                                     plan["cases"][plan["company_case"]]["anchor_tensor_sha256"])):
        raise ValueError("An actual complete 1280-update return changed")
    schedule = plan["tuning_schedule"] if phase == "tune" else plan["schedule"]
    if not any(all(returned[k] == row[k] for k in ("name", "seed", "method", "case")) for row in schedule):
        raise ValueError("The return is outside its declared schedule")
    if (phase == "tune" and returned["seed"] != 20261102):
        raise ValueError("Do not tune on a formal seed")
    selection = read(root / "selection" / name / "RESULT.json")
    if not selection["returns"]:
        raise ValueError("Actual fixed normal/development selection did not finish")
    from_delta = BUDGETS.get(returned["method"], .10)
    factors = {"theta_mse_rad2": round((1 + from_delta)**2, 8),
               "pq_vm_mse_pu2": round((1 + from_delta)**2, 8), "pbl_mean": 1 + from_delta}
    selection_input = read(root / "operations" / name / "selection_input.json")
    if trained["quality_factors"] != factors or selection_input["quality_factors"] != factors:
        raise ValueError("The training penalty and selection gate must use the same physical allowance")
    case = plan["company_case"]
    if trained["starting_tensor_sha256"] != plan["cases"][case]["anchor_tensor_sha256"]:
        raise ValueError("Every trajectory must start from the exact declared N0")
    if trained["target_response_update_or_selection_values"] != 0:
        raise ValueError("No target responses may enter learning or selection")
    for stage in ("train", "select"):
        if read(root / "operations" / (phase + "_" + name) / (stage + "_EXIT.json"))["exit_code"]:
            raise ValueError("An actual required child failed")


def completed_readout(root, run):
    root, run = Path(root), Path(run)
    receipt = read(root / "ANALYSIS_COMPLETE.json")
    scope = read(root / "READOUT_PLAN.json")
    if (receipt["readout_plan_sha256"] != sha(root / "READOUT_PLAN.json")
            or receipt["analysis_sha256"] != sha(root / "analysis.json")
            or scope["training_plan_sha256"] != sha(run / "plan.json")
            or scope["global_return_lock_sha256"] != sha(run / "GLOBAL_RETURN_LOCK.json")
            or receipt["observations_sha256"] != sha(root / "observations.jsonl")):
        raise ValueError("Require the complete unchanged fixed parent readout")
    verify(root, scope["source_bindings"])
    if receipt["optimizer_updates"] or receipt["new_solver_calls"]:
        raise ValueError("A fixed readout may not train or solve")
    return scope


def validate_parent(plan):
    root = Path(plan["matched_parent_run"])
    parent, locked = validate_run(root)
    readout = Path(plan["matched_parent_readout"])
    scope = completed_readout(readout, root)
    if (parent["company_case"] != "case118"
            or plan["matched_parent_plan_sha256"] != sha(root / "plan.json")
            or plan["matched_parent_readout_plan_sha256"] != sha(readout / "READOUT_PLAN.json")
            or plan["matched_parent_analysis_sha256"] != sha(readout / "analysis.json")
            or parent["profiles"] != plan["profiles"] or parent["cases"] != plan["cases"]
            or parent["workspace_input_bindings"] != plan["workspace_input_bindings"]
            or parent["workspace_source_bindings"] != plan["workspace_source_bindings"]
            or parent["reused_source_preparation"] != plan["reused_source_preparation"]
            or plan["budget_readout_population_bindings"] != scope["report_payload_bindings"]):
        raise ValueError("Budgets must inherit the same complete A100118 N0/backend/data and fixed cohorts")
    full = {row["seed"]: row for row in locked["returns"] if row["method"] == "full"}
    if set(full) != set(SEEDS):
        raise ValueError("Do not substitute old CPU Full or omit an unfavorable return")
    for relative, expected in parent["source_bindings"].items():
        if (relative not in ("company_protocol.py", "start_company.py")
                and plan["source_bindings"].get(relative) != expected):
            raise ValueError("The matched parent's numerical training/selection source changed")
    return root, full

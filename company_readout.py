"""Explicit completed-run adapter: no training, reselection, solver calls or CPU substitution."""

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import gc
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time

from company_protocol import (BUDGETS, COUNTS, METRICS, METHODS, SEEDS, WEIGHTS,
                              completed_readout, interval, member, read, risk_ratios,
                              sha, validate_parent, validate_run, verify, write)


HERE = Path(__file__).resolve().parent


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def report_payload(work, case):
    manifests = (["company_reports_case118.json", "additional_inputs.json"] if case == "case118"
                 else ["company_reports_case500.json", "report_inputs.json"])
    for name in manifests:
        manifest_path = HERE / name
        marker = work / (manifest_path.stem.upper() + "_INSTALLED.json")
        if marker.exists() and marker.read_bytes() == manifest_path.read_bytes():
            manifest = read(manifest_path)
            for index, payload in enumerate(manifest["payloads"]):
                if payload["role"].startswith("report_only_" + case):
                    path = work / "downloads" / manifest_path.stem / f"payload{index:02d}.tar.gz"
                    if path.stat().st_size != payload["bytes"] or sha(path) != payload["sha256"]:
                        raise ValueError("A fixed report payload differs")
                    return path, {case: payload["sha256"]}
    raise ValueError("Install the exact case-specific report manifest first")


def extract_reports(archive_path, destination, case):
    prefix = "report_only/" + case + "/"
    with tarfile.open(archive_path, "r:gz") as archive:
        for item in archive:
            if not item.isfile() or not item.name.startswith(prefix):
                raise ValueError("Only ordinary files from the declared report case are admitted")
            path = member(destination, item.name)
            path.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(item) as source, path.open("xb") as target:
                shutil.copyfileobj(source, target, 1024**2)
    return destination / "report_only" / case


def populations(reports, case):
    definitions = reports / "definitions"
    if case == "case118":
        target = [row for row in read(definitions / "readout_target_references.json") if row["case"] == case]
        normal = read(definitions / "readout_inherited_normal_references.json")[case]["records"]
        checks = {(row["domain"], row["group"], row["raw_sha256"]): row["input_sha256"]
                  for row in read(definitions / "readout_checks_case118.json")["checks"]}
    else:
        index = read(definitions / "reference_index.json")
        target, normal = index["records"]["target"], index["records"]["retained_normal"]
        checks = {}
    if (len(target), len(normal)) != COUNTS[case]:
        raise ValueError("The fixed evaluation cohorts are incomplete")
    return {"target": target, "retained_normal": normal}, checks


def aligned_metrics(prediction, inputs, reference):
    """The V3 identity-weighted MSE and four unchanged risks; rPBL is not reported."""
    import torch
    from core.utils.pf_objectives import per_graph_canos_mse, per_graph_power_balance_metrics

    size, bus_batch = inputs.num_graphs, inputs["bus"].batch

    def total(values, indices):
        return values.new_zeros(size).scatter_add_(0, indices, values)

    sums = prediction["bus"].new_zeros(size, dtype=torch.float64)
    counts = sums.clone()
    for kind, key in (("PV", "PV"), ("PQ", "PQ"), ("slack", "slack"), ("branch", "edge_preds")):
        if kind == "branch":
            target = reference["bus", "branch", "bus"].edge_label
            indices = bus_batch[inputs["bus", "branch", "bus"].edge_index[0]]
        else:
            target, indices = reference[kind].y, inputs[kind].batch
        errors = (prediction[key].double() - target.double()).square()
        sums += total(errors.sum(dim=1), indices)
        counts += total(torch.full_like(errors[:, 0], errors.shape[1]), indices)
    ns_mask = torch.ones(inputs["bus"].num_nodes, dtype=torch.bool, device=bus_batch.device)
    ns_mask[inputs["slack", "slack_link", "bus"].edge_index[1]] = False
    angle = (prediction["bus"][:, 0].double() - reference["bus"].bus_voltages[:, 0].double()).square()
    ns_count = total(ns_mask.to(torch.float64), bus_batch)
    pq_bus = inputs["PQ", "PQ_link", "bus"].edge_index[1]
    vm = (prediction["bus"][pq_bus, 1].double() - reference["bus"].bus_voltages[pq_bus, 1].double()).square()
    pq_count = total(torch.ones_like(vm), bus_batch[pq_bus])
    if any(bool((value <= 0).any()) for value in (counts, ns_count, pq_count)):
        raise ValueError("Empty physical/output coordinate set")
    measured = {"mse_entry": sums / counts, "cmse": per_graph_canos_mse(prediction, reference),
                "pbl_mean": per_graph_power_balance_metrics(prediction, inputs)["pbl_mean"],
                "theta_mse_rad2": total(angle * ns_mask, bus_batch) / ns_count,
                "pq_vm_mse_pu2": total(vm, bus_batch[pq_bus]) / pq_count}
    if any(value.shape != (size,) or not bool(torch.isfinite(value).all())
           or bool((value < 0).any()) for value in measured.values()):
        raise FloatingPointError("Invalid aligned fixed-model metric")
    return {key: value.double().cpu().tolist() for key, value in measured.items()}


def csv_rows(path, rows):
    if not rows:
        raise ValueError("An empty table is not a completed result")
    with Path(path).open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def average(rows):
    if not rows:
        raise ValueError("Missing complete fixed population")
    return {key: math.fsum(row[key] for row in rows) / len(rows) for key in METRICS}


def target_average(rows):
    identities, families = defaultdict(list), {}
    for row in rows:
        identity = row["identity"]
        if identity in families and families[identity] != row["family"]:
            raise ValueError("A fixed identity changed family")
        families[identity] = row["family"]
        identities[identity].append(row["metrics"])
    grouped = defaultdict(list)
    for identity, values in identities.items():
        grouped[families[identity]].append(average(values))
    if set(grouped) != set(WEIGHTS):
        raise ValueError("The fixed research mixture lacks a family")
    family_means = {key: average(values) for key, values in grouped.items()}
    return {metric: math.fsum(WEIGHTS[family] * values[metric] for family, values in family_means.items())
            for metric in METRICS}


def aggregate(observations, tensors, case):
    populations_by_tensor, seen = defaultdict(list), set()
    for row in observations:
        key = row["tensor_sha256"], row["domain"], row["input_sha256"]
        if (key in seen or row["case"] != case or set(row["metrics"]) != set(METRICS)
                or any(not math.isfinite(value) or value < 0 for value in row["metrics"].values())):
            raise ValueError("Duplicate or invalid per-condition observation")
        seen.add(key)
        populations_by_tensor[key[:2]].append(row)
    expected = {(tensor, domain) for tensor in tensors for domain in ("target", "retained_normal")}
    if set(populations_by_tensor) != expected:
        raise ValueError("Every locked tensor requires both complete populations")
    values, identities = {}, {}
    for tensor in tensors:
        target = populations_by_tensor[tensor, "target"]
        normal = populations_by_tensor[tensor, "retained_normal"]
        if (len(target), len(normal)) != COUNTS[case]:
            raise ValueError("Never drop or replace an evaluation condition")
        for domain, rows in (("target", target), ("retained_normal", normal)):
            cohort = {(row["input_sha256"], row["identity"], row["group"], row["family"]) for row in rows}
            if domain in identities and identities[domain] != cohort:
                raise ValueError("All models must use identical fixed cohorts")
            identities[domain] = cohort
        values[tensor] = {"target": target_average(target), "C": average([row["metrics"] for row in normal])}
    return values


def analyze(scope, plan, returns, observations, costs):
    case = plan["company_case"]
    n0 = plan["cases"][case]["anchor_tensor_sha256"]
    values = aggregate(observations, {n0, *(row["tensor_sha256"] for row in returns)}, case)
    anchor, logical, indexed = values[n0], [], {}
    for returned in returns:
        scores = values[returned["tensor_sha256"]]
        method, seed = returned["method"], returned["seed"]
        delta = BUDGETS.get(method, .10)
        ratios = risk_ratios(scores["C"], anchor["C"], delta)
        changes = {key: scores["C"][key] / anchor["C"][key] - 1
                   for key in ("theta_mse_rad2", "pq_vm_mse_pu2", "pbl_mean", "cmse")}
        row = {"case": case, "seed": seed, "method": method, "physical_allowance": delta,
               "tensor_sha256": returned["tensor_sha256"], "is_N0": returned["is_N0"],
               "return_kind": returned["kind"],
               **{f"target_{key}": value for key, value in scores["target"].items()},
               **{f"C_{key}": value for key, value in scores["C"].items()},
               **{f"C_change_fraction_{key}": value for key, value in changes.items()},
               **{f"C_budget_ratio_{key}": value for key, value in ratios.items()},
               "C_all_four_budgets_pass": max(ratios.values()) <= 1 + 1e-12,
               "combined_train_seconds": costs[seed, method]["train"],
               "selection_seconds": costs[seed, method]["select"],
               "adapt_plus_selection_seconds": sum(costs[seed, method].values())}
        logical.append(row)
        indexed[seed, method] = row
    methods = tuple(BUDGETS) + ("full",) if scope["budget"] else METHODS
    summary = {}
    for method in methods:
        rows = [indexed[seed, method] for seed in SEEDS]
        summary[method] = {
            "target_mse_entry": interval([row["target_mse_entry"] for row in rows]),
            "target_pbl_mean": interval([row["target_pbl_mean"] for row in rows]),
            "max_four_C_budget_ratio_all_five_returns": max(
                row["C_budget_ratio_" + risk] for row in rows for risk in ("angle", "pq_voltage", "pbl", "supervised")),
            "max_four_C_change_fraction_all_five_returns": max(
                row["C_change_fraction_" + risk] for row in rows
                for risk in ("theta_mse_rad2", "pq_vm_mse_pu2", "pbl_mean", "cmse")),
            "unchanged_N0_count": sum(row["is_N0"] for row in rows),
            "total_adapt_plus_selection_seconds": sum(row["adapt_plus_selection_seconds"] for row in rows),
        }
    contrasts = []
    if not scope["budget"]:
        for comparator in METHODS[1:]:
            for metric in ("mse_entry", "pbl_mean"):
                differences = [indexed[seed, comparator]["target_" + metric] -
                               indexed[seed, "full"]["target_" + metric] for seed in SEEDS]
                contrasts.append({"comparator": comparator, "metric": metric,
                                  "effect": "comparator minus Full; positive favors Full", **interval(differences)})
    return {"case": case, "N0": anchor, "all_logical_returns": logical, "method_summary": summary,
            "paired_raw_contrasts": contrasts, "rPBL_not_implemented_or_imputed": True,
            "statistical_scope": "Five adaptation seeds conditional on one pretrained checkpoint, globally selected L2 coefficient and fixed research mixture. Descriptive non-simultaneous approximate t4 95% intervals, no multiplicity-adjusted significance.",
            "C_report_only_no_reselection": True, "benchmark_previously_informed_development": True,
            "training_wall_clock_includes_training_time_gates": True,
            "selection_seconds_include_gate_development_and_interpolation_not_pure_gate": True,
            "pure_gate_wall_clock_not_separately_instrumented": True,
            "no_AC_solver_or_end_to_end_speedup_claim": True}


def child_costs(run, plan):
    costs, records = {}, []
    for phase, schedule in (("qualify", plan["qualification_schedule"]),
                            ("tune", plan["tuning_schedule"]), ("formal", plan["schedule"])):
        for row in schedule:
            stage_cost = {}
            for stage in (("train",) if phase == "qualify" else ("train", "select")):
                path = run / "operations" / (phase + "_" + row["name"]) / (stage + "_EXIT.json")
                receipt = read(path)
                if receipt["exit_code"] or not math.isfinite(receipt["elapsed_seconds"]) or receipt["elapsed_seconds"] < 0:
                    raise ValueError("A required child did not complete")
                stage_cost[stage] = receipt["elapsed_seconds"]
                records.append({"phase": phase, "name": row["name"], "stage": stage, **receipt})
            if phase == "formal":
                costs[row["seed"], row["method"]] = stage_cost
    return costs, records


def freeze(run, work, output, budget):
    plan, locked = validate_run(run, budget)
    if output == work or work not in output.parents or output == run or run in output.parents:
        raise ValueError("Use a new independent readout directory on the selected data volume, outside the frozen run")
    case = plan["company_case"]
    archive, bindings = report_payload(work, case)
    parent_scope = None
    if budget:
        parent_run, full = validate_parent(plan)
        parent_scope = completed_readout(Path(plan["matched_parent_readout"]), parent_run)
        if bindings != parent_scope["report_payload_bindings"]:
            raise ValueError("Budget readout must use exactly the parent's fixed cohorts")
    anchor_path = Path(plan["profiles"][case]["workspace"]) / "runs" / "tcpa_strong_anchor_v1" / "anchor_assets" / case / "s1" / "model.pt"
    if sha(anchor_path) != plan["cases"][case]["anchor_checkpoint_sha256"]:
        raise ValueError("The original N0 checkpoint changed")
    n0 = plan["cases"][case]["anchor_tensor_sha256"]
    endpoints = {n0: {"path": str(anchor_path), "sha256": sha(anchor_path), "tensor_sha256": n0}}
    for returned in locked["returns"]:
        endpoints.setdefault(returned["tensor_sha256"], {
            "path": str(member(run, returned["file"])), "sha256": returned["sha256"],
            "tensor_sha256": returned["tensor_sha256"]})
    scope = {
        "training_run": str(run), "training_plan_sha256": sha(run / "plan.json"),
        "utc_frozen": datetime.now(timezone.utc).isoformat(), "owner_pid": os.getpid(),
        "global_return_lock_sha256": sha(run / "GLOBAL_RETURN_LOCK.json"), "case": case, "budget": budget,
        "backend": plan["profiles"][case], "report_payload_bindings": bindings,
        "report_archive": str(archive), "target_count": COUNTS[case][0], "normal_C_count": COUNTS[case][1],
        "logical_return_count": 20,
        "distinct_tensors": list(endpoints.values()), "fixed_selection_complete_before_any_test_C_open": True,
        "source_bindings": {name: sha(HERE / name) for name in
                            ("company_readout.py", "company_protocol.py", "start_company.py",
                             "export_results.py", "PROTOCOL_LINEAGE.json")},
        "fresh_same_backend_N0": not budget, "parent_readout_reused_for_budget": parent_scope,
        "maximum_seconds": 14400, "new_optimizer_updates": 0, "new_solver_calls": 0,
    }
    output.mkdir(parents=True, exist_ok=False)
    for name in scope["source_bindings"]:
        shutil.copyfile(HERE / name, output / name)
    write(output / "READOUT_PLAN.json", scope)
    return scope, plan, locked


def execute(scope, plan, locked, output):
    started = time.monotonic()
    run, case = Path(scope["training_run"]), scope["case"]
    sys.path.insert(0, str(run))
    original = module("company_frozen_selection_runtime", run / "selection_reference.py")
    rt, _, _, _, _, _, Counter, versions = original.runtime(plan, case)
    from backoff_core import tensor_sha
    from common import input_fingerprint
    from torch_geometric.data import Batch
    from research.tcpa_objective_budget_ablation.src.run_confirmation_evaluation import build_labeled_graph

    torch = rt.torch
    profile = plan["profiles"][case]
    device = torch.device(profile["device"])
    guarded_profile = dict(profile, maximum_rss_bytes=profile["maximum_worker_rss_bytes"],
                           remaining_ram_floor_bytes=profile["ram_reserve_bytes"],
                           protected_disk_bytes=plan["minimum_free_disk_bytes"])
    guard = original.Guard(guarded_profile, scope["maximum_seconds"])
    reports = extract_reports(Path(scope["report_archive"]), output, case)
    populations_by_domain, prior = populations(reports, case)
    config, source, _, fit, development = rt.datasets(case)
    forbidden = {row["identity_signature"] for dataset in fit + development for row in dataset.records}
    graphs, checks, fingerprints = {}, [], set()
    for domain, records in populations_by_domain.items():
        graphs[domain] = []
        for record in records:
            guard.check()
            path = member(reports / domain, record["file"])
            if sha(path) != record["sha256"] or (domain == "target" and record["identity"] in forbidden):
                raise ValueError("A fixed reference changed or overlaps learning")
            inputs, scan = rt.SYMBOLS["build_strict_generated_canos"](path)
            rt.SYMBOLS["assert_target_scrubbed"](inputs)
            fingerprint = input_fingerprint(inputs)
            if scan["retained_target_response_value_count"] or fingerprint in fingerprints:
                raise ValueError("An input includes a response or repeats a condition")
            if prior and prior[domain, record["group"], record["sha256"]] != fingerprint:
                raise ValueError("The historical strict input fingerprint changed")
            fingerprints.add(fingerprint)
            truth = build_labeled_graph(path, rt.SYMBOLS)
            teacher = {key: truth[key].y for key in ("PV", "PQ", "slack")}
            teacher.update(bus=truth["bus"].bus_voltages, edge_preds=truth["bus", "branch", "bus"].edge_label)
            residual = float(rt.per_graph_power_balance_metrics(teacher, inputs)["pbl_mean"].item())
            if not math.isfinite(residual) or residual > 1e-4:
                raise ValueError("A reference fails the unchanged physical-consistency condition")
            checks.append({"domain": domain, "group": record["group"], "raw_sha256": record["sha256"],
                           "input_sha256": fingerprint, "reference_pbl": residual})
            graphs[domain].append((inputs, truth, record, fingerprint))
    fit = development = None
    gc.collect()
    inherited_observations, parent_returns, parent_costs, parent_cost_records = [], [], {}, []
    if scope["budget"]:
        parent_run, full = validate_parent(plan)
        parent_readout = Path(plan["matched_parent_readout"])
        parent_returns = list(full.values())
        inherited_tensors = {plan["cases"][case]["anchor_tensor_sha256"],
                             *(row["tensor_sha256"] for row in parent_returns)}
        with (parent_readout / "observations.jsonl").open(encoding="utf-8") as stream:
            inherited_observations = [row for line in stream if
                                      (row := json.loads(line))["tensor_sha256"] in inherited_tensors]
        parent_costs, parent_cost_records = child_costs(parent_run, read(parent_run / "plan.json"))
    else:
        inherited_tensors = set()
    endpoints = [row for row in scope["distinct_tensors"] if row["tensor_sha256"] not in inherited_tensors]
    count = len(endpoints) * sum(COUNTS[case])
    if count > plan["maximum_common_readout_graphs"]:
        raise ValueError("The exact fixed-model readout exceeds the frozen graph allowance")
    counter = Counter(output / "execution.jsonl", count, 0)
    model = counter.attach(rt.model_from_anchor(config, source)).to(device).eval().requires_grad_(False)
    observations = list(inherited_observations)

    def protect(_module, _args):
        guard.check()
        if torch.cuda.mem_get_info()[0] < profile["minimum_gpu_free_bytes"]:
            raise MemoryError("Preserve the frozen GPU free-memory margin")

    model.register_forward_pre_hook(protect)
    try:
        with torch.inference_mode(), (output / "observations.jsonl").open("x", encoding="utf-8") as stream:
            for row in inherited_observations:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            for endpoint in endpoints:
                if sha(Path(endpoint["path"])) != endpoint["sha256"]:
                    raise ValueError("A locked model changed")
                state = original.load_state({"kind": "file", **endpoint}, torch)
                if tensor_sha(state) != endpoint["tensor_sha256"]:
                    raise ValueError("A locked tensor identity changed")
                model.load_state_dict(state, strict=True)
                counter.arm = endpoint["tensor_sha256"]
                for domain, population in graphs.items():
                    for offset in range(0, len(population), 32):
                        group = population[offset:offset + 32]
                        inputs = Batch.from_data_list([row[0] for row in group]).to(device)
                        truth = Batch.from_data_list([row[1] for row in group]).to(device)
                        measured = aligned_metrics(rt.split_outputs(model(inputs)), inputs, truth)
                        for index, (_, _, record, fingerprint) in enumerate(group):
                            row = {"case": case, "tensor_sha256": endpoint["tensor_sha256"], "domain": domain,
                                   "group": record["group"], "identity": record["identity"], "family": record["family"],
                                   "raw_sha256": record["sha256"], "input_sha256": fingerprint,
                                   "metrics": {name: measured[name][index] for name in METRICS}}
                            stream.write(json.dumps(row, allow_nan=False) + "\n")
                            observations.append(row)
                        stream.flush()
                del state
            os.fsync(stream.fileno())
        if counter.graphs != count or counter.attempted_graphs != count or counter.updates:
            raise ValueError("The exact zero-update fixed readout was not completed")
    finally:
        counter.close()
    costs, cost_records = child_costs(run, plan)
    costs.update({key: value for key, value in parent_costs.items() if key[1] == "full"})
    analysis = analyze(scope, plan, locked["returns"] + parent_returns, observations, costs)
    analysis.update(child_cost_records=cost_records, reused_parent_cost_records=parent_cost_records,
                    model_graph_forwards=count,
                    readout_elapsed_seconds=time.monotonic() - started, runtime_versions=versions,
                    peak_rss_bytes=guard.peak_rss, forward_latency_not_measured=True,
                    optimizer_updates=0, new_solver_calls=0)
    # Revalidate every authority after inference; scoring cannot change a lock.
    validate_run(run, scope["budget"])
    verify(output, scope["source_bindings"])
    write(output / "reference_checks.json", {"checks": checks})
    csv_rows(output / "logical_returns.csv", analysis["all_logical_returns"])
    write(output / "analysis.json", analysis)
    write(output / "ANALYSIS_COMPLETE.json", {
        "readout_plan_sha256": sha(output / "READOUT_PLAN.json"), "analysis_sha256": sha(output / "analysis.json"),
        "observations_sha256": sha(output / "observations.jsonl"), "optimizer_updates": 0, "new_solver_calls": 0,
        "all_seeds_and_fallbacks_preserved": True, "paper_integration_required": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Explicit completed old500 or new118 directory")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New independent readout directory on data volume")
    parser.add_argument("--budget", action="store_true")
    args = parser.parse_args()
    run, work, output = args.run_dir.resolve(strict=True), args.work_dir.resolve(strict=True), args.output_dir.resolve()
    # No report archive is opened until validate_run has checked every required lock.
    scope, plan, locked = freeze(run, work, output, args.budget)
    os.environ["CUDA_VISIBLE_DEVICES"] = plan["profiles"][plan["company_case"]]["cuda_visible_devices"]
    started = time.monotonic()
    try:
        from start_company import require_idle_gpu
        require_idle_gpu(plan["profiles"][plan["company_case"]]["cuda_visible_devices"])
        scratch = work / "scratch"
        scratch.mkdir(exist_ok=True)
        for key in ("TMPDIR", "TMP", "TEMP"):
            os.environ[key] = str(scratch)
        execute(scope, plan, locked, output)
    except Exception as error:
        write(output / "FAILURE.json", {"error": repr(error), "automatic_retry": False,
                                      "elapsed_seconds": time.monotonic() - started,
                                      "no_N0_restart_authorized": True})
        raise


if __name__ == "__main__":
    main()

"""IO-only protocol/runner tests: synthetic files, no torch, forward, PF or scientific jobs."""

import json
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import company_budget_supervisor
import company_protocol as protocol
import company_readout
import export_results
import start_budget
import start_company


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def factors(delta):
    return {"theta_mse_rad2": round((1 + delta)**2, 8),
            "pq_vm_mse_pu2": round((1 + delta)**2, 8), "pbl_mean": 1 + delta}


def fixture(root, case="case500"):
    workspace = root / "workspace"
    workspace.mkdir()
    (workspace / "input").write_bytes(b"synthetic immutable input")
    anchor = workspace / "runs" / "tcpa_strong_anchor_v1" / "anchor_assets" / case / "s1" / "model.pt"
    anchor.parent.mkdir(parents=True)
    anchor.write_bytes(b"synthetic-N0-no-neural-computation")
    schedule = [{"name": f"{case}_seed{seed}_{method}", "seed": seed, "method": method, "case": case}
                for seed in protocol.SEEDS for method in protocol.METHODS]
    tuning = [{"name": f"{case}_seed20261102_{method}", "seed": 20261102, "method": method, "case": case}
              for method in ("anchor_001", "anchor_01", "anchor_1")]
    plan = {"company_case": case, "durable_output_root": str(root), "seeds": list(protocol.SEEDS),
            "profiles": {case: {"device": "cuda", "workspace": str(workspace), "cpu_affinity": [0],
                                "cuda_visible_devices": "0"}},
            "cases": {case: {"anchor_tensor_sha256": "n0", "anchor_checkpoint_sha256": protocol.sha(anchor)}},
            "tuning_seed": 20261102, "schedule": schedule, "tuning_schedule": tuning,
            "qualification_schedule": [], "formal_methods": list(protocol.METHODS),
            "source_bindings": {}, "workspace_source_bindings": {},
            "workspace_input_bindings": {"input": protocol.sha(workspace / "input")},
            "reused_source_preparation": {case: {"cache_sha256": "fixedB"}}}
    save(root / "plan.json", plan)
    save(root / "BATCH_EXIT.json", {"exit_code": 0})
    digest, formal, pairs = protocol.sha(root / "plan.json"), [], []
    for seed in protocol.SEEDS:
        pairs.append({"seed": seed, "sampling_sha256": None, "calibration_sha256": "calibration"})
    for phase, rows in (("tune", tuning), ("formal", schedule)):
        for i, row in enumerate(rows):
            name, seed, method = row["name"], row["seed"], row["method"]
            state = root / phase / name / "RETURN.pt"
            state.parent.mkdir(parents=True)
            state.write_bytes(("synthetic-state-" + name).encode())
            sample = root / phase / (name + ".sampling.jsonl")
            sample.write_text(str(seed), encoding="utf-8")
            trained = {"optimizer_updates": 1280, "plan_sha256": digest, "seed": seed, "arm": method,
                       "quality_factors": factors(.1), "starting_tensor_sha256": "n0",
                       "target_response_update_or_selection_values": 0, "calibration_sha256": "calibration"}
            training_path = root / phase / (name + ".locked.json")
            save(training_path, trained)
            returned = {**row, "plan_sha256": digest, "target_response_or_C_selection": False,
                        "training_lock_sha256": protocol.sha(training_path),
                        "file": state.relative_to(root).as_posix(), "sha256": protocol.sha(state),
                        "tensor_sha256": "n0" if i == 0 else name, "is_N0": i == 0,
                        "point": {"development_pbl": i + 1}, "kind": "SYNTHETIC_RETURN"}
            save(root / phase / (name + ".return.json"), returned)
            save(root / "selection" / name / "RESULT.json", {"returns": [{"synthetic": True}]})
            save(root / "operations" / name / "selection_input.json", {"quality_factors": factors(.1)})
            for stage in ("train", "select"):
                save(root / "operations" / (phase + "_" + name) / (stage + "_EXIT.json"),
                     {"exit_code": 0, "elapsed_seconds": 2 if stage == "train" else 1})
            if phase == "formal":
                formal.append(returned)
                next(row for row in pairs if row["seed"] == seed)["sampling_sha256"] = protocol.sha(sample)
    selection = {"plan_sha256": digest, "selected_method": "anchor_001",
                 "selection_uses_only_unlabeled_development_PBL_after_normal_gate": True,
                 "tuning_return_bindings": {"tune/" + row["name"] + ".return.json":
                                            protocol.sha(root / "tune" / (row["name"] + ".return.json"))
                                            for row in tuning}}
    save(root / "ANCHOR_SELECTION.json", selection)
    save(root / "GLOBAL_RETURN_LOCK.json", {
        "plan_sha256": digest, "target_and_C_response_values_read": 0, "returns": formal,
        "return_bindings": {"formal/" + row["name"] + ".return.json":
                            protocol.sha(root / "formal" / (row["name"] + ".return.json")) for row in schedule},
        "paired_sampling": pairs, "anchor_selection_sha256": protocol.sha(root / "ANCHOR_SELECTION.json")})
    return plan


class ClosureTests(unittest.TestCase):
    def test_completed_old500_is_accepted_without_modifying_its_frozen_authority(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            fixture(root)
            before = protocol.sha(root / "plan.json")
            plan, locked = protocol.validate_run(root)
            self.assertEqual(plan["company_case"], "case500")
            self.assertEqual(len(locked["returns"]), 20)
            self.assertEqual(protocol.sha(root / "plan.json"), before)

    def test_external_readout_freezes_new_authority_only_after_all_old500_locks(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            root = base / "training"
            root.mkdir()
            fixture(root)
            before = protocol.sha(root / "plan.json")
            output = base / "external_readout"
            with patch("company_readout.report_payload", return_value=(root / "unopened.tar.gz", {"case500": "fixed"})):
                scope, _, _ = company_readout.freeze(root, base, output, False)
            self.assertEqual(scope["logical_return_count"], 20)
            self.assertTrue(scope["fresh_same_backend_N0"])
            self.assertEqual(scope["normal_C_count"], 1019)
            protocol.verify(output, scope["source_bindings"])
            self.assertFalse((output / "report_only").exists())
            self.assertEqual(protocol.sha(root / "plan.json"), before)
            with self.assertRaisesRegex(ValueError, "independent"):
                company_readout.freeze(root, base, root / "do_not_write_old_run", False)

    def test_complete_manual_export_includes_flat_conditions_and_readback_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            fixture(root)
            output = root / "readout"
            output.mkdir()
            save(output / "READOUT_PLAN.json", {
                "training_plan_sha256": protocol.sha(root / "plan.json"),
                "global_return_lock_sha256": protocol.sha(root / "GLOBAL_RETURN_LOCK.json"),
                "source_bindings": {}, "report_payload_bindings": {"case500": "synthetic-fixed"},
                "distinct_tensors": [{"path": "site-path-not-exported", "sha256": "synthetic", "tensor_sha256": "n0"}]})
            save(output / "analysis.json", {"synthetic_IO_test_only": True})
            save(output / "reference_checks.json", {"checks": []})
            observation = {"case": "case500", "tensor_sha256": "n0", "domain": "target",
                           "group": "synthetic", "identity": "synthetic", "family": "generator",
                           "raw_sha256": "raw", "input_sha256": "input",
                           "metrics": {key: 1 for key in protocol.METRICS}}
            (output / "observations.jsonl").write_text(json.dumps(observation) + "\n")
            (output / "execution.jsonl").write_text('{"synthetic_IO_test_only": true}\n')
            (output / "logical_returns.csv").write_text("synthetic_IO_test_only\ntrue\n")
            save(output / "ANALYSIS_COMPLETE.json", {
                "readout_plan_sha256": protocol.sha(output / "READOUT_PLAN.json"),
                "analysis_sha256": protocol.sha(output / "analysis.json"),
                "observations_sha256": protocol.sha(output / "observations.jsonl"),
                "optimizer_updates": 0, "new_solver_calls": 0})
            target = root / "complete.tar.gz"
            result = export_results.export(root, output, target)
            self.assertTrue(result["scientific_result_eligible"])
            with tarfile.open(target) as archive:
                flat = archive.extractfile("readout/per_condition.csv").read().decode("utf-8-sig")
                manifest = json.loads(archive.extractfile("MANIFEST.json").read())
                self.assertIn("raw_sha256,input_sha256,mse_entry", flat)
                self.assertEqual(manifest["readout_members"]["observations.jsonl"]["sha256"],
                                 protocol.sha(output / "observations.jsonl"))
                self.assertNotIn("path", manifest["readout_model_bindings"][0])

    def test_no_report_io_when_any_tune_return_or_batch_is_incomplete(self):
        for failure in ("tune", "batch", "partial", "sampling", "penalty", "gate", "N0"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                base = Path(directory).resolve()
                root = base / "training"
                root.mkdir()
                plan = fixture(root)
                first = plan["schedule"][0]["name"]
                if failure == "tune":
                    (root / "tune" / (plan["tuning_schedule"][0]["name"] + ".return.json")).unlink()
                elif failure == "batch":
                    save(root / "BATCH_EXIT.json", {"exit_code": 1})
                elif failure == "partial":
                    lock = protocol.read(root / "GLOBAL_RETURN_LOCK.json")
                    lock["returns"].pop()
                    save(root / "GLOBAL_RETURN_LOCK.json", lock)
                elif failure == "sampling":
                    (root / "formal" / (first + ".sampling.jsonl")).write_text("wrong stream")
                elif failure == "gate":
                    save(root / "operations" / first / "selection_input.json", {"quality_factors": factors(0)})
                else:
                    path = root / "formal" / (first + ".locked.json")
                    trained = protocol.read(path)
                    if failure == "penalty":
                        trained["quality_factors"] = factors(.2)
                    else:
                        trained["starting_tensor_sha256"] = "old-CPU"
                    save(path, trained)
                with patch("company_readout.report_payload") as payload:
                    with self.assertRaises((ValueError, FileNotFoundError)):
                        company_readout.freeze(root, base, base / "readout", False)
                    payload.assert_not_called()
                    self.assertFalse((base / "readout").exists())

    def test_case118_requires_existing500_identity_before_any_gpu_command(self):
        with patch("sys.argv", ["start_company.py", "--case", "case118", "--work-dir", "unused",
                                "--run-id", "unused"]), patch("start_company.subprocess.check_output") as gpu:
            with self.assertRaisesRegex(ValueError, "after-run-dir"):
                start_company.main()
            gpu.assert_not_called()

    def test_gpu_owner_is_protected_without_killing_or_restarting(self):
        with patch("start_company.subprocess.check_output", return_value="1234\n"):
            with self.assertRaisesRegex(RuntimeError, "do not interrupt"):
                start_company.require_idle_gpu("0")

    def test_budget_plan_is15_actual_trajectories_with_no_newFull_or_tunes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            parent = fixture(root, "case118")
            scope = {"report_payload_bindings": {"case118": "samefixedC"}}
            with patch("start_budget.completed_readout", return_value=scope):
                (root / "READOUT_PLAN.json").write_text("{}")
                (root / "analysis.json").write_text("{}")
                plan = start_budget.budget_plan(parent, root, root, root / "budget")
            self.assertEqual(len(plan["schedule"]), 15)
            self.assertEqual({row["method"] for row in plan["schedule"]}, set(protocol.BUDGETS))
            self.assertEqual(plan["profiles"], parent["profiles"])
            self.assertEqual(plan["cases"], parent["cases"])
            self.assertEqual(plan["qualification_schedule"], [])
            self.assertEqual(plan["tuning_schedule"], [])
            self.assertFalse(plan["budget_parent_Full_retrained"])

    def test_budget_runner_executes_train_then_select_for_each_of15_and_never_parent(self):
        rows = [{"seed": seed, "method": method, "name": f"{seed}_{method}"}
                for seed in protocol.SEEDS for method in protocol.BUDGETS]
        plan = {"schedule": rows, "qualification_schedule": [], "tuning_schedule": [],
                "profiles": {"case118": {"cpu_affinity": [0]}}, "maximum_batch_seconds": 1}
        calls = []
        inherited = SimpleNamespace(run_child=lambda p, row, phase, deadline, selector=False:
                                    calls.append((row["name"], selector)))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "plan.json").write_text("{}")
            (root / "GLOBAL_RETURN_LOCK.json").write_text("{}")
            with patch.dict("sys.modules", {"review_supervisor": inherited}), \
                    patch("company_budget_supervisor.HERE", root), \
                    patch("company_budget_supervisor.validate_parent"), \
                    patch("company_budget_supervisor.lock") as locked, \
                    patch("company_budget_supervisor.os.sched_setaffinity", create=True):
                company_budget_supervisor.run(plan)
            self.assertEqual(calls, [(row["name"], selector) for row in rows for selector in (False, True)])
            locked.assert_called_once_with(plan)
            self.assertFalse((root / "formal").exists())

    def test_approximate_paired_intervals_preserve_negative_and_fallback_values(self):
        values = [-4, -2, 0, 2, 4]
        summary = protocol.interval(values)
        self.assertEqual(summary["mean"], 0)
        self.assertGreater(summary["sample_sd"], 0)
        self.assertLess(summary["approximate_t4_95_percent_interval"][0], 0)
        with self.assertRaises(ValueError):
            protocol.interval(values[:4])

    def test_normal_bound_changes_use_squared_state_and_linear_PBL_sup_unchanged(self):
        anchor = {name: 1.0 for name in protocol.METRICS}
        for delta in (0, .05, .1, .2):
            current = dict(anchor, theta_mse_rad2=(1 + delta)**2,
                           pq_vm_mse_pu2=(1 + delta)**2, pbl_mean=1 + delta, cmse=2.0001)
            self.assertTrue(all(abs(value - 1) < 1e-12
                                for value in protocol.risk_ratios(current, anchor, delta).values()))

    def test_identity_then_family_aggregation_not_naive_graph_mean(self):
        rows = []
        for family in protocol.WEIGHTS:
            for condition, value in enumerate((1, 3) if family == "generator" else (2,)):
                rows.append({"identity": family, "family": family,
                             "metrics": {name: value for name in protocol.METRICS}})
        result = company_readout.target_average(rows)
        self.assertEqual(set(result.values()), {2.0})

    def test_complete_cohorts_cannot_be_cherry_picked_or_duplicated(self):
        row = {"case": "case118", "tensor_sha256": "n0", "domain": "target",
               "input_sha256": "one", "metrics": {key: 1 for key in protocol.METRICS}}
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            company_readout.aggregate([row, row], {"n0"}, "case118")
        with self.assertRaisesRegex(ValueError, "both complete populations"):
            company_readout.aggregate([row], {"n0"}, "case118")

    def test_failure_export_is_ineligible_and_never_includes_native_state_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            fixture(root)
            save(root / "BATCH_EXIT.json", {"exit_code": 1, "error": "synthetic disk protection"})
            target = root / "manual.tar.gz"
            result = export_results.export(root, None, target)
            self.assertFalse(result["scientific_result_eligible"])
            self.assertFalse(result["uploaded"])
            with tarfile.open(target) as archive:
                self.assertFalse(any(name.endswith(".pt") for name in archive.getnames()))
                manifest = json.loads(archive.extractfile("MANIFEST.json").read())
                self.assertTrue(manifest["failure_only_archive_must_not_populate_paper"])
                self.assertTrue(any(row["area"] == "native_retained_not_in_small_archive"
                                    for row in manifest["members"]))
            with self.assertRaises(FileExistsError):
                export_results.export(root, None, target)
            failed_readout = root / "failed_readout"
            save(failed_readout / "FAILURE.json", {"error": "synthetic failed readout", "elapsed_seconds": 1})
            archive_path = root / "readout_failure.tar.gz"
            returned = export_results.export(root, failed_readout, archive_path)
            self.assertFalse(returned["scientific_result_eligible"])
            with tarfile.open(archive_path) as archive:
                manifest = json.loads(archive.extractfile("MANIFEST.json").read())
                self.assertEqual(manifest["failed_readout_members"]["FAILURE.json"]["sha256"],
                                 protocol.sha(failed_readout / "FAILURE.json"))

    def test_analysis_keeps_all5_returns_paired_direction_and_C_worst_case(self):
        returned, observations, costs = [], [], {}
        n0_metrics = {key: 10. for key in protocol.METRICS}
        tensors = {"n0": n0_metrics}
        for seed in protocol.SEEDS:
            for method in protocol.METHODS:
                tensor = "n0" if method == "full" and seed == 1 else f"{seed}_{method}"
                tensors[tensor] = (n0_metrics if tensor == "n0" else
                                   {key: 2. if method == "full" else 4. for key in protocol.METRICS})
                returned.append({"seed": seed, "method": method, "tensor_sha256": tensor,
                                 "is_N0": tensor == "n0", "kind": "SYNTHETIC_KEEP_ALL"})
                costs[seed, method] = {"train": 2., "select": 1.}
        for tensor, metrics in tensors.items():
            for family in protocol.WEIGHTS:
                observations.append({"case": "case118", "tensor_sha256": tensor, "domain": "target",
                                     "identity": family, "family": family, "group": family,
                                     "input_sha256": family, "metrics": metrics})
            observations.append({"case": "case118", "tensor_sha256": tensor, "domain": "retained_normal",
                                 "identity": "normal", "family": "normal", "group": "normal",
                                 "input_sha256": "normal", "metrics": n0_metrics})
        plan = {"company_case": "case118", "cases": {"case118": {"anchor_tensor_sha256": "n0"}}}
        with patch.dict("company_readout.COUNTS", {"case118": (5, 1)}):
            result = company_readout.analyze({"budget": False}, plan, returned, observations, costs)
        self.assertEqual(len(result["all_logical_returns"]), 20)
        self.assertEqual(result["method_summary"]["full"]["unchanged_N0_count"], 1)
        self.assertEqual(result["method_summary"]["full"]["total_adapt_plus_selection_seconds"], 15)
        self.assertAlmostEqual(result["paired_raw_contrasts"][0]["mean"], .4)
        self.assertEqual(len(result["paired_raw_contrasts"]), 6)
        self.assertTrue(result["rPBL_not_implemented_or_imputed"])

    def test_budget_lock_compares_parent_sampling_and_training_and_gate_allowance(self):
        for failure in (None, "parent_stream", "training_penalty", "gate"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                parent = root / "parent"
                output = root / "budget"
                output.mkdir()
                (output / "plan.json").write_text("{}")
                full, schedule = {}, []
                for seed in protocol.SEEDS:
                    name = f"case118_seed{seed}_full"
                    full[seed] = {"name": name}
                    save(parent / "formal" / (name + ".locked.json"), {"calibration_sha256": "cal"})
                    (parent / "formal" / (name + ".sampling.jsonl")).write_text(str(seed))
                    for method, delta in protocol.BUDGETS.items():
                        name = f"case118_seed{seed}_{method}"
                        row = {"name": name, "seed": seed, "method": method}
                        schedule.append(row)
                        model = output / "formal" / name / "RETURN.pt"
                        model.parent.mkdir(parents=True)
                        model.write_bytes(b"synthetic")
                        save(output / "formal" / (name + ".return.json"), {
                            **row, "plan_sha256": protocol.sha(output / "plan.json"),
                            "file": model.relative_to(output).as_posix(), "sha256": protocol.sha(model),
                            "target_response_or_C_selection": False})
                        save(output / "formal" / (name + ".locked.json"), {
                            "optimizer_updates": 1280, "calibration_sha256": "cal",
                            "quality_factors": factors(.1 if failure == "training_penalty" else delta)})
                        (output / "formal" / (name + ".sampling.jsonl")).write_text(
                            "changed" if failure == "parent_stream" else str(seed))
                        save(output / "operations" / name / "selection_input.json", {
                            "quality_factors": factors(.1 if failure == "gate" else delta)})
                plan = {"schedule": schedule, "matched_parent_plan_sha256": "parent"}
                spec = SimpleNamespace(specification=lambda method: {"quality_factors": factors(protocol.BUDGETS[method])})
                with patch("company_budget_supervisor.HERE", output), \
                        patch("company_budget_supervisor.validate_parent", return_value=(parent, full)), \
                        patch.dict("sys.modules", {"study_methods": spec}):
                    if failure:
                        with self.assertRaises(ValueError):
                            company_budget_supervisor.lock(plan)
                        self.assertFalse((output / "GLOBAL_RETURN_LOCK.json").exists())
                    else:
                        company_budget_supervisor.lock(plan)
                        self.assertEqual(len(protocol.read(output / "GLOBAL_RETURN_LOCK.json")["returns"]), 15)


if __name__ == "__main__":
    unittest.main()

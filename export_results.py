"""Create a small manual-return evidence archive; never upload or delete native states."""

import argparse
import csv
import io
import json
from pathlib import Path
import tarfile

from company_protocol import completed_readout, read, sha, validate_run


def export(run, readout, target, budget=False):
    run = Path(run).resolve(strict=True)
    target = Path(target).resolve()
    if target.exists():
        raise FileExistsError("Never overwrite an earlier manual return")
    plan = read(run / "plan.json")
    complete = False
    if readout is not None and (Path(readout) / "ANALYSIS_COMPLETE.json").exists():
        validate_run(run, budget)
        readout = Path(readout).resolve(strict=True)
        completed_readout(readout, run)
        complete = True
    elif readout is not None:
        readout = Path(readout).resolve(strict=True)
        if not (readout / "FAILURE.json").exists():
            raise ValueError("An incomplete readout needs its actual failure receipt, not a live process snapshot")
    members = []
    for path in sorted(run.rglob("*")):
        if readout is not None and (path == readout or readout in path.parents):
            continue
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(run).as_posix()
        if relative.startswith(("workspace/", "prepared/")):
            continue
        # Model/RNG/optimizer bytes stay on the data volume; every native member is hashed.
        if path.suffix in (".pt", ".pth", ".xz"):
            members.append({"area": "native_retained_not_in_small_archive", "path": relative,
                            "bytes": path.stat().st_size, "sha256": sha(path)})
        elif path.suffix in (".json", ".jsonl", ".csv", ".log") and path.name != "plan.json":
            if path.name.endswith(("_input.json", "_OWNER.json")):
                area = "retained_site_specific_evidence"
            else:
                area = "included" if path.stat().st_size <= 2 * 1024**2 else "retained_oversize_evidence"
            members.append({"area": area, "path": relative, "bytes": path.stat().st_size, "sha256": sha(path)})
    receipt = {
        "schema_version": 1, "training_plan_sha256": sha(run / "plan.json"),
        "complete_fixed_readout": complete, "scientific_result_eligible": complete,
        "case": plan["company_case"], "budget": budget,
        "source_bindings": plan["source_bindings"],
        "workspace_source_bindings": plan["workspace_source_bindings"],
        "workspace_input_bindings": plan["workspace_input_bindings"],
        "N0_and_case_bindings": plan["cases"],
        "seeds": plan["seeds"], "updates_per_actual_trajectory": 1280,
        "members": members, "native_states_removed": 0, "automatic_upload": False,
        "numerical_current_state_resume_verified": False,
        "failure_only_archive_must_not_populate_paper": not complete,
        "paths_and_native_states_remain_on_company_volume": True,
        "export_source_bindings": {"export_results.py": sha(Path(__file__)),
                                  "company_protocol.py": sha(Path(__file__).with_name("company_protocol.py"))},
    }
    if complete:
        receipt["readout_plan_sha256"] = sha(readout / "READOUT_PLAN.json")
        receipt["readout_complete_sha256"] = sha(readout / "ANALYSIS_COMPLETE.json")
        receipt["readout_report_payload_bindings"] = read(readout / "READOUT_PLAN.json")["report_payload_bindings"]
        receipt["readout_model_bindings"] = read(readout / "READOUT_PLAN.json")["distinct_tensors"]
        # Return portable model bindings, not the company's absolute filesystem layout.
        receipt["readout_model_bindings"] = [{k: v for k, v in row.items() if k != "path"}
                                             for row in receipt["readout_model_bindings"]]
        receipt["readout_source_bindings"] = read(readout / "READOUT_PLAN.json")["source_bindings"]
        receipt["readout_members"] = {
            name: {"bytes": (readout / name).stat().st_size, "sha256": sha(readout / name)}
            for name in ("ANALYSIS_COMPLETE.json", "analysis.json", "logical_returns.csv",
                         "observations.jsonl", "reference_checks.json", "execution.jsonl")}
        if budget:
            parent = Path(plan["matched_parent_run"])
            full = [row for row in read(parent / "GLOBAL_RETURN_LOCK.json")["returns"] if row["method"] == "full"]
            receipt["matched_parent_bindings"] = {
                "training_plan_sha256": plan["matched_parent_plan_sha256"],
                "readout_plan_sha256": plan["matched_parent_readout_plan_sha256"],
                "analysis_sha256": plan["matched_parent_analysis_sha256"],
                "all_five_Full_returns": [{key: row[key] for key in
                                          ("seed", "method", "sha256", "tensor_sha256", "is_N0", "kind")}
                                         for row in full]}
    elif readout is not None:
        receipt["failed_readout_members"] = {
            name: {"bytes": (readout / name).stat().st_size, "sha256": sha(readout / name)}
            for name in ("FAILURE.json", "execution.jsonl", "observations.jsonl")
            if (readout / name).is_file()}
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as raw, tarfile.open(fileobj=raw, mode="w:gz") as archive:
        def add_bytes(name, data):
            item = tarfile.TarInfo(name)
            item.size, item.mode = len(data), 0o644
            archive.addfile(item, io.BytesIO(data))

        add_bytes("MANIFEST.json", json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False).encode())
        protocol_fields = ("company_case", "seeds", "tuning_seed", "formal_methods", "methods",
                           "quality_factors", "beta", "saved_update_budgets", "budget_parent_Full_retrained")
        add_bytes("PROTOCOL.json", json.dumps({key: plan[key] for key in protocol_fields if key in plan},
                                              indent=2, sort_keys=True).encode())
        for row in members:
            if row["area"] == "included":
                archive.add(run / row["path"], arcname="training/" + row["path"], recursive=False)
        if complete:
            for name in ("ANALYSIS_COMPLETE.json", "analysis.json", "logical_returns.csv",
                         "observations.jsonl", "reference_checks.json", "execution.jsonl"):
                archive.add(readout / name, arcname="readout/" + name, recursive=False)
            flat = io.StringIO(newline="")
            fields = ["case", "tensor_sha256", "domain", "group", "identity", "family",
                      "raw_sha256", "input_sha256", "mse_entry", "cmse", "pbl_mean",
                      "theta_mse_rad2", "pq_vm_mse_pu2"]
            writer = csv.DictWriter(flat, fieldnames=fields)
            writer.writeheader()
            with (readout / "observations.jsonl").open(encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    writer.writerow({**{key: row[key] for key in fields[:8]}, **row["metrics"]})
            add_bytes("readout/per_condition.csv", flat.getvalue().encode("utf-8-sig"))
        elif readout is not None:
            for name in receipt["failed_readout_members"]:
                archive.add(readout / name, arcname="failed_readout/" + name, recursive=False)
    # Read back the small archive, not just its existence.
    with tarfile.open(target, "r:gz") as archive:
        if json.loads(archive.extractfile("MANIFEST.json").read()) != receipt:
            raise ValueError("The manual-return archive failed readback")
        for row in members:
            if row["area"] == "included":
                import hashlib
                digest = hashlib.sha256(archive.extractfile("training/" + row["path"]).read()).hexdigest()
                if digest != row["sha256"]:
                    raise ValueError("A returned evidence member changed")
        if complete:
            for name, binding in receipt["readout_members"].items():
                import hashlib
                digest = hashlib.sha256(archive.extractfile("readout/" + name).read()).hexdigest()
                if digest != binding["sha256"]:
                    raise ValueError("A fixed readout evidence member changed during manual export")
        elif readout is not None:
            for name, binding in receipt["failed_readout_members"].items():
                import hashlib
                digest = hashlib.sha256(archive.extractfile("failed_readout/" + name).read()).hexdigest()
                if digest != binding["sha256"]:
                    raise ValueError("A failed readout evidence member changed during manual export")
    return {"archive": str(target), "bytes": target.stat().st_size, "sha256": sha(target),
            "scientific_result_eligible": complete, "uploaded": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--readout-dir", type=Path, help="Omit only for an explicitly incomplete/failure evidence return")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--budget", action="store_true")
    args = parser.parse_args()
    print(json.dumps(export(args.run_dir, args.readout_dir, args.archive, args.budget), indent=2))

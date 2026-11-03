"""Start one actual 20-cell matched study and three independent-seed L2 tunes."""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


HERE = Path(__file__).resolve().parent


def validate_run_id(value):
    if not value or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in value):
        raise ValueError("Use a simple unique run name")


def require_idle_gpu(gpu):
    clients = subprocess.check_output(
        ["nvidia-smi", "-i", gpu, "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True, timeout=20,
    ).strip()
    if clients:
        raise RuntimeError("The selected company GPU already has workloads; do not interrupt them: " + clients)


def native_admission_bytes(anchor_bytes, cells):
    # Native optimizer/RNG/rolling states and selection candidates stay raw.
    return max(20 * 1024**3, 5 * 1024**3 + 48 * anchor_bytes * cells)


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True, help="A new unique run name; never restart an old claim")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--case", choices=("case118", "case30", "case500"), default="case500")
    parser.add_argument("--after-run-dir", type=Path,
                        help="Required for new118: the completed existing company500, never a new rerun")
    args = parser.parse_args()
    scope = json.loads((HERE / "CURRENT_SCOPE.json").read_bytes())
    if args.case not in scope["company_dispatch_cases"]:
        raise ValueError("Not in the author's current nonduplicate company scope: " + args.case)
    from company_protocol import validate_run
    if args.case == "case500":
        raise ValueError("Company500 is reported started; reuse its explicit directory, do not rerun it")
    if args.after_run_dir is None:
        raise ValueError("--after-run-dir must identify the completed existing company500")
    previous, _ = validate_run(args.after_run_dir)
    if previous["company_case"] != "case500":
        raise ValueError("Serialize new118 after the existing company500 completes")
    if datetime.now(timezone(timedelta(hours=8))).date().isoformat() > scope["deadline_rule"]["last_new_block_start"]:
        raise ValueError("The author's last-new-block date passed")
    if os.name != "posix":
        raise OSError("This is the Ubuntu company entry point, not a local Windows GPU launcher")
    validate_run_id(args.run_id)
    root = args.work_dir.resolve(strict=True)
    manifest_name = "scientific_inputs" if args.case == "case118" else "additional_inputs"
    if (root / (manifest_name.upper() + "_INSTALLED.json")).read_bytes() != (HERE / (manifest_name + ".json")).read_bytes():
        raise ValueError("Install the exact declared input release first")
    package = root / "science"
    if args.case != "case118":
        package = package / "systems" / args.case
    anchor = package / "workspace" / "runs" / "tcpa_strong_anchor_v1" / "anchor_assets" / args.case / "s1" / "model.pt"
    admission_bytes = native_admission_bytes(anchor.stat().st_size, 24)
    if shutil.disk_usage(root).free < admission_bytes:
        raise OSError(f"Preserve the complete raw-native study and5GiB write-stop margin; require {admission_bytes} free bytes")
    require_idle_gpu(args.gpu)
    for folder in ("scratch", "cache"):
        (root / folder).mkdir(exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu, TMPDIR=str(root / "scratch"),
                      TMP=str(root / "scratch"), TEMP=str(root / "scratch"),
                      XDG_CACHE_HOME=str(root / "cache"),
                      TORCH_HOME=str(root / "cache" / "torch"),
                      TORCH_EXTENSIONS_DIR=str(root / "cache" / "torch_extensions"),
                      CUDA_CACHE_PATH=str(root / "cache" / "cuda"),
                      PYTHONDONTWRITEBYTECODE="1")
    import torch
    import numpy
    import torch_geometric
    if not torch.cuda.is_available():
        raise RuntimeError("The selected existing environment cannot use CUDA")
    cpus = sorted(os.sched_getaffinity(0))[:8]
    runs = root / "runs"
    runs.mkdir(exist_ok=True)
    destination = runs / args.run_id
    if destination.exists():
        raise FileExistsError("Preserve the previous run; do not repeat or reset its science")
    shutil.copytree(package / "runtime", destination)
    shutil.copyfile(HERE / "company_supervisor.py", destination / "company_supervisor.py")
    for name in ("start_company.py", "company_protocol.py"):
        shutil.copyfile(HERE / name, destination / name)
    plan = json.loads((package / "plan_template.json").read_bytes())
    profile = {
        "cpu_affinity": cpus, "cuda_visible_devices": args.gpu, "device": "cuda",
        "dependencies": str(package / "workspace"), "workspace": str(package / "workspace"),
        "prepared": str(package / "prepared" / args.case), "python": sys.executable,
        "threads": len(cpus), "maximum_arm_seconds": 14400,
        "maximum_worker_rss_bytes": (64 if args.case == "case500" else 8) * 1024**3,
        "ram_reserve_bytes": 32 * 1024**3,
        "gpu_reserved_limit_bytes": min(64 * 1024**3, int(torch.cuda.get_device_properties(0).total_memory * 0.85)),
        "minimum_gpu_free_bytes": 4 * 1024**3,
        "versions": {"python": platform.python_version(), "torch": str(torch.__version__),
                     "numpy": numpy.__version__, "pyg": torch_geometric.__version__},
    }
    plan.update(profiles={args.case: profile}, company_case=args.case,
                durable_output_root=str(destination),
                utc_frozen=datetime.now(timezone.utc).isoformat(),
                serialized_after_company500_plan_sha256=sha(args.after_run_dir / "plan.json"),
                company_backend_scope="Independent matched A100118 block; never merged with CPU partials",
                native_storage_admission={"required_free_bytes": admission_bytes,
                                          "conservative_raw_N0_multiples_per_cell": 48,
                                          "whole_volume_quota_verified": False},
                closed_checkpoint_retention="All raw native states retained on the company data volume; no deletion")
    plan["source_bindings"]["company_supervisor.py"] = sha(destination / "company_supervisor.py")
    for name in ("start_company.py", "company_protocol.py"):
        plan["source_bindings"][name] = sha(destination / name)
    if args.case == "case500":
        shutil.copyfile(HERE / "prepare_company_normal.py", destination / "prepare_company_normal.py")
        plan["source_bindings"]["prepare_company_normal.py"] = sha(destination / "prepare_company_normal.py")
        # Prepare before freezing the run authority, using only identified normal B.
        preparation_input = destination / "preparation_input.json"
        preparation_input.write_text(json.dumps({"plan": plan, "package": str(package)}, indent=2),
                                     encoding="utf-8")
        subprocess.run([sys.executable, "-B", str(destination / "prepare_company_normal.py"),
                        "--input", str(preparation_input)], check=True)
        prepared_receipt = destination / "prepared" / args.case / "result.json"
        prepared = json.loads(prepared_receipt.read_bytes())
        profile["prepared"] = str(prepared_receipt.parent)
        plan["reused_source_preparation"] = {args.case: {
            "cache_sha256": prepared["cache_sha256"], "receipt_sha256": sha(prepared_receipt)}}
    for field in ("qualification_schedule", "tuning_schedule", "schedule"):
        for row in plan[field]:
            row["cpus"] = cpus
    with (destination / "plan.json").open("x", encoding="utf-8") as output:
        json.dump(plan, output, indent=2, sort_keys=True, allow_nan=False)
        output.flush()
        os.fsync(output.fileno())
    print("REAL_STUDY", destination, "8 actual qualification updates + 3x1280 tuning + 20x1280 formal", flush=True)
    print("The initial qualification is not a formal result. No test/C scoring before all locks.", flush=True)
    os.execv(sys.executable, [sys.executable, "-B", str(destination / "company_supervisor.py"),
                            "--plan-sha256", sha(destination / "plan.json")])


if __name__ == "__main__":
    main()

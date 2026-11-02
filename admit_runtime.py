"""Check a writable workspace and bounded CUDA primitives, not research training."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from preflight import inventory


PHASES = {"import_torch", "cuda_backward", "pyg_scatter", "complete"}


def workspace_check(work_dir):
    if not str(work_dir).strip():
        raise ValueError("An existing authorized workspace is required")
    root = Path(work_dir).resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError("The authorized workspace must be an existing directory")
    disk = shutil.disk_usage(root)
    system_root = Path(Path.home().anchor)
    descriptor, name = tempfile.mkstemp(prefix=".research-write-check-", dir=root)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(b"write-readback-check\n")
            handle.flush()
            os.fsync(handle.fileno())
        if Path(name).read_bytes() != b"write-readback-check\n":
            raise RuntimeError("Workspace write/read-back check failed")
    finally:
        Path(name).unlink()
    return {
        "write_readback_ok": True,
        "disk_free_bytes_snapshot": disk.free,
        "same_filesystem_as_system_root": root.stat().st_dev == system_root.stat().st_dev,
        "quota_verified": False,
        "scientific_storage_budget_verified": False,
    }


def cuda_worker(device, check_pyg):
    print("PHASE=import_torch", file=sys.stderr, flush=True)
    report = inventory(require_cuda=True, device=device)
    import torch

    print("PHASE=cuda_backward", file=sys.stderr, flush=True)
    values = torch.tensor([1.0, 2.0, 3.0], device=f"cuda:{device}", requires_grad=True)
    (values * values).sum().backward()
    torch.cuda.synchronize(device)
    if values.grad.cpu().tolist() != [2.0, 4.0, 6.0]:
        raise RuntimeError("CUDA backward produced an unexpected gradient")
    report["cuda_backward_checked"] = True
    report["pyg_scatter_backward_checked"] = False
    if check_pyg:
        print("PHASE=pyg_scatter", file=sys.stderr, flush=True)
        from torch_geometric.utils import scatter

        source = torch.tensor([1.0, 2.0, 3.0, 4.0], device=f"cuda:{device}", requires_grad=True)
        index = torch.tensor([0, 1, 0, 1], device=f"cuda:{device}")
        summed = scatter(source, index, dim=0, dim_size=2, reduce="sum")
        summed.sum().backward()
        torch.cuda.synchronize(device)
        if summed.detach().cpu().tolist() != [4.0, 6.0] or source.grad.cpu().tolist() != [1.0] * 4:
            raise RuntimeError("PyG CUDA scatter/backward produced an unexpected result")
        report["pyg_scatter_backward_checked"] = True
    report["scientific_model_qualified"] = False
    print("PHASE=complete", file=sys.stderr, flush=True)
    print(json.dumps(report), flush=True)


def run_check(work_dir, device=0, timeout=180, check_pyg=False):
    if device < 0 or not 1 <= timeout <= 600:
        raise ValueError("Use a nonnegative device index and a timeout between 1 and 600 seconds")
    workspace = workspace_check(work_dir)
    root = Path(work_dir).resolve(strict=True)
    run = Path(tempfile.mkdtemp(prefix="cuda-admission-", dir=root))
    cache_paths = {name: run / name for name in ("tmp", "cache", "torch", "cuda-cache")}
    for path in cache_paths.values():
        path.mkdir()
    env = os.environ.copy()
    env.update(
        PYTHONDONTWRITEBYTECODE="1",
        TMPDIR=str(cache_paths["tmp"]), TMP=str(cache_paths["tmp"]), TEMP=str(cache_paths["tmp"]),
        XDG_CACHE_HOME=str(cache_paths["cache"]),
        TORCH_HOME=str(cache_paths["torch"]), CUDA_CACHE_PATH=str(cache_paths["cuda-cache"]),
    )
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--_worker", "--device", str(device)]
    if check_pyg:
        command.append("--check-pyg")
    result = {
        "schema_version": 1,
        "run_id": run.name,
        "workspace": workspace,
        "status": "not_completed",
        "scientific_execution_qualified": False,
        "scientific_training_updates": 0,
    }
    try:
        completed = subprocess.run(
            command, cwd=run, env=env, capture_output=True, text=True, timeout=timeout,
        )
        stdout, stderr = completed.stdout, completed.stderr
        result["worker_exit_code"] = completed.returncode
        if completed.returncode:
            result["status"] = "worker_failed"
        else:
            try:
                report = json.loads(stdout)
            except json.JSONDecodeError:
                report = None
            valid = (
                isinstance(report, dict)
                and report.get("cuda_backward_checked") is True
                and report.get("scientific_model_qualified") is False
                and (not check_pyg or report.get("pyg_scatter_backward_checked") is True)
            )
            result["status"] = "primitive_checks_passed" if valid else "invalid_worker_report"
            if valid:
                result["runtime"] = report
    except subprocess.TimeoutExpired as error:
        def as_text(value):
            return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value or ""

        stdout, stderr = as_text(error.stdout), as_text(error.stderr)
        result["status"] = "worker_timed_out"
        result["timeout_seconds"] = timeout
    phases = [line.partition("=")[2] for line in stderr.splitlines()
              if line.startswith("PHASE=") and line.partition("=")[2] in PHASES]
    result["last_phase"] = phases[-1] if phases else "unknown"
    for name, content in (("stdout.txt", stdout), ("stderr.txt", stderr)):
        with (run / name).open("x", encoding="utf-8") as handle:
            handle.write(content)
    with (run / "result.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--check-pyg", action="store_true")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args._worker:
        cuda_worker(args.device, args.check_pyg)
        return
    if args.work_dir is None or not args.work_dir.strip():
        parser.error("--work-dir must name an existing authorized writable directory")
    print("Checking workspace and CUDA primitives; no research training will start.", file=sys.stderr, flush=True)
    report = run_check(args.work_dir, args.device, args.timeout, args.check_pyg)
    print(json.dumps(report, indent=2))
    if report["status"] != "primitive_checks_passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

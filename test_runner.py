import importlib.metadata
import json
from pathlib import Path
import sys
import tempfile
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import preflight
import smoke
import admit_runtime


class RunnerTests(unittest.TestCase):
    def test_unchecked_cuda_cannot_admit_training(self):
        with patch("preflight.importlib.metadata.version",
                   side_effect=importlib.metadata.PackageNotFoundError):
            report = preflight.inventory()
        self.assertEqual(report["cuda"], {"status": "not_checked"})
        self.assertFalse(report["training_qualified"])
        self.assertNotIn("hostname", report)
        self.assertEqual(set(report["packages"].values()), {None})

    def test_cuda_failure_is_explicit(self):
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with patch.dict(sys.modules, {"torch": fake_torch}):
            with self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
                preflight.inventory(require_cuda=True)

    def test_cuda_inventory_still_does_not_qualify_training(self):
        fake_cuda = SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            get_device_properties=lambda device: SimpleNamespace(name="Synthetic GPU", major=8, minor=0),
            mem_get_info=lambda device: (64, 80),
        )
        fake_torch = SimpleNamespace(cuda=fake_cuda, version=SimpleNamespace(cuda="synthetic"))
        with patch.dict(sys.modules, {"torch": fake_torch}):
            report = preflight.inventory(require_cuda=True)
            with self.assertRaises(ValueError):
                preflight.inventory(require_cuda=True, device=1)
        self.assertEqual(report["cuda"]["status"], "available")
        self.assertFalse(report["training_qualified"])

    def test_receipt_is_synthetic_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "receipt.json"
            receipt = smoke.write_receipt(output)
            self.assertEqual(receipt, json.loads(output.read_text(encoding="utf-8")))
            self.assertTrue(receipt["synthetic"])
            self.assertFalse(receipt["scientific_execution_qualified"])
            with self.assertRaises(FileExistsError):
                smoke.write_receipt(output)

    def test_workspace_requires_existing_directory_and_reads_back(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                admit_runtime.workspace_check(root / "absent")
            self.assertFalse((root / "absent").exists())
            existing = root / "file"
            existing.write_text("unchanged", encoding="utf-8")
            with self.assertRaises(NotADirectoryError):
                admit_runtime.workspace_check(existing)
            report = admit_runtime.workspace_check(root)
            self.assertTrue(report["write_readback_ok"])
            self.assertFalse(report["quota_verified"])
            self.assertFalse(report["scientific_storage_budget_verified"])
            self.assertEqual(list(root.iterdir()), [existing])

    def test_workspace_permission_failure_is_not_admitted(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("admit_runtime.tempfile.mkstemp", side_effect=PermissionError("denied")):
                with self.assertRaises(PermissionError):
                    admit_runtime.workspace_check(directory)

    def test_cuda_timeout_retains_failure_and_redirects_caches(self):
        failure = subprocess.TimeoutExpired(["worker"], 180, output=b"", stderr=b"PHASE=import_torch\n")
        with tempfile.TemporaryDirectory() as directory:
            with patch("admit_runtime.subprocess.run", side_effect=failure) as run:
                report = admit_runtime.run_check(directory)
            self.assertEqual(report["status"], "worker_timed_out")
            self.assertEqual(report["last_phase"], "import_torch")
            self.assertFalse(report["scientific_execution_qualified"])
            retained = Path(directory) / report["run_id"]
            self.assertEqual(json.loads((retained / "result.json").read_text(encoding="utf-8")), report)
            child_env = run.call_args.kwargs["env"]
            for name in ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "TORCH_HOME", "CUDA_CACHE_PATH"):
                self.assertTrue(Path(child_env[name]).is_relative_to(retained))
            self.assertEqual(child_env["PYTHONDONTWRITEBYTECODE"], "1")

    def test_nonzero_and_invalid_cuda_reports_never_pass(self):
        for returncode, output in ((1, ""), (0, "not json"), (0, "{}")):
            with self.subTest(returncode=returncode, output=output):
                with tempfile.TemporaryDirectory() as directory:
                    completed = subprocess.CompletedProcess(["worker"], returncode, output, "PHASE=cuda_backward\n")
                    with patch("admit_runtime.subprocess.run", return_value=completed):
                        report = admit_runtime.run_check(directory)
                    self.assertNotEqual(report["status"], "primitive_checks_passed")
                    self.assertFalse(report["scientific_execution_qualified"])

    def test_pyg_requirement_and_primitive_success_do_not_qualify_model(self):
        worker = {"cuda_backward_checked": True, "pyg_scatter_backward_checked": False,
                  "scientific_model_qualified": False}
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.CompletedProcess(["worker"], 0, json.dumps(worker), "PHASE=complete\n")
            with patch("admit_runtime.subprocess.run", return_value=completed):
                rejected = admit_runtime.run_check(directory, check_pyg=True)
                accepted = admit_runtime.run_check(directory, check_pyg=False)
            self.assertEqual(rejected["status"], "invalid_worker_report")
            self.assertEqual(accepted["status"], "primitive_checks_passed")
            self.assertFalse(accepted["scientific_execution_qualified"])


if __name__ == "__main__":
    unittest.main()

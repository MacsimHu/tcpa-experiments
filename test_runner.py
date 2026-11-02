import importlib.metadata
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import preflight
import smoke


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


if __name__ == "__main__":
    unittest.main()

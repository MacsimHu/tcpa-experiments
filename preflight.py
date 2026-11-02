"""Print a minimal runtime inventory without starting research workloads."""

import argparse
import importlib.metadata
import json
import platform


def inventory(require_cuda=False, device=0):
    if device < 0:
        raise ValueError("The device index must be nonnegative")
    packages = {}
    for package in ("torch", "numpy", "torch-geometric"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    result = {
        "schema_version": 1,
        "operating_system": platform.system(),
        "python": platform.python_version(),
        "packages": packages,
        "cuda": {"status": "not_checked"},
        "training_qualified": False,
    }
    if require_cuda:
        try:
            import torch
        except ModuleNotFoundError as error:
            if error.name != "torch":
                raise
            raise RuntimeError("CUDA preflight requires an existing PyTorch installation") from error
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; scientific execution is not admitted")
        if device >= torch.cuda.device_count():
            raise ValueError("The selected CUDA device does not exist")
        properties = torch.cuda.get_device_properties(device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        result["cuda"] = {
            "status": "available",
            "selected_device": device,
            "name": properties.name,
            "capability": [properties.major, properties.minor],
            "free_bytes_snapshot": free_bytes,
            "total_bytes": total_bytes,
            "torch_cuda_version": torch.version.cuda,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(inventory(args.require_cuda, args.device), indent=2))


if __name__ == "__main__":
    main()

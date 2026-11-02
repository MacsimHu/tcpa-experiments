"""Write a synthetic receipt; never treat it as a scientific result."""

import argparse
import json
from pathlib import Path


def write_receipt(output):
    receipt = {
        "schema_version": 1,
        "synthetic": True,
        "check": "standard_library_receipt_write",
        "training_updates": 0,
        "neural_forwards": 0,
        "solver_calls": 0,
        "scientific_execution_qualified": False,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(write_receipt(args.output), indent=2))


if __name__ == "__main__":
    main()

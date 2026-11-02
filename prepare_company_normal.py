"""Rebuild original case500 normal B from identified references, without NN or solves."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.input.read_bytes())
    plan, package = request["plan"], Path(request["package"])
    case = plan["company_case"]
    if case != "case500" or plan["normal_B_count"] != 1022:
        raise ValueError("This bridge is only for the original admitted 1022 normal references")
    sys.path.insert(0, str(HERE))
    import runtime_support
    from common import input_fingerprint, sha
    from coverage import labeled_source
    _, rt, versions = runtime_support.runtime(plan, case)
    torch = rt.torch
    from research.tcpa_objective_budget_ablation.src.run_confirmation_evaluation import build_labeled_graph
    config, historical, subset, fit, development = rt.datasets(case)
    template = historical[0]
    source = package / "normal_B"
    identities = json.loads((source / "ORIGINAL_INPUT_IDENTITIES.json").read_bytes())["identities"]
    expected = {row["group"]: row for row in identities}
    references = json.loads((source / "REFERENCES.json").read_bytes())["records"]
    if len(references) != len(expected) or len(expected) != 1022:
        raise ValueError("The original fixed cohort changed")
    graphs, seen = [], set()
    for row in references:
        path = source.joinpath(*row["file"].split("/"))
        if sha(path) != row["sha256"] or row["group"] in seen:
            raise ValueError("An original normal reference changed or is duplicated")
        inputs, scan = rt.SYMBOLS["build_strict_generated_canos"](path)
        rt.SYMBOLS["assert_target_scrubbed"](inputs)
        if input_fingerprint(inputs) != expected[row["group"]]["input_sha256"]:
            raise ValueError("The reconstructed allowed normal input differs from its original identity")
        truth = build_labeled_graph(path, rt.SYMBOLS)
        graphs.append(labeled_source(inputs, truth, template, torch))
        seen.add(row["group"])
    destination = HERE / "prepared" / case
    destination.mkdir(parents=True, exist_ok=False)
    cache = destination / "normal_labeled.pt"
    with cache.open("xb") as stream:
        torch.save(graphs, stream)
        stream.flush()
        os.fsync(stream.fileno())
    receipt = {"cache_sha256": sha(cache), "conditions": len(graphs),
               "plan_sha256": plan["parent_training_plan_sha256"],
               "source_identity_manifest_sha256": sha(source / "ORIGINAL_INPUT_IDENTITIES.json"),
               "source_reference_manifest_sha256": sha(source / "REFERENCES.json"),
               "runtime_versions": versions, "model_forwards": 0, "optimizer_updates": 0,
               "solver_calls": 0, "new_serialization_not_the_original_CPU_cache_bytes": True}
    with (destination / "result.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
    print("ORIGINAL_NORMAL_B_PREPARED", len(graphs), "0 model forwards; 0 PF solves", flush=True)


if __name__ == "__main__":
    main()

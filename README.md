# Research execution handoff

This repository is a **bootstrap**, not a complete scientific training package.
It contains newly authored, generic environment checks and a synthetic smoke
receipt. It includes no research models, training implementations, datasets,
checkpoints, manuscripts, real experimental results, credentials or internal
machine configuration. Cloning it alone cannot train the research models.

## Check the destination machine

Use an existing Python 3.10+ environment:

```powershell
python preflight.py
python preflight.py --require-cuda --device 0
python smoke.py --output results\synthetic-smoke.json
python -m unittest -v test_runner
```

The CUDA command requires an existing working PyTorch installation and fails
explicitly if CUDA is unavailable. It inspects only the selected device and
does not launch model training. The smoke command uses only the standard
library; it does not exercise CUDA, model gradients, checkpoints or resumption.
A passing smoke is not evidence that scientific experiments can run or resume.

The preflight omits usernames, hostnames, environment variables, credentials
and filesystem paths. Its free-memory snapshot is not a resource reservation,
ownership check, concurrency qualification or permission to interrupt work.

## Before scientific execution

Confirm the organization's compute, software, networking and data-transfer
policies. Obtain the project-specific implementation and inputs through an
approved channel; verify immutable manifests, dataset roles, model identities,
software versions, resource ownership and complete-state recovery.

An upstream project's public license does not automatically clear local
unpublished modifications, weights, derived data or results for publication.
No project-specific source, inputs, logs or real result bundles are included
or automatically downloaded, executed or published here.

## Result handling

Keep actual artifacts and results in approved storage. The ignored `results`
directory is local only. There is deliberately no automatic GitHub uploader,
remote job dispatcher or arbitrary-shell job executor.

Only separately reviewed, non-sensitive release material may be committed.
Raw logs, model states, reference labels and company configuration must not be
added merely to make another machine able to reproduce a run.

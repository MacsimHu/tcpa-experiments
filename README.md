# Research execution handoff

This repository is a **bootstrap**, not a complete scientific training package.
It contains newly authored, generic environment checks and a synthetic smoke
receipt. It includes no research models, training implementations, datasets,
checkpoints, manuscripts, real experimental results, credentials or internal
machine configuration. Cloning it alone cannot train the research models.

**Company operator: start with [RUNBOOK.md](RUNBOOK.md).**
It defines the experiment order, actual missing research assets and input
fingerprints. `jobs.jsonl` contains the finite planned queue; it is not an
automatic dispatcher and none of its jobs is marked execution-ready.

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

## Bounded workspace and real CUDA check

Select an existing, authorized, writable work directory on a volume with enough
space for the intended research inputs, temporary files and checkpoints.
Do not use a nearly full system volume simply because another data volume has
free space. Do not change mount permissions, delete files or use elevated
privileges to bypass an unwritable directory. Ask its administrator to provide
one user-writable subdirectory if needed.

In the existing Python environment intended for the experiments, set
`WORK_DIR` to that directory, then run:

```text
python admit_runtime.py --work-dir "$WORK_DIR" --device 0 --timeout 180 --check-pyg
```

`$WORK_DIR` is shell-specific: on Linux it is a shell variable; on Windows
PowerShell define `$WORK_DIR` first. Do not pass an unset or empty variable.
No packages are installed or upgraded by this command.

This checks a tiny write/read-back and runs a separate, bounded child process
for an actual CUDA backward operation and an optional PyG scatter/backward.
It preserves the failed or successful attempt in a new `cuda-admission-*`
directory under the workspace. Its child temporary and common Torch/CUDA
cache locations are directed there, not to the system-volume defaults.
Other global/site-specific cache settings and pre-existing import hooks are
not overridden. Keep the attempt's logs private; do not automatically commit
them to GitHub.

The summary distinguishes timeout, worker failure and primitive success.
Import delays can be diagnosed from the retained phase markers.
Primitive success does **not** validate the research model, native graph
extensions, legacy checkpoint loading, complete-state resumption, filesystem
quota, a full training storage budget or resource ownership.

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

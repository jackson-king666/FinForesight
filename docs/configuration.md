# Configuration and live prerequisites

Run documented commands from the repository root. TOML dataset/trace paths are
resolved relative to the TOML file by the existing configuration loader. Experiment
manifest path semantics are schema-specific; inspect the deterministic plan before
execution. Absolute external-source paths are supplied explicitly through environment
variables. Copy `.env.example` to a private `.env` and select an exact model identity.
Do not commit populated secrets or use placeholder credentials for a live run.

```sh
uv run finforesight data-sync --config configs/strict.toml
uv run finforesight preflight --config configs/strict.toml
uv run finforesight run --help
```

The first command explicitly fetches the legacy pinned Hub snapshot. These
commands need network and are distinct from the offline example. Pin validation
must fail on count/hash drift; do not change counts to accept a mutable revision.

For controlled research, supply Volc credentials and a separate filter model:

```sh
uv run finforesight search-service start --config configs/search-service.example.toml
uv run finforesight search-service status --config configs/search-service.example.toml
uv run finforesight search-service stop --config configs/search-service.example.toml
```

The service is explicitly managed and loopback-bound. Run does not auto-start it.
Missing service, credentials or immutable source must be resolved before paid calls.
Never switch provider, model snapshot or web policy to bypass a failed gate.

FinanceHarness uses a pinned external checkout and Python 3.12 environment.
Inspect `finforesight external-sync --help` and its source licensing before an
explicit sync. Preflight/run verify external inputs and never clone or install.
See `configs/external/README.md` for ODR/GPT-Researcher metadata and source hashes;
these systems use manifest-driven experiments, not legacy baseline aliases.

`full-release-v4.example.yaml` and other experiment YAML files retain versioned
input contracts. They require separately supplied temporal/cohort/review/result
inputs and may contain placeholder hashes. They are templates and are not executable
quickstarts or evidence of completed experiments. Archived calibration, QC, and
Path B command families require explicitly supplied matching bundles; no historical
calibration or financial page payloads are distributed here. Use their CLI help to
supply roots and policy paths, rather than relying on unbundled default locations.

Keep logs and generated artifacts outside version control. An error should identify
the missing input without printing a credential. Child workers receive stripped
credentials and only the mediated capabilities allowed by their profile.

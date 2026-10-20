# Reproduce the shipped workflows

Run commands from this repository root. Install the locked environment first:

```sh
uv sync --locked --python 3.11
uv run finforesight --help
uv run benchmark-harness --help
uv run python scripts/offline_demo.py --output results/offline-demo
uv run python scripts/check_release.py
```

Both console commands use `benchmark_harness.cli:main`. The offline demo writes
`predictions.jsonl`, `metrics_summary.json`, and `manifest.json` inside the supplied
output directory. The manifest binds fixture, evaluator and output hashes. The
summary separates forecasting and diagnostic populations and uses the frozen
reference metrics, identified as `frozen-reference/v1` for this example. Invalid
parser examples remain in the predictions and evaluator counts; none are selected away.
Reusing a nonempty output directory fails. A missing or malformed fixture fails.

To verify the offline suite and package:

```sh
uv lock --check
uv run ruff check .
uv run ruff format --check .
uv run pytest -m "not live"
uv build
git diff --check
```

The optional authoritative evaluator parity tests require an explicit
`BENCHMARK_REFERENCE_EVALUATION_SOURCE_ROOT` at the revision recorded in the
evaluator manifest. Their absence is reported as a skip. Installed packages
include the evaluator manifest, notice, and Search+Filter JSON resources.

## Live framework execution

Use [configuration](docs/configuration.md) to supply data, credentials, external
sources, and a service. Start with preflight, not a paid run. Native profile and
baseline choices are listed by `finforesight run --help`. External ODR and
GPT-Researcher use `finforesight experiment --help` and verified manifests; their
source checkouts are never installed automatically during preflight/run.

## Paper tables

The offline example is synthetic and does not regenerate manuscript numbers.
The bundled legacy compatibility configuration pins 339 records. The v4 example
names a different 1,119-record contract and has zero hash placeholders: validation
must reject it until every required input is supplied and bound.

Exact manuscript table regeneration requires the author's final dataset/schema
and temporal assignments, all final model/system configurations and result
bundles, and the versioned table exporter with scoring/rounding rules. These
inputs are not bundled. `formal-utility/v1` is preserved as an existing analysis
utility; it is not asserted to equal the manuscript aggregation. In particular,
its FUND transformation differs from per-instance clipped manuscript utility.
No command here claims complete paper-table reproduction.

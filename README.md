# FinForesight

Research code for **FinForesight: Can Deep Research Systems Truly Forecast Financial Markets?**
FinForesight separates model-visible tasks, mediated research tools, held-out
labels, and host-side evaluation to study financial forecasting under temporal controls.

| Task | Category | Purpose |
|---|---|---|
| VAL | A1 | Valuation forecasts |
| RANK | A2-F / A2-T / A2-H | Ranking with three input variants |
| EVT | B | Event direction forecasts |
| FUND | C | Fundamental value forecasts |
| CF | D | Counterfactual diagnostic |
| CALC | E | Calculation diagnostic |

T1, T2, and T3 remain separate temporal populations; CF and CALC are diagnostic
controls and do not enter forecasting aggregates. Pending T3 outcomes remain pending.

## Install and try offline

Python 3.11+ and [uv](https://docs.astral.sh/uv/) are required.

```sh
uv sync --locked --python 3.11
uv run finforesight --help
uv run python scripts/offline_demo.py --output results/offline-demo
```

After installation the example needs no network, API keys, financial data, or
external source checkout. Its fixed synthetic responses exercise the real frozen
parser, evaluator, and atomic artifact writer. They are not model performance results.
Use a new output directory for each invocation.

## Execution paths

| Path | Scope | Inputs |
|---|---|---|
| Offline example | All six tasks, including F/T/H ranking variants | Bundled synthetic fixtures |
| `run` with `one_turn_strict` | Existing native adapter registry | Explicit pinned data and model credentials |
| `run` with `agentic_web_search` | Controlled Volc search | Data, credentials, resident Search+Filter service |
| `run` with `deep_research` | FinanceHarness baseline | Above plus pinned external source and isolated Python 3.12 environment |
| Manifest-driven `experiment` | FinanceHarness, ODR, GPT-Researcher under their gates | Immutable manifest, source/model/data identities and required evidence |

ODR and GPT-Researcher are not legacy `run --baseline` choices. This release does
not supply Claude Code or Codex integrations or certify all manuscript configurations.
The full-release manifest is an input template, not a runnable quickstart.

See [REPRODUCE.md](REPRODUCE.md), [data](docs/data.md),
[configuration](docs/configuration.md), [tasks](docs/tasks.md),
[architecture](docs/architecture.md), and [reproducibility](docs/reproducibility.md).

## License

Project code retains [Apache-2.0](LICENSE). The frozen evaluator retains its
CC-BY-4.0 attribution. See [third-party notices](THIRD_PARTY_NOTICES.md).

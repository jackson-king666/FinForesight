# Data inputs

The offline example uses only `examples/offline/fixtures.json`, with explicit
synthetic provenance and A1, A2-F/T/H, B, C, D and E examples (including malformed
responses). It contains no financial source documents.

The strict and native live example configurations use the existing immutable
Hub identity `sselaine27/benchmark-research` at
`e941b978af5f8a38fbeb4670ee2d77671a5b551e`, split `train`. Its eight configuration
names and enforced counts are defined in `benchmark_harness/dataset_sources/`:

| Config | Count |
|---|---:|
| a1 | 72 |
| a2_f | 10 |
| a2_t | 10 |
| a2_h | 10 |
| b | 33 |
| c | 180 |
| d | 12 |
| e | 12 |

These 339 records are a legacy compatibility contract, not the manuscript's final
1,119-instance release. The v4 template records a separate pin and per-config
counts; matching its total alone does not establish the final paper dataset.
No full paper dataset download is advertised here.

Use the explicit `data-sync` step in the configuration guide. Cache data resides
under the config's data directory (normally `data/.huggingface`); result paths are
explicit in TOML or the demo command. The bundled `data/metrics` and `data/temporal`
files retain their own version identity and are not universal cutoff authority.

Only task_id, prompt and cutoff_date enter system TaskInput. Ground truth, temporal
assignments, settlement state and analysis metadata remain evaluator/analysis-side.
Underlying financial datasets and source documents are not redistributed.

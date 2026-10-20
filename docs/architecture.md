# Architecture

`config.py` and `profile_registry.py` define immutable configuration and supported
profiles. `dataset_sources/` and `dataset_bridge.py` load pinned inputs.
`contracts.py` limits model-visible TaskInput to task_id, prompt and cutoff_date.
`adapters/`, `systems/` and `external_harnesses/` execute systems under host-owned
model/search boundaries; they do not receive labels or evaluator state.

`search/` and `search_service/` enforce cutoff filtering, model/search isolation and
restricted capabilities. `runner.py` owns execution and `artifacts.py` owns atomic
local persistence. `reference_evaluation/` contains the frozen host-side parser,
evaluator and attribution. Its engine hash is pinned in the evaluator manifest.
`experiment/`, `temporal/`, `analysis/` and `reporting/` bind identities and produce
versioned summaries without changing the frozen evaluator.

There are exactly three legacy profiles: one_turn_strict, agentic_web_search and
deep_research. Profile registry permissions, ranking eligibility, service contracts,
zero-egress workers and failure gates remain authoritative. Generic retrieval
regimes are not implicitly the manuscript's Search/PIT/PIT+Filter ablation.

Calibration, evidence-QC and Path B modules are retained reusable utilities with
external bundle requirements. Their historical input archives are not shipped.

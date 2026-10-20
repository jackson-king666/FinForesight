## External adapter provenance

The adapter modules under `benchmark_harness/systems/adapters/` are
integration-only gate implementations. They do not add a ranking population,
live unrestricted-web execution, or Phase 5 artifacts.

- ODR uses the exact upstream revision/tree and upstream `uv.lock`/project/
  license hashes declared in `odr.py`. The disposable worker verifies the
  pinned source hooks before launch, constructs a LangChain structured
  `search_filter_http` tool, and binds `search_api=none`, `mcp_config=null`,
  and `allow_clarification=false` through the parent v3 channel.
- GPT-Researcher uses revision `5d84d2f5553e70a2765a8ff3a0d2672d60437ce8`
  and its benchmark-owned Python 3.12 lock at
  `configs/external/gpt-researcher/uv.lock`. `LOCK_METADATA.json` records the
  resolver, runtime, and supported platform identities. The lock and metadata
  are verify-only at preflight/run; no implicit sync or install is performed by
  the gate. Its worker binds `RETRIEVER=custom`, raises before scraper/browser
  access, and routes all prompt-visible dates through the declared clock.

Both gates require a task-scoped v3 worker, a declared experiment clock, and
the parent-owned model/search capabilities. The default worker command is
stdio-only and one-task; the child environment contains no provider
credentials. Source and license verification is verify-only; callers must
supply the exact external checkout or an attested source verifier. Missing
collaborator inputs are not defaulted.

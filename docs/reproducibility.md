# Reproducibility contracts

Pin dataset revision, task counts, temporal index, cutoff registry, external source,
model snapshot, metric catalog and configuration hashes before execution. Planning
is deterministic; resume appends attempts rather than selecting a better score or
overwriting prior evidence. Raw outputs and receipts are the audit source of truth.
Keep failure/exclusion states and their denominators; do not replace unavailable
scores with zero or omit invalid responses silently.

Strict execution has one prompt, no tools/history and no non-model network calls.
Controlled research uses the explicitly managed Search+Filter service. External
workers do not receive provider credentials, labels, a general network client or
unrestricted tools. Unknown identities, malformed filter responses, missing hashes
and failed capability gates fail closed. Budgets and retries remain visible.

The evaluator hash and notice establish scoring identity. `formal-utility/v1` and
exploratory utilities retain their original definitions; none is renamed to imply
manuscript equivalence. CF/CALC and unrestricted-current-web runs remain separate
from forecasting rankings; T1/T2/T3 and unresolved settlement states remain explicit.

The release manifest binds shipped payload hashes. The synthetic demo manifest
binds fixture/evaluator/output hashes and declares synthetic status. Those artifacts
verify the code path, not scientific performance. See REPRODUCE.md for supplied
workflows and the input contract for exact paper-table regeneration.

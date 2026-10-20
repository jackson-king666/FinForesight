"""Framework-neutral Deep Research system contracts.

The contracts are additive to the legacy FinanceHarness v2 path.  They define
what a future external system may observe; they do not expose a new runtime
adapter or ranking population by themselves.
"""

from benchmark_harness.systems.adapters import (
    FinanceHarnessSystem,
    GptResearcherAdapter,
    OdrAdapter,
)
from benchmark_harness.systems.capabilities import (
    CapabilityError,
    SearchInvocation,
    TypedSearchCapability,
)
from benchmark_harness.systems.clock import ClockPolicyError, DeclaredExperimentClock
from benchmark_harness.systems.contracts import (
    BudgetSpec,
    DeepResearchSystem,
    ExperimentBinding,
    PreflightResult,
    SystemDescriptor,
    TaskCapability,
)
from benchmark_harness.systems.external_stdio import (
    BudgetLedger,
    FinanceHarnessV2CompatibilityBridge,
    V3CapabilityManifest,
    V3Channel,
    V3Envelope,
    V3ProtocolError,
    V3SearchRequest,
    V3SearchResponse,
)
from benchmark_harness.systems.fakes import (
    FAKE_SYSTEM_IDS,
    FakeDeepResearchSystem,
    fake_system_registry,
)
from benchmark_harness.systems.provenance import SystemSourceProvenance, verify_system_source
from benchmark_harness.systems.regimes import (
    RetrievalPolicy,
    RetrievalPolicyError,
    RetrievalRegime,
    build_retrieval_policy,
)
from benchmark_harness.systems.registry import SystemRegistry, SystemRegistryError
from benchmark_harness.systems.runtime import (
    AdapterBoundaryError,
    AdapterGateReport,
    AdapterRun,
    V3HostAdapter,
)

__all__ = [
    "BudgetSpec",
    "CapabilityError",
    "ClockPolicyError",
    "DeclaredExperimentClock",
    "BudgetLedger",
    "DeepResearchSystem",
    "FAKE_SYSTEM_IDS",
    "ExperimentBinding",
    "FakeDeepResearchSystem",
    "FinanceHarnessSystem",
    "FinanceHarnessV2CompatibilityBridge",
    "PreflightResult",
    "SystemDescriptor",
    "TaskCapability",
    "TypedSearchCapability",
    "SystemRegistry",
    "SystemRegistryError",
    "V3CapabilityManifest",
    "V3Channel",
    "V3Envelope",
    "V3ProtocolError",
    "V3SearchRequest",
    "V3SearchResponse",
    "SearchInvocation",
    "RetrievalPolicy",
    "RetrievalPolicyError",
    "RetrievalRegime",
    "build_retrieval_policy",
    "AdapterBoundaryError",
    "AdapterGateReport",
    "AdapterRun",
    "SystemSourceProvenance",
    "V3HostAdapter",
    "GptResearcherAdapter",
    "OdrAdapter",
    "fake_system_registry",
    "verify_system_source",
]

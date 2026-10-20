"""Five deterministic fake systems used only for boundary/analysis fixtures."""

from __future__ import annotations

import json
from datetime import date
from typing import Any

from benchmark_harness.contracts import AdapterResult, TaskInput, canonical_input_hash
from benchmark_harness.systems.contracts import (
    ExperimentBinding,
    PreflightResult,
    SystemDescriptor,
    TaskCapability,
)
from benchmark_harness.systems.registry import SystemRegistry


class FakeDeepResearchSystem:
    """A record-free adapter fixture that emits only observable JSON output."""

    def __init__(self, descriptor: SystemDescriptor, *, score: float) -> None:
        self.descriptor = descriptor
        self.score = score
        self.closed = False

    def preflight(self, binding: ExperimentBinding) -> PreflightResult:
        if binding.descriptor_hash != self.descriptor.descriptor_hash:
            return PreflightResult(False, ("descriptor_hash_mismatch",))
        return PreflightResult(
            True,
            provenance={
                "system_id": self.descriptor.system_id,
                "protocol": "external_research_stdio/v3",
                "ranking_eligible": False,
            },
        )

    def run_task(
        self,
        task: TaskInput,
        context: Any,
        capability: TaskCapability,
        binding: ExperimentBinding | None = None,
    ) -> AdapterResult:
        if self.closed:
            raise RuntimeError("fake system is closed")
        payload = {
            "system_id": self.descriptor.system_id,
            "task_id": task.task_id,
            "score": self.score,
            "retrieval_regime": self.descriptor.retrieval_regime,
        }
        raw_text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return AdapterResult(
            raw_text=raw_text,
            usage={"input_tokens": 1, "output_tokens": 1},
            latency_seconds=0.0,
            canonical_input_hash=canonical_input_hash(task.prompt),
            effective_input_hash=canonical_input_hash(task.prompt),
            effective_messages=tuple(task.messages),
            effective_request={"messages": task.messages, "protocol": "external_research_stdio/v3"},
            model_call_count=1,
            tool_calls=(),
            network_events=(),
            trace_events=tuple(context.events),
            trace_artifact_ref="trajectory_events.jsonl",
        )

    def close(self) -> None:
        self.closed = True


FAKE_SYSTEM_IDS = (
    "financeharness-fixture",
    "odr-fixture",
    "gpt-researcher-fixture",
    "hf-open-deep-research-fixture",
    "agents-sdk-fixture",
)


def fake_system_registry(
    *,
    backbone_training_cutoff: date,
    retrieval_regime: str = "cutoff_controlled",
) -> SystemRegistry:
    """Create the five fixture systems; all remain non-ranking by construction."""

    registry = SystemRegistry()
    for index, system_id in enumerate(FAKE_SYSTEM_IDS):
        descriptor = SystemDescriptor(
            system_id=system_id,
            framework_id=system_id.removesuffix("-fixture"),
            framework_revision=f"fixture-framework-{index + 1}",
            strategy_id=f"fixture-strategy-{index + 1}",
            strategy_revision="fixture-v1",
            backbone_id="fixture/model@v1",
            backbone_training_cutoff=backbone_training_cutoff,
            cutoff_source="fixture",
            adapter_id="fake-external-stdio-v3",
            adapter_revision="fixture-v1",
            profile="deep_research",
            retrieval_regime=retrieval_regime,
            ranking_eligible=False,
            license_status="MIT",
        )
        registry.register(
            descriptor,
            lambda descriptor=descriptor, score=0.4 + index * 0.1: FakeDeepResearchSystem(
                descriptor, score=score
            ),
        )
    return registry

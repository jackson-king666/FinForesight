"""Phase 3A generic registry, v3 IPC, budget, and isolation gates."""

from __future__ import annotations

import io
from datetime import date
from pathlib import Path

import pytest

from benchmark_harness.config import ExperimentConfig, ModelConfig
from benchmark_harness.contracts import RunContext, TaskInput
from benchmark_harness.external_harnesses.ipc import Envelope, encode_envelope
from benchmark_harness.systems import (
    BudgetLedger,
    BudgetSpec,
    ExperimentBinding,
    FinanceHarnessV2CompatibilityBridge,
    TaskCapability,
    V3Channel,
    V3Envelope,
    V3ProtocolError,
    fake_system_registry,
)
from benchmark_harness.systems.isolation import IsolationSpec, build_isolated_environment


def _binding(descriptor_hash: str) -> ExperimentBinding:
    return ExperimentBinding(
        experiment_id="fixture-experiment",
        plan_id="fixture-plan",
        plan_hash="fixture-plan-hash",
        cell_id="cell-1",
        attempt_id="cell-1:attempt-1",
        descriptor_hash=descriptor_hash,
        task_id="A1-00001",
        retrieval_regime="cutoff_controlled",
        trial=1,
        budget=BudgetSpec(2, 2, 2, 2, 100, 100, 30.0),
    )


def _capability() -> TaskCapability:
    return TaskCapability(
        service_url="http://127.0.0.1:8765",
        service_token="task-token",
        model_broker_capability="broker-token",
        budget=BudgetSpec(2, 2, 2, 2, 100, 100, 30.0),
        allow_calculation=True,
        allow_citations=True,
        require_structured_output=True,
        run_id="run-1",
        trial_id="trial-1",
        trace_mode="hash_only",
    )


def test_five_fake_systems_are_registry_constructed_and_non_ranking() -> None:
    registry = fake_system_registry(backbone_training_cutoff=date(2024, 1, 1))
    assert len(registry.descriptors()) == 5
    assert all(descriptor.ranking_eligible is False for descriptor in registry.descriptors())

    profile = ExperimentConfig(model=ModelConfig(mode="mock")).resolve_model(require_ready=True)
    assert profile is not None
    for descriptor in registry.descriptors():
        system = registry.create(descriptor.system_id)
        binding = _binding(descriptor.descriptor_hash)
        assert system.preflight(binding).ready is True
        context = RunContext(
            run_id="run-1",
            framework=descriptor.framework_id,
            adapter_version="fixture-v1",
            model=profile,
            trace_content_mode="hash_only",
        ).for_task("A1-00001")
        result = system.run_task(TaskInput("A1-00001", "fixture prompt"), context, _capability())
        assert result.network_events == ()
        assert result.trace_artifact_ref == "trajectory_events.jsonl"
        system.close()


def test_v3_ipc_is_closed_bound_and_rejects_temporal_leakage() -> None:
    binding = _binding("descriptor")
    writer = io.BytesIO()
    channel = V3Channel(io.BytesIO(), writer, binding=binding)
    sent = channel.send("task.start", {"capability_manifest_sha256": "cap"}, request_id="request-1")
    frame = writer.getvalue()
    assert sent.binding_hash == binding.plan_hash

    reader = io.BytesIO(frame)
    received = V3Channel(reader, io.BytesIO(), binding=binding).read()
    assert received.type == "task.start"
    assert received.seq == 0

    with pytest.raises(V3ProtocolError, match="time_band"):
        V3Envelope(
            type="trajectory.event",
            request_id="request-2",
            task_id="A1-00001",
            seq=1,
            binding_hash=binding.plan_hash,
            payload={"time_band": "T2"},
        )


def test_v3_budget_counts_shared_limits_and_v2_bridge_remains_explicit() -> None:
    ledger = BudgetLedger(BudgetSpec(1, 1, 1, 1, 10, 10, 30.0))
    ledger.consume(model_calls=1, input_tokens=2, output_tokens=3)
    assert ledger.remaining()["model_calls"] == 0
    with pytest.raises(V3ProtocolError, match="model_calls"):
        ledger.consume(model_calls=1)

    legacy = Envelope(
        type="hello",
        request_id="legacy-1",
        task_id="A1-00001",
        seq=0,
        payload={"protocol": "external_research_stdio/v2"},
    )
    decoded = FinanceHarnessV2CompatibilityBridge().decode_legacy(encode_envelope(legacy))
    assert decoded.schema_version == "external_research_stdio/v2"


def test_generic_isolation_strips_credentials_and_requires_literal_loopback(tmp_path: Path) -> None:
    spec = IsolationSpec(
        source_dir=tmp_path.resolve(),
        repo_root=tmp_path.resolve(),
        loopback_url="http://127.0.0.1:8765",
    )
    environment = build_isolated_environment(
        spec,
        base_environment={
            "PATH": "/bin",
            "OPENAI_API_KEY": "secret",
            "VOLC_ACCESS_KEY_ID": "secret",
            "LANG": "C",
        },
    )
    assert "OPENAI_API_KEY" not in environment
    assert "VOLC_ACCESS_KEY_ID" not in environment
    assert environment["PYTHONPATH"].split(":")[0] == str(tmp_path.resolve())

    with pytest.raises(ValueError, match="127.0.0.1"):
        IsolationSpec(
            source_dir=tmp_path.resolve(),
            repo_root=tmp_path.resolve(),
            loopback_url="http://example.com:8765",
        )

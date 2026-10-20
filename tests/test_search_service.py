"""Offline contract tests for the resident Search+Filter service."""

from __future__ import annotations

import asyncio
import json
import stat
from pathlib import Path
from typing import Any

import pytest

from benchmark_harness.config import (
    FINANCEHARNESS_PROTOCOL,
    DatasetConfig,
    DeepResearchConfig,
    ExecutionConfig,
    ExperimentConfig,
    ExternalHarnessConfig,
    ModelConfig,
    SearchConfig,
    TraceConfig,
)
from benchmark_harness.contracts import AdapterResult, NetworkEvent, ToolCall
from benchmark_harness.external_harnesses.financeharness.adapter import DeepRunOutput
from benchmark_harness.runner import BenchmarkRunner
from benchmark_harness.search.volc import VolcHttpResponse
from benchmark_harness.search_service import SearchServiceClient, SearchServiceSession
from benchmark_harness.search_service.api import SEARCH_SERVICE_PROTOCOL, SearchServiceError
from benchmark_harness.search_service.config import SearchServiceClientConfig, SearchServiceConfig
from benchmark_harness.search_service.engine import SearchFilterEngine
from benchmark_harness.search_service.identities import filter_model_identity, service_manifest
from benchmark_harness.search_service.server import _filter_profile, create_app
from benchmark_harness.systems.capabilities import TypedSearchCapability
from benchmark_harness.systems.contracts import BudgetSpec
from benchmark_harness.systems.external_stdio import BudgetLedger


class _FakeVolcTransport:
    def __init__(self) -> None:
        self.queries: list[str] = []
        self.time_ranges: list[str] = []

    def post(
        self,
        url: str,
        *,
        json: dict[str, Any],
        headers: dict[str, str],
        timeout_seconds: float,
    ) -> VolcHttpResponse:
        del url, headers, timeout_seconds
        self.queries.append(str(json["Query"]))
        self.time_ranges.append(str(json["TimeRange"]))
        return VolcHttpResponse(
            status_code=200,
            request_id=f"req-{len(self.queries)}",
            payload={
                "Result": {
                    "WebResults": [
                        {
                            "Url": "https://source.example/before-cutoff",
                            "Title": "Evidence",
                            "Content": "Evidence published before the cutoff.",
                            "PublishTime": "2024-01-01T00:00:00Z",
                        }
                    ]
                }
            },
        )


class _TestClientTransport:
    def __init__(self, app: Any) -> None:
        from fastapi.testclient import TestClient

        self.client = TestClient(app)

    def request(
        self, method: str, path: str, body: dict[str, Any] | None, auth: str | None
    ) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {auth}"} if auth else {}
        response = self.client.request(method, path, json=body, headers=headers)
        payload = response.json()
        if response.status_code >= 400:
            error = payload.get("error") if isinstance(payload, dict) else None
            code = error.get("code") if isinstance(error, dict) else "request_failed"
            raise SearchServiceError(str(code), status_code=response.status_code)
        return payload


def test_service_cohort_binds_filter_budget_and_temperature() -> None:
    base = {
        "mode": "openai_compatible",
        "endpoint": "https://filter.example/v1",
        "model_name": "filter-model",
        "temperature": 0.0,
        "timeout_seconds": 60.0,
        "retries": 2,
        "api_key": "must-not-be-bound",
    }
    low_budget = {**base, "max_tokens": 120}
    high_budget = {**base, "max_tokens": 1024}

    low = service_manifest(filter_model=low_budget)
    high = service_manifest(filter_model=high_budget)

    assert low.service_cohort_sha256 != high.service_cohort_sha256
    assert filter_model_identity(high_budget) == {
        "mode": "openai_compatible",
        "endpoint": "https://filter.example/v1",
        "model_name": "filter-model",
        "temperature": 0.0,
        "max_tokens": 1024,
        "timeout_seconds": 60.0,
        "retries": 2,
    }


def test_w1_filter_profile_and_service_cohort_are_pinned(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "SEARCH_FILTER_MODEL_ENDPOINT", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    monkeypatch.setenv("SEARCH_FILTER_MODEL_NAME", "deepseek-v4-flash")
    profile = _filter_profile(SearchServiceConfig())

    assert profile is not None
    assert profile.max_tokens == 2048
    assert (
        service_manifest(filter_model=profile.public_dict()).service_cohort_sha256
        == "77b9d73ae159d8e327a7cae3ded29091b828aa5a2b82b448533f47c5f66f8e15"
    )


@pytest.fixture
def service(tmp_path: Path):
    transport = _FakeVolcTransport()
    config = SearchServiceConfig(state_dir=tmp_path / "state", test_only=True)
    engine = SearchFilterEngine(transport=transport, test_only=True)
    app = create_app(config, engine=engine)
    return app, app.state.search_service, transport


def test_service_is_closed_loopback_api_with_auth_and_exact_fields(service: Any) -> None:
    app, state, _transport = service
    from fastapi.testclient import TestClient

    client = TestClient(app)
    assert client.get("/healthz").status_code == 200
    assert client.get("/readyz").status_code == 401
    ready = client.get("/readyz", headers={"Authorization": f"Bearer {state.admin_token}"})
    assert ready.status_code == 200
    assert ready.json()["protocol"] == SEARCH_SERVICE_PROTOCOL
    assert ready.json()["test_only"] is True
    invalid = client.post(
        "/v1/sessions",
        headers={"Authorization": f"Bearer {state.admin_token}"},
        json={"unexpected": True},
    )
    assert invalid.status_code == 400
    assert invalid.json()["error"]["protocol"] == SEARCH_SERVICE_PROTOCOL


def test_search_route_offloads_the_blocking_engine_to_a_worker_thread(
    service: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, state, _transport = service
    descriptor = state.open_session(
        _principal_session_body("financeharness", state.manifest.service_cohort_sha256)
    )["session"]
    calls: list[tuple[Any, tuple[Any, ...]]] = []

    async def fake_to_thread(function: Any, *args: Any) -> Any:
        calls.append((function, args))
        return function(*args)

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)
    from fastapi.testclient import TestClient

    response = TestClient(app).post(
        f"/v1/sessions/{descriptor['session_id']}/search",
        headers={"Authorization": f"Bearer {descriptor['task_token']}"},
        json={
            "schema_version": SEARCH_SERVICE_PROTOCOL,
            "request_id": "offload-request",
            "session_id": descriptor["session_id"],
            "run_id": descriptor["run_id"],
            "task_id": descriptor["task_id"],
            "profile": descriptor["profile"],
            "baseline": descriptor["baseline"],
            "cutoff_date": descriptor["cutoff_date"],
            "call_index": 1,
            "query": "offloaded query",
        },
    )

    assert response.status_code == 200
    assert len(calls) == 1
    assert calls[0][0].__self__ is state
    assert calls[0][0].__name__ == "search"
    assert calls[0][1][0] == descriptor["session_id"]


def test_session_admission_requires_the_external_cohort_before_side_effects(service: Any) -> None:
    _app, state, _transport = service
    before = state._active_sessions()
    with pytest.raises(SearchServiceError) as missing:
        state.open_session(_principal_session_body("financeharness", ""))
    assert missing.value.code == "service_cohort_invalid"
    with pytest.raises(SearchServiceError) as wrong:
        state.open_session(_principal_session_body("financeharness", "0" * 64))
    assert wrong.value.code == "service_cohort_mismatch"
    assert state._active_sessions() == before


def test_client_search_is_idempotent_and_close_exports_receipts(service: Any) -> None:
    app, state, transport = service
    client = SearchServiceClient(
        SearchServiceClientConfig(
            base_url="http://127.0.0.1:8765",
            admin_token_file=state.config.admin_token_file,
            expected_service_cohort_sha256=state.manifest.service_cohort_sha256,
        ),
        transport=_TestClientTransport(app),
    )
    ready = client.ready()
    session = client.open_session(
        run_id="run-1",
        task_id="A1-1",
        profile="agentic_web_search",
        baseline="langgraph",
        cutoff_date="2024-06-01",
        policy_version="agentic_web_search/v2",
        expected_service_cohort_sha256=ready["service_cohort_sha256"],
    )
    first = client.search(
        session.descriptor, request_id="same-request", call_index=1, query="evidence"
    )
    replay = client.search(
        session.descriptor, request_id="same-request", call_index=1, query="evidence"
    )
    assert first.receipt.receipt_sha256 == replay.receipt.receipt_sha256
    assert replay.receipt.replayed is True
    assert len(transport.queries) == 1
    close = client.close_session(session.descriptor, (first.receipt,))
    assert close["comparable"] is True
    assert close["receipt_count"] == 1
    assert close["receipts"][0]["receipt_sha256"] == first.receipt.receipt_sha256


def test_child_receipts_can_be_handed_to_parent_session_close(service: Any) -> None:
    app, state, _transport = service
    client = SearchServiceClient(
        SearchServiceClientConfig(
            admin_token_file=state.config.admin_token_file,
        ),
        transport=_TestClientTransport(app),
    )
    ready = client.ready()
    child = client.open_session(
        run_id="run-child",
        task_id="A1-child",
        profile="deep_research",
        baseline="financeharness",
        cutoff_date="2024-06-01",
        policy_version="deep_research/v2",
        expected_service_cohort_sha256=ready["service_cohort_sha256"],
    )
    outcome = child.search("child receipt")

    parent = SearchServiceSession(client, child.descriptor)
    assert parent.receipts == ()
    close = parent.close(receipts=child.receipts)

    assert close["comparable"] is True
    assert close["receipt_count"] == 1
    assert close["receipt_chain_head"] == outcome.receipt.receipt_sha256


def test_search_service_preserves_query_invalid_http_400(service: Any) -> None:
    app, state, _transport = service
    client = SearchServiceClient(
        SearchServiceClientConfig(
            admin_token_file=state.config.admin_token_file,
            expected_service_cohort_sha256=state.manifest.service_cohort_sha256,
        ),
        transport=_TestClientTransport(app),
    )
    session = client.open_session(
        run_id="run-query-invalid",
        task_id="A1-query-invalid",
        profile="deep_research",
        baseline="financeharness",
        cutoff_date="2024-06-01",
        policy_version="deep_research/v2",
        expected_service_cohort_sha256=state.manifest.service_cohort_sha256,
    )

    with pytest.raises(SearchServiceError) as exc:
        client.search(
            session.descriptor,
            request_id="query-too-long",
            call_index=1,
            query="q" * 2049,
        )

    assert exc.value.code == "query_invalid"
    assert exc.value.status_code == 400


def test_typed_search_capability_persists_service_query_error_and_status() -> None:
    def handler(_query: str, _token: str, _call_index: int) -> dict[str, Any]:
        raise SearchServiceError("query_invalid", status_code=400)

    ledger = BudgetLedger(BudgetSpec(1, 1, 0, 0, 100, 100, 5.0))
    capability = TypedSearchCapability(
        handler,
        capability_token="parent-only-token",
        ledger=ledger,
    )

    response = capability.invoke("valid query")

    assert response.error == "search_handler_error:query_invalid:http_400"
    assert response.results == ()
    assert capability.calls[0].result_count == 0


def test_service_binds_cutoff_per_session_and_keeps_filter_budget_separate(service: Any) -> None:
    app, state, transport = service
    client = SearchServiceClient(
        SearchServiceClientConfig(
            admin_token_file=state.config.admin_token_file,
            expected_service_cohort_sha256=state.manifest.service_cohort_sha256,
        ),
        transport=_TestClientTransport(app),
    )
    first = client.open_session(
        run_id="run-a",
        task_id="A1-a",
        profile="deep_research",
        baseline="financeharness",
        cutoff_date="2024-01-01",
        policy_version="deep_research/v2",
        expected_service_cohort_sha256=state.manifest.service_cohort_sha256,
    )
    second = client.open_session(
        run_id="run-b",
        task_id="A1-b",
        profile="deep_research",
        baseline="financeharness",
        cutoff_date="2024-06-01",
        policy_version="deep_research/v2",
        expected_service_cohort_sha256=state.manifest.service_cohort_sha256,
    )
    first.search("same query")
    second.search("same query")
    assert len(transport.time_ranges) == 2
    assert transport.time_ranges[0] != transport.time_ranges[1]
    assert first.budget.model_calls == 0
    assert first.budget.tool_calls == 1
    assert first.outcomes[0].filter_model_call_count == 0


def _principal_session_body(
    principal: str,
    expected_cohort: str,
    request_id: str = "req-open-1",
) -> dict[str, Any]:
    return {
        "schema_version": SEARCH_SERVICE_PROTOCOL,
        "request_id": request_id,
        "run_id": "run-a",
        "task_id": "A1-1",
        "profile": "deep_research",
        "baseline": principal,
        "cutoff_date": "2024-06-01",
        "policy_version": "deep_research/v2",
        "expected_service_cohort_sha256": expected_cohort,
    }


def test_deep_search_principals_are_admitted_with_true_identity(service: Any) -> None:
    app, state, _transport = service
    for index, principal in enumerate(("financeharness", "odr", "gpt-researcher")):
        response = state.open_session(
            _principal_session_body(
                principal,
                state.manifest.service_cohort_sha256,
                request_id=f"req-open-{index}",
            )
        )
        session = response["session"]
        assert session["baseline"] == principal
        assert session["profile"] == "deep_research"
    assert state._active_sessions() == 3


def test_unknown_or_aliased_search_principals_reject_before_side_effects(service: Any) -> None:
    app, state, _transport = service
    before = state._active_sessions()
    for index, principal in enumerate(
        ("open_deep_research", "gpt_researcher", "finance-harness", "pydantic-ai")
    ):
        with pytest.raises(SearchServiceError) as exc:
            state.open_session(
                _principal_session_body(
                    principal,
                    state.manifest.service_cohort_sha256,
                    request_id=f"req-reject-{index}",
                )
            )
        assert exc.value.code == "profile_search_principal_incompatible"
        assert exc.value.status_code == 400
    assert state._active_sessions() == before


def test_wrong_profile_search_principals_reject_before_side_effects(service: Any) -> None:
    app, state, _transport = service
    before = state._active_sessions()
    agentic = {
        **_principal_session_body("odr", state.manifest.service_cohort_sha256),
        "policy_version": "agentic_web_search/v2",
    }
    agentic["profile"] = "agentic_web_search"
    with pytest.raises(SearchServiceError) as exc:
        state.open_session(agentic)
    assert exc.value.code == "profile_search_principal_incompatible"
    strict = {
        **_principal_session_body("odr", state.manifest.service_cohort_sha256),
        "policy_version": "one_turn_strict/v1",
    }
    strict["profile"] = "one_turn_strict"
    with pytest.raises(SearchServiceError) as exc:
        state.open_session(strict)
    assert exc.value.code == "profile_baseline_incompatible"
    assert state._active_sessions() == before


def test_odr_principal_is_preserved_through_search_receipts_and_close(service: Any) -> None:
    app, state, transport = service
    client = SearchServiceClient(
        SearchServiceClientConfig(
            admin_token_file=state.config.admin_token_file,
            expected_service_cohort_sha256=state.manifest.service_cohort_sha256,
        ),
        transport=_TestClientTransport(app),
    )
    ready = client.ready()
    session = client.open_session(
        run_id="run-odr",
        task_id="A1-odr",
        profile="deep_research",
        baseline="odr",
        cutoff_date="2024-06-01",
        policy_version="deep_research/v2",
        expected_service_cohort_sha256=ready["service_cohort_sha256"],
    )
    assert session.descriptor.baseline == "odr"
    outcome = session.search("odr evidence")
    assert outcome.framework == "odr"
    assert outcome.candidates
    assert all(audit.framework == "odr" for audit in outcome.audit_records)
    close = client.close_session(session.descriptor, session.receipts)
    assert close["comparable"] is True
    assert close["session"]["baseline"] == "odr"
    assert all(audit["framework"] == "odr" for audit in close["search_audit"])
    assert all(audit["baseline"] == "odr" for audit in close["search_audit"])


def test_service_rejects_non_loopback_client_urls() -> None:
    with pytest.raises(SearchServiceError, match="service_url"):
        SearchServiceClient(base_url="http://localhost:8765")
    with pytest.raises(SearchServiceError, match="service_url"):
        SearchServiceClient(base_url="http://127.0.0.1:8765/v1")


def test_agentic_runner_uses_service_session_and_exports_local_ledger(
    service: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, state, _transport = service
    client = SearchServiceClient(
        SearchServiceClientConfig(
            admin_token_file=state.config.admin_token_file,
            expected_service_cohort_sha256=state.manifest.service_cohort_sha256,
        ),
        transport=_TestClientTransport(app),
    )

    class Source:
        def preflight(self) -> dict[str, Any]:
            return self.provenance()

        def provenance(self) -> dict[str, Any]:
            return {"source": "test", "dataset_revision": "test", "record_count": 1}

        def load_records(
            self, *, categories: list[str] | None, status: str | None
        ) -> list[dict[str, Any]]:
            del categories, status
            return [
                {
                    "task_id": "A1-service",
                    "category": "A1",
                    "status": "ready",
                    "time_band": "T1",
                    "prompt": "Research evidence.",
                    "seed": {"cutoff_date": "2024-06-01"},
                }
            ]

    class Evaluation:
        def parse_prediction(self, category: str, raw_text: str) -> dict[str, Any]:
            return {"category": category, "raw": raw_text}

        def evaluate_prediction(
            self, category: str, parsed: dict[str, Any], record: dict[str, Any]
        ) -> dict[str, Any]:
            del category, record
            return {"ok": bool(parsed)}

        def summarize_results(
            self,
            predictions: list[dict[str, Any]],
            skipped: list[dict[str, Any]],
            *,
            agent: str,
            model: str,
        ) -> dict[str, Any]:
            return {
                "agent": agent,
                "model": model,
                "evaluated": len(predictions),
                "skipped": len(skipped),
            }

        def mock_response(self, record: dict[str, Any]) -> str:
            return str(record["task_id"])

    class Adapter:
        name = "langgraph"
        adapter_version = "service-test/v1"

        def run(self, task: Any, context: Any) -> AdapterResult:
            del task, context
            raise AssertionError("agentic test must use run_agentic")

        def run_agentic(self, task: Any, context: Any, search_session: Any) -> AdapterResult:
            search_session.record_model_call()
            outcome = search_session.search("model selected query")
            event = NetworkEvent(
                host="model.example",
                method="POST",
                path="/v1/chat/completions",
                body_hash="model-body",
                status_code=200,
                latency_seconds=0.01,
                is_model_endpoint=True,
            )
            return AdapterResult(
                raw_text="service answer",
                usage={},
                latency_seconds=0.01,
                canonical_input_hash=task.input_hash,
                effective_input_hash=task.input_hash,
                effective_messages=tuple(task.messages),
                effective_request={"agentic_web_search": {"service": True}},
                model_call_count=1,
                tool_calls=(ToolCall(name="search_web", result_hash=outcome.response_sha256),),
                network_events=(event,),
                trace_events=context.events,
            )

    import benchmark_harness.runner as runner_module

    monkeypatch.setattr(runner_module, "_make_adapter", lambda _baseline, mock_responses: Adapter())
    config = ExperimentConfig(
        dataset=DatasetConfig(),
        model=ModelConfig(mode="mock"),
        search=SearchConfig(test_only=False),
        execution=ExecutionConfig(input_policy="agentic_web_search"),
        trace=TraceConfig(content_mode="hash_only", results_dir=tmp_path / "results-agentic"),
    )
    result = BenchmarkRunner(
        config,
        dataset_source_factory=lambda _config: Source(),
        evaluation_engine=Evaluation(),
        search_service_client_factory=lambda _config: client,
    ).run(baselines=["langgraph"], run_id="service-e2e", isolated=False)
    output_dir = tmp_path / "results-agentic" / "service-e2e" / "langgraph"
    assert result["baselines"][0]["comparable_task_count"] == 1
    assert (output_dir / "search_service_manifest.json").is_file()
    ledger = [
        json.loads(line) for line in (output_dir / "network_ledger.jsonl").read_text().splitlines()
    ]
    assert {row.get("network_class") for row in ledger} >= {
        "web_search",
        "search_service_local",
    }


def test_deep_runner_uses_service_session_and_exports_service_receipt(
    service: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, state, _transport = service
    client = SearchServiceClient(
        SearchServiceClientConfig(
            admin_token_file=state.config.admin_token_file,
            expected_service_cohort_sha256=state.manifest.service_cohort_sha256,
        ),
        transport=_TestClientTransport(app),
    )

    class Source:
        def preflight(self) -> dict[str, Any]:
            return self.provenance()

        def provenance(self) -> dict[str, Any]:
            return {"source": "test", "dataset_revision": "test", "record_count": 1}

        def load_records(
            self, *, categories: list[str] | None, status: str | None
        ) -> list[dict[str, Any]]:
            del categories, status
            return [
                {
                    "task_id": "A1-deep-service",
                    "category": "A1",
                    "status": "ready",
                    "time_band": "T2",
                    "prompt": "Research historical evidence.",
                    "seed": {"cutoff_date": "2024-06-01"},
                }
            ]

    class Evaluation:
        def parse_prediction(self, category: str, raw_text: str) -> dict[str, Any]:
            return {"category": category, "raw": raw_text}

        def evaluate_prediction(
            self, category: str, parsed: dict[str, Any], record: dict[str, Any]
        ) -> dict[str, Any]:
            del category, record
            return {"ok": bool(parsed)}

        def summarize_results(
            self,
            predictions: list[dict[str, Any]],
            skipped: list[dict[str, Any]],
            *,
            agent: str,
            model: str,
        ) -> dict[str, Any]:
            return {
                "agent": agent,
                "model": model,
                "evaluated": len(predictions),
                "skipped": len(skipped),
            }

    external_config = ExternalHarnessConfig(checkout_root=tmp_path / "external")
    external_manifest = {
        "schema_version": "1",
        "adapter": "financeharness",
        "adapter_version": "v2",
        "repo_url": external_config.repo_url,
        "revision": external_config.revision,
        "tree_sha": external_config.tree_sha,
        "uv_lock_sha256": external_config.uv_lock_sha256,
        "pyproject_sha256": external_config.pyproject_sha256,
        "python_version": "3.12",
        "protocol": FINANCEHARNESS_PROTOCOL,
        "license_status": "UNDECLARED",
        "license_acknowledged": True,
        "source_relative": "data/external-harnesses/financeharness",
    }

    class FakeManager:
        config = external_config

        def verify(self) -> dict[str, Any]:
            return external_manifest

    class Adapter:
        adapter_version = "service-test/deep-v2"

        def __init__(self, search_service: SearchServiceClient) -> None:
            self.search_service = search_service

        def run_deep(self, task: Any, context: Any) -> DeepRunOutput:
            ready = self.search_service.preflight()
            session = self.search_service.open_session(
                run_id=context.run_id,
                task_id=task.task_id,
                profile="deep_research",
                baseline="financeharness",
                cutoff_date=task.cutoff_date,
                policy_version="deep_research/v2",
                expected_service_cohort_sha256=ready["service_cohort_sha256"],
            )
            outcome = session.search("model selected historical evidence")
            close = session.close()
            local_rows = tuple(
                {
                    "run_id": context.run_id,
                    "task_id": task.task_id,
                    "framework": "financeharness",
                    "network_class": "search_service_local",
                    "provider": "local",
                    "host": "127.0.0.1",
                    "method": "POST",
                    "path": f"/v1/sessions/{session.descriptor.session_id}/search",
                    "body_hash": receipt.receipt_sha256,
                    "status_code": 200,
                    "latency_seconds": None,
                    "is_model_endpoint": False,
                    "session_id": session.descriptor.session_id,
                    "service_request_id": receipt.service_request_id,
                    "call_index": receipt.call_index,
                    "receipt_sha256": receipt.receipt_sha256,
                }
                for receipt in session.receipts
            )
            model_event = NetworkEvent(
                host="mock.parent.model",
                method="POST",
                path="/v1/chat/completions",
                body_hash="deep-service-model-request",
                status_code=200,
                latency_seconds=0.0,
                is_model_endpoint=True,
            )
            result = AdapterResult(
                raw_text="service deep answer",
                usage={},
                latency_seconds=0.0,
                canonical_input_hash=task.input_hash,
                effective_input_hash=task.input_hash,
                effective_messages=tuple(task.messages),
                effective_request={
                    "messages": task.messages,
                    "messages_sha256": task.input_hash,
                    "deep_research": {
                        "termination": "answer",
                        "rounds": 1,
                        "filter_model_call_count": 0,
                    },
                },
                model_call_count=1,
                tool_calls=(ToolCall(name="search"),),
                network_events=(model_event,),
                trace_events=context.events,
            )
            return DeepRunOutput(
                result=result,
                search_outcomes=(outcome,),
                search_ledger_rows=tuple(close["network_ledger"]) + local_rows,
                research_audit_rows=(
                    {
                        "task_id": task.task_id,
                        "event_type": "task_started",
                        "network_call_count": 0,
                    },
                    {
                        "task_id": task.task_id,
                        "event_type": "task_finished",
                        "network_call_count": 0,
                    },
                ),
                citations=("https://source.example/before-cutoff",),
                external_manifest=external_manifest,
                child_network_call_count=0,
                child_search_service_call_count=1,
                child_forbidden_network_call_count=0,
                service_close=close,
            )

    import benchmark_harness.runner as runner_module

    monkeypatch.setattr(
        runner_module, "_make_external_harness_manager", lambda _config: FakeManager()
    )
    config = ExperimentConfig(
        dataset=DatasetConfig(),
        model=ModelConfig(mode="mock"),
        search=SearchConfig(test_only=False),
        execution=ExecutionConfig(input_policy="deep_research"),
        trace=TraceConfig(content_mode="hash_only", results_dir=tmp_path / "results-deep-research"),
        deep_research=DeepResearchConfig(),
        external_harnesses={"financeharness": external_config},
    )
    result = BenchmarkRunner(
        config,
        dataset_source_factory=lambda _config: Source(),
        evaluation_engine=Evaluation(),
        search_service_client_factory=lambda _config: client,
        external_harness_adapter_factory=lambda **kwargs: Adapter(kwargs["search_service"]),
    ).run(baselines=["financeharness"], run_id="deep-service-e2e", isolated=False)
    output_dir = tmp_path / "results-deep-research" / "deep-service-e2e" / "financeharness"
    assert result["baselines"][0]["comparable_task_count"] == 1
    prediction = json.loads((output_dir / "predictions.jsonl").read_text().strip())
    assert prediction["service_receipt_count"] == 1
    assert prediction["child_search_service_call_count"] == 1
    assert json.loads((output_dir / "metrics_summary.json").read_text())["by_time_band"] == {
        "T2": 1
    }
    assert (output_dir / "search_service_manifest.json").is_file()


def test_exclusive_lock_sets_secure_mode_and_rejects_second_holder(tmp_path: Path) -> None:
    from benchmark_harness.search_service.lifecycle import _exclusive_lock

    config = SearchServiceConfig(state_dir=tmp_path / "state")
    handle = _exclusive_lock(config)
    try:
        assert stat.S_IMODE(config.lock_file.stat().st_mode) == 0o600
        with pytest.raises(SearchServiceError, match="service_lock_unavailable"):
            _exclusive_lock(config)
    finally:
        handle.close()

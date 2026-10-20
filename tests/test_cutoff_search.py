"""Focused regression coverage for the Volc cutoff gateway and filter."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from benchmark_harness.config import ResolvedModelProfile
from benchmark_harness.contracts import AdapterResult, NetworkEvent
from benchmark_harness.search import (
    AgenticWebSearchPolicyGuard,
    BEventDateError,
    CutoffSearchGateway,
    LlmTemporalVerdict,
    SearchGatewayError,
    ToolContract,
    VolcHttpResponse,
    filter_volc_candidates,
    validate_b_cutoff_not_after_event,
    write_search_audit_jsonl,
)
from benchmark_harness.search.volc import AliyunTemporalFilter


class FakeVolcTransport:
    def __init__(self, response: VolcHttpResponse) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def post(
        self,
        url: str,
        *,
        json: dict[str, object],
        headers: dict[str, str],
        timeout_seconds: float,
    ) -> VolcHttpResponse:
        self.calls.append(
            {
                "url": url,
                "body": dict(json),
                "headers": dict(headers),
                "timeout_seconds": timeout_seconds,
            }
        )
        return self.response


def _gateway(transport: FakeVolcTransport) -> CutoffSearchGateway:
    return CutoffSearchGateway(transport=transport, test_only=True)


def test_volc_gateway_sets_time_range_and_postfilters_candidates() -> None:
    transport = FakeVolcTransport(
        VolcHttpResponse(
            status_code=200,
            request_id="volc-request-1",
            payload={
                "Result": {
                    "WebResults": [
                        {
                            "Url": "https://source.example/on-cutoff",
                            "Title": "On cutoff",
                            "Content": "safe retained source content",
                            "PublishTime": "2024-06-01T23:59:59Z",
                        },
                        {
                            "Url": "https://source.example/after-cutoff",
                            "Title": "After cutoff",
                            "Content": "must not reach model",
                            "PublishTime": "2024-06-02T00:00:00Z",
                        },
                        {
                            "Url": "https://source.example/no-date",
                            "Title": "Undated",
                            "Content": "must not reach model",
                        },
                    ]
                }
            },
        )
    )

    outcome = _gateway(transport).search(
        task_id="A1-00001",
        framework="pydantic-ai",
        query="historic event",
        cutoff_date="2024-06-01",
        category="a1",
    )

    assert transport.calls == [
        {
            "url": "https://mercury.volcengineapi.com/",
            "body": {
                "Query": "historic event",
                "SearchType": "web",
                "Count": 10,
                "Filter": {"NeedContent": True, "NeedUrl": True},
                "TimeRange": "2019-06-01..2024-06-01",
            },
            "headers": {"Content-Type": "application/json"},
            "timeout_seconds": 30.0,
        }
    ]
    assert [candidate.url for candidate in outcome.candidates] == [
        "https://source.example/on-cutoff"
    ]
    assert [audit.reason for audit in outcome.audit_records] == [
        "accepted",
        "volc_publish_time_after_cutoff",
        "ambiguous_without_llm",
    ]
    assert [audit.filter_stage for audit in outcome.audit_records] == ["rule", "rule", "rule"]
    assert all(audit.time_range == "2019-06-01..2024-06-01" for audit in outcome.audit_records)
    assert all(
        "safe retained source content" not in str(audit.as_dict())
        for audit in outcome.audit_records
    )
    assert outcome.test_only is True
    assert outcome.ranking_eligible is False
    assert outcome.network_ledger(profile="agentic_web_search")["cutoff_compliant"] is True


def test_gateway_is_fail_closed_for_missing_live_volc_keys() -> None:
    gateway = CutoffSearchGateway(
        access_key_env="ABSENT_VOLC_ACCESS_KEY",
        secret_key_env="ABSENT_VOLC_SECRET_KEY",
    )

    with pytest.raises(SearchGatewayError, match="volc_access_key_missing"):
        gateway.preflight()


def test_filter_uses_mocked_llm_for_ambiguous_future_marker_only() -> None:
    requests = []

    def reviewer(request):
        requests.append(request)
        return LlmTemporalVerdict(verdict="keep", reason="historical comparison context")

    result = filter_volc_candidates(
        [
            {
                "Url": "https://source.example/ambiguous",
                "Title": "2025 outlook that quotes 2024 data",
                "Content": "A historical comparison published before cutoff.",
                "PublishTime": "2024-06-01",
            },
            {
                "Url": "https://source.example/late",
                "Title": "Late source",
                "Content": "Not eligible.",
                "PublishTime": "2024-06-03",
            },
        ],
        "2024-06-01",
        llm_filter=reviewer,
    )

    assert [candidate.url for candidate in result.kept] == ["https://source.example/ambiguous"]
    assert len(requests) == 1
    assert result.decisions[0].reason == "llm_keep"
    assert result.decisions[0].llm_verdict == "keep"
    assert result.decisions[1].reason == "volc_publish_time_after_cutoff"


def test_filter_fails_closed_when_ambiguous_llm_review_errors() -> None:
    def unavailable(_request):
        raise TimeoutError("filter backend unavailable")

    result = filter_volc_candidates(
        [
            {
                "Url": "https://source.example/no-date",
                "Title": "Undated evidence",
                "Content": "Potentially relevant source.",
            }
        ],
        "2024-06-01",
        llm_filter=unavailable,
    )

    assert result.kept == ()
    assert result.decisions[0].reason == "llm_filter_error"


@pytest.mark.parametrize("content", [None, ["not a text response"]])
def test_empty_or_nontext_temporal_review_rejects_only_the_ambiguous_candidate(
    monkeypatch: pytest.MonkeyPatch, content: object
) -> None:
    class FakeOpenAI:
        def __init__(self, **_kwargs: object) -> None:
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **_request: SimpleNamespace(
                        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
                    )
                )
            )

    import openai

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    reviewer = AliyunTemporalFilter(
        ResolvedModelProfile(
            mode="openai_compatible",
            endpoint="https://models.example.test/v1",
            model_name="test-filter-model",
            api_key="test-only",
            temperature=0.0,
            max_tokens=120,
            timeout_seconds=1.0,
            retries=0,
            seed=None,
        )
    )
    result = filter_volc_candidates(
        [
            {
                "Url": "https://source.example/ambiguous",
                "Title": "Future outlook",
                "Content": "Forecast for 2025.",
                "PublishTime": "2024-06-01",
            },
            {
                "Url": "https://source.example/safe",
                "Title": "Historical source",
                "Content": "Evidence available before the cutoff.",
                "PublishTime": "2024-06-01",
            },
        ],
        "2024-06-01",
        llm_filter=reviewer,
    )

    assert [candidate.url for candidate in result.kept] == ["https://source.example/safe"]
    assert result.decisions[0].reason == "llm_filter_error"
    assert result.decisions[0].accepted is False
    assert result.decisions[1].reason == "rule_keep"
    assert result.decisions[1].accepted is True


def test_temporal_filter_uses_raised_profile_budget_for_nonempty_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[dict[str, object]] = []

    class FakeOpenAI:
        def __init__(self, **_kwargs: object) -> None:
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self._create),
            )

        def _create(self, **request: object) -> SimpleNamespace:
            requests.append(request)
            content = "KEEP: historical comparison context" if request["max_tokens"] == 1024 else ""
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
            )

    import openai

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    reviewer = AliyunTemporalFilter(
        ResolvedModelProfile(
            mode="openai_compatible",
            endpoint="https://models.example.test/v1",
            model_name="test-filter-model",
            api_key="test-only",
            temperature=0.0,
            max_tokens=1024,
            timeout_seconds=1.0,
            retries=0,
            seed=None,
        )
    )

    result = filter_volc_candidates(
        [
            {
                "Url": "https://source.example/ambiguous",
                "Title": "Future outlook",
                "Content": "Forecast for 2025.",
                "PublishTime": "2024-06-01",
            }
        ],
        "2024-06-01",
        llm_filter=reviewer,
    )

    assert [candidate.url for candidate in result.kept] == ["https://source.example/ambiguous"]
    assert len(requests) == 1
    assert requests[0]["max_tokens"] == 1024


def test_gateway_rejects_bad_cutoff_before_transport() -> None:
    transport = FakeVolcTransport(
        VolcHttpResponse(status_code=200, payload={"Result": {"WebResults": []}})
    )

    with pytest.raises(SearchGatewayError, match="cutoff_date_invalid"):
        _gateway(transport).search(
            task_id="A1-00001",
            framework="langgraph",
            query="historic event",
            cutoff_date="2024-02-30",
        )

    assert transport.calls == []


def test_b_seed_cutoff_cannot_postdate_event_and_never_replaces_search_bound() -> None:
    window = validate_b_cutoff_not_after_event("2024-06-01", "2024-06-02")
    assert window.cutoff_date == "2024-06-01"
    assert window.exclusive_bound == "2024-06-02T00:00:00Z"

    with pytest.raises(BEventDateError, match="b_cutoff_after_event_date"):
        validate_b_cutoff_not_after_event("2024-06-03", "2024-06-02")
    with pytest.raises(BEventDateError, match="b_event_date_invalid"):
        validate_b_cutoff_not_after_event("2024-06-01", "June 2, 2024")


def test_session_enforces_shared_three_tool_call_budget() -> None:
    transport = FakeVolcTransport(
        VolcHttpResponse(status_code=200, payload={"Result": {"WebResults": []}})
    )
    session = _gateway(transport).new_session(
        task_id="C-00001",
        framework="langgraph",
        cutoff_date="2024-06-01",
    )

    for index in range(3):
        outcome = session.search(f"query {index}")
        assert outcome.tool_call_count == index + 1
    with pytest.raises(SearchGatewayError, match="tool_call_budget_exceeded"):
        session.search("too many")

    assert len(transport.calls) == 3


def test_session_enforces_shared_four_model_call_budget() -> None:
    transport = FakeVolcTransport(
        VolcHttpResponse(status_code=200, payload={"Result": {"WebResults": []}})
    )
    session = _gateway(transport).new_session(
        task_id="C-00001",
        framework="pydantic-ai",
        cutoff_date="2024-06-01",
    )

    for _ in range(4):
        session.record_model_call()
    with pytest.raises(SearchGatewayError, match="model_call_budget_exceeded"):
        session.record_model_call()


def test_gateway_rejects_a_mutable_or_renamed_native_tool_contract() -> None:
    with pytest.raises(ValueError, match="tool schema and policy are fixed"):
        CutoffSearchGateway(
            transport=FakeVolcTransport(
                VolcHttpResponse(status_code=200, payload={"Result": {"WebResults": []}})
            ),
            test_only=True,
            tool_contract=ToolContract(name="web_search"),
        )


def test_candidate_audit_is_jsonl_safe_and_does_not_store_query_or_content(tmp_path) -> None:
    transport = FakeVolcTransport(
        VolcHttpResponse(
            status_code=200,
            payload={
                "Result": {
                    "WebResults": [
                        {
                            "Url": "https://source.example/one",
                            "Title": "One",
                            "Content": "private source passage",
                            "PublishTime": "2024-06-01",
                        }
                    ]
                }
            },
        )
    )
    outcome = _gateway(transport).search(
        task_id="B-00001",
        framework="pydantic-ai",
        query="private search terms",
        cutoff_date="2024-06-01",
    )
    path = write_search_audit_jsonl(tmp_path / "search_audit.jsonl", outcome.audit_records)
    text = path.read_text(encoding="utf-8")
    row = json.loads(text)

    assert "private search terms" not in text
    assert "private source passage" not in text
    assert row["content_sha256"]
    assert row["query_sha256"] == outcome.query_sha256
    assert row["response_sha256"] == outcome.response_sha256

    hash_only_path = write_search_audit_jsonl(
        tmp_path / "hash-only-search-audit.jsonl",
        outcome.audit_records,
        hash_only=True,
    )
    hash_only_text = hash_only_path.read_text(encoding="utf-8")
    hash_only_row = json.loads(hash_only_text)
    assert "https://source.example/one" not in hash_only_text
    assert "url" not in hash_only_row
    assert hash_only_row["url_sha256"]


def _model_network_event() -> NetworkEvent:
    return NetworkEvent(
        host="models.example.test",
        method="POST",
        path="/v1/chat/completions",
        body_hash="model-request",
        status_code=200,
        latency_seconds=0.01,
        is_model_endpoint=True,
    )


def _agentic_adapter_result(
    model_calls: int = 1,
    *,
    network_events: tuple[NetworkEvent, ...] | None = None,
) -> AdapterResult:
    return AdapterResult(
        raw_text="answer",
        usage={},
        latency_seconds=0.01,
        canonical_input_hash="canonical",
        effective_input_hash="effective",
        effective_messages=(),
        effective_request={},
        model_call_count=model_calls,
        tool_calls=(),
        network_events=network_events or (_model_network_event(),),
        trace_events=(),
    )


def test_agentic_guard_requires_audit_ledger_and_allows_zero_activity_no_cutoff_skip() -> None:
    guard = AgenticWebSearchPolicyGuard()
    no_cutoff_task = SimpleNamespace(cutoff_date=None)
    skipped = guard.check(no_cutoff_task, None, skipped_no_cutoff=True)
    assert skipped.comparable is True
    assert skipped.skipped_no_cutoff is True

    transport = FakeVolcTransport(
        VolcHttpResponse(status_code=200, payload={"Result": {"WebResults": []}})
    )
    outcome = _gateway(transport).search(
        task_id="A1-00001",
        framework="pydantic-ai",
        query="historical query",
        cutoff_date="2024-06-01",
    )
    valid_task = SimpleNamespace(cutoff_date="2024-06-01")
    missing_ledger = guard.check(
        valid_task,
        _agentic_adapter_result(),
        search_outcomes=(outcome,),
    )
    assert missing_ledger.comparable is False
    assert "missing_search_network_ledger" in missing_ledger.violations

    accepted = guard.check(
        valid_task,
        _agentic_adapter_result(),
        search_outcomes=(outcome,),
        network_ledger_rows=(outcome.network_ledger(profile="agentic_web_search"),),
    )
    assert accepted.comparable is True


def test_agentic_guard_rejects_more_than_four_model_calls() -> None:
    guard = AgenticWebSearchPolicyGuard()
    task = SimpleNamespace(cutoff_date="2024-06-01")
    decision = guard.check(task, _agentic_adapter_result(model_calls=5))

    assert decision.comparable is False
    assert "model_call_budget_exceeded=5" in decision.violations


def test_agentic_guard_rejects_adapter_non_model_network_traffic() -> None:
    guard = AgenticWebSearchPolicyGuard()
    task = SimpleNamespace(cutoff_date="2024-06-01")
    result = _agentic_adapter_result(
        network_events=(
            _model_network_event(),
            NetworkEvent(
                host="unapproved.example.test",
                method="POST",
                path="/lookup",
                body_hash="unexpected-network-call",
                status_code=200,
                latency_seconds=0.01,
                is_model_endpoint=False,
            ),
        ),
    )

    decision = guard.check(task, result)

    assert decision.comparable is False
    assert "adapter_network_not_model_allowlisted=1" in decision.violations


def test_agentic_guard_rejects_unaccounted_or_over_budget_model_network_traffic() -> None:
    guard = AgenticWebSearchPolicyGuard()
    task = SimpleNamespace(cutoff_date="2024-06-01")
    result = _agentic_adapter_result(
        model_calls=1,
        network_events=(_model_network_event(),) * 5,
    )

    decision = guard.check(task, result)

    assert decision.comparable is False
    assert "model_network_call_budget_exceeded=5" in decision.violations
    assert "model_network_ledger_mismatch=reported:1,observed:5" in decision.violations

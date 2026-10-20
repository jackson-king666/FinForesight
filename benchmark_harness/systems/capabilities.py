"""Host-owned typed capabilities for external research workers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from benchmark_harness.search_service.api import SearchServiceError
from benchmark_harness.systems.external_stdio import BudgetLedger, V3ProtocolError, V3SearchResponse


class CapabilityError(RuntimeError):
    """Raised when a child requests a capability outside its typed contract."""


@dataclass(frozen=True, slots=True)
class SearchInvocation:
    """Auditable, token-free identity of one host-owned search call."""

    query_sha256: str
    call_index: int
    receipt_sha256: str
    result_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_sha256": self.query_sha256,
            "call_index": self.call_index,
            "receipt_sha256": self.receipt_sha256,
            "result_count": self.result_count,
        }


SearchHandler = Callable[[str, str, int], Mapping[str, Any] | list[Mapping[str, Any]]]


class TypedSearchCapability:
    """Inject a search function without exposing a client, URL, or token.

    The token is passed only to the parent-owned handler.  It is never returned
    by this object, placed in a v3 payload, or included in the invocation log.
    """

    protocol = "search_filter_http/v1"

    def __init__(
        self,
        handler: SearchHandler,
        *,
        capability_token: str,
        ledger: BudgetLedger,
    ) -> None:
        if not callable(handler):
            raise CapabilityError("search handler must be callable")
        if not isinstance(capability_token, str) or not capability_token.strip():
            raise CapabilityError("search capability token must be non-empty")
        self._handler = handler
        self._capability_token = capability_token
        self._ledger = ledger
        self._calls: list[SearchInvocation] = []

    @property
    def calls(self) -> tuple[SearchInvocation, ...]:
        return tuple(self._calls)

    def invoke(self, query: str, *, max_results: int = 5) -> V3SearchResponse:
        if not isinstance(query, str) or not query.strip():
            raise CapabilityError("search query must be a non-empty string")
        if (
            not isinstance(max_results, int)
            or isinstance(max_results, bool)
            or not 1 <= max_results <= 100
        ):
            raise CapabilityError("search max_results must be an integer in [1, 100]")
        call_index = len(self._calls) + 1
        self._ledger.consume(search_calls=1)
        try:
            raw = self._handler(query.strip(), self._capability_token, call_index)
            results = self._normalize_results(raw)
            error = None
        except SearchServiceError as exc:
            results = ()
            error = f"search_handler_error:{exc.code}:http_{exc.status_code}"
        except Exception as exc:
            results = ()
            error = f"search_handler_error:{type(exc).__name__}"
        receipt_sha256 = _receipt_hash(query.strip(), results, error=error, call_index=call_index)
        invocation = SearchInvocation(
            query_sha256=_sha256(query.strip()),
            call_index=call_index,
            receipt_sha256=receipt_sha256,
            result_count=len(results),
        )
        self._calls.append(invocation)
        return V3SearchResponse(
            results=results,
            receipt_sha256=receipt_sha256,
            call_index=call_index,
            error=error,
        )

    @staticmethod
    def _normalize_results(
        raw: Mapping[str, Any] | list[Mapping[str, Any]],
    ) -> tuple[dict[str, Any], ...]:
        if isinstance(raw, Mapping):
            raw_results = raw.get("results", raw.get("candidates", []))
        else:
            raw_results = raw
        if not isinstance(raw_results, list | tuple):
            raise CapabilityError("search handler must return a results list")
        results: list[dict[str, Any]] = []
        for result in raw_results:
            if not isinstance(result, Mapping):
                raise CapabilityError("search result must be an object")
            results.append(dict(result))
        return tuple(results)

    def public_manifest(self) -> dict[str, Any]:
        """Return metadata safe to send to a worker; deliberately no token."""

        return {
            "protocol": self.protocol,
            "tool": "search",
            "capability_token": None,
            "token_persisted": False,
        }


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _receipt_hash(
    query: str,
    results: tuple[dict[str, Any], ...],
    *,
    error: str | None,
    call_index: int,
) -> str:
    payload = {
        "call_index": call_index,
        "error": error,
        "query": query,
        "results": list(results),
    }
    try:
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise V3ProtocolError("search result is not JSON serializable") from exc
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


__all__ = ["CapabilityError", "SearchInvocation", "TypedSearchCapability"]

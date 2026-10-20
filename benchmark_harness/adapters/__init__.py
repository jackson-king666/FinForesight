"""Framework adapters. Each accepts only TaskInput and RunContext."""

from benchmark_harness.adapters.base import DirectTransportAdapter, MappingMockTransport
from benchmark_harness.adapters.langgraph import LangGraphAdapter
from benchmark_harness.adapters.pydantic_ai import PydanticAIAdapter

__all__ = [
    "DirectTransportAdapter",
    "LangGraphAdapter",
    "MappingMockTransport",
    "PydanticAIAdapter",
]

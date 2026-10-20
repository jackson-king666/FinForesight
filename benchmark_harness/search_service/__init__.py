"""Resident, loopback-only Search+Filter service contracts and adapters."""

from .api import (
    SEARCH_SERVICE_PROTOCOL,
    SearchReceiptRef,
    SearchServiceError,
    SearchSessionDescriptor,
)
from .client import SearchServiceClient, SearchServiceSession
from .config import SearchServiceClientConfig, SearchServiceConfig, load_search_service_config
from .lifecycle import SearchServiceLifecycle, ServiceStatus

__all__ = [
    "SEARCH_SERVICE_PROTOCOL",
    "SearchReceiptRef",
    "SearchServiceClient",
    "SearchServiceClientConfig",
    "SearchServiceConfig",
    "SearchServiceError",
    "SearchServiceLifecycle",
    "SearchServiceSession",
    "SearchSessionDescriptor",
    "ServiceStatus",
    "load_search_service_config",
]

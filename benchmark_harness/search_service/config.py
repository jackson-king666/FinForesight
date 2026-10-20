"""Daemon and client configuration for the loopback Search+Filter service."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .api import SEARCH_SERVICE_BASE_URL, SEARCH_SERVICE_PROTOCOL


class SearchServiceConfigError(ValueError):
    """Raised for an unsafe or ambiguous service configuration."""


@dataclass(frozen=True, slots=True)
class SearchServiceClientConfig:
    base_url: str = SEARCH_SERVICE_BASE_URL
    admin_token_file: Path = Path("data/search-service/admin.token")
    connect_timeout_seconds: float = 2.0
    request_timeout_seconds: float = 1800.0
    # Declared by the run specification; never inferred from /readyz.
    expected_service_cohort_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class SearchServiceConfig:
    protocol: str = SEARCH_SERVICE_PROTOCOL
    bind_host: str = "127.0.0.1"
    port: int = 8765
    state_dir: Path = Path("data/search-service")
    max_inflight_searches: int = 8
    queue_limit: int = 64
    drain_timeout_seconds: float = 30.0
    volc_access_key_env: str = "VOLC_ACCESS_KEY_ID"
    volc_secret_key_env: str = "VOLC_SECRET_ACCESS_KEY"
    filter_endpoint_env: str = "SEARCH_FILTER_MODEL_ENDPOINT"
    filter_name_env: str = "SEARCH_FILTER_MODEL_NAME"
    filter_api_key_env: str = "SEARCH_FILTER_MODEL_API_KEY"
    filter_timeout_seconds: float = 60.0
    filter_retries: int = 2
    test_only: bool = False

    @property
    def admin_token_file(self) -> Path:
        return self.state_dir / "admin.token"

    @property
    def state_file(self) -> Path:
        return self.state_dir / "state.json"

    @property
    def lock_file(self) -> Path:
        return self.state_dir / "service.lock"

    @property
    def log_file(self) -> Path:
        return self.state_dir / "service.log"

    @property
    def audit_dir(self) -> Path:
        return self.state_dir / "audit"

    def public_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "bind_host": self.bind_host,
            "port": self.port,
            "state_dir": str(self.state_dir),
            "max_inflight_searches": self.max_inflight_searches,
            "queue_limit": self.queue_limit,
            "drain_timeout_seconds": self.drain_timeout_seconds,
            "volc_access_key_env": self.volc_access_key_env,
            "volc_secret_key_env": self.volc_secret_key_env,
            "filter_endpoint_env": self.filter_endpoint_env,
            "filter_name_env": self.filter_name_env,
            "filter_api_key_env": self.filter_api_key_env,
            "filter_timeout_seconds": self.filter_timeout_seconds,
            "filter_retries": self.filter_retries,
            "test_only": self.test_only,
        }


def load_search_service_config(path: str | Path) -> SearchServiceConfig:
    config_path = Path(path).expanduser().resolve()
    try:
        with config_path.open("rb") as handle:
            raw = tomllib.load(handle)
    except FileNotFoundError as exc:
        raise SearchServiceConfigError(f"service config does not exist: {config_path}") from exc
    if not isinstance(raw, dict):
        raise SearchServiceConfigError("service config must be a TOML table")
    allowed_top = {"service", "volc", "temporal_filter_model"}
    unknown = sorted(set(raw).difference(allowed_top))
    if unknown:
        raise SearchServiceConfigError("unsupported service section(s): " + ", ".join(unknown))
    service = _table(raw, "service")
    volc = _table(raw, "volc")
    filter_model = _table(raw, "temporal_filter_model")
    _reject(
        service,
        "[service]",
        {
            "protocol",
            "bind_host",
            "port",
            "state_dir",
            "max_inflight_searches",
            "queue_limit",
            "drain_timeout_seconds",
            "test_only",
        },
    )
    _reject(volc, "[volc]", {"access_key_env", "secret_key_env"})
    _reject(
        filter_model,
        "[temporal_filter_model]",
        {
            "endpoint_env",
            "name_env",
            "api_key_env",
            "timeout_seconds",
            "retries",
        },
    )
    protocol = _text(service.get("protocol", SEARCH_SERVICE_PROTOCOL), "service.protocol")
    if protocol != SEARCH_SERVICE_PROTOCOL:
        raise SearchServiceConfigError("service.protocol must be search_filter_http/v1")
    bind_host = _text(service.get("bind_host", "127.0.0.1"), "service.bind_host")
    if bind_host != "127.0.0.1":
        raise SearchServiceConfigError("service.bind_host must be literal IPv4 127.0.0.1")
    port = _int(service.get("port", 8765), "service.port", minimum=1)
    if port > 65535:
        raise SearchServiceConfigError("service.port must be <= 65535")
    state_raw = _text(service.get("state_dir", "../data/search-service"), "service.state_dir")
    state_dir = Path(state_raw).expanduser()
    if not state_dir.is_absolute():
        state_dir = (config_path.parent / state_dir).resolve()
    return SearchServiceConfig(
        protocol=protocol,
        bind_host=bind_host,
        port=port,
        state_dir=state_dir,
        max_inflight_searches=_int(
            service.get("max_inflight_searches", 8), "service.max_inflight_searches", minimum=1
        ),
        queue_limit=_int(service.get("queue_limit", 64), "service.queue_limit", minimum=1),
        drain_timeout_seconds=_float(
            service.get("drain_timeout_seconds", 30.0),
            "service.drain_timeout_seconds",
            minimum=0.001,
        ),
        volc_access_key_env=_env_name(
            volc.get("access_key_env", "VOLC_ACCESS_KEY_ID"),
            "volc.access_key_env",
            prefix="VOLC_",
        ),
        volc_secret_key_env=_env_name(
            volc.get("secret_key_env", "VOLC_SECRET_ACCESS_KEY"),
            "volc.secret_key_env",
            prefix="VOLC_",
        ),
        filter_endpoint_env=_env_name(
            filter_model.get("endpoint_env", "SEARCH_FILTER_MODEL_ENDPOINT"),
            "temporal_filter_model.endpoint_env",
            prefix="SEARCH_FILTER_MODEL_",
        ),
        filter_name_env=_env_name(
            filter_model.get("name_env", "SEARCH_FILTER_MODEL_NAME"),
            "temporal_filter_model.name_env",
            prefix="SEARCH_FILTER_MODEL_",
        ),
        filter_api_key_env=_env_name(
            filter_model.get("api_key_env", "SEARCH_FILTER_MODEL_API_KEY"),
            "temporal_filter_model.api_key_env",
            prefix="SEARCH_FILTER_MODEL_",
        ),
        filter_timeout_seconds=_float(
            filter_model.get("timeout_seconds", 60.0),
            "temporal_filter_model.timeout_seconds",
            minimum=0.001,
        ),
        filter_retries=_int(
            filter_model.get("retries", 2), "temporal_filter_model.retries", minimum=0
        ),
        test_only=_bool(service.get("test_only", False), "service.test_only"),
    )


def _table(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key, {})
    if not isinstance(value, dict):
        raise SearchServiceConfigError(f"{key} must be a TOML table")
    return value


def _reject(value: dict[str, Any], field: str, allowed: set[str]) -> None:
    unknown = sorted(set(value).difference(allowed))
    if unknown:
        raise SearchServiceConfigError(f"unsupported {field} key(s): {', '.join(unknown)}")


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SearchServiceConfigError(f"{field} must be a non-empty string")
    return value.strip()


def _env_name(value: Any, field: str, *, prefix: str | None = None) -> str:
    name = _text(value, field)
    if not name.replace("_", "").isalnum() or not name[0].isalpha():
        raise SearchServiceConfigError(f"{field} must be an environment variable name")
    if prefix is not None and not name.startswith(prefix):
        raise SearchServiceConfigError(f"{field} must start with {prefix}")
    return name


def _int(value: Any, field: str, *, minimum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise SearchServiceConfigError(f"{field} must be an integer >= {minimum}")
    return value


def _float(value: Any, field: str, *, minimum: float) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool) or float(value) < minimum:
        raise SearchServiceConfigError(f"{field} must be numeric >= {minimum}")
    return float(value)


def _bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise SearchServiceConfigError(f"{field} must be boolean")
    return value


def resolved_service_env(config: SearchServiceConfig) -> dict[str, str]:
    """Return only service-provider values; never pass the complete parent env."""

    names = (
        config.volc_access_key_env,
        config.volc_secret_key_env,
        config.filter_endpoint_env,
        config.filter_name_env,
        config.filter_api_key_env,
    )
    return {name: os.environ[name] for name in names if os.environ.get(name)}


__all__ = [
    "SearchServiceClientConfig",
    "SearchServiceConfig",
    "SearchServiceConfigError",
    "load_search_service_config",
    "resolved_service_env",
]

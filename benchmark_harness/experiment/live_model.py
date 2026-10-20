"""Live ``openai_compatible`` runtime resolution from manifest + environment.

Secrets stay parent-owned environment variables and are never serialized. The
scientific snapshot identity and provider wire name are validated separately;
the dated mini snapshot may use its explicitly authorized Azure wire binding.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from benchmark_harness.config import ResolvedModelProfile
from benchmark_harness.experiment.manifest import ExperimentManifest, ManifestError
from benchmark_harness.experiment.planner import PlanCell
from benchmark_harness.model_identity import (
    MINI_SNAPSHOT_BACKBONE_ID,
    MINI_SNAPSHOT_MODEL_NAME,
    MINI_WIRE_MODEL_NAME,
    MINI_WIRE_NAME_BINDING,
)

MODEL_SEED_NAMESPACE = "model-seed-v1"
AZURE_OPENAI_HOST_SUFFIX = ".services.ai.azure.com"
AZURE_OPENAI_API_PATH = "/openai/v1"
ENDPOINT_ENV = "BENCHMARK_MODEL_ENDPOINT"
NAME_ENV = "BENCHMARK_MODEL_NAME"
API_KEY_ENV = "BENCHMARK_MODEL_API_KEY"
SEARCH_SERVICE_URL_ENV = "BENCHMARK_SEARCH_SERVICE_URL"
SEARCH_SERVICE_TOKEN_FILE_ENV = "BENCHMARK_SEARCH_SERVICE_ADMIN_TOKEN_FILE"
SEARCH_SERVICE_COHORT_ENV = "BENCHMARK_EXPECTED_SERVICE_COHORT_SHA256"
SOURCE_ENV_BY_SYSTEM = {
    "financeharness": "BENCHMARK_FINANCE_SOURCE",
    "finance-harness": "BENCHMARK_FINANCE_SOURCE",
    "odr": "BENCHMARK_ODR_SOURCE",
    "open_deep_research": "BENCHMARK_ODR_SOURCE",
    "gpt-researcher": "BENCHMARK_GPT_SOURCE",
    "gpt_researcher": "BENCHMARK_GPT_SOURCE",
}


class LiveRuntimeError(RuntimeError):
    """Raised when the declared live runtime cannot be resolved exactly."""


def derive_model_seed(seed: int, cell_id: str) -> int:
    """Deterministic per-cell provider seed declared by the manifest."""

    payload = f"{MODEL_SEED_NAMESPACE}|{seed}|{cell_id}"
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16)


def _optional_string(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def declared_model_identity(manifest: ExperimentManifest) -> dict[str, str | None]:
    """Return the manifest's scientific snapshot and transport identities.

    The mini tuple is intentionally strict: its backbone, dated snapshot,
    Azure wire name, and user-authorized binding must agree in both the model
    declaration and the static provenance section.
    """

    model = manifest.v2_metadata.get("model", {})
    backbone = manifest.v2_metadata.get("backbone", {})
    if not isinstance(model, Mapping) or not isinstance(backbone, Mapping):
        raise ManifestError(f"{manifest.schema_version} model identity is missing")
    if manifest.schema_version in {"experiment/v2", "experiment/v4"} and not model and not backbone:
        raise ManifestError(f"{manifest.schema_version} model identity is missing")
    model_backbone = _optional_string(model.get("backbone_id"))
    backbone_id = model_backbone or _optional_string(backbone.get("backbone_id"))
    snapshot_model_name = _optional_string(model.get("snapshot_model_name"))
    wire_model_name = _optional_string(model.get("api_model_name"))
    wire_name_binding = _optional_string(model.get("wire_name_binding"))
    if backbone_id == MINI_SNAPSHOT_BACKBONE_ID:
        if backbone.get("backbone_id") != MINI_SNAPSHOT_BACKBONE_ID:
            raise ManifestError(
                "mini model and backbone declarations must share the dated backbone"
            )
        if model.get("backbone_id") != MINI_SNAPSHOT_BACKBONE_ID:
            raise ManifestError("mini model must declare the exact dated backbone")
        if model.get("snapshot_model_name") != MINI_SNAPSHOT_MODEL_NAME:
            raise ManifestError("mini snapshot_model_name must be the exact dated snapshot")
        if model.get("provider") != "azure_openai":
            raise ManifestError("mini model provider must be azure_openai")
        if model.get("api_model_name") != MINI_WIRE_MODEL_NAME:
            raise ManifestError("mini api_model_name must be the exact Azure wire name")
        if model.get("api_model_name_semantics") != "azure_request_model":
            raise ManifestError("mini api_model_name_semantics must be azure_request_model")
        if model.get("wire_name_binding") != MINI_WIRE_NAME_BINDING:
            raise ManifestError("mini wire_name_binding is missing or invalid")
        provenance = manifest.v2_metadata.get("provenance", {})
        if not isinstance(provenance, Mapping):
            raise ManifestError("mini manifest provenance is missing")
        expected_provenance = {
            "backbone_id": MINI_SNAPSHOT_BACKBONE_ID,
            "snapshot_model_name": MINI_SNAPSHOT_MODEL_NAME,
            "wire_model_name": MINI_WIRE_MODEL_NAME,
            "wire_name_binding": MINI_WIRE_NAME_BINDING,
        }
        if any(provenance.get(key) != value for key, value in expected_provenance.items()):
            raise ManifestError("mini manifest model provenance does not match its model identity")
    return {
        "backbone_id": backbone_id,
        "snapshot_model_name": snapshot_model_name or wire_model_name,
        "wire_model_name": wire_model_name,
        "wire_name_binding": wire_name_binding,
    }


def declared_model_name(manifest: ExperimentManifest) -> str:
    identity = declared_model_identity(manifest)
    name = identity["wire_model_name"]
    if not name:
        raise ManifestError("experiment/v2 model.api_model_name must be a non-empty wire name")
    return name


def declared_endpoint(manifest: ExperimentManifest) -> str:
    model = manifest.v2_metadata.get("model", {})
    endpoint = str(model.get("endpoint", "")).strip() if isinstance(model, Mapping) else ""
    if not endpoint:
        raise ManifestError("experiment/v2 model.endpoint is missing")
    return endpoint


def _normalize_azure_openai_endpoint(endpoint: str) -> str:
    """Validate and canonicalize the Azure OpenAI-compatible base URL."""

    try:
        parsed = urlsplit(endpoint)
        host = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise LiveRuntimeError("live Azure endpoint is malformed") from exc
    if (
        parsed.scheme.lower() != "https"
        or not host
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or port is not None
    ):
        raise LiveRuntimeError("live Azure endpoint must be an HTTPS URL without credentials")
    normalized_host = host.lower()
    if not normalized_host.endswith(
        AZURE_OPENAI_HOST_SUFFIX
    ) or normalized_host == AZURE_OPENAI_HOST_SUFFIX.lstrip("."):
        raise LiveRuntimeError("live endpoint must use an Azure services.ai.azure.com host")
    normalized_path = parsed.path.rstrip("/") or "/"
    if normalized_path != AZURE_OPENAI_API_PATH:
        raise LiveRuntimeError("live Azure endpoint path must end with /openai/v1")
    return f"https://{normalized_host}{AZURE_OPENAI_API_PATH}"


def resolve_live_model_profile(
    manifest: ExperimentManifest, *, cell_id: str
) -> ResolvedModelProfile:
    """Resolve one cell's snapshot and transport profile; secrets stay in env."""

    model = manifest.v2_metadata.get("model", {})
    if not isinstance(model, Mapping):
        raise ManifestError("experiment/v2 model identity is missing")
    identity = declared_model_identity(manifest)
    name = identity["wire_model_name"]
    if not name:
        raise ManifestError("experiment/v2 model.api_model_name must be a non-empty wire name")
    endpoint_env = os.environ.get(ENDPOINT_ENV)
    name_env = os.environ.get(NAME_ENV)
    requires_azure_wire_env = identity["backbone_id"] == MINI_SNAPSHOT_BACKBONE_ID
    if requires_azure_wire_env and not name_env:
        raise LiveRuntimeError(f"{NAME_ENV} is required and must be exactly {MINI_WIRE_MODEL_NAME}")
    if requires_azure_wire_env and not endpoint_env:
        raise LiveRuntimeError(
            f"{ENDPOINT_ENV} is required and must be an Azure /openai/v1 endpoint"
        )
    if requires_azure_wire_env and name_env != name:
        raise LiveRuntimeError(
            f"{NAME_ENV} must be exactly the Azure wire name {name!r}; "
            "arbitrary aliases are forbidden"
        )
    if not requires_azure_wire_env and name_env and name_env.strip() != name:
        raise LiveRuntimeError(
            f"{NAME_ENV} alias is forbidden; the manifest pins {name!r}, got {name_env!r}"
        )
    endpoint = (endpoint_env or declared_endpoint(manifest)).strip()
    if requires_azure_wire_env:
        endpoint = _normalize_azure_openai_endpoint(endpoint)
    elif not endpoint.startswith("https://") or "mock" in endpoint.lower():
        raise LiveRuntimeError(f"live endpoint must be an https provider URL, got {endpoint!r}")
    api_key = os.environ.get(API_KEY_ENV)
    if not api_key or not api_key.strip():
        raise LiveRuntimeError(
            f"{API_KEY_ENV} is required; live execution never falls back to mock"
        )
    temperature = model.get("temperature", 0.2)
    top_p = model.get("top_p")
    max_tokens = int(model.get("max_output_tokens", 32768))
    seed = derive_model_seed(manifest.seed, cell_id)
    return ResolvedModelProfile(
        mode="openai_compatible",
        endpoint=endpoint,
        model_name=name,
        api_key=api_key,
        temperature=float(temperature),
        max_tokens=max_tokens,
        timeout_seconds=600.0,
        retries=0,
        seed=seed,
        top_p=float(top_p) if top_p is not None else None,
        snapshot_model_name=identity["snapshot_model_name"],
        backbone_id=identity["backbone_id"],
        wire_name_binding=identity["wire_name_binding"],
    )


def _source_checkout_paths(manifest: ExperimentManifest) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for system in manifest.systems:
        env_name = SOURCE_ENV_BY_SYSTEM.get(system.system_id)
        if env_name is None:
            raise LiveRuntimeError(f"no source environment binding for {system.system_id!r}")
        raw = os.environ.get(env_name)
        if not raw or not raw.strip():
            raise LiveRuntimeError(f"{env_name} is required for exact {system.system_id} checkout")
        path = Path(raw).expanduser().resolve()
        if not path.is_dir():
            raise LiveRuntimeError(f"{env_name} checkout is unavailable: {path}")
        result[system.system_id] = path
    return result


def _search_service_binding() -> tuple[str, Path]:
    url = os.environ.get(SEARCH_SERVICE_URL_ENV, "http://127.0.0.1:8765")
    token_file = os.environ.get(
        SEARCH_SERVICE_TOKEN_FILE_ENV,
        str(Path.cwd() / "data/search-service/admin.token"),
    )
    token_path = Path(token_file).expanduser().resolve()
    if not token_path.is_file():
        raise LiveRuntimeError(f"search service admin token file is unavailable: {token_path}")
    return url, token_path


def _declared_service_cohort() -> str:
    value = _optional_string(os.environ.get(SEARCH_SERVICE_COHORT_ENV))
    if (
        value is None
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise LiveRuntimeError(
            f"{SEARCH_SERVICE_COHORT_ENV} must be an externally declared lowercase SHA-256 digest"
        )
    return value


def _verify_declared_service_cohort(identity: Mapping[str, Any], expected: str) -> None:
    if identity.get("service_cohort_sha256") != expected:
        raise LiveRuntimeError(
            "search service cohort mismatch; the externally declared cohort does not match "
            "the resident service"
        )


def validate_live_runtime(
    manifest: ExperimentManifest,
    cells: Iterable[PlanCell],
    *,
    search_service_check: bool = True,
) -> dict[str, Any]:
    """Fail closed before cell 1 on any live runtime shortfall.

    Returns sanitized provenance; the API key itself is never included.
    """

    identity = declared_model_identity(manifest)
    name = identity["wire_model_name"]
    if not name:
        raise ManifestError("experiment/v2 model.api_model_name must be a non-empty wire name")
    profile = resolve_live_model_profile(manifest, cell_id="preflight")
    sources = _source_checkout_paths(manifest)
    provenance: dict[str, Any] = {
        "mode": "openai_compatible",
        "model_name": name,
        "wire_model_name": identity["wire_model_name"],
        "snapshot_model_name": identity["snapshot_model_name"],
        "backbone_id": identity["backbone_id"],
        "wire_name_binding": identity["wire_name_binding"],
        "provider_response_model_names": [],
        "system_fingerprints": [],
        "endpoint": profile.public_dict()["endpoint"],
        "sources": {system_id: str(path) for system_id, path in sorted(sources.items())},
        "model_seed_namespace": MODEL_SEED_NAMESPACE,
    }
    if not any(cell.retrieval_regime == "cutoff_controlled" for cell in cells):
        return provenance
    url, token_path = _search_service_binding()
    provenance["search_service_url"] = url
    provenance["search_service_admin_token_file"] = str(token_path)
    if search_service_check:
        from benchmark_harness.search_service import SearchServiceClient
        from benchmark_harness.search_service.config import SearchServiceClientConfig

        expected_cohort = _declared_service_cohort()
        client = SearchServiceClient(
            SearchServiceClientConfig(
                base_url=url,
                admin_token_file=token_path,
                expected_service_cohort_sha256=expected_cohort,
            )
        )
        ready = client.ready()
        identity = _verify_service_identity(ready)
        _verify_declared_service_cohort(identity, expected_cohort)
        provenance["service_cohort_sha256"] = expected_cohort
    return provenance


def _filter_service_config() -> Any:
    """Resolve the resident service config without reading provider secrets."""

    from benchmark_harness.search_service.config import (
        SearchServiceConfig,
        SearchServiceConfigError,
        load_search_service_config,
    )

    service_config = SearchServiceConfig()
    configured_token = os.environ.get(SEARCH_SERVICE_TOKEN_FILE_ENV)
    token_path = (
        Path(configured_token).expanduser()
        if configured_token
        else Path.cwd() / "data/search-service/admin.token"
    )
    config_path = token_path.resolve().parent / "service-config.toml"
    if config_path.is_file():
        try:
            service_config = load_search_service_config(config_path)
        except SearchServiceConfigError as exc:
            raise LiveRuntimeError("search service config is invalid") from exc
    return service_config


def _verify_service_identity(ready: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed when a stale resident daemon serves older identities.

    Session-principal admission is part of the committed service API and
    implementation identity, so a pre-change daemon must be rejected before
    cell 1 instead of looking merely "ready".  The Volc/cutoff capability
    hash must remain identical to the local committed identity.
    """

    from benchmark_harness.search_service.api import SEARCH_SERVICE_PROTOCOL
    from benchmark_harness.search_service.identities import (
        filter_model_identity,
        service_manifest,
    )
    from benchmark_harness.search_service.server import _filter_profile

    test_only = ready.get("test_only", False)
    if not isinstance(test_only, bool):
        raise LiveRuntimeError("search service ready payload has invalid test_only")
    filter_profile = _filter_profile(_filter_service_config())
    filter_model = filter_model_identity(
        filter_profile.public_dict() if filter_profile is not None else None
    )
    local = service_manifest(test_only=test_only, filter_model=filter_model)
    expected = {
        "api_contract_sha256": local.api_contract_sha256,
        "implementation_manifest_sha256": local.implementation_manifest_sha256,
        "capability_sha256": local.capability_sha256,
        "service_cohort_sha256": local.service_cohort_sha256,
    }
    observed = {key: ready.get(key) for key in expected}
    if any(not isinstance(value, str) or not value for value in observed.values()):
        raise LiveRuntimeError("search service ready payload is missing identity hashes")
    if ready.get("protocol") != SEARCH_SERVICE_PROTOCOL:
        raise LiveRuntimeError("search service ready payload has an invalid protocol")
    if observed != expected:
        raise LiveRuntimeError(
            "search service identity mismatch; restart the resident Search+Filter "
            "service after deploying this build before running the experiment: "
            + "; ".join(f"{key}={value}" for key, value in sorted(observed.items()))
        )
    instance_id = ready.get("instance_id")
    return {
        "protocol": SEARCH_SERVICE_PROTOCOL,
        **expected,
        "filter_model": filter_model,
        "instance_id": instance_id if isinstance(instance_id, str) and instance_id else None,
    }


def build_live_search_session_factory(manifest: ExperimentManifest) -> Any:
    """Build a per-cell SearchService session factory for ODR/GPT.

    FinanceHarness owns its session lifecycle internally through the same
    service, so its cells receive ``None`` here.  Sessions are task-scoped,
    cutoff-bound, and closed by the executor with their receipts preserved.
    """

    url, token_file = _search_service_binding()
    from benchmark_harness.search_service import SearchServiceClient
    from benchmark_harness.search_service.config import SearchServiceClientConfig

    expected_cohort = _declared_service_cohort()
    client = SearchServiceClient(
        SearchServiceClientConfig(
            base_url=url,
            admin_token_file=token_file,
            expected_service_cohort_sha256=expected_cohort,
        )
    )
    ready = client.ready()
    identity = _verify_service_identity(ready)
    _verify_declared_service_cohort(identity, expected_cohort)

    def factory(binding: Any, task: Any, system_id: str) -> Any:
        if system_id in {"financeharness", "finance-harness"}:
            return None
        if not task.cutoff_date:
            raise LiveRuntimeError(
                f"cutoff_controlled cell {binding.cell_id} requires a task cutoff_date"
            )
        return client.open_session(
            run_id=binding.experiment_id,
            task_id=task.task_id,
            profile="deep_research",
            baseline=system_id,
            cutoff_date=task.cutoff_date,
            policy_version="deep_research/v2",
            expected_service_cohort_sha256=expected_cohort,
        )

    return factory


__all__ = [
    "API_KEY_ENV",
    "ENDPOINT_ENV",
    "LiveRuntimeError",
    "NAME_ENV",
    "build_live_search_session_factory",
    "declared_model_identity",
    "declared_model_name",
    "derive_model_seed",
    "resolve_live_model_profile",
    "validate_live_runtime",
]

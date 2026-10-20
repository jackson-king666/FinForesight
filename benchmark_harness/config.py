"""Versioned, credential-free configuration for benchmark experiments.

The default dataset path is the pinned Hugging Face Hub revision.  The legacy
``external_checkout`` source is intentionally explicit: it exists only to
migrate historical fixtures and never becomes a silent fallback for Hub data.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from benchmark_harness.profile_registry import ProfileRegistryError, ProfileSpec, get_profile

HF_DATASET_REPO_ID = "sselaine27/benchmark-research"
HF_DATASET_REVISION = "e941b978af5f8a38fbeb4670ee2d77671a5b551e"
EVALUATOR_SOURCE_COMMIT = "6c3ba942a33bb1ce941a88e4b5a78a11c3153f32"

# FinanceHarness is deliberately not a project dependency.  These values are
# the reproducibility boundary for the optional external population; the
# checkout manager verifies them again before a worker can be started.
FINANCEHARNESS_REPO_URL = "https://github.com/Yijia-Xiao/FinanceHarness.git"
FINANCEHARNESS_REVISION = "c0eb9371c9299616bd39a5080377694ed960688b"
FINANCEHARNESS_TREE_SHA = "4939d3b3dec91460b76ea01731ef892bd0e9c2f0"
FINANCEHARNESS_UV_LOCK_SHA256 = "1a3091a1e24a9c90d83af103e96016eb921c8d93656645be7c8a4bc8ac1e15c7"
FINANCEHARNESS_PYPROJECT_SHA256 = "85ac39ffd947e16ba93afbc8807bd6ceb5614a6e3f1bd909d1518280c78fa213"
FINANCEHARNESS_PROTOCOL = "external_research_stdio/v2"
FINANCEHARNESS_LICENSE_STATUS = "UNDECLARED"

# Kept as an import-compatible name for callers that previously referred to a
# dataset revision constant.  It is now the Hub dataset revision, not a local
# Git checkout revision.
DATASET_REVISION = HF_DATASET_REVISION
DATASET_CONFIG_MAP: dict[str, str] = {
    "A1": "a1",
    "A2-F": "a2_f",
    "A2-T": "a2_t",
    "A2-H": "a2_h",
    "B": "b",
    "C": "c",
    "D": "d",
    "E": "e",
}
DATASET_EXPECTED_COUNTS: dict[str, int] = {
    "a1": 72,
    "a2_f": 10,
    "a2_t": 10,
    "a2_h": 10,
    "b": 33,
    "c": 180,
    "d": 12,
    "e": 12,
}
_ENV_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class ConfigError(ValueError):
    """Raised when an experiment configuration is invalid or incomplete."""


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """Dataset loader configuration.

    ``root`` exists solely for explicit ``external_checkout`` migration use.
    New configurations must use the Hugging Face default and keep runtime
    caches under ``data/.huggingface``.
    """

    source: str = "huggingface"
    repo_id: str = HF_DATASET_REPO_ID
    revision: str = HF_DATASET_REVISION
    split: str = "train"
    data_dir: Path = Path("data")
    offline: bool = False
    config_map: dict[str, str] = field(default_factory=lambda: dict(DATASET_CONFIG_MAP))
    expected_counts: dict[str, int] = field(default_factory=lambda: dict(DATASET_EXPECTED_COUNTS))
    validate_on_preflight: bool = True
    root: Path | None = None

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / ".huggingface"


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Frozen evaluator provenance, independent of dataset loading."""

    manifest_path: Path = Path("benchmark_harness/reference_evaluation/EVALUATOR_MANIFEST.json")
    source_commit: str = EVALUATOR_SOURCE_COMMIT
    require_golden_parity: bool = True


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """A provider-neutral OpenAI-compatible model profile."""

    mode: str = "openai_compatible"
    endpoint_env: str = "BENCHMARK_MODEL_ENDPOINT"
    name_env: str = "BENCHMARK_MODEL_NAME"
    api_key_env: str = "BENCHMARK_MODEL_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout_seconds: float = 60.0
    retries: int = 2
    seed: int | None = None


@dataclass(frozen=True, slots=True)
class ResolvedModelProfile:
    """Runtime-only profile. It is never serialized with its API key."""

    mode: str
    endpoint: str
    model_name: str
    api_key: str | None
    temperature: float
    max_tokens: int
    timeout_seconds: float
    retries: int
    seed: int | None
    top_p: float | None = None
    snapshot_model_name: str | None = None
    backbone_id: str | None = None
    wire_name_binding: str | None = None

    def public_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "endpoint": sanitized_endpoint(self.endpoint),
            "model_name": self.model_name,
            "wire_model_name": self.model_name,
            "snapshot_model_name": self.snapshot_model_name,
            "backbone_id": self.backbone_id,
            "wire_name_binding": self.wire_name_binding,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "timeout_seconds": self.timeout_seconds,
            "retries": self.retries,
            "seed": self.seed,
        }

    def provenance_dict(
        self,
        *,
        provider_response_model_names: tuple[str, ...] | list[str] = (),
        system_fingerprints: tuple[str, ...] | list[str] = (),
    ) -> dict[str, Any]:
        """Return sanitized runtime identity, including observed provider data."""

        response_names = sorted(set(provider_response_model_names))
        return {
            **self.public_dict(),
            "provider_response_model_names": response_names,
            "provider_model_names": response_names,
            "system_fingerprints": sorted(set(system_fingerprints)),
        }


@dataclass(frozen=True, slots=True)
class SearchConfig:
    """Shared, cutoff-bound Volc web-search contract for non-strict profiles."""

    provider: str = "volc"
    access_key_env: str = "VOLC_ACCESS_KEY_ID"
    secret_key_env: str = "VOLC_SECRET_ACCESS_KEY"
    lookback_years: int = 5
    max_tool_calls: int = 3
    max_model_calls: int = 4
    test_only: bool = False


@dataclass(frozen=True, slots=True)
class SearchServiceExperimentConfig:
    """Client-only service binding; provider credentials stay daemon-owned."""

    base_url: str = "http://127.0.0.1:8765"
    admin_token_file: Path = Path("data/search-service/admin.token")
    connect_timeout_seconds: float = 2.0
    request_timeout_seconds: float = 1800.0
    expected_service_cohort_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    input_policy: str = "one_turn_strict"
    concurrency: int = 1
    continue_on_error: bool = True


@dataclass(frozen=True, slots=True)
class TraceConfig:
    content_mode: str = "full"
    results_dir: Path = Path("results")


@dataclass(frozen=True, slots=True)
class DeepResearchConfig:
    """Fail-closed policy and IPC limits for the FinanceHarness population."""

    mode: str = "research"
    grounding_review: bool = True
    max_rounds: int = 24
    max_model_calls: int = 32
    max_search_queries: int = 8
    max_visit_urls: int = 12
    max_tool_calls: int = 40
    task_timeout_seconds: float = 1800.0
    max_frame_bytes: int = 8 * 1024 * 1024
    max_stderr_bytes: int = 1024 * 1024


@dataclass(frozen=True, slots=True)
class ExternalHarnessConfig:
    """Pinned metadata for one explicitly synchronized external harness."""

    adapter: str = "financeharness"
    version: str = "v2"
    repo_url: str = FINANCEHARNESS_REPO_URL
    revision: str = FINANCEHARNESS_REVISION
    tree_sha: str = FINANCEHARNESS_TREE_SHA
    uv_lock_sha256: str = FINANCEHARNESS_UV_LOCK_SHA256
    pyproject_sha256: str = FINANCEHARNESS_PYPROJECT_SHA256
    python_version: str = "3.12"
    protocol: str = FINANCEHARNESS_PROTOCOL
    checkout_root: Path = Path("data/external-harnesses")
    offline: bool = False
    license_status: str = FINANCEHARNESS_LICENSE_STATUS

    @property
    def source_dir(self) -> Path:
        return self.checkout_root / self.adapter


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    search_service: SearchServiceExperimentConfig | None = None
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    trace: TraceConfig = field(default_factory=TraceConfig)
    deep_research: DeepResearchConfig | None = None
    external_harnesses: dict[str, ExternalHarnessConfig] = field(default_factory=dict)
    source_path: Path | None = None

    @property
    def config_hash(self) -> str:
        payload = json.dumps(
            self.public_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def is_strict(self) -> bool:
        return self.execution.input_policy == "one_turn_strict"

    @property
    def is_agentic_web_search(self) -> bool:
        return self.execution.input_policy == "agentic_web_search"

    @property
    def is_deep_research(self) -> bool:
        return self.execution.input_policy == "deep_research"

    @property
    def profile_spec(self) -> ProfileSpec:
        return get_profile(self.execution.input_policy)

    def public_dict(self) -> dict[str, Any]:
        return {
            "dataset": {
                "source": self.dataset.source,
                "repo_id": self.dataset.repo_id,
                "revision": self.dataset.revision,
                "split": self.dataset.split,
                "data_dir": str(self.dataset.data_dir),
                "offline": self.dataset.offline,
                "config_map": dict(self.dataset.config_map),
                "expected_counts": dict(self.dataset.expected_counts),
                "validate_on_preflight": self.dataset.validate_on_preflight,
                "root": str(self.dataset.root) if self.dataset.root else None,
            },
            "evaluation": {
                "manifest_path": str(self.evaluation.manifest_path),
                "source_commit": self.evaluation.source_commit,
                "require_golden_parity": self.evaluation.require_golden_parity,
            },
            "model": asdict(self.model),
            "search": asdict(self.search),
            **(
                {
                    "search_service": {
                        "protocol": "search_filter_http/v1",
                        "service_client_only": True,
                        "expected_service_cohort_sha256": (
                            self.search_service.expected_service_cohort_sha256
                        ),
                    }
                }
                if self.search_service is not None and not self.is_strict
                else {}
            ),
            "execution": asdict(self.execution),
            "trace": {
                "content_mode": self.trace.content_mode,
                "results_dir": str(self.trace.results_dir),
            },
            **(
                {
                    "deep_research": asdict(self.deep_research),
                    "external_harnesses": {
                        name: {
                            **asdict(harness),
                            "checkout_root": str(harness.checkout_root),
                        }
                        for name, harness in sorted(self.external_harnesses.items())
                    },
                }
                if self.is_deep_research and self.deep_research is not None
                else {}
            ),
        }

    def resolve_model(self, *, require_ready: bool) -> ResolvedModelProfile | None:
        """Resolve endpoint settings only when an adapter needs a real model."""

        if self.model.mode == "mock":
            return ResolvedModelProfile(
                mode="mock",
                endpoint="mock://local",
                model_name="ground-truth-mock (non-performance)",
                api_key=None,
                temperature=self.model.temperature,
                max_tokens=self.model.max_tokens,
                timeout_seconds=_resolved_timeout(self.model.timeout_seconds),
                retries=_resolved_retries(self.model.retries),
                seed=self.model.seed,
            )

        if self.model.mode != "openai_compatible":
            raise ConfigError(f"Unsupported model.mode: {self.model.mode!r}")

        # Explicit benchmark variables retain priority for backwards
        # compatibility.  A normal cloud deployment can instead use the
        # concise Aliyun/DashScope `.env` mapping documented in .env.example.
        endpoint = _first_environment_value(
            self.model.endpoint_env,
            "BENCHMARK_MODEL_ENDPOINT",
            "ALIYUN_BASE_URL",
            "DASHSCOPE_BASE_URL",
        )
        model_name = _first_environment_value(
            self.model.name_env,
            "BENCHMARK_MODEL_NAME",
            "ALIYUN_MODEL",
            "DASHSCOPE_MODEL",
        )
        if endpoint and not model_name:
            model_name = "deepseek-v4-flash"
        api_key = (
            _first_environment_value(
                self.model.api_key_env,
                "BENCHMARK_MODEL_API_KEY",
                "ALIYUN_API_KEY",
                "DASHSCOPE_API_KEY",
                "BAILIAN_API_KEY",
            )
            or None
        )
        missing = [
            name
            for name, value in (
                (self.model.endpoint_env + " or ALIYUN_BASE_URL", endpoint),
                (self.model.name_env + " or ALIYUN_MODEL", model_name),
            )
            if not value
        ]
        if missing:
            if require_ready:
                raise ConfigError(
                    "Real endpoint is not configured; set "
                    + ", ".join(missing)
                    + " (and optionally "
                    + self.model.api_key_env
                    + " / ALIYUN_API_KEY"
                    + " for authenticated providers)."
                )
            return None

        return ResolvedModelProfile(
            mode="openai_compatible",
            endpoint=endpoint,
            model_name=model_name,
            api_key=api_key,
            temperature=self.model.temperature,
            max_tokens=self.model.max_tokens,
            timeout_seconds=_resolved_timeout(self.model.timeout_seconds),
            retries=_resolved_retries(self.model.retries),
            seed=self.model.seed,
        )


def sanitized_endpoint(endpoint: str) -> str:
    """Drop query/credentials before endpoint provenance is persisted."""

    if "://" not in endpoint:
        return endpoint.split("?", 1)[0]
    scheme, rest = endpoint.split("://", 1)
    authority_and_path = rest.split("?", 1)[0]
    authority, separator, path = authority_and_path.partition("/")
    authority = authority.rsplit("@", 1)[-1]
    return f"{scheme}://{authority}{separator}{path}".rstrip("/")


def _first_environment_value(*names: str) -> str:
    """Return the first non-empty named environment value, preserving precedence."""

    for name in dict.fromkeys(names):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def _resolved_timeout(configured: float) -> float:
    """Apply endpoint-neutral timeout aliases without persisting secret settings."""

    raw = _first_environment_value("BENCHMARK_MODEL_TIMEOUT", "ALIYUN_TIMEOUT", "DASHSCOPE_TIMEOUT")
    if not raw:
        return configured
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError("BENCHMARK_MODEL_TIMEOUT / ALIYUN_TIMEOUT must be numeric.") from exc
    if value <= 0:
        raise ConfigError("BENCHMARK_MODEL_TIMEOUT / ALIYUN_TIMEOUT must be positive.")
    return value


def _resolved_retries(configured: int) -> int:
    """Let an environment override set an operational retry policy per deployment."""

    raw = _first_environment_value("BENCHMARK_MODEL_RETRIES")
    if not raw:
        return configured
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError("BENCHMARK_MODEL_RETRIES must be an integer.") from exc
    if value < 0:
        raise ConfigError("BENCHMARK_MODEL_RETRIES must be >= 0.")
    return value


def _required_mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"Expected [{key}] table in configuration.")
    return value


def _optional_mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key, {})
    if not isinstance(value, dict):
        raise ConfigError(f"[{key}] must be a TOML table when present.")
    return value


def _reject_unknown_keys(data: dict[str, Any], *, field: str, allowed: set[str]) -> None:
    unknown = sorted(set(data).difference(allowed))
    if unknown:
        raise ConfigError(f"Unsupported {field} key(s): {', '.join(unknown)}.")


def _sha256_value(value: Any, *, field: str) -> str:
    parsed = _expand_required(value, field=field)
    if not re.fullmatch(r"[0-9a-f]{64}", parsed):
        raise ConfigError(f"{field} must be a lowercase SHA-256 hex digest.")
    return parsed


def _git_sha_value(value: Any, *, field: str) -> str:
    parsed = _expand_required(value, field=field)
    if not re.fullmatch(r"[0-9a-f]{40}", parsed):
        raise ConfigError(f"{field} must be a 40-character lowercase Git SHA.")
    return parsed


def _expand_required(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field} must be a non-empty string.")

    missing: list[str] = []

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        resolved = os.environ.get(name)
        if resolved is None:
            missing.append(name)
            return match.group(0)
        return resolved

    expanded = _ENV_REFERENCE.sub(replace, value)
    if missing:
        raise ConfigError(f"{field} references unset environment variable(s): {', '.join(missing)}")
    return expanded


def _as_bool(value: Any, *, field: str, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ConfigError(f"{field} must be a boolean.")
    return value


def _as_int(value: Any, *, field: str, default: int, minimum: int | None = None) -> int:
    if value is None:
        return default
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"{field} must be an integer.")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{field} must be >= {minimum}.")
    return value


def _as_float(value: Any, *, field: str, default: float, minimum: float | None = None) -> float:
    if value is None:
        return default
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ConfigError(f"{field} must be numeric.")
    parsed = float(value)
    if minimum is not None and parsed < minimum:
        raise ConfigError(f"{field} must be >= {minimum}.")
    return parsed


def _as_optional_int(value: Any, *, field: str) -> int | None:
    if value is None:
        return None
    return _as_int(value, field=field, default=0)


def _string_map(value: Any, *, field: str, default: dict[str, str]) -> dict[str, str]:
    if value is None:
        return dict(default)
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(item, str) and item for key, item in value.items()
    ):
        raise ConfigError(f"{field} must be a non-empty-string mapping.")
    return dict(value)


def _count_map(value: Any, *, field: str, default: dict[str, int]) -> dict[str, int]:
    if value is None:
        return dict(default)
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(item, int) and not isinstance(item, bool) and item >= 0
        for key, item in value.items()
    ):
        raise ConfigError(f"{field} must map strings to non-negative integer counts.")
    return dict(value)


def _resolve_path(value: Any, *, field: str, config_path: Path, default: Path) -> Path:
    raw = default if value is None else value
    if not isinstance(raw, str | Path) or not str(raw):
        raise ConfigError(f"{field} must be a non-empty path string.")
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def _assert_no_external_secret_values(value: Any, *, field: str) -> None:
    """Reject credential-shaped keys in the external checkout declaration."""

    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            if any(
                marker in normalized
                for marker in ("api_key", "apikey", "secret", "token", "password", "credential")
            ):
                raise ConfigError(f"{field}.{key} must not contain credentials.")
            _assert_no_external_secret_values(item, field=f"{field}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_no_external_secret_values(item, field=f"{field}[{index}]")


def _parse_deep_configuration(
    *,
    deep_raw: dict[str, Any],
    external_harnesses_raw: dict[str, Any],
    config_path: Path,
    profile_spec: ProfileSpec,
) -> tuple[DeepResearchConfig | None, dict[str, ExternalHarnessConfig]]:
    """Parse the opt-in external population without adding a host dependency."""

    if not profile_spec.requires_external_harness:
        if deep_raw or external_harnesses_raw:
            raise ConfigError(
                "Unsupported top-level configuration section(s): deep_research, external_harnesses."
            )
        return None, {}

    _reject_unknown_keys(
        deep_raw,
        field="[deep_research]",
        allowed={
            "mode",
            "grounding_review",
            "max_rounds",
            "max_model_calls",
            "max_search_queries",
            "max_visit_urls",
            "max_tool_calls",
            "task_timeout_seconds",
            "max_frame_bytes",
            "max_stderr_bytes",
        },
    )
    if deep_raw.get("mode", "research") != "research":
        raise ConfigError("deep_research.mode must be 'research'.")
    deep = DeepResearchConfig(
        mode="research",
        grounding_review=_as_bool(
            deep_raw.get("grounding_review"),
            field="deep_research.grounding_review",
            default=True,
        ),
        max_rounds=_as_int(
            deep_raw.get("max_rounds"), field="deep_research.max_rounds", default=24, minimum=1
        ),
        max_model_calls=_as_int(
            deep_raw.get("max_model_calls"),
            field="deep_research.max_model_calls",
            default=32,
            minimum=1,
        ),
        max_search_queries=_as_int(
            deep_raw.get("max_search_queries"),
            field="deep_research.max_search_queries",
            default=8,
            minimum=1,
        ),
        max_visit_urls=_as_int(
            deep_raw.get("max_visit_urls"),
            field="deep_research.max_visit_urls",
            default=12,
            minimum=1,
        ),
        max_tool_calls=_as_int(
            deep_raw.get("max_tool_calls"),
            field="deep_research.max_tool_calls",
            default=40,
            minimum=1,
        ),
        task_timeout_seconds=_as_float(
            deep_raw.get("task_timeout_seconds"),
            field="deep_research.task_timeout_seconds",
            default=1800.0,
            minimum=0.001,
        ),
        max_frame_bytes=_as_int(
            deep_raw.get("max_frame_bytes"),
            field="deep_research.max_frame_bytes",
            default=8 * 1024 * 1024,
            minimum=1024,
        ),
        max_stderr_bytes=_as_int(
            deep_raw.get("max_stderr_bytes"),
            field="deep_research.max_stderr_bytes",
            default=1024 * 1024,
            minimum=1024,
        ),
    )
    locked_budget = asdict(DeepResearchConfig())
    for field_name, expected in locked_budget.items():
        if asdict(deep)[field_name] != expected:
            raise ConfigError(
                f"deep_research.{field_name} is locked to {expected!r} for policy deep_research/v2."
            )

    if set(external_harnesses_raw) != {"financeharness"}:
        raise ConfigError(
            "deep_research requires exactly one [external_harnesses.financeharness] table."
        )
    finance_raw = external_harnesses_raw["financeharness"]
    if not isinstance(finance_raw, dict):
        raise ConfigError("[external_harnesses.financeharness] must be a TOML table.")
    _assert_no_external_secret_values(finance_raw, field="external_harnesses.financeharness")
    _reject_unknown_keys(
        finance_raw,
        field="[external_harnesses.financeharness]",
        allowed={
            "adapter",
            "version",
            "repo_url",
            "revision",
            "tree_sha",
            "tree",
            "uv_lock_sha256",
            "lockfile_sha256",
            "pyproject_sha256",
            "project_sha256",
            "python_version",
            "python",
            "protocol",
            "checkout_root",
            "offline",
            "license_status",
        },
    )

    adapter = _expand_required(
        finance_raw.get("adapter", "financeharness"),
        field="external_harnesses.financeharness.adapter",
    )
    if adapter != "financeharness":
        raise ConfigError("Only the financeharness external adapter is supported.")
    version = _expand_required(
        finance_raw.get("version", "v2"), field="external_harnesses.financeharness.version"
    )
    if version != "v2":
        raise ConfigError("external_harnesses.financeharness.version must be 'v2'.")
    repo_url = _expand_required(
        finance_raw.get("repo_url", FINANCEHARNESS_REPO_URL),
        field="external_harnesses.financeharness.repo_url",
    )
    if repo_url != FINANCEHARNESS_REPO_URL:
        raise ConfigError("FinanceHarness repo_url must use the locked upstream repository.")
    revision = _git_sha_value(
        finance_raw.get("revision", FINANCEHARNESS_REVISION),
        field="external_harnesses.financeharness.revision",
    )
    if revision != FINANCEHARNESS_REVISION:
        raise ConfigError("FinanceHarness revision must use the locked upstream SHA.")
    tree_sha = _git_sha_value(
        finance_raw.get("tree_sha", finance_raw.get("tree", FINANCEHARNESS_TREE_SHA)),
        field="external_harnesses.financeharness.tree_sha",
    )
    if tree_sha != FINANCEHARNESS_TREE_SHA:
        raise ConfigError("FinanceHarness tree_sha does not match the locked checkout.")
    uv_lock_sha256 = _sha256_value(
        finance_raw.get(
            "uv_lock_sha256", finance_raw.get("lockfile_sha256", FINANCEHARNESS_UV_LOCK_SHA256)
        ),
        field="external_harnesses.financeharness.uv_lock_sha256",
    )
    if uv_lock_sha256 != FINANCEHARNESS_UV_LOCK_SHA256:
        raise ConfigError("FinanceHarness uv.lock hash does not match the locked checkout.")
    pyproject_sha256 = _sha256_value(
        finance_raw.get(
            "pyproject_sha256",
            finance_raw.get("project_sha256", FINANCEHARNESS_PYPROJECT_SHA256),
        ),
        field="external_harnesses.financeharness.pyproject_sha256",
    )
    if pyproject_sha256 != FINANCEHARNESS_PYPROJECT_SHA256:
        raise ConfigError("FinanceHarness pyproject.toml hash does not match the locked checkout.")
    python_version = _expand_required(
        finance_raw.get("python_version", finance_raw.get("python", "3.12")),
        field="external_harnesses.financeharness.python_version",
    )
    if python_version != "3.12":
        raise ConfigError("FinanceHarness requires the isolated Python 3.12 environment.")
    protocol = _expand_required(
        finance_raw.get("protocol", FINANCEHARNESS_PROTOCOL),
        field="external_harnesses.financeharness.protocol",
    )
    if protocol != FINANCEHARNESS_PROTOCOL:
        raise ConfigError("FinanceHarness protocol must be external_research_stdio/v2.")
    license_status = _expand_required(
        finance_raw.get("license_status", FINANCEHARNESS_LICENSE_STATUS),
        field="external_harnesses.financeharness.license_status",
    )
    if license_status != FINANCEHARNESS_LICENSE_STATUS:
        raise ConfigError("FinanceHarness license_status must remain UNDECLARED.")
    harness = ExternalHarnessConfig(
        adapter=adapter,
        version=version,
        repo_url=repo_url,
        revision=revision,
        tree_sha=tree_sha,
        uv_lock_sha256=uv_lock_sha256,
        pyproject_sha256=pyproject_sha256,
        python_version=python_version,
        protocol=protocol,
        checkout_root=_resolve_path(
            finance_raw.get("checkout_root", "../data/external-harnesses"),
            field="external_harnesses.financeharness.checkout_root",
            config_path=config_path,
            default=Path("data/external-harnesses"),
        ),
        offline=_as_bool(
            finance_raw.get("offline"),
            field="external_harnesses.financeharness.offline",
            default=False,
        ),
        license_status=license_status,
    )
    return deep, {adapter: harness}


def _parse_search_service_configuration(
    raw: dict[str, Any], *, config_path: Path, profile_spec: ProfileSpec
) -> SearchServiceExperimentConfig | None:
    """Parse only the client binding; daemon credentials never belong here."""

    if not raw:
        return (
            None
            if not profile_spec.requires_search
            else SearchServiceExperimentConfig(
                admin_token_file=(
                    config_path.parent / "../data/search-service/admin.token"
                ).resolve()
            )
        )
    if not profile_spec.requires_search:
        raise ConfigError("[search_service] is not allowed for one_turn_strict.")
    _reject_unknown_keys(
        raw,
        field="[search_service]",
        allowed={
            "base_url",
            "admin_token_file",
            "connect_timeout_seconds",
            "request_timeout_seconds",
            "expected_service_cohort_sha256",
        },
    )
    base_url = raw.get("base_url", "http://127.0.0.1:8765")
    if not isinstance(base_url, str) or base_url != base_url.strip():
        raise ConfigError("search_service.base_url must be a literal loopback URL.")
    from benchmark_harness.search_service.client import _validate_base_url

    try:
        normalized_base_url = _validate_base_url(base_url)
    except Exception as exc:
        raise ConfigError("search_service.base_url must be http://127.0.0.1:<port>.") from exc
    token_file = _resolve_path(
        raw.get("admin_token_file", "../data/search-service/admin.token"),
        field="search_service.admin_token_file",
        config_path=config_path,
        default=Path("data/search-service/admin.token"),
    )
    connect_timeout = _as_float(
        raw.get("connect_timeout_seconds"),
        field="search_service.connect_timeout_seconds",
        default=2.0,
        minimum=0.001,
    )
    request_timeout = _as_float(
        raw.get("request_timeout_seconds"),
        field="search_service.request_timeout_seconds",
        default=1800.0,
        minimum=0.001,
    )
    expected_cohort = raw.get("expected_service_cohort_sha256")
    if expected_cohort is not None and (
        not isinstance(expected_cohort, str)
        or len(expected_cohort) != 64
        or any(character not in "0123456789abcdef" for character in expected_cohort)
    ):
        raise ConfigError(
            "search_service.expected_service_cohort_sha256 must be a lowercase SHA-256 hex digest."
        )
    return SearchServiceExperimentConfig(
        base_url=normalized_base_url,
        admin_token_file=token_file,
        connect_timeout_seconds=connect_timeout,
        request_timeout_seconds=request_timeout,
        expected_service_cohort_sha256=expected_cohort,
    )


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load TOML and validate the reproducibility-bearing experiment contract."""

    config_path = Path(path).resolve()
    try:
        with config_path.open("rb") as handle:
            raw = tomllib.load(handle)
    except FileNotFoundError as exc:
        raise ConfigError(f"Configuration file does not exist: {config_path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid TOML in {config_path}: {exc}") from exc

    dataset_raw = _required_mapping(raw, "dataset")
    model_raw = _required_mapping(raw, "model")
    evaluation_raw = _optional_mapping(raw, "evaluation")
    search_raw = _optional_mapping(raw, "search")
    execution_raw = _optional_mapping(raw, "execution")
    trace_raw = _optional_mapping(raw, "trace")
    search_service_raw = _optional_mapping(raw, "search_service")
    deep_raw = _optional_mapping(raw, "deep_research")
    external_harnesses_raw = _optional_mapping(raw, "external_harnesses")
    supported_sections = {
        "dataset",
        "model",
        "evaluation",
        "search",
        "search_service",
        "execution",
        "trace",
        "deep_research",
        "external_harnesses",
    }
    unsupported_sections = sorted(set(raw).difference(supported_sections))
    if unsupported_sections:
        raise ConfigError(
            "Unsupported top-level configuration section(s): "
            + ", ".join(unsupported_sections)
            + "."
        )

    source = dataset_raw.get("source", "huggingface")
    if source not in {"huggingface", "external_checkout"}:
        raise ConfigError("dataset.source must be 'huggingface' or 'external_checkout'.")
    root: Path | None = None
    if source == "external_checkout":
        root = _resolve_path(
            _expand_required(dataset_raw.get("root"), field="dataset.root"),
            field="dataset.root",
            config_path=config_path,
            default=Path("."),
        )
    repo_id = dataset_raw.get("repo_id", HF_DATASET_REPO_ID)
    revision = dataset_raw.get("revision", HF_DATASET_REVISION)
    split = dataset_raw.get("split", "train")
    if not all(isinstance(item, str) and item for item in (repo_id, revision, split)):
        raise ConfigError("dataset.repo_id, dataset.revision, and dataset.split must be strings.")
    data_dir = _resolve_path(
        dataset_raw.get("data_dir", "../data"),
        field="dataset.data_dir",
        config_path=config_path,
        default=Path("data"),
    )
    config_map = _string_map(
        dataset_raw.get("config_map"), field="dataset.config_map", default=DATASET_CONFIG_MAP
    )
    expected_counts = _count_map(
        dataset_raw.get("expected_counts"),
        field="dataset.expected_counts",
        default=DATASET_EXPECTED_COUNTS,
    )
    if set(config_map.values()) != set(expected_counts):
        raise ConfigError("dataset.config_map values and dataset.expected_counts keys must match.")
    if config_map == DATASET_CONFIG_MAP and expected_counts != DATASET_EXPECTED_COUNTS:
        raise ConfigError("The default v0.6 dataset map requires the locked expected counts.")
    dataset = DatasetConfig(
        source=source,
        repo_id=repo_id,
        revision=revision,
        split=split,
        data_dir=data_dir,
        offline=_as_bool(dataset_raw.get("offline"), field="dataset.offline", default=False),
        config_map=config_map,
        expected_counts=expected_counts,
        validate_on_preflight=_as_bool(
            dataset_raw.get("validate_on_preflight"),
            field="dataset.validate_on_preflight",
            default=True,
        ),
        root=root,
    )
    if dataset.source == "huggingface" and dataset.revision != HF_DATASET_REVISION:
        raise ConfigError(
            "dataset.revision must pin the locked v0.6 Hugging Face SHA " + HF_DATASET_REVISION
        )

    default_manifest_path = "../benchmark_harness/reference_evaluation/EVALUATOR_MANIFEST.json"
    manifest_path = _resolve_path(
        evaluation_raw.get("manifest_path", default_manifest_path),
        field="evaluation.manifest_path",
        config_path=config_path,
        default=Path("benchmark_harness/reference_evaluation/EVALUATOR_MANIFEST.json"),
    )
    source_commit = evaluation_raw.get("source_commit", EVALUATOR_SOURCE_COMMIT)
    if not isinstance(source_commit, str) or not source_commit:
        raise ConfigError("evaluation.source_commit must be a non-empty string.")
    require_golden_parity = _as_bool(
        evaluation_raw.get("require_golden_parity"),
        field="evaluation.require_golden_parity",
        default=True,
    )
    if not require_golden_parity:
        raise ConfigError("evaluation.require_golden_parity cannot be disabled for v0.3 scores.")
    evaluation = EvaluationConfig(
        manifest_path=manifest_path,
        source_commit=source_commit,
        require_golden_parity=require_golden_parity,
    )

    mode = model_raw.get("mode", "openai_compatible")
    if mode not in {"openai_compatible", "mock"}:
        raise ConfigError("model.mode must be 'openai_compatible' or 'mock'.")
    model = ModelConfig(
        mode=mode,
        endpoint_env=_expand_required(
            model_raw.get("endpoint_env", "BENCHMARK_MODEL_ENDPOINT"), field="model.endpoint_env"
        ),
        name_env=_expand_required(
            model_raw.get("name_env", "BENCHMARK_MODEL_NAME"), field="model.name_env"
        ),
        api_key_env=_expand_required(
            model_raw.get("api_key_env", "BENCHMARK_MODEL_API_KEY"), field="model.api_key_env"
        ),
        temperature=_as_float(
            model_raw.get("temperature"), field="model.temperature", default=0.0, minimum=0.0
        ),
        max_tokens=_as_int(
            model_raw.get("max_tokens"), field="model.max_tokens", default=1024, minimum=1
        ),
        timeout_seconds=_as_float(
            model_raw.get("timeout_seconds"),
            field="model.timeout_seconds",
            default=60.0,
            minimum=0.001,
        ),
        retries=_as_int(model_raw.get("retries"), field="model.retries", default=2, minimum=0),
        seed=_as_optional_int(model_raw.get("seed"), field="model.seed"),
    )

    search = SearchConfig(
        provider=search_raw.get("provider", "volc"),
        access_key_env=_expand_required(
            search_raw.get("access_key_env", "VOLC_ACCESS_KEY_ID"),
            field="search.access_key_env",
        ),
        secret_key_env=_expand_required(
            search_raw.get("secret_key_env", "VOLC_SECRET_ACCESS_KEY"),
            field="search.secret_key_env",
        ),
        lookback_years=_as_int(
            search_raw.get("lookback_years"), field="search.lookback_years", default=5, minimum=1
        ),
        max_tool_calls=_as_int(
            search_raw.get("max_tool_calls"), field="search.max_tool_calls", default=3, minimum=1
        ),
        max_model_calls=_as_int(
            search_raw.get("max_model_calls"),
            field="search.max_model_calls",
            default=4,
            minimum=1,
        ),
        test_only=_as_bool(search_raw.get("test_only"), field="search.test_only", default=False),
    )
    if search.max_tool_calls != 3 or search.max_model_calls != 4:
        raise ConfigError("search.max_tool_calls and search.max_model_calls are locked to 3 and 4.")

    execution = ExecutionConfig(
        input_policy=execution_raw.get("input_policy", "one_turn_strict"),
        concurrency=_as_int(
            execution_raw.get("concurrency"), field="execution.concurrency", default=1, minimum=1
        ),
        continue_on_error=_as_bool(
            execution_raw.get("continue_on_error"),
            field="execution.continue_on_error",
            default=True,
        ),
    )
    try:
        profile_spec = get_profile(execution.input_policy)
    except ProfileRegistryError as exc:
        raise ConfigError(str(exc)) from exc
    if search.provider != "volc":
        if profile_spec.requires_search and search_raw:
            raise ConfigError(
                "[search] is a legacy non-strict section; configure the resident service in "
                "[search_service]."
            )
        raise ConfigError(
            "search.provider must be 'volc'; no alternate production backend is allowed."
        )
    if profile_spec.requires_search and search_raw:
        raise ConfigError(
            "[search] is a legacy non-strict section; configure the resident service in "
            "[search_service]."
        )
    if not profile_spec.requires_external_harness and (
        "deep_research" in raw or "external_harnesses" in raw
    ):
        raise ConfigError(
            "Unsupported top-level configuration section(s): deep_research, external_harnesses."
        )
    if execution.input_policy == "one_turn_strict":
        if execution.concurrency != 1:
            raise ConfigError("one_turn_strict requires execution.concurrency = 1 for parity.")

    deep_research, external_harnesses = _parse_deep_configuration(
        deep_raw=deep_raw,
        external_harnesses_raw=external_harnesses_raw,
        config_path=config_path,
        profile_spec=profile_spec,
    )
    search_service = _parse_search_service_configuration(
        search_service_raw, config_path=config_path, profile_spec=profile_spec
    )

    content_mode = trace_raw.get("content_mode", "full")
    if content_mode not in {"full", "hash_only"}:
        raise ConfigError("trace.content_mode must be 'full' or 'hash_only'.")
    results_dir = _resolve_path(
        trace_raw.get("results_dir", "../results"),
        field="trace.results_dir",
        config_path=config_path,
        default=Path("results"),
    )
    if profile_spec.requires_separate_results and results_dir.name == "results":
        raise ConfigError(
            f"{profile_spec.name} requires a separate trace.results_dir "
            "(for example results-agentic-web-search)."
        )
    if profile_spec.requires_external_harness and results_dir.name != "results-deep-research":
        raise ConfigError(
            "deep_research requires trace.results_dir to end in results-deep-research."
        )
    trace = TraceConfig(content_mode=content_mode, results_dir=results_dir)

    return ExperimentConfig(
        dataset=dataset,
        evaluation=evaluation,
        model=model,
        search=search,
        search_service=search_service,
        execution=execution,
        trace=trace,
        deep_research=deep_research,
        external_harnesses=external_harnesses,
        source_path=config_path,
    )

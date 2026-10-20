"""Explicit search-service lifecycle with stale-PID and secure-state checks."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .api import SearchServiceError
from .client import SearchServiceClient
from .config import SearchServiceConfig, resolved_service_env


@dataclass(frozen=True, slots=True)
class ServiceStatus:
    state: str
    ready: bool
    pid: int | None
    instance_id: str | None
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "ready": self.ready,
            "pid": self.pid,
            "instance_id": self.instance_id,
            "reason": self.reason,
        }


class SearchServiceLifecycle:
    def __init__(self, config: SearchServiceConfig) -> None:
        self.config = config

    def serve(self) -> None:
        from .server import run_server

        _ensure_loopback(self.config)
        _ensure_runtime_dir(self.config)
        run_server(self.config)

    def start(self, *, wait_seconds: float = 10.0) -> ServiceStatus:
        _ensure_loopback(self.config)
        current = self.status()
        if current.ready:
            raise SearchServiceError("service_already_running", status_code=409)
        _ensure_runtime_dir(self.config)
        env = _service_environment(self.config)
        command = [
            sys.executable,
            "-m",
            "benchmark_harness.search_service.server",
            "--config",
            str(_config_path_for_process(self.config)),
        ]
        log = self.config.log_file.open("ab")
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                cwd=str(Path.cwd()),
                env=env,
                start_new_session=True,
            )
        finally:
            log.close()
        deadline = time.monotonic() + wait_seconds
        while time.monotonic() < deadline:
            status = self.status()
            if status.ready:
                return status
            if process.poll() is not None:
                break
            time.sleep(0.05)
        if process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=3.0)
            except (OSError, subprocess.TimeoutExpired):
                process.kill()
        raise SearchServiceError("service_start_timeout", status_code=503)

    def status(self) -> ServiceStatus:
        try:
            state = json.loads(self.config.state_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return ServiceStatus("stopped", False, None, None, "state_unavailable")
        pid = state.get("pid")
        instance_id = state.get("instance_id")
        if not isinstance(pid, int) or not isinstance(instance_id, str):
            return ServiceStatus("unhealthy", False, None, None, "state_invalid")
        if not _pid_matches(pid, state.get("pid_start_marker")):
            return ServiceStatus("stopped", False, pid, instance_id, "stale_pid")
        try:
            client = SearchServiceClient(
                base_url=f"http://127.0.0.1:{self.config.port}",
                admin_token_file=self.config.admin_token_file,
            )
            health = client.health()
            if health.get("instance_id") != instance_id:
                return ServiceStatus("unhealthy", False, pid, instance_id, "instance_mismatch")
            if state.get("draining"):
                return ServiceStatus("draining", False, pid, instance_id, "draining")
            ready = client.ready()
            if ready.get("instance_id") != instance_id:
                return ServiceStatus(
                    "unhealthy", False, pid, instance_id, "ready_instance_mismatch"
                )
            for key in (
                "api_contract_sha256",
                "implementation_manifest_sha256",
                "capability_sha256",
                "service_cohort_sha256",
            ):
                if state.get(key) != ready.get(key):
                    return ServiceStatus("unhealthy", False, pid, instance_id, f"{key}_mismatch")
            return ServiceStatus("ready", True, pid, instance_id)
        except Exception as exc:
            return ServiceStatus("unhealthy", False, pid, instance_id, type(exc).__name__)

    def stop(self, *, force: bool = False, wait_seconds: float | None = None) -> ServiceStatus:
        status = self.status()
        if not status.pid or not status.instance_id:
            return status
        client = SearchServiceClient(
            base_url=f"http://127.0.0.1:{self.config.port}",
            admin_token_file=self.config.admin_token_file,
        )
        try:
            client.drain()
        except Exception:
            if not force:
                raise
        timeout = self.config.drain_timeout_seconds if wait_seconds is None else wait_seconds
        if not force:
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline and _active_sessions(self.config) > 0:
                time.sleep(0.05)
            active = _active_sessions(self.config)
            if active:
                return ServiceStatus(
                    "draining",
                    False,
                    status.pid,
                    status.instance_id,
                    f"active_sessions={active}",
                )
        if not _pid_matches(status.pid, _state_pid_marker(self.config)):
            raise SearchServiceError("stale_pid_refused", status_code=409)
        os.kill(status.pid, signal.SIGTERM)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not _pid_matches(status.pid, _state_pid_marker(self.config)):
                return ServiceStatus("stopped", False, status.pid, status.instance_id)
            time.sleep(0.05)
        if force:
            os.kill(status.pid, signal.SIGKILL)
            return ServiceStatus("stopped", False, status.pid, status.instance_id, "forced")
        raise SearchServiceError("service_stop_timeout", status_code=504)


def _ensure_loopback(config: SearchServiceConfig) -> None:
    if config.bind_host != "127.0.0.1":
        raise SearchServiceError("service_bind_not_loopback", status_code=400)


def _ensure_runtime_dir(config: SearchServiceConfig) -> None:
    config.state_dir.mkdir(parents=True, exist_ok=True)
    config.audit_dir.mkdir(parents=True, exist_ok=True)
    config.state_dir.chmod(0o700)
    config.audit_dir.chmod(0o700)


def _exclusive_lock(config: SearchServiceConfig):
    _ensure_runtime_dir(config)
    try:
        import fcntl

        handle = config.lock_file.open("a+")
        os.fchmod(handle.fileno(), 0o600)
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return handle
    except (OSError, ImportError) as exc:
        raise SearchServiceError("service_lock_unavailable", status_code=409) from exc


def _service_environment(config: SearchServiceConfig) -> dict[str, str]:
    allowed = {"PATH", "PYTHONPATH", "PYTHONUNBUFFERED", "PYTHONNOUSERSITE", "LANG", "LC_ALL"}
    env = {key: value for key, value in os.environ.items() if key in allowed}
    for key, value in resolved_service_env(config).items():
        env[key] = value
    return env


def _config_path_for_process(config: SearchServiceConfig) -> Path:
    # Detached starts serialize the validated config into a private TOML file
    # so the child does not depend on the caller's current working directory.
    path = config.state_dir / "service-config.toml"
    lines = [
        "[service]",
        f'protocol = "{config.protocol}"',
        f'bind_host = "{config.bind_host}"',
        f"port = {config.port}",
        f'state_dir = "{config.state_dir}"',
        f"max_inflight_searches = {config.max_inflight_searches}",
        f"queue_limit = {config.queue_limit}",
        f"drain_timeout_seconds = {config.drain_timeout_seconds}",
        f"test_only = {str(config.test_only).lower()}",
        "",
        "[volc]",
        f'access_key_env = "{config.volc_access_key_env}"',
        f'secret_key_env = "{config.volc_secret_key_env}"',
        "",
        "[temporal_filter_model]",
        f'endpoint_env = "{config.filter_endpoint_env}"',
        f'name_env = "{config.filter_name_env}"',
        f'api_key_env = "{config.filter_api_key_env}"',
        f"timeout_seconds = {config.filter_timeout_seconds}",
        f"retries = {config.filter_retries}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o600)
    return path


def _state_pid_marker(config: SearchServiceConfig) -> str | None:
    try:
        value = json.loads(config.state_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    marker = value.get("pid_start_marker")
    return marker if isinstance(marker, str) else None


def _active_sessions(config: SearchServiceConfig) -> int:
    try:
        value = json.loads(config.state_file.read_text(encoding="utf-8")).get("active_sessions", 0)
    except (OSError, json.JSONDecodeError):
        return 0
    return value if isinstance(value, int) and value >= 0 else 0


def _pid_matches(pid: int, marker: Any) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    if marker is None:
        return True
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[-1].split()
        return len(fields) <= 19 or fields[19] == marker
    except OSError:
        return True


__all__ = ["SearchServiceLifecycle", "ServiceStatus"]

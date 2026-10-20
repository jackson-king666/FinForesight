"""FastAPI application and foreground entry point for Search+Filter service."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from typing import Any

from fastapi import Request

from benchmark_harness.config import ResolvedModelProfile
from benchmark_harness.profile_registry import ProfileRegistryError

from .api import (
    MAX_REQUEST_BYTES,
    SEARCH_SERVICE_PROTOCOL,
    SearchServiceError,
    redact_error_body,
)
from .config import SearchServiceConfig, load_search_service_config
from .engine import SearchFilterEngine
from .state import SearchServiceState


def create_app(
    config: SearchServiceConfig,
    *,
    engine: Any | None = None,
    state: SearchServiceState | None = None,
):
    """Create a closed-route FastAPI app; no routes are registered implicitly."""

    try:
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse
    except ImportError as exc:  # pragma: no cover - dependency gate
        raise RuntimeError("FastAPI is required for search-service") from exc

    service_state = state or SearchServiceState(
        config, engine=engine or _engine_from_environment(config)
    )
    app = FastAPI(
        title="Benchmark Search+Filter Service",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.search_service = service_state

    async def body(request: Request) -> dict[str, Any]:
        raw = await request.body()
        if len(raw) > MAX_REQUEST_BYTES:
            raise SearchServiceError("request_too_large", status_code=413)
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SearchServiceError("json_invalid", status_code=400) from exc
        if not isinstance(value, dict):
            raise SearchServiceError("json_object_required", status_code=400)
        return value

    def auth(request: Request, *, admin: bool) -> str:
        value = request.headers.get("authorization", "")
        scheme, _, token = value.partition(" ")
        if scheme.lower() != "bearer" or not token or len(token) > 512:
            raise SearchServiceError("authorization_required", status_code=401)
        if admin:
            import hmac

            if not hmac.compare_digest(service_state.admin_token, token):
                raise SearchServiceError("admin_token_invalid", status_code=401)
        return token

    @app.exception_handler(SearchServiceError)
    async def service_error_handler(_request: Request, exc: SearchServiceError):
        return JSONResponse(status_code=exc.status_code, content=redact_error_body(exc.code))

    @app.exception_handler(ProfileRegistryError)
    async def registry_error_handler(_request: Request, _exc: ProfileRegistryError):
        return JSONResponse(status_code=400, content=redact_error_body("profile_invalid"))

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {
            "status": "draining" if service_state.draining else "ok",
            "protocol": SEARCH_SERVICE_PROTOCOL,
            "instance_id": service_state.instance_id,
        }

    @app.get("/readyz")
    async def readyz(request: Request) -> dict[str, Any]:
        auth(request, admin=True)
        return service_state.ready_payload()

    @app.post("/v1/sessions")
    async def open_session(request: Request) -> dict[str, Any]:
        auth(request, admin=True)
        return service_state.open_session(await body(request))

    @app.post("/v1/sessions/{session_id}/search")
    async def search(request: Request, session_id: str) -> dict[str, Any]:
        token = auth(request, admin=False)
        # Search performs blocking provider I/O and bounded retry sleeps.  Keep
        # that synchronous engine contract off FastAPI's event loop so health,
        # readiness, and unrelated task sessions remain responsive under load.
        return await asyncio.to_thread(service_state.search, session_id, token, await body(request))

    @app.post("/v1/sessions/{session_id}/close")
    async def close(request: Request, session_id: str) -> dict[str, Any]:
        auth(request, admin=True)
        return service_state.close(session_id, await body(request))

    @app.post("/v1/admin/drain")
    async def drain(request: Request) -> dict[str, Any]:
        auth(request, admin=True)
        return service_state.drain(await body(request))

    return app


def _engine_from_environment(config: SearchServiceConfig) -> SearchFilterEngine:
    profile = _filter_profile(config)
    return SearchFilterEngine(
        access_key_env=config.volc_access_key_env,
        secret_key_env=config.volc_secret_key_env,
        access_key=os.environ.get(config.volc_access_key_env),
        secret_key=os.environ.get(config.volc_secret_key_env),
        test_only=config.test_only,
        model_profile=profile,
    )


def _filter_profile(config: SearchServiceConfig) -> ResolvedModelProfile | None:
    endpoint = os.environ.get(config.filter_endpoint_env, "").strip()
    model_name = os.environ.get(config.filter_name_env, "").strip()
    api_key = os.environ.get(config.filter_api_key_env, "").strip() or None
    if not endpoint or not model_name:
        return None
    return ResolvedModelProfile(
        mode="openai_compatible",
        endpoint=endpoint,
        model_name=model_name,
        api_key=api_key,
        temperature=0.0,
        # HUA-174's real filter stage needs enough output budget to return a
        # bounded KEEP/DROP verdict for long candidate snippets.  This is a
        # service-owned profile, not an agent model budget.
        max_tokens=2048,
        timeout_seconds=config.filter_timeout_seconds,
        retries=config.filter_retries,
        seed=None,
    )


def run_server(config: SearchServiceConfig) -> None:
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - dependency gate
        raise RuntimeError("Uvicorn is required for search-service") from exc
    if config.bind_host != "127.0.0.1":
        raise RuntimeError("search-service refuses non-loopback bind")
    # Keep the advisory lock alive for the complete foreground server
    # lifetime.  This makes `start`/`serve` race-safe without a PID-only
    # assumption, and releases it when Uvicorn exits.
    from .lifecycle import _ensure_runtime_dir, _exclusive_lock

    _ensure_runtime_dir(config)
    lock = _exclusive_lock(config)
    try:
        app = create_app(config)
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=config.port,
            log_config=None,
            access_log=False,
            server_header=False,
        )
    finally:
        lock.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="benchmark-search-service")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    run_server(load_search_service_config(args.config))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["create_app", "main", "run_server"]

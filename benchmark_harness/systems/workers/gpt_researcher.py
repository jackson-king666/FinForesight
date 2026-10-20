"""Disposable one-task worker for pinned GPT-Researcher ``deep`` mode.

The worker imports the pinned ``Config``, custom retriever, scraper, prompt,
agent, and ``DeepResearchSkill`` modules from the read-only source checkout.
Optional upstream dependencies are represented by inert import shims so the
offline benchmark environment cannot accidentally open a network client.
The real source classes remain the execution path: only their provider seams
are replaced with the typed parent retriever/model broker and declared clock.
"""

from __future__ import annotations

import argparse
import asyncio
import builtins
import importlib.util
import json
import os
import sys
import types
from collections.abc import Callable, Mapping
from datetime import date as _Date
from datetime import datetime as _DateTime
from datetime import time as _Time
from datetime import timedelta as _Timedelta
from datetime import timezone as _Timezone
from pathlib import Path
from typing import Any

from benchmark_harness.systems.adapters.gpt_researcher import (
    GPT_RESEARCHER_BENCHMARK_LOCK,
    GPT_RESEARCHER_BENCHMARK_PROJECT,
    GPT_RESEARCHER_BENCHMARK_PROJECT_SHA256,
    GPT_RESEARCHER_LICENSE_SHA256,
    GPT_RESEARCHER_LOCK_METADATA,
    GPT_RESEARCHER_LOCK_METADATA_SHA256,
    GPT_RESEARCHER_PROVENANCE,
)
from benchmark_harness.systems.clock import DeclaredExperimentClock
from benchmark_harness.systems.fixtures import verify_fixture_source
from benchmark_harness.systems.provenance import (
    verify_declared_file_hashes,
    verify_system_source,
)
from benchmark_harness.systems.workers.common import (
    SEARCH_QUERY_MAX_CHARS,
    WorkerChannel,
    emit_trajectory,
    json_context,
    request_task_bound_prediction,
    response_content,
    send_task_error,
    source_import_path,
    task_start,
)
from benchmark_harness.systems.workers.guards import verify_gpt_researcher_hooks
from benchmark_harness.systems.workers.output_contract import (
    reject_fixture_sentinel,
)


class NetworkDisabledError(RuntimeError):
    """Raised before a scraper or browser can create network side effects."""


class _DeclaredDateTime(_DateTime):
    """datetime implementation whose ``now`` is bound to the task clock."""

    _clock: DeclaredExperimentClock | None = None

    @classmethod
    def now(cls, tz: Any = None) -> _DateTime:
        if cls._clock is None:  # pragma: no cover - construction guard.
            raise NetworkDisabledError("declared_clock_not_installed")
        value = _DateTime.combine(cls._clock.today(), _Time(12, 0, 0))
        return value.replace(tzinfo=tz) if tz is not None else value


class _DeclaredDate(_Date):
    """date implementation whose ``today`` is bound to the task clock."""

    _clock: DeclaredExperimentClock | None = None

    @classmethod
    def today(cls) -> _Date:
        if cls._clock is None:  # pragma: no cover - construction guard.
            raise NetworkDisabledError("declared_clock_not_installed")
        return cls._clock.today()


def blocked_scraper(*_args: Any, **_kwargs: Any) -> None:
    """Guard installed over the source scraper before it can touch a URL."""

    raise NetworkDisabledError("gpt_scraper_disabled_before_network")


def blocked_browser(*_args: Any, **_kwargs: Any) -> None:
    """Guard installed over the source browser before it can start."""

    raise NetworkDisabledError("gpt_browser_disabled_before_network")


def _completion_role(messages: list[dict[str, str]]) -> str:
    """Map pinned deep-research prompt families to the allow-listed roles."""

    text = "\n".join(message.get("content", "") for message in messages)
    if '"questions"' in text or "generate targeted questions" in text:
        return "strategic"
    if '"researchGoal"' in text or ("generate" in text and "search quer" in text):
        return "fast"
    return "smart"


def _fixture_completion_response(role: str, task_prompt: str) -> str:
    """Keep deterministic responses strictly behind explicit fixture mode."""

    if role == "strategic":
        return json.dumps({"questions": [task_prompt]}, ensure_ascii=False)
    if role == "fast":
        return json.dumps(
            [{"query": task_prompt, "researchGoal": "task-bound evidence"}],
            ensure_ascii=False,
        )
    return json.dumps(
        {
            "learnings": [{"insight": task_prompt, "sourceUrl": ""}],
            "followUpQuestions": [],
        },
        ensure_ascii=False,
    )


def _verify_source(source_dir: Path, repo_root: Path, *, fixture: bool = False) -> dict[str, Any]:
    if fixture:
        evidence = verify_fixture_source(
            source_dir,
            GPT_RESEARCHER_PROVENANCE,
            system="gpt_researcher",
            hook_verifier=verify_gpt_researcher_hooks,
        )
    else:
        evidence = verify_system_source(
            source_dir,
            GPT_RESEARCHER_PROVENANCE,
            lock_filename=None,
            project_filename="pyproject.toml",
        )
    evidence = {
        **evidence,
        "license_files": verify_declared_file_hashes(
            source_dir,
            {"LICENSE": GPT_RESEARCHER_LICENSE_SHA256},
        ),
        "worker_hooks": verify_gpt_researcher_hooks(source_dir),
    }
    evidence["benchmark_lock"] = verify_declared_file_hashes(
        repo_root,
        {
            Path(GPT_RESEARCHER_BENCHMARK_LOCK).as_posix(): (
                GPT_RESEARCHER_PROVENANCE.dependency_lock_sha256
            ),
            Path(GPT_RESEARCHER_BENCHMARK_PROJECT).as_posix(): (
                GPT_RESEARCHER_BENCHMARK_PROJECT_SHA256
            ),
            Path(GPT_RESEARCHER_LOCK_METADATA).as_posix(): GPT_RESEARCHER_LOCK_METADATA_SHA256,
        },
    )
    return evidence


def _load_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"gpt_pinned_module_spec_missing:{name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    parent_name, _, child_name = name.rpartition(".")
    if parent_name and parent_name in sys.modules:
        setattr(sys.modules[parent_name], child_name, module)
    return module


def _package(name: str, path: Path | None = None) -> types.ModuleType:
    module = types.ModuleType(name)
    if path is not None:
        module.__path__ = [str(path)]  # type: ignore[attr-defined]
    else:
        module.__path__ = []  # type: ignore[attr-defined]
    module.__package__ = name
    sys.modules[name] = module
    parent_name, _, child_name = name.rpartition(".")
    if parent_name and parent_name in sys.modules:
        setattr(sys.modules[parent_name], child_name, module)
    return module


def _install_import_shims() -> None:
    """Install only dependency shims required to import pinned source files."""

    json_repair = types.ModuleType("json_repair")
    json_repair.loads = json.loads
    sys.modules.setdefault("json_repair", json_repair)

    colorama = types.ModuleType("colorama")
    colorama.Fore = types.SimpleNamespace(YELLOW="", GREEN="", RED="")
    colorama.init = lambda **_kwargs: None
    sys.modules.setdefault("colorama", colorama)

    class _ReasoningEffort:
        High = types.SimpleNamespace(value="high")
        Medium = types.SimpleNamespace(value="medium")
        Low = types.SimpleNamespace(value="low")

    base = types.ModuleType("gpt_researcher.llm_provider.generic.base")
    base.ReasoningEfforts = _ReasoningEffort
    base._SUPPORTED_PROVIDERS = {"openai", "anthropic", "custom", "azure_openai"}
    sys.modules["gpt_researcher.llm_provider.generic.base"] = base

    embeddings = types.ModuleType("gpt_researcher.memory.embeddings")
    embeddings._SUPPORTED_PROVIDERS = {"openai", "custom"}
    sys.modules["gpt_researcher.memory.embeddings"] = embeddings

    retriever_utils = types.ModuleType("gpt_researcher.retrievers.utils")
    retriever_utils.get_all_retriever_names = lambda: ["custom"]
    sys.modules["gpt_researcher.retrievers.utils"] = retriever_utils

    llm = types.ModuleType("gpt_researcher.utils.llm")

    async def inert_completion(*_args: Any, **_kwargs: Any) -> str:
        return "{}"

    llm.create_chat_completion = inert_completion
    sys.modules["gpt_researcher.utils.llm"] = llm

    query_processing = types.ModuleType("gpt_researcher.actions.query_processing")
    query_processing.get_search_results = None
    sys.modules["gpt_researcher.actions.query_processing"] = query_processing

    workers = types.ModuleType("gpt_researcher.utils.workers")

    class InertWorkerPool:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            self.executor = None

    workers.WorkerPool = InertWorkerPool
    sys.modules["gpt_researcher.utils.workers"] = workers

    views = types.ModuleType("multi_agents.agents.utils.views")
    views.print_agent_output = lambda *_args, **_kwargs: None
    sys.modules["multi_agents.agents.utils.views"] = views
    agent_llms = types.ModuleType("multi_agents.agents.utils.llms")
    agent_llms.call_model = inert_completion
    sys.modules["multi_agents.agents.utils.llms"] = agent_llms
    json5 = types.ModuleType("json5")
    json5.loads = json.loads
    sys.modules.setdefault("json5", json5)


def _make_host_custom_retriever(
    upstream_type: type[Any],
    channel: WorkerChannel,
    *,
    closed_book: bool = False,
    fixture: bool = False,
    citation_urls: list[str] | None = None,
    search_skip_reasons: list[str] | None = None,
    audit_search_skip: Callable[[str], None] | None = None,
) -> type[Any]:
    """Subclass the pinned CustomRetriever while replacing only its transport."""

    class HostCustomRetriever(upstream_type):
        def __init__(
            self,
            query: str,
            query_domains: Any = None,
            *,
            audit_stage: str = "search",
        ) -> None:
            del query_domains
            self.query = query
            self.audit_stage = audit_stage
            self._channel = channel
            self.endpoint = "host://search_filter_http/v1"
            self.params: dict[str, str] = {}

        def search(self, max_results: int = 5) -> list[dict[str, Any]]:
            if not fixture:
                reject_fixture_sentinel(self.query, field="gpt_search_query")
            if closed_book:
                return []
            if isinstance(self.query, str) and len(self.query.strip()) > SEARCH_QUERY_MAX_CHARS:
                if search_skip_reasons is not None:
                    reason = f"{self.audit_stage}_skipped_query_over_limit"
                    search_skip_reasons.append(reason)
                    if audit_search_skip is not None:
                        audit_search_skip(reason)
                return []
            payload = self._channel.request_search(query=self.query, max_results=max_results)
            if not fixture:
                reject_fixture_sentinel(payload, field="gpt_search_response")
            results = payload.get("results")
            if not isinstance(results, list):
                raise NetworkDisabledError("custom_retriever_response_malformed")
            normalized = [dict(value) for value in results if isinstance(value, Mapping)]
            if citation_urls is not None:
                citation_urls.extend(
                    value["url"]
                    for value in normalized
                    if isinstance(value.get("url"), str) and value["url"].strip()
                )
            return normalized

    return HostCustomRetriever


def _declared_datetime_module(clock: DeclaredExperimentClock) -> types.ModuleType:
    _DeclaredDateTime._clock = clock
    _DeclaredDate._clock = clock
    module = types.ModuleType("datetime")
    module.date = _DeclaredDate
    module.datetime = _DeclaredDateTime
    module.time = _Time
    module.timedelta = _Timedelta
    module.timezone = _Timezone
    return module


def _load_pinned_runtime(
    source: Path,
    channel: WorkerChannel,
    clock: DeclaredExperimentClock,
    *,
    closed_book: bool,
    task_prompt: str,
    fixture: bool,
    audit_search_skip: Callable[[str], None] | None = None,
) -> tuple[Any, Any, list[list[dict[str, str]]], list[str], list[str], list[str], list[str]]:
    """Load source modules and wire their documented seams to the host."""

    _install_import_shims()
    package_root = source / "gpt_researcher"
    root = _package("gpt_researcher", package_root)
    for name, relative in (
        ("config", "config"),
        ("config.variables", "config/variables"),
        ("retrievers", "retrievers"),
        ("retrievers.custom", "retrievers/custom"),
        ("utils", "utils"),
        ("llm_provider", "llm_provider"),
        ("llm_provider.generic", "llm_provider/generic"),
        ("actions", "actions"),
        ("skills", "skills"),
        ("memory", "memory"),
        ("scraper", "scraper"),
    ):
        _package(f"gpt_researcher.{name}", package_root / relative)

    _package("multi_agents", source / "multi_agents")
    _package("multi_agents.agents", source / "multi_agents/agents")
    _package("multi_agents.agents.utils", source / "multi_agents/agents/utils")
    _package("multi_agents.memory", source / "multi_agents/memory")
    _package("multi_agents.ag2", source / "multi_agents/ag2")
    _package("multi_agents.ag2.agents", source / "multi_agents/ag2/agents")
    agents_package = sys.modules["multi_agents.agents"]
    for name in ("ResearchAgent", "ReviewerAgent", "ReviserAgent"):
        setattr(agents_package, name, type(name, (), {}))
    draft = types.ModuleType("multi_agents.memory.draft")
    draft.DraftState = dict
    sys.modules["multi_agents.memory.draft"] = draft

    sys.path.insert(0, str(source_import_path(source)))
    base_config = _load_module(
        "gpt_researcher.config.variables.base", package_root / "config/variables/base.py"
    )
    default_config = _load_module(
        "gpt_researcher.config.variables.default", package_root / "config/variables/default.py"
    )
    _ = (base_config, default_config)
    config_module = _load_module("gpt_researcher.config.config", package_root / "config/config.py")
    config_package = sys.modules["gpt_researcher.config"]
    config_package.Config = config_module.Config
    custom_module = _load_module(
        "gpt_researcher.retrievers.custom.custom",
        package_root / "retrievers/custom/custom.py",
    )
    citation_urls: list[str] = []
    search_skip_reasons: list[str] = []
    host_retriever = _make_host_custom_retriever(
        custom_module.CustomRetriever,
        channel,
        closed_book=closed_book,
        fixture=fixture,
        citation_urls=citation_urls,
        search_skip_reasons=search_skip_reasons,
        audit_search_skip=audit_search_skip,
    )

    # Import all prompt date constants under the declared clock, then patch
    # the deep path's direct datetime reference as well.
    old_datetime = sys.modules.get("datetime")
    sys.modules["datetime"] = _declared_datetime_module(clock)
    try:
        prompt_module = _load_module("gpt_researcher.prompts", package_root / "prompts.py")
    finally:
        if old_datetime is None:
            sys.modules.pop("datetime", None)
        else:
            sys.modules["datetime"] = old_datetime
    deep_module = _load_module(
        "gpt_researcher.skills.deep_research",
        package_root / "skills/deep_research.py",
    )
    writer_module = _load_module(
        "multi_agents.agents.writer", source / "multi_agents/agents/writer.py"
    )
    editor_module = _load_module(
        "multi_agents.agents.editor", source / "multi_agents/agents/editor.py"
    )
    ag2_editor_module = _load_module(
        "multi_agents.ag2.agents.editor", source / "multi_agents/ag2/agents/editor.py"
    )
    deep_module.datetime = _DeclaredDateTime
    prompt_module.datetime = _DeclaredDateTime
    prompt_module.date = _DeclaredDate
    writer_module.datetime = _DeclaredDateTime
    editor_module.datetime = _DeclaredDateTime
    ag2_editor_module.datetime = _DeclaredDateTime

    # Load the real scraper dispatcher and install guards over its browser and
    # scraper classes before any source method can receive a URL.
    scraper_package = sys.modules["gpt_researcher.scraper"]
    blocked_browser_class = type(
        "BlockedUpstreamBrowser",
        (),
        {"__init__": lambda self, *_args, **_kwargs: blocked_browser()},
    )
    blocked_scraper_class = type(
        "BlockedUpstreamScraper",
        (),
        {"__init__": lambda self, *_args, **_kwargs: blocked_scraper()},
    )
    browser_names = {"BrowserScraper", "NoDriverScraper"}
    scraper_names = {
        "BeautifulSoupScraper",
        "WebBaseLoaderScraper",
        "ArxivScraper",
        "PyMuPDFScraper",
        "TavilyExtract",
        "FireCrawl",
    }
    for name in browser_names:
        setattr(scraper_package, name, blocked_browser_class)
    for name in scraper_names:
        setattr(scraper_package, name, blocked_scraper_class)
    scraper_module = _load_module(
        "gpt_researcher.scraper.scraper", package_root / "scraper/scraper.py"
    )
    for name in browser_names:
        setattr(scraper_module, name, blocked_browser_class)
    for name in scraper_names:
        setattr(scraper_module, name, blocked_scraper_class)
    scraper_package.Scraper = scraper_module.Scraper

    class GuardPool:
        executor = None

    dispatcher = scraper_module.Scraper([], "benchmark", "browser", GuardPool())
    if dispatcher.get_scraper("https://must-not-be-fetched.example") is not blocked_browser_class:
        raise NetworkDisabledError("gpt_scraper_guard_not_installed")
    try:
        blocked_browser_class("https://must-not-be-fetched.example")
    except NetworkDisabledError:
        pass
    else:  # pragma: no cover - defensive invariant.
        raise NetworkDisabledError("gpt_browser_guard_not_installed")
    try:
        dispatcher.scraper = "bs"
        dispatcher.get_scraper("https://must-not-be-fetched.example")(
            "https://must-not-be-fetched.example"
        )
    except NetworkDisabledError:
        pass
    else:  # pragma: no cover - defensive invariant.
        raise NetworkDisabledError("gpt_scraper_guard_not_installed")

    actions_utils = types.ModuleType("gpt_researcher.actions.utils")

    async def inert_stream_output(*_args: Any, **_kwargs: Any) -> None:
        return None

    actions_utils.stream_output = inert_stream_output
    sys.modules["gpt_researcher.actions.utils"] = actions_utils
    web_scraping = types.ModuleType("gpt_researcher.actions.web_scraping")

    async def blocked_scrape_urls(*_args: Any, **_kwargs: Any) -> None:
        raise NetworkDisabledError("gpt_scraper_disabled_before_network")

    web_scraping.scrape_urls = blocked_scrape_urls
    sys.modules["gpt_researcher.actions.web_scraping"] = web_scraping
    scraper_utils = types.ModuleType("gpt_researcher.scraper.utils")
    scraper_utils.get_image_hash = lambda *_args, **_kwargs: None
    sys.modules["gpt_researcher.scraper.utils"] = scraper_utils
    browser_module = _load_module(
        "gpt_researcher.skills.browser", package_root / "skills/browser.py"
    )
    browser_module.scrape_urls = blocked_scrape_urls
    browser_probe = types.SimpleNamespace(
        cfg=types.SimpleNamespace(max_scraper_workers=1, scraper_rate_limit_delay=0),
        verbose=False,
        websocket=None,
    )
    try:
        asyncio.run(
            browser_module.BrowserManager(browser_probe).browse_urls(
                ["https://must-not-be-fetched.example"]
            )
        )
    except NetworkDisabledError:
        pass
    else:  # pragma: no cover - defensive invariant.
        raise NetworkDisabledError("gpt_browser_guard_not_installed")

    # Import the real GPTResearcher agent with offline component seams.  The
    # deep skill still constructs this pinned class for every nested query.
    actions = sys.modules["gpt_researcher.actions"]
    for name in (
        "add_references",
        "extract_headers",
        "extract_sections",
        "table_of_contents",
        "get_search_results",
        "generate_draft_section_titles",
        "generate_report",
        "write_report_introduction",
        "write_conclusion",
        "summarize_url",
    ):
        setattr(actions, name, lambda *_args, **_kwargs: [])

    async def choose_agent(*_args: Any, **_kwargs: Any) -> tuple[str, str]:
        return "researcher", "researcher"

    actions.choose_agent = choose_agent
    actions.get_retrievers = lambda *_args, **_kwargs: [host_retriever]

    llm_provider = sys.modules["gpt_researcher.llm_provider"]
    llm_provider.GenericLLMProvider = type("GenericLLMProvider", (), {})
    memory = sys.modules["gpt_researcher.memory"]
    memory.Memory = type("Memory", (), {"__init__": lambda self, *_args, **_kwargs: None})
    vector_store = types.ModuleType("gpt_researcher.vector_store")
    vector_store.VectorStoreWrapper = type("VectorStoreWrapper", (), {})
    sys.modules["gpt_researcher.vector_store"] = vector_store

    def _component(name: str) -> type[Any]:
        return type(name, (), {"__init__": lambda self, *_args, **_kwargs: None})

    context_manager = types.ModuleType("gpt_researcher.skills.context_manager")
    context_manager.ContextManager = _component("ContextManager")
    sys.modules["gpt_researcher.skills.context_manager"] = context_manager
    curator = types.ModuleType("gpt_researcher.skills.curator")
    curator.SourceCurator = _component("SourceCurator")
    sys.modules["gpt_researcher.skills.curator"] = curator
    image_generator = types.ModuleType("gpt_researcher.skills.image_generator")
    image_generator.ImageGenerator = type(
        "ImageGenerator",
        (),
        {
            "__init__": lambda self, *_args, **_kwargs: None,
            "is_enabled": lambda self: False,
        },
    )
    sys.modules["gpt_researcher.skills.image_generator"] = image_generator

    completion_log: list[list[dict[str, str]]] = []
    completion_outputs: list[str] = []

    def _message_payload(messages: Any) -> list[dict[str, str]]:
        result: list[dict[str, str]] = []
        for message in messages if isinstance(messages, list) else []:
            if isinstance(message, Mapping):
                role = message.get("role", "user")
                content = message.get("content", "")
            else:
                role = "user"
                content = str(message)
            result.append(
                {
                    "role": role if isinstance(role, str) else "user",
                    "content": content if isinstance(content, str) else str(content),
                }
            )
        return result

    async def host_completion(*, messages: Any, **_kwargs: Any) -> str:
        payload = [
            {
                "role": "system",
                "content": f"Declared experiment date: {clock.prompt_date()}",
            },
            *_message_payload(messages),
        ]
        completion_log.append(payload)
        role = _completion_role(payload)
        if not fixture:
            reject_fixture_sentinel(payload, field="gpt_model_request")
        broker_response = response_content(channel.request_model(role=role, messages=payload))
        if not fixture:
            reject_fixture_sentinel(broker_response, field="gpt_model_response")
        completion_outputs.append(broker_response)
        if fixture:
            response = _fixture_completion_response(role, task_prompt)
            # Fixture mode is explicit and deterministic; production mode
            # always returns the parent broker's actual model content below.
            return response
        return broker_response

    deep_module.create_chat_completion = host_completion

    async def host_get_search_results(query: str, _retriever: Any, *args: Any, **kwargs: Any):
        del args
        return host_retriever(query, audit_stage="pre_search").search(
            max_results=int(kwargs.get("max_results", 3))
        )

    deep_module.get_search_results = host_get_search_results
    query_processing = sys.modules["gpt_researcher.actions.query_processing"]
    query_processing.get_search_results = host_get_search_results

    class HostResearchConductor:
        def __init__(self, researcher: Any) -> None:
            self.researcher = researcher

        async def conduct_research(self) -> list[str]:
            if closed_book:
                self.researcher.visited_urls = set()
                self.researcher.research_sources = []
                return []
            results = host_retriever(
                self.researcher.query,
                audit_stage="generated_search",
            ).search(max_results=5)
            self.researcher.visited_urls = {
                value["url"]
                for value in results
                if isinstance(value.get("url"), str) and value["url"].strip()
            }
            self.researcher.research_sources = results
            return [json_context({"query": self.researcher.query, "results": results})]

    researcher_module = types.ModuleType("gpt_researcher.skills.researcher")
    researcher_module.ResearchConductor = HostResearchConductor
    sys.modules["gpt_researcher.skills.researcher"] = researcher_module
    writer_skill = types.ModuleType("gpt_researcher.skills.writer")
    writer_skill.ReportGenerator = _component("ReportGenerator")
    sys.modules["gpt_researcher.skills.writer"] = writer_skill

    # Keep prompt construction on the pinned module while making the factory
    # inert; all deep-path model calls use the host completion above.
    prompt_module.get_prompt_family = lambda *_args, **_kwargs: None
    agent_module = _load_module("gpt_researcher.agent", package_root / "agent.py")
    agent_module.get_retrievers = lambda *_args, **_kwargs: [host_retriever]
    agent_module.ResearchConductor = HostResearchConductor
    root.GPTResearcher = agent_module.GPTResearcher

    return (
        agent_module.GPTResearcher,
        deep_module.DeepResearchSkill,
        completion_log,
        completion_outputs,
        [
            "gpt_researcher.prompts",
            "gpt_researcher.skills.deep_research",
            "multi_agents.agents.writer",
            "multi_agents.agents.editor",
            "multi_agents.ag2.agents.editor",
        ],
        citation_urls,
        search_skip_reasons,
    )


def _request_task_bound_prediction(
    channel: WorkerChannel,
    *,
    task_id: str,
    task_prompt: str,
    research_context: str,
    citations: list[str],
    fixture: bool,
) -> tuple[str, list[dict[str, str]]]:
    return request_task_bound_prediction(
        channel,
        task_id=task_id,
        task_prompt=task_prompt,
        research_context=research_context,
        citations=citations,
        role="smart",
        fixture=fixture,
        error_prefix="gpt",
    )


def run_one_task(
    source_dir: str | Path,
    *,
    repo_root: str | Path,
    channel: WorkerChannel | None = None,
    fixture: bool = False,
) -> None:
    source = Path(source_dir).expanduser().resolve()
    root = Path(repo_root).expanduser().resolve()
    _verify_source(source, root, fixture=fixture)
    sys.path.insert(0, str(source_import_path(source)))
    os.environ["RETRIEVER"] = "custom"

    channel = channel or WorkerChannel()
    start, payload = task_start(channel, adapter_id="gpt-researcher-typed-host-v3")
    configuration = payload.get("configuration")
    if not isinstance(configuration, dict):
        raise RuntimeError("gpt_worker_configuration_missing")
    required = {
        "retriever": "custom",
        "fallback_retriever": False,
        "scraper": "disabled",
        "embeddings": "disabled",
        "source_curation": "disabled",
        "telemetry": False,
        "native_browser": False,
        "public_network": False,
        "listener": False,
        "clock_injection": "declared_experiment_clock/v1",
    }
    if any(configuration.get(key) != value for key, value in required.items()):
        raise RuntimeError("gpt_worker_configuration_drift")
    if configuration.get("retriever") != os.environ["RETRIEVER"]:
        raise RuntimeError("gpt_retriever_environment_drift")
    if payload.get("mcp_config") is not None:
        raise RuntimeError("gpt_mcp_config_must_be_none")
    policy = payload.get("policy")
    if not isinstance(policy, Mapping) or policy.get("regime") not in {
        "closed_book",
        "cutoff_controlled",
        "unrestricted_current_web",
    }:
        raise RuntimeError("gpt_retrieval_policy_missing")
    closed_book = policy.get("regime") == "closed_book"
    search_capability = payload.get("search_capability")
    if closed_book and search_capability is not None:
        raise RuntimeError("gpt_closed_book_search_capability_present")
    if not closed_book and not isinstance(search_capability, dict):
        raise RuntimeError("gpt_search_capability_missing")

    channel.send(
        "hello",
        {
            "protocol": "external_research_stdio/v3",
            "adapter_id": "gpt-researcher-typed-host-v3",
            "child_network": "forbidden",
            "mcp_enabled": False,
            "mcp_config": None,
            "retriever": "custom",
            "retriever_env": "custom",
            "fallback_retriever": False,
            "scraper": "disabled",
            "embeddings": "disabled",
            "source_curation": "disabled",
            "telemetry": False,
            "native_browser": False,
            "public_network": False,
            "listener": False,
            "clock_injection": "declared_experiment_clock/v1",
            "injected_search": None if closed_book else "search_filter_http/v1",
            "pinned_deep_path": "gpt_researcher.skills.deep_research.DeepResearchSkill",
            "pinned_agent": "gpt_researcher.agent.GPTResearcher",
            "source_revision": GPT_RESEARCHER_PROVENANCE.revision,
            "license_status": "CONFLICTING",
            "ranking_eligible": False,
        },
        request_id=start.request_id,
    )

    task_prompt = payload.get("prompt")
    if not isinstance(task_prompt, str) or not task_prompt.strip():
        raise RuntimeError("gpt_task_prompt_missing")
    if not fixture:
        reject_fixture_sentinel(task_prompt, field="gpt_task_prompt")
    task_id = payload.get("task_id")
    if not isinstance(task_id, str) or not task_id.strip():
        raise RuntimeError("gpt_task_id_missing")
    clock = DeclaredExperimentClock(payload["experiment_as_of"])
    # A closed-book retriever still satisfies the pinned custom-retriever
    # interface, but returns no candidates and never asks the parent for a
    # search capability.  Cutoff-controlled runs use the same pinned class
    # with the typed parent channel transport.
    (
        GPTResearcher,
        DeepResearchSkill,
        completion_log,
        completion_outputs,
        clock_paths,
        citation_urls,
        search_skip_reasons,
    ) = _load_pinned_runtime(
        source,
        channel,
        clock,
        closed_book=closed_book,
        task_prompt=task_prompt,
        fixture=fixture,
        audit_search_skip=lambda reason: emit_trajectory(
            channel,
            node="gpt_search",
            detail=reason,
        ),
    )
    config = GPTResearcher(
        query=task_prompt,
        report_type="deep",
        report_source="web",
        config_path=None,
        mcp_configs=None,
        mcp_strategy="disabled",
        verbose=False,
    )
    config.cfg.deep_research_breadth = 1
    config.cfg.deep_research_depth = 1
    config.cfg.deep_research_concurrency = 1
    skill = DeepResearchSkill(config)
    protocol_print = builtins.print
    builtins.print = lambda *_args, **_kwargs: None
    try:
        result = asyncio.run(skill.run())
    finally:
        builtins.print = protocol_print
    if not isinstance(result, str):
        raise RuntimeError("gpt_pinned_deep_research_context_missing")
    if not result.strip():
        # The pinned deep skill may complete without extracting a learning
        # from an otherwise valid model response.  Preserve that real broker
        # content as evidence for the task-bound answer stage; the final
        # response still goes through the unchanged strict evaluator parser.
        result = "\n\n".join(output for output in completion_outputs if output.strip())
    if not completion_log or not clock_paths:
        raise RuntimeError("gpt_pinned_clock_paths_not_executed")

    citations = list(dict.fromkeys(citation_urls))
    prediction, final_messages = _request_task_bound_prediction(
        channel,
        task_id=task_id,
        task_prompt=task_prompt,
        research_context=result,
        citations=citations,
        fixture=fixture,
    )
    emit_trajectory(channel, node="pinned_deep_research", detail="DeepResearchSkill")
    channel.send(
        "task.result",
        {
            "prediction": prediction,
            "messages": final_messages,
            "citations": citations,
            "usage": {
                "worker_model_roles": 3,
                "worker_model_calls": len(completion_log) + 1,
                "worker_search_tools": 0 if closed_book else 1,
                "search_skip_reasons": list(search_skip_reasons),
                "pinned_deep_path": True,
                "scraper_guard": True,
                "clock_paths": clock_paths,
            },
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--fixture", action="store_true")
    args = parser.parse_args(argv)
    channel = WorkerChannel()
    try:
        run_one_task(
            args.source,
            repo_root=args.repo_root,
            channel=channel,
            fixture=args.fixture,
        )
    except BaseException as exc:
        send_task_error(channel, exc)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the stdio smoke.
    raise SystemExit(main())


__all__ = [
    "NetworkDisabledError",
    "blocked_browser",
    "blocked_scraper",
    "main",
    "run_one_task",
]

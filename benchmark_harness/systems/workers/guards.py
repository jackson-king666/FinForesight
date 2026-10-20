"""Pinned-source and hook checks shared by adapter preflight and workers."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path


class WorkerSourceError(ValueError):
    """Raised when a pinned upstream hook is absent or has drifted."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_required_markers(
    source_dir: str | Path,
    markers: Mapping[str, tuple[str, ...]],
) -> dict[str, object]:
    """Verify compatibility markers without writing to the upstream tree."""

    source = Path(source_dir).expanduser().resolve()
    files: dict[str, str] = {}
    checked: dict[str, list[str]] = {}
    for relative, required in markers.items():
        target = source / relative
        if not target.is_file() or target.is_symlink():
            raise WorkerSourceError(f"worker_hook_file_missing:{relative}")
        text = target.read_text(encoding="utf-8")
        missing = [marker for marker in required if marker not in text]
        if missing:
            raise WorkerSourceError(f"worker_hook_drift:{relative}:" + ",".join(missing))
        files[relative] = _sha256(target)
        checked[relative] = list(required)
    return {"files": files, "markers": checked}


ODR_HOOK_MARKERS: dict[str, tuple[str, ...]] = {
    "src/open_deep_research/configuration.py": (
        "class Configuration",
        "allow_clarification",
        "search_api",
        "mcp_config",
    ),
    "src/open_deep_research/utils.py": (
        "async def get_search_tool",
        "async def get_all_tools",
        "SearchAPI.NONE",
    ),
    "src/open_deep_research/deep_researcher.py": (
        "get_all_tools",
        "allow_clarification",
        "get_today_str",
        "researcher_subgraph",
        "deep_researcher",
    ),
    "src/open_deep_research/prompts.py": ("date",),
}


GPT_RESEARCHER_HOOK_MARKERS: dict[str, tuple[str, ...]] = {
    "gpt_researcher/config/config.py": (
        "class Config",
        "RETRIEVER",
        "_set_attributes",
    ),
    "gpt_researcher/config/variables/default.py": (
        '"RETRIEVER": "tavily"',
        '"EMBEDDING":',
    ),
    "gpt_researcher/retrievers/custom/custom.py": (
        "class CustomRetriever",
        "RETRIEVER_ENDPOINT",
        "requests.get",
    ),
    "gpt_researcher/scraper/scraper.py": (
        "class Scraper",
        '"browser": BrowserScraper',
        "def get_scraper",
        "def extract_data_from_url",
    ),
    "gpt_researcher/skills/browser.py": (
        "class BrowserManager",
        "async def browse_urls",
        "scrape_urls",
    ),
    "gpt_researcher/skills/deep_research.py": (
        "class DeepResearchSkill",
        "async def deep_research",
        "GPTResearcher(",
    ),
    "gpt_researcher/agent.py": (
        "class GPTResearcher",
        "async def conduct_research",
    ),
    "gpt_researcher/prompts.py": (
        "datetime.now",
        "date.today",
    ),
    "multi_agents/agents/writer.py": ("datetime.now",),
    "multi_agents/agents/editor.py": ("class EditorAgent", "datetime.now"),
    "multi_agents/ag2/agents/editor.py": ("class EditorAgent", "datetime.now"),
}


def verify_odr_hooks(source_dir: str | Path) -> dict[str, object]:
    return verify_required_markers(source_dir, ODR_HOOK_MARKERS)


def verify_gpt_researcher_hooks(source_dir: str | Path) -> dict[str, object]:
    return verify_required_markers(source_dir, GPT_RESEARCHER_HOOK_MARKERS)


__all__ = [
    "GPT_RESEARCHER_HOOK_MARKERS",
    "ODR_HOOK_MARKERS",
    "WorkerSourceError",
    "verify_gpt_researcher_hooks",
    "verify_odr_hooks",
    "verify_required_markers",
]

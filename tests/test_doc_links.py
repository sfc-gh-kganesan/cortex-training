from __future__ import annotations

import re
import subprocess
from pathlib import Path
from urllib.parse import unquote

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Paths linked from launch material outside this repo (blog, Snowflake docs,
# OpenHands skill, Harbor cookbook). GitHub does not redirect renamed folders,
# so moving one of these breaks those links; update the outside links first.
PUBLIC_PATHS = (
    "README.md",
    "docs/getting-started/setup.md",
    "docs/reference/model-compatibility.md",
    "model-catalog/README.md",
    "recipes/README.md",
    "recipes/api_quickstart/README.md",
    "recipes/sft_quickstart",
    "recipes/rl_quickstart",
    "recipes/distillation",
    "recipes/inference",
    "recipes/model_deployment",
    "recipes/skyrl/gsm8k_grpo/README.md",
    "recipes/tinker/README.md",
    "recipes/openhands/README.md",
    "recipes/bring-your-own-agent.md",
)

_SELF_URL = re.compile(
    r"https://github\.com/snowflakedb/cortex-training/(?:blob|tree)/main/([^\s)\"'>#?]+)"
)
_INLINE_LINK = re.compile(r"!?\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
_REFERENCE_LINK = re.compile(r"^\s{0,3}\[[^\]]+\]:\s*<?(\S+?)>?(?:\s|$)", re.MULTILINE)
_FENCE = re.compile(r"^(```|~~~).*?^\1", re.MULTILINE | re.DOTALL)


def _tracked(*patterns: str) -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "--", *patterns], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout
    return [REPO_ROOT / line for line in out.splitlines() if (REPO_ROOT / line).is_file()]


def _markdown_files() -> list[Path]:
    return _tracked("*.md")


def _relative_targets(markdown: Path) -> list[str]:
    text = _FENCE.sub("", markdown.read_text(encoding="utf-8"))
    targets = _INLINE_LINK.findall(text) + _REFERENCE_LINK.findall(text)
    return [t for t in targets if not re.match(r"^([a-z][a-z0-9+.-]*:|#|/|\{)", t, re.IGNORECASE)]


@pytest.mark.parametrize("markdown", _markdown_files(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_relative_markdown_links_point_at_files_that_exist(markdown):
    broken = [
        target
        for target in _relative_targets(markdown)
        if not (markdown.parent / unquote(target.split("#", 1)[0].split("?", 1)[0])).exists()
    ]
    assert broken == [], f"{markdown.relative_to(REPO_ROOT)} links to missing paths: {broken}"


def test_links_to_this_repo_on_github_point_at_paths_that_exist():
    broken = []
    for path in _tracked("*.md", "*.py", "*.ipynb", "*.yaml", "*.yml", "*.toml", "*.json"):
        for target in _SELF_URL.findall(path.read_text(encoding="utf-8", errors="ignore")):
            if not (REPO_ROOT / unquote(target.rstrip("/.,"))).exists():
                broken.append(f"{path.relative_to(REPO_ROOT)}: {target}")
    assert broken == []


@pytest.mark.parametrize("path", PUBLIC_PATHS)
def test_publicly_linked_paths_still_exist(path):
    assert (REPO_ROOT / path).exists(), (
        f"{path} is linked from outside this repo; update those links before moving it"
    )

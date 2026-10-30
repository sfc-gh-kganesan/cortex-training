"""The example notebooks, checked without running them against a cluster."""

from __future__ import annotations

import ast
import importlib.util
import json
import re
import sys
import types
from pathlib import Path

import pytest

from recipes.utils import make_config

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = {
    "inference": REPO_ROOT / "recipes/inference/inference_walkthrough.ipynb",
    "sft": REPO_ROOT / "recipes/sft_quickstart/conversational/qwen3_8b_sft_training.ipynb",
    "multiplex": REPO_ROOT / "recipes/sft_quickstart/conversational/qwen3_8b_sft_training_multiplex.ipynb",
}
_UUID = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.IGNORECASE)


def _notebook(name: str) -> dict:
    return json.loads(NOTEBOOKS[name].read_text(encoding="utf-8"))


def _source(cell: dict) -> str:
    return "".join(cell["source"])


def _guard(name: str) -> str:
    """The first cell up to the make_config import: the part that runs offline."""
    source = _source(_notebook(name)["cells"][0])
    end = source.index("from recipes.utils import make_config")
    return source[: source.index("\n", end) + 1]


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebooks_are_valid_nbformat_4(name):
    notebook = _notebook(name)

    assert (notebook["nbformat"], notebook["nbformat_minor"]) == (4, 4)
    assert set(notebook) == {"cells", "metadata", "nbformat", "nbformat_minor"}
    for cell in notebook["cells"]:
        assert cell["cell_type"] == "code"
        assert set(cell) == {"cell_type", "execution_count", "metadata", "outputs", "source"}
        ast.parse(_source(cell))


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_the_first_cell_puts_the_repository_first_on_sys_path(name, monkeypatch):
    monkeypatch.setattr(sys, "path", [entry for entry in sys.path if Path(entry or ".").resolve() != REPO_ROOT])
    monkeypatch.chdir(NOTEBOOKS[name].parent)
    namespace: dict = {}

    exec(compile(_guard(name), str(NOTEBOOKS[name]), "exec"), namespace)

    assert sys.path[0] == str(REPO_ROOT)
    assert namespace["make_config"] is make_config


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_the_first_cell_fails_clearly_when_examples_resolves_elsewhere(name, monkeypatch, tmp_path):
    shadow = types.ModuleType("recipes")
    shadow.__file__ = str(tmp_path / "site-packages" / "recipes" / "__init__.py")
    monkeypatch.setitem(sys.modules, "recipes", shadow)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.chdir(NOTEBOOKS[name].parent)

    with pytest.raises(RuntimeError, match="`recipes` resolves to .*site-packages.*, outside .*Another package named"):
        exec(compile(_guard(name), str(NOTEBOOKS[name]), "exec"), {})


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_the_first_cell_builds_the_client_from_the_whole_connection_file(name, monkeypatch, tmp_path):
    pytest.importorskip("torch")
    import cortex_training

    captured: dict = {}

    def fake_from_pat(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(cortex_training.CortexTrainingClient, "_from_pat", staticmethod(fake_from_pat))
    config_file = tmp_path / "conn.json"
    config_file.write_text(
        json.dumps(
            {
                "host": "acct.example.com",
                "pat": "tok-123",
                "database": "MY_DB",
                "schema": "MY_SCHEMA",
                "endpoint": "my-endpoint",
                "verify_ssl": False,
                "poll_interval": 2.5,
                "poll_timeout": 42.0,
            }
        )
    )
    monkeypatch.setenv("CORTEX_TRAINING_CONFIG", str(config_file))
    monkeypatch.setattr(sys, "path", [entry for entry in sys.path if Path(entry or ".").resolve() != REPO_ROOT])
    monkeypatch.chdir(NOTEBOOKS[name].parent)

    exec(compile(_source(_notebook(name)["cells"][0]), str(NOTEBOOKS[name]), "exec"), {})

    # The whole mapping propagates from the file, not the old hard-coded placeholders.
    assert captured["host"] == "acct.example.com"
    assert captured["pat"] == "tok-123"
    assert captured["database"] == "MY_DB"
    assert captured["schema"] == "MY_SCHEMA"
    assert captured["endpoint"] == "my-endpoint"
    assert captured["verify_ssl"] is False
    assert captured["poll_interval"] == 2.5
    assert captured["poll_timeout"] == 42.0
    # No placeholder identifiers or hard-coded from_pat call remain in the cell.
    source = _source(_notebook(name)["cells"][0])
    assert "ACCOUNT.snowflakecomputing.com" not in source
    assert "from_pat(" not in source


def _job_body(name: str) -> dict:
    namespace = {"make_config": make_config}
    exec(compile(_source(_notebook(name)["cells"][1]), str(NOTEBOOKS[name]), "exec"), namespace)
    return namespace["job_body"]


def test_the_inference_notebook_shrinks_the_catalog_job_for_its_demo():
    body = _job_body("inference")
    expected = make_config("Qwen/Qwen3-8B", "standard", inference=True)
    expected["sub_job_configs"][0]["inference_config"].update(n_gpus=1, max_seq_len=2048)

    assert body == expected


@pytest.mark.parametrize("name", ["sft", "multiplex"])
def test_the_sft_notebooks_shrink_the_catalog_job_for_their_demo(name):
    body = _job_body(name)
    expected = make_config("Qwen/Qwen3-8B", "standard")
    expected["sub_job_configs"][0]["training_config"].update(max_seq_len=128, train_batch_size=8)
    if name == "multiplex":
        expected["sub_job_configs"][0]["training_config"]["multiplex_job_id"] = (
            "00000000-0000-0000-0000-000000000000:training:0"
        )

    assert body == expected


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_no_notebook_defines_its_job_inline(name):
    sources = [_source(cell) for cell in _notebook(name)["cells"]]

    assert not any("job_body = {" in source for source in sources)
    assert "make_config(" in sources[1]


def _imported_modules(source: str) -> set[str]:
    modules = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module)
    return modules


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_every_notebook_import_resolves(name, monkeypatch):
    monkeypatch.setattr(sys, "path", [str(REPO_ROOT), *sys.path])
    modules = set().union(*(_imported_modules(_source(cell)) for cell in _notebook(name)["cells"]))

    assert "recipes.utils" in modules
    assert {module for module in modules if importlib.util.find_spec(module) is None} == set()


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_committed_outputs_are_sanitized(name):
    texts = [
        json.dumps(cell["outputs"]) + _source(cell)
        for cell in _notebook(name)["cells"]
    ]
    blob = "\n".join(texts)

    assert {uuid for uuid in _UUID.findall(blob)} <= {"00000000-0000-0000-0000-000000000000"}
    assert set(re.findall(r"[\w.-]+\.snowflakecomputing\.com", blob)) <= {"ACCOUNT.snowflakecomputing.com"}

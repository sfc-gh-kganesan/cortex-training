from __future__ import annotations

import json

import pytest

from cortex_training.cli import commands as _cli
from cortex_training.client import core as client_module
from cortex_training.client import CortexTrainingClient
from cortex_training.connection import resolve_connection
from recipes import utils

HOST = "ACCOUNT.snowflakecomputing.com"
PROFILE_BASE_URL = f"https://{HOST}"
PAT = "pat-secret-value"

ENV_NAMES = (
    "CORTEX_TRAINING_BASE_URL",
    "CORTEX_TRAINING_HOST",
    "SNOWFLAKE_HOST",
    "CORTEX_TRAINING_PAT",
    "SNOWFLAKE_PAT",
    "CORTEX_TRAINING_DATABASE",
    "SNOWFLAKE_DATABASE",
    "CORTEX_TRAINING_SCHEMA",
    "SNOWFLAKE_SCHEMA",
    "CORTEX_TRAINING_ENDPOINT",
    "CORTEX_TRAINING_CONFIG",
    "CORTEX_TRAINING_CONNECTION",
)


class FakeProfileAuth:
    """Stands in for the connector-backed profile session (no network, no connector)."""

    instances: list["FakeProfileAuth"] = []

    def __init__(self, connection_name=None, **_kwargs):
        self.connection_name = connection_name
        self.base_url = PROFILE_BASE_URL
        self.database = "PROFILE_DB"
        self.schema = None
        self.role = None
        FakeProfileAuth.instances.append(self)

    def get_token(self):
        return "session-token"

    def refresh(self, observed_token=None):
        return "session-token"

    def invalidate(self):
        pass

    def open_artifact_connection(self):
        raise AssertionError("not used")

    def close(self):
        pass


@pytest.fixture
def env(monkeypatch, tmp_path):
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CORTEX_TRAINING_DISABLE_TELEMETRY", "1")
    monkeypatch.setenv("CORTEX_TRAINING_LOGIN_FILE", str(tmp_path / "no-login.json"))
    FakeProfileAuth.instances = []
    monkeypatch.setattr(client_module, "SnowflakeProfileAuth", FakeProfileAuth)
    return monkeypatch


def write_json(tmp_path, payload, name="config.json"):
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return str(path)


def pat_config(tmp_path, **extra):
    payload = {"host": HOST, "pat": PAT, "database": "TRAIN_DB", "schema": "PUBLIC"}
    payload.update(extra)
    return write_json(tmp_path, payload)


# Config file and PAT ---------------------------------------------------------


def test_json_config_uses_pat_and_reports_its_source(env, tmp_path):
    resolved = utils.resolve_example_connection(pat_config(tmp_path))

    assert resolved.source == "config file"
    assert resolved.pat == PAT
    assert resolved.host == HOST
    assert (resolved.database, resolved.schema) == ("TRAIN_DB", "PUBLIC")
    assert resolved.client.base_url == f"https://{HOST}"
    assert resolved.client._session.headers["Authorization"] == f"Bearer {PAT}"
    assert FakeProfileAuth.instances == []


def test_describe_names_the_target_and_source_but_never_the_pat(env, tmp_path):
    line = utils.resolve_example_connection(pat_config(tmp_path)).describe()

    assert line == f"connected to {HOST} (TRAIN_DB.PUBLIC) via config file"
    assert PAT not in line


def test_resolved_connection_repr_never_contains_the_pat(env, tmp_path):
    resolved = utils.resolve_example_connection(pat_config(tmp_path))

    assert resolved.pat == PAT
    assert PAT not in repr(resolved)


def test_json_config_without_database_keeps_the_legacy_default(env, tmp_path):
    path = write_json(tmp_path, {"host": HOST, "pat": PAT})

    resolved = utils.resolve_example_connection(path)

    assert resolved.database == utils.LEGACY_DEFAULT_DATABASE
    assert resolved.schema == "PUBLIC"


def test_json_config_tolerates_unknown_keys_as_examples_always_have(env, tmp_path):
    path = write_json(tmp_path, {"host": HOST, "pat": PAT, "database": "TRAIN_DB", "notes": "x"})

    assert utils.resolve_example_connection(path).pat == PAT


def test_json_config_without_pat_is_rejected(env, tmp_path):
    path = write_json(tmp_path, {"host": HOST, "database": "TRAIN_DB"})

    with pytest.raises(ValueError, match="no PAT found"):
        utils.resolve_example_connection(path)


def test_make_client_still_returns_a_pat_client(env, tmp_path):
    client = utils.make_client(pat_config(tmp_path))

    assert isinstance(client, CortexTrainingClient)
    assert client.database == "TRAIN_DB"


def test_poll_timeout_override_wins_over_the_file(env, tmp_path):
    path = pat_config(tmp_path, poll_timeout=900)

    client = utils.make_client(path, poll_timeout=42.0)

    assert client.poll_timeout == 42.0


# Explicit profile -------------------------------------------------------------


def test_explicit_connection_profile_is_used_without_a_config(env):
    resolved = utils.resolve_example_connection(connection="training")

    assert [auth.connection_name for auth in FakeProfileAuth.instances] == ["training"]
    assert resolved.source == "connection profile 'training'"
    assert resolved.pat is None
    assert resolved.database == "PROFILE_DB"
    assert resolved.client._session.headers.get("Authorization") is None


def test_profile_keeps_its_database_instead_of_the_example_default(env):
    resolved = utils.resolve_example_connection(connection="training")

    assert resolved.database == "PROFILE_DB"
    assert resolved.database != utils.LEGACY_DEFAULT_DATABASE


def test_connection_env_var_selects_the_profile(env):
    env.setenv("CORTEX_TRAINING_CONNECTION", "from-env")

    resolved = utils.resolve_example_connection()

    assert [auth.connection_name for auth in FakeProfileAuth.instances] == ["from-env"]
    assert resolved.source == "connection profile 'from-env'"


def test_profile_ignores_pat_in_the_environment(env):
    env.setenv("SNOWFLAKE_PAT", "env-pat")
    env.setenv("SNOWFLAKE_HOST", "env-host.example.com")

    resolved = utils.resolve_example_connection(connection="training")

    assert resolved.pat is None
    assert resolved.source == "connection profile 'training'"


def test_config_and_connection_together_are_rejected(env, tmp_path):
    with pytest.raises(ValueError, match="either config= or connection="):
        utils.resolve_example_connection(pat_config(tmp_path), connection="training")


# No arguments: CLI resolution order -------------------------------------------


def test_no_arguments_falls_back_to_the_default_profile(env):
    resolved = utils.resolve_example_connection()

    assert [auth.connection_name for auth in FakeProfileAuth.instances] == [None]
    assert resolved.source == "default Snowflake connection profile"
    assert resolved.pat is None


def test_config_env_var_beats_the_default_profile(env, tmp_path):
    env.setenv("CORTEX_TRAINING_CONFIG", pat_config(tmp_path))

    resolved = utils.resolve_example_connection()

    assert resolved.source == "config file"
    assert resolved.pat == PAT
    assert FakeProfileAuth.instances == []


def test_complete_direct_env_credentials_beat_the_saved_login(env, tmp_path):
    login = tmp_path / "login.json"
    login.write_text(json.dumps({"config_path": pat_config(tmp_path, pat="login-pat")}))
    env.setenv("CORTEX_TRAINING_LOGIN_FILE", str(login))
    env.setenv("CORTEX_TRAINING_HOST", HOST)
    env.setenv("CORTEX_TRAINING_PAT", PAT)
    env.setenv("CORTEX_TRAINING_DATABASE", "ENV_DB")

    resolved = utils.resolve_example_connection()

    assert resolved.source == "environment variables or arguments"
    assert resolved.pat == PAT
    assert resolved.database == "ENV_DB"


def test_saved_login_beats_the_default_profile(env, tmp_path):
    login = tmp_path / "login.json"
    login.write_text(json.dumps({"config_path": pat_config(tmp_path)}))
    env.setenv("CORTEX_TRAINING_LOGIN_FILE", str(login))

    resolved = utils.resolve_example_connection()

    assert resolved.source == "saved login (cortex-training login)"
    assert resolved.pat == PAT
    assert FakeProfileAuth.instances == []


# Parity with the CLI ----------------------------------------------------------


@pytest.mark.parametrize(
    "setup",
    ["profile-env", "config-env", "default"],
)
def test_example_resolution_matches_the_cli_resolver(env, tmp_path, setup):
    if setup == "profile-env":
        env.setenv("CORTEX_TRAINING_CONNECTION", "training")
    elif setup == "config-env":
        env.setenv("CORTEX_TRAINING_CONFIG", pat_config(tmp_path))

    args = _cli.parse_args(["list"])
    resolved = utils.resolve_example_connection()

    assert resolved.pat == (args.pat if not args.use_connection_profile else None)
    assert resolved.verify_ssl is (not args.no_verify_ssl)


def test_shared_resolver_is_what_the_cli_uses(env):
    resolved = resolve_connection(connection="training")

    assert resolved.source == "connection profile 'training'"
    assert [auth.connection_name for auth in FakeProfileAuth.instances] == ["training"]


# Connector-level config parsing -----------------------------------------------


def test_cli_loader_rejects_unknown_keys_unless_non_strict(tmp_path):
    path = write_json(tmp_path, {"host": HOST, "notes": "x"})

    with pytest.raises(ValueError, match="unknown config key"):
        _cli._load_config(path)
    assert _cli._load_config(path, strict=False) == {"host": HOST}


def test_example_command_hints_render_the_connection_argument():
    assert utils.example_connection_args("/p/config.json", None) == "config=/p/config.json"
    assert utils.example_connection_args(None, "training") == "connection=training"
    assert utils.example_connection_args(None, None) == ""


# Every example entry point ----------------------------------------------------

# Modules that call chz.nested_entrypoint (Protocol-only modules are excluded).
EXAMPLE_ENTRYPOINTS = [
    "recipes.distillation.on_policy.train",
    "recipes.inference.evaluate",
    "recipes.inference.generate",
    "recipes.inference.serve",
    "recipes.model_deployment.deploy",
    "recipes.rl_quickstart.math_grpo.train",
    "recipes.sft_quickstart.conversational.train",
    "recipes.sft_quickstart.dare.evaluate",
    "recipes.sft_quickstart.dare.train",
]


def test_no_example_requires_a_config_file():
    """A required `config: str` anywhere under recipes/ would force a JSON file again."""
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[1] / "recipes"
    required = re.compile(r"^    config: str\s*$", re.M)
    offenders = [
        str(path.relative_to(root.parent))
        for path in sorted(root.rglob("*.py"))
        if required.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []


@pytest.mark.parametrize("module_name", EXAMPLE_ENTRYPOINTS)
def test_example_config_is_optional_and_takes_a_connection(module_name):
    import importlib
    import inspect

    pytest.importorskip("chz")
    if module_name == "recipes.sft_quickstart.dare.evaluate":
        pytest.importorskip("sklearn")
    module = importlib.import_module(module_name)
    parameters = inspect.signature(module.Config).parameters

    assert parameters["config"].default is None
    assert parameters["connection"].default is None


# --- config= must behave exactly as it did before connection profiles ---------------


def _capture_from_pat(env):
    calls = []
    real = CortexTrainingClient._from_pat.__func__

    def spy(cls, **kwargs):
        calls.append(kwargs)
        return real(cls, **kwargs)

    env.setattr(CortexTrainingClient, "_from_pat", classmethod(spy))
    return calls


def _legacy_kwargs(payload, **overrides):
    """What the pre-profile make_client passed to CortexTrainingClient.from_pat."""
    config = payload.get("connection", payload)
    kwargs = dict(
        host=config["host"],
        pat=config["pat"],
        verify_ssl=bool(config.get("verify_ssl", True)),
        database=config.get("database", "CORTEX_TRAINING_DB"),
        schema=config.get("schema", "PUBLIC"),
        endpoint=config.get("endpoint", "cortex-training"),
        poll_interval=float(config.get("poll_interval", 0.5)),
        poll_timeout=float(config.get("poll_timeout", 1800.0)),
    )
    kwargs.update(overrides)
    return kwargs


@pytest.mark.parametrize(
    "payload",
    [
        {"host": HOST, "pat": PAT},
        {"host": HOST, "pat": PAT, "database": "D", "schema": "S", "endpoint": "e",
         "poll_interval": 2, "poll_timeout": 9, "verify_ssl": False},
        {"connection": {"host": HOST, "pat": PAT, "database": "NESTED_DB"}},
        {"host": HOST, "pat": PAT, "base_url": "http://localhost:1", "unknown_key": 1},
    ],
)
def test_config_ignores_the_environment_and_matches_legacy_make_client(env, tmp_path, payload):
    calls = _capture_from_pat(env)
    for name, value in {
        "CORTEX_TRAINING_CONNECTION": "some-profile",
        "CORTEX_TRAINING_CONFIG": str(tmp_path / "other.json"),
        "CORTEX_TRAINING_BASE_URL": "http://localhost:2",
        "SNOWFLAKE_HOST": "other.snowflakecomputing.com",
        "SNOWFLAKE_PAT": "other-pat",
        "SNOWFLAKE_DATABASE": "ENV_DB",
        "SNOWFLAKE_SCHEMA": "ENV_SCHEMA",
        "CORTEX_TRAINING_ENDPOINT": "env-endpoint",
    }.items():
        env.setenv(name, value)
    path = write_json(tmp_path, payload)

    resolved = utils.resolve_example_connection(path)

    assert calls == [_legacy_kwargs(payload)]
    assert resolved.source == "config file"
    assert resolved.pat == PAT
    assert FakeProfileAuth.instances == []


def test_config_overrides_are_passed_through_like_before(env, tmp_path):
    calls = _capture_from_pat(env)
    payload = {"host": HOST, "pat": PAT}
    utils.make_client(write_json(tmp_path, payload), poll_timeout=5.0, max_retries=2)
    assert calls == [_legacy_kwargs(payload, poll_timeout=5.0, max_retries=2)]


def test_config_and_connection_together_are_rejected(env, tmp_path):
    with pytest.raises(ValueError, match="either config= or connection="):
        utils.resolve_example_connection(pat_config(tmp_path), "training")

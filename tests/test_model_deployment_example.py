from __future__ import annotations

import json
from pathlib import Path

import pytest

from cortex_training.connection import ResolvedConnection
from cortex_training.model_deployment import ModelDeploymentSpec
from recipes.model_deployment import streaming

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = REPO_ROOT / "recipes/model_deployment"
DEPLOYMENT_ID = "sf-ct-00000000-0000-4000-8000-000000000000"
BASE_URL = "https://ACCOUNT.snowflakecomputing.com/api/v2/databases/DB/schemas/PUBLIC/cortex-training/inference/v1"


def sse(*chunks: dict, done: bool = True) -> list[str]:
    lines = [": keep-alive", ""]
    for chunk in chunks:
        lines += [f"data: {json.dumps(chunk)}", ""]
    if done:
        lines.append("data: [DONE]")
    return lines


def delta(text: str | None, finish_reason: str | None = None) -> dict:
    return {
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {"content": text} if text is not None else {}, "finish_reason": finish_reason}],
    }


class FakeResponse:
    def __init__(self, status_code=200, lines=(), body=None, text=""):
        self.status_code = status_code
        self.ok = status_code < 400
        self._lines = list(lines)
        self._body = body
        self.text = text

    def iter_lines(self, decode_unicode=False):
        return iter(self._lines)

    def json(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response


@pytest.mark.parametrize("name", ["base-model.json", "checkpoint.json"])
def test_shipped_specs_are_valid(name):
    spec = json.loads((EXAMPLE_DIR / "configs" / name).read_text())
    assert ModelDeploymentSpec.from_dict(spec).to_public_spec() == spec


def test_iter_sse_chunks_skips_comments_and_stops_at_done():
    lines = sse(delta("a"), delta("b")) + [f"data: {json.dumps(delta('after'))}"]
    texts = [streaming.chunk_text(chunk) for chunk in streaming.iter_sse_chunks(lines)]
    assert texts == ["a", "b"]


def test_iter_sse_chunks_decodes_bytes():
    lines = [line.encode() for line in sse(delta("x"))]
    assert [streaming.chunk_text(c) for c in streaming.iter_sse_chunks(lines)] == ["x"]


@pytest.mark.parametrize(
    ("lines", "message"),
    [
        (sse(delta("a"), {"error": {"message": "engine died"}}), "returned an error: engine died"),
        (sse(delta("a"), done=False) + ['data: {"choices": [', ""], "not JSON"),
        (sse(delta("a"), delta("b"), done=False), "ended without \\[DONE\\]"),
    ],
    ids=["error-chunk", "malformed-chunk", "truncated"],
)
def test_iter_sse_chunks_raises_on_failures_after_the_200(lines, message):
    with pytest.raises(streaming.DeploymentCallError, match=message):
        list(streaming.iter_sse_chunks(lines))


def test_stream_chat_does_not_report_a_truncated_stream_as_a_reply():
    session = FakeSession(FakeResponse(lines=sse(delta("partial"), done=False)))
    seen = []
    with pytest.raises(streaming.DeploymentCallError, match="without"):
        streaming.stream_chat(
            BASE_URL,
            "PAT",
            model=DEPLOYMENT_ID,
            messages=streaming.chat_messages("Hi"),
            max_completion_tokens=16,
            temperature=0.0,
            on_delta=seen.append,
            session=session,
        )
    assert seen == ["partial"]


def test_complete_chat_raises_on_an_error_body():
    session = FakeSession(FakeResponse(body={"error": {"message": "overloaded"}}))
    with pytest.raises(streaming.DeploymentCallError, match="overloaded"):
        streaming.complete_chat(
            BASE_URL,
            "PAT",
            model=DEPLOYMENT_ID,
            messages=streaming.chat_messages("Hi"),
            max_completion_tokens=16,
            temperature=0.0,
            session=session,
        )


def test_chunk_text_handles_empty_choices_and_role_only_delta():
    assert streaming.chunk_text({"choices": []}) == ""
    assert streaming.chunk_text({"choices": [{"delta": {"role": "assistant"}}]}) == ""


def test_stream_chat_sends_stream_request_and_collects_deltas():
    usage = {"prompt_tokens": 5, "completion_tokens": 2}
    session = FakeSession(
        FakeResponse(
            lines=sse(delta(None), delta("Hel"), delta("lo", "stop"), {"choices": [], "usage": usage})
        )
    )
    seen = []

    result = streaming.stream_chat(
        BASE_URL + "/",
        "PAT",
        model=DEPLOYMENT_ID,
        messages=streaming.chat_messages("Hi", system="Be brief."),
        max_completion_tokens=16,
        temperature=0.0,
        on_delta=seen.append,
        session=session,
    )

    assert seen == ["Hel", "lo"]
    assert result.text == "Hello"
    assert result.chunks == 4
    assert result.finish_reason == "stop"
    assert result.usage == usage
    assert result.time_to_first_token is not None
    url, kwargs = session.calls[0]
    assert url == f"{BASE_URL}/chat/completions"
    assert kwargs["stream"] is True
    assert kwargs["json"] == {
        "model": DEPLOYMENT_ID,
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hi"},
        ],
        "max_completion_tokens": 16,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    assert kwargs["headers"]["Authorization"] == "Bearer PAT"
    assert kwargs["headers"]["X-Snowflake-Authorization-Token-Type"] == "PROGRAMMATIC_ACCESS_TOKEN"
    assert kwargs["headers"]["Accept"] == "text/event-stream"


def test_complete_chat_reads_message_content():
    body = {"choices": [{"message": {"content": "Hello"}, "finish_reason": "stop"}]}
    session = FakeSession(FakeResponse(body=body))

    result = streaming.complete_chat(
        BASE_URL,
        "PAT",
        model=DEPLOYMENT_ID,
        messages=streaming.chat_messages("Hi"),
        max_completion_tokens=16,
        temperature=0.0,
        session=session,
    )

    assert result.text == "Hello"
    assert result.time_to_first_token is None
    sent = session.calls[0][1]["json"]
    assert "stream" not in sent and "max_tokens" not in sent
    assert sent["max_completion_tokens"] == 16


@pytest.mark.parametrize(
    ("status", "message"),
    [(404, "inference may not be enabled"), (400, "HTTP 400: bad request")],
)
def test_http_errors_raise_with_status(status, message):
    session = FakeSession(FakeResponse(status_code=status, text="bad request"))
    with pytest.raises(streaming.DeploymentCallError, match=message):
        streaming.stream_chat(
            BASE_URL,
            "PAT",
            model=DEPLOYMENT_ID,
            messages=streaming.chat_messages("Hi"),
            max_completion_tokens=16,
            temperature=0.0,
            session=session,
        )


class FakeClient:
    model_deployment_base_url = BASE_URL

    def __init__(self, wait_error=None, drop_error=None):
        self.wait_error = wait_error
        self.drop_error = drop_error
        self.created = []
        self.dropped = []

    def create_model_deployment(self, spec):
        self.created.append(spec)
        return {"deployment": DEPLOYMENT_ID, "status": "PENDING"}

    def wait_for_model_deployment(self, deployment_id):
        if self.wait_error:
            raise self.wait_error
        return {"deployment": deployment_id, "status": "READY", "base_url": BASE_URL}

    def drop_model_deployment(self, deployment_id):
        self.dropped.append(deployment_id)
        if self.drop_error:
            raise self.drop_error
        return {"deployment": deployment_id, "dropped": True}


@pytest.fixture
def deploy(monkeypatch, tmp_path):
    pytest.importorskip("chz")
    from recipes.model_deployment import deploy as module

    connection = tmp_path / "config.json"
    connection.write_text(json.dumps({"host": "ACCOUNT.snowflakecomputing.com", "pat": "PAT"}))
    state = {"client": FakeClient(), "calls": []}

    def fake_resolve(config=None, connection=None, **overrides):
        state["resolved"] = (config, connection, overrides)
        return ResolvedConnection(
            client=state["client"],
            source=f"connection profile '{connection}'" if connection else "config file",
            host="ACCOUNT.snowflakecomputing.com",
            database="DB",
            schema="PUBLIC",
            verify_ssl=True,
            pat=state.get("pat", "PAT"),
        )

    monkeypatch.setattr(module, "resolve_example_connection", fake_resolve)

    def fake_stream_chat(base_url, pat, **kwargs):
        state["calls"].append((base_url, pat, kwargs))
        kwargs["on_delta"]("Hello")
        return streaming.ChatResult(text="Hello", time_to_first_token=0.1, total_seconds=0.2, chunks=1)

    monkeypatch.setattr(module, "stream_chat", fake_stream_chat)
    state["config"] = lambda **kw: module.Config(config=str(connection), **kw)
    state["main"] = module.main
    state["module"] = module
    return state


def test_deploy_creates_streams_and_drops(deploy, capsys):
    deploy["main"](deploy["config"]())

    client = deploy["client"]
    assert client.created == [json.loads((EXAMPLE_DIR / "configs/base-model.json").read_text())]
    assert client.dropped == [DEPLOYMENT_ID]
    base_url, pat, kwargs = deploy["calls"][0]
    assert (base_url, pat, kwargs["model"]) == (BASE_URL, "PAT", DEPLOYMENT_ID)
    assert capsys.readouterr().out == "Hello\n"


def test_deploy_drops_when_wait_fails(deploy):
    deploy["client"] = FakeClient(wait_error=TimeoutError("still INITIALIZING"))
    with pytest.raises(TimeoutError):
        deploy["main"](deploy["config"]())
    assert deploy["client"].dropped == [DEPLOYMENT_ID]
    assert deploy["calls"] == []


def test_failed_drop_does_not_mask_the_wait_error(deploy, caplog):
    deploy["client"] = FakeClient(
        wait_error=TimeoutError("still INITIALIZING"), drop_error=ConnectionError("reset")
    )
    with pytest.raises(TimeoutError, match="still INITIALIZING"):
        deploy["main"](deploy["config"]())
    assert f"cortex-training model-deployment drop {DEPLOYMENT_ID}" in caplog.text


def test_failed_drop_after_a_good_reply_is_raised(deploy):
    deploy["client"] = FakeClient(drop_error=ConnectionError("reset"))
    with pytest.raises(ConnectionError):
        deploy["main"](deploy["config"]())


def test_deploy_keep_deployment_leaves_it_running(deploy):
    deploy["main"](deploy["config"](keep_deployment=True))
    assert deploy["client"].dropped == []


def test_deploy_existing_deployment_is_not_created_or_dropped(deploy):
    deploy["main"](deploy["config"](deployment=DEPLOYMENT_ID))
    assert deploy["client"].created == []
    assert deploy["client"].dropped == []
    assert deploy["calls"][0][2]["model"] == DEPLOYMENT_ID


def test_load_spec_resolves_relative_to_example(deploy, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    spec = deploy["module"].load_spec("configs/checkpoint.json")
    assert spec["source"]["type"] == "checkpoint"


def test_deploy_is_an_entrypoint():
    source = (EXAMPLE_DIR / "deploy.py").read_text()
    assert 'if __name__ == "__main__":' in source
    assert "chz.nested_entrypoint(main)" in source


def test_deploy_without_pat_fails_before_creating_a_deployment(deploy):
    deploy["pat"] = None
    with pytest.raises(ValueError, match="calls the inference endpoint with a PAT"):
        deploy["main"](deploy["config"]())
    assert deploy["client"].created == []
    assert deploy["calls"] == []


def test_deploy_profile_without_pat_names_the_profile(deploy):
    deploy["pat"] = None
    module = deploy["module"]
    config = module.Config(connection="training")
    with pytest.raises(ValueError, match="connection profile 'training'"):
        deploy["main"](config)
    assert deploy["client"].created == []

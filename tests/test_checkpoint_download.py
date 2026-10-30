"""Offline coverage of checkpoint downloads through the CLI."""

from __future__ import annotations

import io
import json
from unittest.mock import MagicMock

import pytest
import requests

from cortex_training import _checkpoint_download as download
from cortex_training.cli import commands as cli
from cortex_training.cli import build_parser


@pytest.fixture
def client():
    client = MagicMock()
    client.list_checkpoints.return_value = [{"checkpoint_id": "CHECKPOINT_ID"}]
    client.export_checkpoint.return_value = {
        "checkpoint_id": "CHECKPOINT_ID",
        "expires_at": "2026-10-27T00:00:00Z",
        "files": [
            {
                "filename": "weights/model.safetensors",
                "url": "https://download.example/model?signature=SECRET",
                "size_bytes": 4,
            }
        ],
    }
    return client


def _invoke(client, root, *options, stderr=None):
    stdout = io.StringIO()
    stderr = io.StringIO() if stderr is None else stderr
    status = cli.main(
        [
            "--base-url",
            "https://ACCOUNT.snowflakecomputing.com",
            "--database",
            "CORTEX_TRAINING_DB",
            "--compact",
            "download-checkpoint",
            "JOB_ID",
            "CHECKPOINT_ID",
            "--output-dir",
            str(root),
            *options,
        ],
        client_factory=lambda _: client,
        stdout=stdout,
        stderr=stderr,
    )
    return status, stdout.getvalue(), stderr.getvalue()


def _response(chunks=(b"abcd",), status=200, content_range=None):
    response = MagicMock()
    response.__enter__.return_value = response
    response.status_code = status
    response.headers = {} if content_range is None else {"Content-Range": content_range}
    response.iter_content.return_value = iter(chunks)
    return response


def test_command_is_available_from_public_parser_and_requires_output_dir():
    parser = build_parser()
    args = parser.parse_args(
        ["download-checkpoint", "JOB_ID", "CHECKPOINT_ID", "--output-dir", "out"]
    )
    assert args.include == []
    assert not hasattr(args, "progress")
    assert args.output_dir == "out"
    with pytest.raises(SystemExit):
        parser.parse_args(["download-checkpoint", "JOB_ID", "CHECKPOINT_ID"])
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "download-checkpoint",
                "JOB_ID",
                "CHECKPOINT_ID",
                "--output-dir",
                "out",
                "--progress",
            ]
        )


def test_downloads_nested_file_and_prints_summary(client, tmp_path, monkeypatch):
    get = MagicMock(return_value=_response())
    monkeypatch.setattr(download.requests, "get", get)
    status, stdout, stderr = _invoke(client, tmp_path)
    target = tmp_path / "weights/model.safetensors"
    assert status == 0
    assert stderr == "Checkpoint [####################] 100% 4/4 bytes (1/1 files)\n"
    assert target.read_bytes() == b"abcd"
    assert not target.with_name(target.name + ".partial").exists()
    client.list_checkpoints.assert_called_once_with("JOB_ID")
    client.export_checkpoint.assert_called_once_with("JOB_ID", "CHECKPOINT_ID")
    get.assert_called_once_with(
        "https://download.example/model?signature=SECRET",
        allow_redirects=False,
        headers={},
        stream=True,
        timeout=(30, 300),
    )
    assert "SECRET" not in stdout
    assert json.loads(stdout) == {
        "checkpoint_id": "CHECKPOINT_ID",
        "expires_at": "2026-10-27T00:00:00Z",
        "job_id": "JOB_ID",
        "output_dir": str(tmp_path),
        "files": [
            {
                "filename": "weights/model.safetensors",
                "saved_path": str(target),
                "size_bytes": 4,
                "status": "downloaded",
            }
        ],
    }


def test_decimal_size_and_repeatable_include_patterns(client, tmp_path, monkeypatch):
    files = client.export_checkpoint.return_value["files"]
    files[0]["size_bytes"] = "4"
    files.extend(
        [
            {
                "filename": "config.json",
                "url": "https://download.example/config",
                "size_bytes": 4,
            },
            {
                "filename": "optimizer.bin",
                "url": "https://download.example/optimizer",
                "size_bytes": 4,
            },
        ]
    )
    get = MagicMock(side_effect=[_response(), _response()])
    monkeypatch.setattr(download.requests, "get", get)
    status, stdout, _ = _invoke(
        client, tmp_path, "--include", "*.safetensors", "--include", "config.json"
    )
    assert status == 0
    assert get.call_count == 2
    assert not (tmp_path / "optimizer.bin").exists()
    assert [entry["size_bytes"] for entry in json.loads(stdout)["files"]] == [4, 4]


def test_no_matches_is_an_error_before_creating_output(client, tmp_path):
    root = tmp_path / "new"
    status, stdout, stderr = _invoke(client, root, "--include", "*.bin")
    assert status == 1
    assert stdout == ""
    assert "no checkpoint export files matched" in stderr
    assert not root.exists()


@pytest.mark.parametrize(
    "listing",
    [
        None,
        {},
        [None],
        [{}],
        [{"checkpoint_id": ""}],
        [{"checkpoint_id": True}],
        [{"checkpoint_id": "OTHER"}],
    ],
)
def test_invalid_listing_does_not_export(client, tmp_path, listing):
    client.list_checkpoints.return_value = listing
    status, stdout, stderr = _invoke(client, tmp_path)
    assert status == 1
    assert stdout == ""
    assert "checkpoint" in stderr
    client.export_checkpoint.assert_not_called()


@pytest.mark.parametrize(
    "field,value",
    [
        ("checkpoint_id", "OTHER"),
        ("checkpoint_id", None),
        ("expires_at", None),
        ("expires_at", ""),
        ("files", None),
        ("files", []),
        ("files", [None]),
        ("files", [{}]),
    ],
)
def test_invalid_export_does_not_download(client, tmp_path, monkeypatch, field, value):
    client.export_checkpoint.return_value[field] = value
    get = MagicMock()
    monkeypatch.setattr(download.requests, "get", get)
    status, stdout, _ = _invoke(client, tmp_path)
    assert status == 1
    assert stdout == ""
    get.assert_not_called()


@pytest.mark.parametrize(
    "filename",
    [
        "",
        "/absolute",
        "../outside",
        "weights/../outside",
        "weights//model",
        "./model",
        "weights\\model",
        "C:/model",
        "model\x00",
    ],
)
def test_unsafe_export_filename_is_rejected(client, tmp_path, monkeypatch, filename):
    client.export_checkpoint.return_value["files"][0]["filename"] = filename
    get = MagicMock()
    monkeypatch.setattr(download.requests, "get", get)
    status, _, stderr = _invoke(client, tmp_path)
    assert status == 1
    assert "unsafe checkpoint export filename" in stderr
    get.assert_not_called()


@pytest.mark.parametrize(
    "field,value",
    [
        ("url", "http://download.example/model"),
        ("url", "https://"),
        ("url", None),
        ("url", "https://user:SECRET@download.example/model"),
        ("size_bytes", True),
        ("size_bytes", -1),
        ("size_bytes", 1.5),
        ("size_bytes", "-1"),
        ("size_bytes", "４"),
        ("size_bytes", None),
    ],
)
def test_invalid_file_metadata_is_rejected(client, tmp_path, field, value):
    client.export_checkpoint.return_value["files"][0][field] = value
    status, _, stderr = _invoke(client, tmp_path)
    assert status == 1
    assert field in stderr
    assert "SECRET" not in stderr


@pytest.mark.parametrize("suffix", ["", ".partial"])
def test_duplicate_or_partial_collision_is_rejected(client, tmp_path, suffix):
    files = client.export_checkpoint.return_value["files"]
    files.append(dict(files[0], filename=files[0]["filename"] + suffix))
    status, _, stderr = _invoke(client, tmp_path)
    assert status == 1
    assert "duplicate" in stderr or "conflict" in stderr


@pytest.mark.parametrize("kind", ["directory", "final", "partial"])
def test_symlink_escape_is_rejected(client, tmp_path, kind):
    root = tmp_path / "output"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    if kind == "directory":
        (root / "weights").symlink_to(outside, target_is_directory=True)
    else:
        (root / "weights").mkdir()
        suffix = ".partial" if kind == "partial" else ""
        (root / ("weights/model.safetensors" + suffix)).symlink_to(outside / "model")
    status, _, stderr = _invoke(client, root)
    assert status == 1
    assert "unsafe checkpoint download path" in stderr
    assert list(outside.iterdir()) == []


def test_partial_transfer_resumes_on_next_invocation(client, tmp_path, monkeypatch):
    response = _response()

    def interrupted():
        yield b"ab"
        raise requests.ConnectionError(
            "https://download.example/model?signature=SECRET"
        )

    response.iter_content.return_value = interrupted()
    get = MagicMock(return_value=response)
    monkeypatch.setattr(download.requests, "get", get)
    status, stdout, stderr = _invoke(client, tmp_path)
    partial = tmp_path / "weights/model.safetensors.partial"
    assert status == 1
    assert stdout == ""
    assert "SECRET" not in stderr
    assert partial.read_bytes() == b"ab"
    get.return_value = _response((b"cd",), status=206, content_range="bytes 2-3/4")
    status, stdout, stderr = _invoke(client, tmp_path)
    assert status == 0
    assert stderr == "Checkpoint [####################] 100% 4/4 bytes (1/1 files)\n"
    assert get.call_args.kwargs["headers"] == {"Range": "bytes=2-"}
    assert (tmp_path / "weights/model.safetensors").read_bytes() == b"abcd"
    assert not partial.exists()
    assert json.loads(stdout)["files"][0]["status"] == "resumed"


@pytest.mark.parametrize(
    "status,content_range",
    [(200, None), (206, None), (206, "bytes 0-1/4"), (206, "bytes 2-3/5")],
)
def test_bad_resume_response_keeps_partial_unchanged(
    client, tmp_path, monkeypatch, status, content_range
):
    partial = tmp_path / "weights/model.safetensors.partial"
    partial.parent.mkdir()
    partial.write_bytes(b"ab")
    monkeypatch.setattr(
        download.requests,
        "get",
        MagicMock(return_value=_response((b"cd",), status, content_range)),
    )
    status, _, stderr = _invoke(client, tmp_path)
    assert status == 1
    assert "expected HTTP" in stderr or "Content-Range" in stderr
    assert partial.read_bytes() == b"ab"
    assert not (tmp_path / "weights/model.safetensors").exists()


@pytest.mark.parametrize("chunks", [(b"ab",), (b"abcde",)])
def test_size_mismatch_is_not_promoted(client, tmp_path, monkeypatch, chunks):
    monkeypatch.setattr(
        download.requests, "get", MagicMock(return_value=_response(chunks))
    )
    status, _, stderr = _invoke(client, tmp_path)
    assert status == 1
    assert "expected size" in stderr
    assert not (tmp_path / "weights/model.safetensors").exists()
    assert (tmp_path / "weights/model.safetensors.partial").exists()


@pytest.mark.parametrize(
    "existing,expected",
    [("final", "existing"), ("partial", "resumed"), ("empty", "downloaded")],
)
def test_complete_files_and_zero_byte_files_need_no_get(
    client, tmp_path, monkeypatch, existing, expected
):
    target = tmp_path / "weights/model.safetensors"
    if existing == "empty":
        client.export_checkpoint.return_value["files"][0]["size_bytes"] = 0
    else:
        target.parent.mkdir()
        path = (
            target
            if existing == "final"
            else target.with_name(target.name + ".partial")
        )
        path.write_bytes(b"abcd")
    get = MagicMock()
    monkeypatch.setattr(download.requests, "get", get)
    status, stdout, _ = _invoke(client, tmp_path)
    assert status == 0
    assert json.loads(stdout)["files"][0]["status"] == expected
    get.assert_not_called()
    assert target.read_bytes() == (b"" if existing == "empty" else b"abcd")


@pytest.mark.parametrize("partial", [False, True])
def test_oversized_existing_files_are_not_overwritten(
    client, tmp_path, monkeypatch, partial
):
    path = tmp_path / ("weights/model.safetensors" + (".partial" if partial else ""))
    path.parent.mkdir()
    path.write_bytes(b"abcde")
    get = MagicMock()
    monkeypatch.setattr(download.requests, "get", get)
    status, _, stderr = _invoke(client, tmp_path)
    assert status == 1
    assert "move it aside" in stderr
    assert path.read_bytes() == b"abcde"
    get.assert_not_called()


def test_http_error_does_not_print_presigned_url(client, tmp_path, monkeypatch):
    response = _response(status=403)
    response.raise_for_status.side_effect = requests.HTTPError(
        "https://download.example/model?signature=SECRET", response=response
    )
    monkeypatch.setattr(download.requests, "get", MagicMock(return_value=response))
    status, stdout, stderr = _invoke(client, tmp_path)
    assert status == 1
    assert stdout == ""
    assert "HTTP 403" in stderr
    assert "SECRET" not in stderr


def test_redirect_is_refused(client, tmp_path, monkeypatch):
    monkeypatch.setattr(
        download.requests, "get", MagicMock(return_value=_response(status=302))
    )
    status, _, stderr = _invoke(client, tmp_path)
    assert status == 1
    assert "HTTP 200, got 302" in stderr


def test_keyboard_interrupt_keeps_partial_and_reports_resume(
    client, tmp_path, monkeypatch
):
    response = _response()

    def interrupted():
        yield b"ab"
        raise KeyboardInterrupt()

    response.iter_content.return_value = interrupted()
    monkeypatch.setattr(download.requests, "get", MagicMock(return_value=response))
    status, stdout, stderr = _invoke(client, tmp_path)
    assert status == 130
    assert stdout == ""
    assert "rerun the same command to resume" in stderr
    assert (tmp_path / "weights/model.safetensors.partial").read_bytes() == b"ab"


class _TtyStderr(io.StringIO):
    def isatty(self):
        return True


def test_progress_updates_terminal_without_changing_json(client, tmp_path, monkeypatch):
    monkeypatch.setattr(
        download.requests, "get", MagicMock(return_value=_response((b"ab", b"cd")))
    )
    status, stdout, stderr = _invoke(client, tmp_path, stderr=_TtyStderr())
    assert status == 0
    assert "\rCheckpoint [##########----------]" in stderr
    assert "50% 2/4 bytes (0/1 files)" in stderr
    assert stderr.endswith("100% 4/4 bytes (1/1 files)\n")
    assert "Checkpoint [" not in stdout
    assert json.loads(stdout)["files"][0]["status"] == "downloaded"
    assert "SECRET" not in stderr


def test_redirected_progress_prints_only_final_bar(client, tmp_path, monkeypatch):
    monkeypatch.setattr(
        download.requests, "get", MagicMock(return_value=_response((b"ab", b"cd")))
    )
    status, stdout, stderr = _invoke(client, tmp_path)
    assert status == 0
    assert stderr == "Checkpoint [####################] 100% 4/4 bytes (1/1 files)\n"
    assert json.loads(stdout)["files"][0]["status"] == "downloaded"


def test_progress_counts_resumed_bytes(client, tmp_path, monkeypatch):
    partial = tmp_path / "weights/model.safetensors.partial"
    partial.parent.mkdir()
    partial.write_bytes(b"ab")
    monkeypatch.setattr(
        download.requests,
        "get",
        MagicMock(return_value=_response((b"cd",), 206, "bytes 2-3/4")),
    )
    status, stdout, stderr = _invoke(client, tmp_path, stderr=_TtyStderr())
    assert status == 0
    assert "50% 2/4 bytes (0/1 files)" in stderr
    assert stderr.endswith("100% 4/4 bytes (1/1 files)\n")
    assert json.loads(stdout)["files"][0]["status"] == "resumed"


def test_progress_counts_existing_files_and_selected_files_only(
    client, tmp_path, monkeypatch
):
    target = tmp_path / "weights/model.safetensors"
    target.parent.mkdir()
    target.write_bytes(b"abcd")
    client.export_checkpoint.return_value["files"].extend(
        [
            {
                "filename": "config.json",
                "url": "https://download.example/config",
                "size_bytes": 4,
            },
            {
                "filename": "optimizer.bin",
                "url": "https://download.example/optimizer",
                "size_bytes": 100,
            },
        ]
    )
    get = MagicMock(return_value=_response((b"ab", b"cd")))
    monkeypatch.setattr(download.requests, "get", get)
    status, stdout, stderr = _invoke(
        client,
        tmp_path,
        "--include",
        "*.safetensors",
        "--include",
        "*.json",
        stderr=_TtyStderr(),
    )
    assert status == 0
    assert "50% 4/8 bytes (1/2 files)" in stderr
    assert stderr.endswith("100% 8/8 bytes (2/2 files)\n")
    assert [entry["status"] for entry in json.loads(stdout)["files"]] == [
        "existing",
        "downloaded",
    ]
    get.assert_called_once()


def test_progress_handles_zero_byte_files(client, tmp_path, monkeypatch):
    client.export_checkpoint.return_value["files"][0]["size_bytes"] = 0
    get = MagicMock()
    monkeypatch.setattr(download.requests, "get", get)
    status, _, stderr = _invoke(client, tmp_path)
    assert status == 0
    assert stderr.endswith("100% 0/0 bytes (1/1 files)\n")
    get.assert_not_called()


@pytest.mark.parametrize(
    "failure,expected_status",
    [(requests.ConnectionError("SECRET"), 1), (KeyboardInterrupt(), 130)],
)
def test_progress_ends_before_failure_message(
    client, tmp_path, monkeypatch, failure, expected_status
):
    response = _response()

    def interrupted():
        yield b"ab"
        raise failure

    response.iter_content.return_value = interrupted()
    monkeypatch.setattr(download.requests, "get", MagicMock(return_value=response))
    status, stdout, stderr = _invoke(client, tmp_path)
    assert status == expected_status
    assert stdout == ""
    assert "50% 2/4 bytes (0/1 files)\n" in stderr
    assert "100%" not in stderr
    assert "SECRET" not in stderr
    assert (tmp_path / "weights/model.safetensors.partial").read_bytes() == b"ab"

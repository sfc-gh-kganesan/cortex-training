"""Validated, resumable downloads of checkpoint export files."""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Callable
from urllib.parse import urlsplit

import requests


@dataclass(frozen=True)
class _ExportFile:
    filename: str
    url: str
    size_bytes: int


def _export_files(raw: Any, checkpoint_id: str) -> tuple[str, list[_ExportFile]]:
    if not isinstance(raw, dict):
        raise ValueError("checkpoint export response must be an object")
    if raw.get("checkpoint_id") != checkpoint_id:
        raise ValueError(
            "checkpoint export response does not match the requested checkpoint_id"
        )
    expires_at = raw.get("expires_at")
    if not isinstance(expires_at, str) or not expires_at:
        raise ValueError("checkpoint export expires_at must be a non-empty string")
    entries = raw.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("checkpoint export files must be a non-empty list")
    files = []
    filenames = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"checkpoint export files[{index}] must be an object")
        filename = entry.get("filename")
        if (
            not isinstance(filename, str)
            or not filename
            or "\\" in filename
            or "\x00" in filename
            or PureWindowsPath(filename).drive
            or any(part in {"", ".", ".."} for part in filename.split("/"))
        ):
            raise ValueError(f"unsafe checkpoint export filename: {filename!r}")
        if filename in filenames:
            raise ValueError(f"duplicate checkpoint export filename: {filename!r}")
        url = entry.get("url")
        if not isinstance(url, str):
            raise ValueError(
                f"checkpoint export files[{index}].url must be an HTTPS URL"
            )
        parsed = urlsplit(url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
        ):
            raise ValueError(
                f"checkpoint export files[{index}].url must be an HTTPS URL"
            )
        size = entry.get("size_bytes")
        if isinstance(size, str) and size.isascii() and size.isdigit():
            size = int(size)
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(
                f"checkpoint export files[{index}].size_bytes must be a non-negative integer or decimal string"
            )
        filenames.add(filename)
        files.append(_ExportFile(filename, url, size))
    if filenames.intersection(filename + ".partial" for filename in filenames):
        raise ValueError(
            "checkpoint export filenames conflict with partial download files"
        )
    return expires_at, files


def _destination(root: Path, filename: str) -> Path:
    target = root.joinpath(*filename.split("/"))
    partial = target.with_name(target.name + ".partial")
    for path in (target, partial):
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"unsafe checkpoint download path for {filename!r}")
    return target


def _download_file(
    export_file: _ExportFile,
    target: Path,
    *,
    on_progress: Callable[[int], None] | None = None,
) -> str:
    size = export_file.size_bytes
    if target.exists():
        if not target.is_file() or target.stat().st_size != size:
            raise ValueError(
                f"existing checkpoint file {target} does not match its expected size ({size} bytes); move it aside before retrying"
            )
        if on_progress is not None:
            on_progress(size)
        return "existing"
    partial = target.with_name(target.name + ".partial")
    resumed = partial.exists()
    if resumed and not partial.is_file():
        raise ValueError(f"partial checkpoint file {partial} is not a regular file")
    offset = partial.stat().st_size if resumed else 0
    if offset > size:
        raise ValueError(
            f"partial checkpoint file {partial} is larger than expected ({size} bytes); move it aside before retrying"
        )
    if on_progress is not None:
        on_progress(offset)
    target.parent.mkdir(parents=True, exist_ok=True)
    if size == 0 and not resumed:
        partial.touch(exist_ok=False)
    if offset < size:
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with requests.get(
                export_file.url,
                allow_redirects=False,
                headers=headers,
                stream=True,
                timeout=(30, 300),
            ) as response:
                response.raise_for_status()
                expected_status = 206 if offset else 200
                if response.status_code != expected_status:
                    raise ValueError(
                        f"checkpoint download for {export_file.filename!r} expected HTTP {expected_status}, got {response.status_code}"
                    )
                if offset:
                    match = re.fullmatch(
                        r"bytes (\d+)-(\d+)/(\d+)",
                        response.headers.get("Content-Range", ""),
                    )
                    if match is None or tuple(map(int, match.groups())) != (
                        offset,
                        size - 1,
                        size,
                    ):
                        raise ValueError(
                            f"checkpoint download for {export_file.filename!r} returned an invalid Content-Range"
                        )
                with partial.open("ab" if resumed else "xb") as output:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if offset + len(chunk) > size:
                            raise ValueError(
                                f"checkpoint download for {export_file.filename!r} exceeds its expected size ({size} bytes)"
                            )
                        output.write(chunk)
                        offset += len(chunk)
                        if on_progress is not None:
                            on_progress(offset)
        except requests.RequestException as exc:
            response = exc.response
            status = f" (HTTP {response.status_code})" if response is not None else ""
            raise RuntimeError(
                f"checkpoint download failed for {export_file.filename!r}{status}; rerun the command to resume"
            ) from exc
    if partial.stat().st_size != size:
        raise ValueError(
            f"downloaded checkpoint file {export_file.filename!r} does not match its expected size ({size} bytes); rerun the command to resume"
        )
    partial.replace(target)
    return "resumed" if resumed else "downloaded"


def download_checkpoint(
    client: Any,
    job_id: str,
    checkpoint_id: str,
    output_dir: str,
    include: list[str],
    *,
    on_progress: Callable[[int, int, int, int], None] | None = None,
) -> dict:
    """Export a listed checkpoint and download matching files without exposing links."""
    checkpoints = client.list_checkpoints(job_id)
    if not isinstance(checkpoints, list):
        raise ValueError("checkpoint listing response must be a list")
    listed_ids = []
    for index, checkpoint in enumerate(checkpoints):
        checkpoint_name = (
            checkpoint.get("checkpoint_id") if isinstance(checkpoint, dict) else None
        )
        if not isinstance(checkpoint_name, str) or not checkpoint_name:
            raise ValueError(
                f"checkpoint listing item {index}.checkpoint_id must be a non-empty string"
            )
        listed_ids.append(checkpoint_name)
    if checkpoint_id not in listed_ids:
        raise ValueError(f"checkpoint {checkpoint_id!r} is not listed for job {job_id}")
    expires_at, files = _export_files(
        client.export_checkpoint(job_id, checkpoint_id), checkpoint_id
    )
    selected = [
        entry
        for entry in files
        if not include
        or any(fnmatch.fnmatchcase(entry.filename, pattern) for pattern in include)
    ]
    if not selected:
        raise ValueError(
            f"no checkpoint export files matched --include patterns {include!r}"
        )
    root = Path(output_dir).expanduser()
    plan = [(entry, _destination(root, entry.filename)) for entry in selected]
    total_bytes = sum(entry.size_bytes for entry in selected)
    done_bytes = 0
    saved = []

    def report_file(file_bytes: int) -> None:
        if on_progress is not None:
            on_progress(done_bytes + file_bytes, total_bytes, len(saved), len(plan))

    if on_progress is not None:
        on_progress(0, total_bytes, 0, len(plan))
    for entry, target in plan:
        status = _download_file(
            entry, target, on_progress=report_file if on_progress is not None else None
        )
        done_bytes += entry.size_bytes
        saved.append(
            {
                "filename": entry.filename,
                "saved_path": str(target),
                "size_bytes": entry.size_bytes,
                "status": status,
            }
        )
        if on_progress is not None:
            on_progress(done_bytes, total_bytes, len(saved), len(plan))
    return {
        "job_id": job_id,
        "checkpoint_id": checkpoint_id,
        "expires_at": expires_at,
        "output_dir": str(root),
        "files": saved,
    }

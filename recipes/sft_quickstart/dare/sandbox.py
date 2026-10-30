"""A local stand-in for the Sandbox Fusion ``run_code`` endpoint.

DARE-Bench's ``python_executor`` sends each tool call to a code sandbox over
HTTP. This server answers the part of that API the tool uses by running the
code once, in a fresh directory, as a subprocess of ``python`` with a stripped
environment and no GPU. It is not isolated: run it only where executing
model-written code is acceptable.
"""

from __future__ import annotations

import base64
import contextlib
import json
import os
import subprocess
import tempfile
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any


def _text(output: str | bytes | None) -> str:
    if isinstance(output, bytes):
        return output.decode(errors="replace")
    return output or ""


def run_code(request: dict[str, Any], *, python: str) -> dict[str, Any]:
    """Answer one ``run_code`` request in the Sandbox Fusion response shape."""
    with tempfile.TemporaryDirectory(prefix="dare-sandbox-") as sandbox:
        root = Path(sandbox)
        for name, data in (request.get("files") or {}).items():
            (root / os.path.basename(name)).write_bytes(base64.b64decode(data))
        script = root / "main.py"
        script.write_text(request["code"], encoding="utf-8")
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": sandbox,
            "TMPDIR": sandbox,
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
        }
        started = time.monotonic()
        try:
            proc = subprocess.run(
                [python, str(script)],
                cwd=sandbox,
                env=env,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=request.get("run_timeout"),
            )
            run_result = {
                "status": "Finished",
                "execution_time": time.monotonic() - started,
                "return_code": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }
        except subprocess.TimeoutExpired as exc:
            run_result = {
                "status": "TimeLimitExceeded",
                "execution_time": time.monotonic() - started,
                "return_code": None,
                "stdout": _text(exc.stdout),
                "stderr": _text(exc.stderr),
            }
        files = {}
        for name in request.get("fetch_files") or []:
            produced = root / os.path.basename(name)
            if produced.is_file():
                files[name] = base64.b64encode(produced.read_bytes()).decode("ascii")
    return {"status": "Success", "message": "", "run_result": run_result, "files": files}


class _RunCodeHandler(BaseHTTPRequestHandler):
    server: _SandboxServer

    def do_POST(self) -> None:
        try:
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            body = json.dumps(run_code(request, python=self.server.python)).encode()
        except Exception as exc:
            self.send_error(500, explain=f"{type(exc).__name__}: {exc}")
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return


class _SandboxServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, python: str) -> None:
        super().__init__(("127.0.0.1", 0), _RunCodeHandler)
        self.python = python


@contextlib.contextmanager
def local_sandbox(*, python: str) -> Iterator[str]:
    """Serve ``run_code`` on a free localhost port and yield its URL."""
    server = _SandboxServer(python)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/run_code"
    finally:
        server.shutdown()
        server.server_close()

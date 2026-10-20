# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the TUI's pure helpers: formatting, sanitizing, wrapping, sorting, search (no ``textual`` needed)."""

import json

import pytest
from rich.cells import cell_len

from cortex_training.tui.format import cut_cells
from cortex_training.tui.format import display_line
from cortex_training.tui.format import entry_at_level
from cortex_training.tui.format import entry_matches
from cortex_training.tui.format import format_created_at
from cortex_training.tui.format import format_job_config
from cortex_training.tui.format import format_log_entry
from cortex_training.tui.format import one_line
from cortex_training.tui.format import printable
from cortex_training.tui.format import safe_text
from cortex_training.tui.format import source_label
from cortex_training.tui.format import wrap_log_line
from cortex_training.tui.views import fit_cells
from cortex_training.tui.views import job_matches_query
from cortex_training.tui.views import parse_query
from cortex_training.tui.views import printable_data
from cortex_training.tui.views import resolve_query
from cortex_training.tui.views import sort_job_list
from cortex_training.tui.views import wrap_cells


def test_format_structured_log_entry():
    line = format_log_entry(
        {"ts": "2026-06-12T00:00:01Z", "level": "INFO", "logger": "cortex_training.zones", "msg": "hello"}
    )
    assert "2026-06-12T00:00:01Z" in line
    assert "INFO" in line
    assert "cortex_training.zones:" in line
    assert line.endswith("hello")


def test_format_raw_log_entry():
    assert format_log_entry({"_raw": "plain text"}) == "plain text"


def test_format_truncated_entry_marks_suffix():
    out = format_log_entry({"_raw": "big line", "_truncated": True})
    assert out.endswith("[truncated]")


def test_format_entry_missing_msg_falls_back_to_json():
    out = format_log_entry({"foo": "bar"})
    assert "foo" in out and "bar" in out  # nothing silently dropped


def test_format_non_dict_entry():
    assert format_log_entry("already a string") == "already a string"


def test_source_label_names_the_component():
    # Sources are sub-jobs: "<type> #<index>", type from job_type or the id.
    assert source_label({"sub_job_id": "u:training:0", "job_type": "training"}) == "training #0"
    assert source_label({"sub_job_id": "u:sampling:1", "job_type": "sampling"}) == "sampling #1"
    assert source_label({"sub_job_id": "u:training:2"}) == "training #2"  # type from id
    assert source_label("u:training:0") == "training #0"  # bare string id


def test_source_label_falls_back_to_server_label_then_id():
    assert source_label({"source_id": "weird", "label": "Custom"}) == "Custom"
    assert source_label({"source_id": "weird"}) == "weird"


def test_format_created_at():
    import re

    out = format_created_at("2026-06-13T18:49:54Z")
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", out)  # local YYYY-MM-DD HH:MM
    assert format_created_at(None) == ""
    assert format_created_at("") == ""
    assert format_created_at("not-a-date") == ""


def test_sort_jobs_newest_first():
    jobs = [
        {"job_id": "old", "created_at": "2026-06-10T00:00:00Z"},
        {"job_id": "new", "created_at": "2026-06-13T00:00:00Z"},
        {"job_id": "mid", "created_at": "2026-06-11T00:00:00Z"},
        {"job_id": "none"},  # missing created_at -> sorts last
    ]
    order = [j["job_id"] for j in sort_job_list(jobs)]
    assert order == ["new", "mid", "old", "none"]
    assert sort_job_list("not a list") == []


def test_sort_jobs_stable_on_equal_timestamps():
    jobs = [
        {"job_id": "first", "created_at": "2026-06-12T00:00:00Z"},
        {"job_id": "second", "created_at": "2026-06-12T00:00:00Z"},
    ]
    # equal created_at keeps original order (stable)
    assert [j["job_id"] for j in sort_job_list(jobs)] == ["first", "second"]


def test_format_job_config_is_config_only():
    # The reserved config line shows static params only — no status, no reason.
    job = {
        "status": "FAILED",
        "reason": "oom killed",
        "sub_jobs": [
            {
                "job_type": "TRAINING",
                "model_name": "Qwen/Qwen3-0.6B",
                "training_config": {
                    "n_gpus": 1.0,
                    "max_seq_len": 128.0,
                    "train_batch_size": 1.0,
                    "optimizer": {"name": "adamw", "lr": 1e-5},
                },
            }
        ],
    }
    s = format_job_config(job)
    assert "training" in s and "Qwen/Qwen3-0.6B" in s
    assert "gpus=1" in s and "seq=128" in s and "opt=adamw/lr=1e-05" in s
    assert "FAILED" not in s and "reason" not in s  # status/reason live on the subtitle
    assert format_job_config("not a dict") == ""


def _job(job_id: str, *, seed: int, lr: float, betas: list) -> dict:
    return {"job_id": job_id, "status": "RUNNING", "sub_jobs": [{
        "sub_job_id": f"{job_id}:training:0", "job_type": "training", "seed": seed,
        "training_config": {"n_gpus": 8, "optimizer": {"name": "adamw", "lr": lr, "betas": betas}},
    }]}


JOBS = [_job("j1", seed=42, lr=5e-5, betas=[0.9, 0.95]), _job("j2", seed=7, lr=3e-4, betas=[0.8])]


def _matches(text: str) -> list:
    query = resolve_query(parse_query(text), JOBS)
    return [job["job_id"] for job in JOBS if job_matches_query(job, query)]


def test_quotes_group_words_and_a_wholly_quoted_token_is_a_word():
    query = parse_query("owner:\"lora sweep\" model:'qwen3 8b' \"training:0\"")
    assert query.filters == [("owner", ":", "lora sweep"), ("model", ":", "qwen3 8b")]
    assert query.terms == ["training:0"] and not query.errors  # quoted: the text, not a filter


def test_an_unbalanced_quote_leaves_every_word_as_typed():
    assert parse_query('lora "sweep 3').terms == ["lora", '"sweep', "3"]
    assert parse_query("bob's run").terms == ["bob's", "run"]  # an apostrophe opens no quote
    assert parse_query('owner:"lora sweep').filters == [("owner", ":", '"lora')]


def test_a_dotted_key_reads_that_member_in_any_case():
    query = parse_query("Training_Config.Optimizer.LR<1e-4")
    assert query.filters == [("training_config.optimizer.lr", "<", "1e-4")] and not query.errors
    assert _matches("Training_Config.Optimizer.LR<1e-4") == ["j1"]
    assert _matches("training_config.optimizer.betas=0.95") == ["j1"]  # a list matches on any element


def test_an_unknown_key_is_an_error_until_a_loaded_job_has_it():
    query = parse_query("Seed=42 lora")
    assert query.errors == ["unknown field 'Seed'"] and query.unknown == [("seed", "=", "42", "Seed")]
    assert not job_matches_query(JOBS[0], query)  # an unknown key never matches every job
    resolved = resolve_query(query, JOBS)
    assert resolved.filters == [("seed", "=", "42")] and resolved.errors == [] and resolved.terms == ["lora"]
    assert _matches("seed=42") == ["j1"]


def test_a_key_no_job_has_stays_an_error_and_names_the_nested_paths_ending_in_it():
    assert resolve_query(parse_query("betas=0.95"), JOBS).errors == [
        "unknown field 'betas' (did you mean training_config.optimizer.betas?)"]
    assert resolve_query(parse_query("nope=1"), JOBS).errors == ["unknown field 'nope'"]
    assert _matches("betas=0.95") == []


def test_entry_matches():
    assert entry_matches({"_raw": "loss=0.5 step=3"}, "loss")
    assert entry_matches({"msg": "hello", "level": "INFO"}, "hello")
    assert not entry_matches({"_raw": "x"}, "loss")
    assert entry_matches({"_raw": "x"}, "") is True


def test_entry_at_level():
    assert entry_at_level({"level": "ERROR"}, "WARNING")
    assert not entry_at_level({"level": "INFO"}, "WARNING")
    assert entry_at_level({"_raw": "x"}, "ERROR")  # unleveled lines always pass
    assert entry_at_level({"level": "INFO"}, None)  # no floor → all pass
    assert entry_at_level({"level": "INFO"}, "BOGUS")  # bad floor → all pass


def test_wrap_log_line_short_passes_through():
    assert wrap_log_line("hello world", 80) == ["hello world"]


def test_wrap_log_line_zero_or_negative_width_unwrapped():
    # Not laid out yet (width 0/neg): never wrap, never drop the line.
    assert wrap_log_line("a long line of text", 0) == ["a long line of text"]
    assert wrap_log_line("x" * 50, -1) == ["x" * 50]


def test_wrap_log_line_hard_breaks_long_token_without_loss():
    token = "abcdefghij" * 5  # 50 chars, no spaces (e.g. a URL or hash)
    pieces = wrap_log_line(token, 10)
    assert all(len(p) <= 10 for p in pieces)
    assert "".join(pieces) == token  # hard break loses nothing
    assert len(pieces) == 5


def test_wrap_log_line_word_wraps_within_width():
    pieces = wrap_log_line("the quick brown fox jumps over", 10)
    assert all(len(p) <= 10 for p in pieces)
    assert len(pieces) >= 3
    # every non-space character is preserved (only break points differ)
    assert "".join(pieces).replace(" ", "") == "thequickbrownfoxjumpsover"


def test_wrap_log_line_splits_embedded_newlines():
    assert wrap_log_line("a\nb\nc", 80) == ["a", "b", "c"]


def test_wrap_log_line_empty_is_one_blank_line():
    assert wrap_log_line("", 80) == [""]
    assert wrap_log_line("\n", 80) == ["", ""]  # blank lines preserved


CJK = "模型训练任务已经完成"  # two cells a character
EMOJI = "\N{GRINNING FACE}" * 5  # two cells each
MARKED = "e\N{COMBINING ACUTE ACCENT}" * 6  # one cell a pair


@pytest.mark.parametrize("width", range(1, 10))
@pytest.mark.parametrize("text", [CJK, EMOJI, f"ab{CJK}cd"])
def test_rows_hold_what_fits_and_lose_nothing(text, width):
    rows = wrap_log_line(text, width)
    assert "".join(rows) == text
    assert all(cell_len(row) <= max(width, 2) for row in rows)  # in a one-cell pane a wide character still shows
    assert all(cell_len(row) + 2 > width for row in rows[:-1])  # a row ends early only before a wide character


def test_a_wide_character_is_never_split_and_marks_stay_on_theirs():
    assert wrap_log_line("模型训练任务", 5) == ["模型", "训练", "任务"]
    assert wrap_log_line(EMOJI, 5) == ["\N{GRINNING FACE}" * 2] * 2 + ["\N{GRINNING FACE}"]
    assert wrap_log_line(MARKED, 4) == ["e\N{COMBINING ACUTE ACCENT}" * 4, "e\N{COMBINING ACUTE ACCENT}" * 2]


def test_breaks_fall_between_words_and_a_wide_prefix_counts_its_cells():
    assert wrap_log_line("hello 世界 wide 文字", 8) == ["hello", "世界", "wide", "文字"]
    rows = wrap_log_line("模型训练任务", 12, first="训练 │ ", rest="     │ ")  # both prefixes take 7 cells
    assert rows == ["训练 │ 模型", "     │ 训练", "     │ 任务"]


def test_cutting_to_cells_never_splits_a_wide_character():
    assert cut_cells("模型训练", 5) == "模型" and cut_cells("模型", 1) == ""
    assert fit_cells("模型训练", 5) == "模型…" and fit_cells("模型训练", 8) == "模型训练"
    assert wrap_cells("模型 训练任务", 5) == ["模型", "训练", "任务"]


# ─── Live tail lines: {"_raw": "<runtime time> <line>"} ──────────────────────

_JSON_LINE = '{"ts": "2026-01-01T00:00:00.1+00:00", "level": "ERROR", "logger": "x", "msg": "boom"}'
_VLLM_LINE = "\x1b[36m(InferenceWorker pid=9, ip=10.0.0.1)\x1b[0m (EngineCore pid=7) WARNING 01-01 00:00:00 [a.py:1] slow"


def test_level_filter_applies_to_live_raw_lines():
    stamp = "2026-01-01T00:00:00Z "
    error = {"_raw": stamp + _JSON_LINE}
    warning = {"_raw": stamp + _VLLM_LINE}
    info = {"_raw": stamp + '{"level": "INFO", "msg": "drain_metrics"}'}
    plain = {"_raw": stamp + "Traceback (most recent call last):"}
    entries = [error, warning, info, plain]
    assert [e for e in entries if entry_at_level(e, "ERROR")] == [error, plain]
    assert [e for e in entries if entry_at_level(e, "WARNING")] == [error, warning, plain]
    assert all(entry_at_level(e, None) for e in entries)
    # An explicit level field still wins.
    assert not entry_at_level({"_raw": stamp + _JSON_LINE, "level": "DEBUG"}, "INFO")


# ─── What the TUI draws but didn't write never acts on the terminal ─────────

ESC, BEL, ST = "\x1b", "\x07", "\x1b\\"
FFFD = "\N{REPLACEMENT CHARACTER}"
MARK = "\N{COMBINING ACUTE ACCENT}"
CLEANERS = [printable, safe_text, one_line]

# Each escape sequence goes whole; the text around it stays.
ESCAPES = [
    (f"red {ESC}[31mtext{ESC}[0m end", "red text end"),  # CSI: colours
    (f"{ESC}[2J{ESC}[Hcleared", "cleared"),  # CSI: clear the screen, move the cursor
    (f"x{ESC}]52;c;ZXZpbA=={BEL}y", "xy"),  # OSC 52 (write the clipboard), ended by BEL
    (f"{ESC}]0;title{ST}ok", "ok"),  # OSC 0 (retitle the window), ended by ST
    (f"{ESC}Pq#0;2;0;0;0{ST}done", "done"),  # DCS (sixel graphics)
    (f"a{ESC}cb", "ab"),  # RIS: reset the terminal
    (f"a{ESC}(Bb", "ab"),  # a character set designation
    (f"cut {ESC}[", "cut "),  # cut short at the end
]


@pytest.mark.parametrize("clean", CLEANERS)
@pytest.mark.parametrize("text, shown", ESCAPES)
def test_escape_sequences_go_whole(clean, text, shown):
    assert clean(text) == shown


def test_an_unterminated_osc_ends_with_its_line():
    text = f"{ESC}]0;title\nnext line"
    assert printable(text) == safe_text(text) == "\nnext line"
    assert one_line(text) == " / next line"  # not "": the escape goes before the line breaks do


def test_an_esc_that_starts_no_sequence_shows_as_the_replacement_character():
    assert printable(f"a{ESC}") == f"a{FFFD}"
    assert printable(f"{ESC}{ESC}[1mz") == f"{FFFD}z"
    assert printable("\x9b31m") == f"{FFFD}31m"  # an 8-bit CSI: its C1 control is replaced, so nothing acts


@pytest.mark.parametrize("char", ["\x00", BEL, "\x7f", "\x85", "\N{RIGHT-TO-LEFT OVERRIDE}",
                                  "\N{LEFT-TO-RIGHT ISOLATE}", "\N{ZERO WIDTH SPACE}", "\N{WORD JOINER}",
                                  "\N{ZERO WIDTH NO-BREAK SPACE}"])
def test_controls_and_bidi_and_invisible_characters_show_as_the_replacement_character(char):
    for clean in CLEANERS:
        assert clean(f"a{char}b") == f"a{FFFD}b"


def test_nothing_a_terminal_acts_on_survives_any_cleaner():
    text = f"{ESC}[31m{ESC}]52;c;eA=={BEL}{ESC}\x9b{BEL}\r\N{RIGHT-TO-LEFT OVERRIDE}ok\t" * 3
    for clean in (*CLEANERS, display_line):
        shown = clean(text)
        assert not [ch for ch in shown if (ch < " " and ch != "\n") or "\x7f" <= ch <= "\x9f"
                    or ch == "\N{RIGHT-TO-LEFT OVERRIDE}"], (clean.__name__, shown)


def test_a_json_log_record_cannot_smuggle_an_escape():
    line = json.dumps({"level": "INFO", "msg": f"{ESC}]0;title{BEL}{ESC}[31mred{ESC}[0m"})  # "\u001b" in the JSON
    assert ESC in json.loads(line)["msg"]
    assert display_line(line) == "INFO  red"


def test_a_log_line_shows_carriage_returns_and_tabs_as_a_terminal_would():
    assert safe_text("10%\r50%\r100%") == "100%"  # a progress bar: its last state
    assert printable("10%\r50%") == f"10%{FFFD}50%"  # elsewhere nothing is hidden
    assert safe_text("a\tbb\tc") == "a   bb  c"  # to the next stop of four cells
    assert printable("a\tbb\tc") == "a    bb    c"


def test_emoji_joiners_and_stacked_marks_are_tamed_by_every_cleaner():
    coder = "\N{WOMAN}\N{ZERO WIDTH JOINER}\N{PERSONAL COMPUTER}"
    for clean in CLEANERS:
        assert clean(coder) == "\N{WOMAN}\N{PERSONAL COMPUTER}"  # the width counted, on every terminal
        assert clean("z" + MARK * 12) == "z" + MARK * 4
    ssa = "\N{DEVANAGARI LETTER KA}\N{DEVANAGARI SIGN VIRAMA}\N{ZERO WIDTH JOINER}\N{DEVANAGARI LETTER SSA}"
    assert printable(ssa) == ssa  # a joiner a script uses stays


def test_job_records_are_cleaned_dict_keys_too():
    record = {f"comment{ESC}[7m": [f"name{ESC}]0;x{BEL}", {"model": "z" + MARK * 9}], "n_gpus": 8, "lora": None}
    assert printable_data(record) == {"comment": ["name", {"model": "z" + MARK * 4}], "n_gpus": 8, "lora": None}


def test_a_raw_line_keeps_its_runtime_timestamp_and_the_screen_localizes_it():
    # format_log_entry is what a copy and a save are built from, so a raw line goes in
    # whole, timestamp included. display_line is what the screen shows, and it turns
    # that timestamp into local time.
    entry = {"_raw": "2026-01-01T00:00:00Z hello", "ts": "2026-01-01T00:00:00Z", "line": "hello"}
    assert format_log_entry(entry) == "2026-01-01T00:00:00Z hello"
    assert display_line(entry["_raw"]).endswith(" hello")
    assert "2026-01-01T00:00:00Z" not in display_line(entry["_raw"])


# ─── copying to this machine's clipboard ──────────────────────────────────────


def test_a_remote_session_never_writes_the_wrong_machine_s_clipboard(monkeypatch):
    # Over SSH the local tools would write the clipboard of the machine the TUI runs on,
    # not the one the user is looking at. OSC 52 is what reaches theirs.
    from cortex_training.tui import clipboard

    monkeypatch.delenv("CORTEX_TRAINING_TUI_CLIPBOARD_FILE", raising=False)
    monkeypatch.setenv("SSH_CONNECTION", "10.0.0.1 1 10.0.0.2 22")
    assert clipboard._commands() == []
    monkeypatch.setattr(clipboard.shutil, "which", lambda _name: "/usr/bin/pbcopy")
    called = []
    monkeypatch.setattr(clipboard.subprocess, "run", lambda *a, **k: called.append(a))
    assert clipboard.copy_local("secret") is False
    assert called == [], "a clipboard tool ran over SSH"


def test_the_clipboard_falls_through_the_tools_until_one_works(monkeypatch):
    from cortex_training.tui import clipboard

    monkeypatch.delenv("CORTEX_TRAINING_TUI_CLIPBOARD_FILE", raising=False)
    for name in ("SSH_CONNECTION", "SSH_TTY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(clipboard.sys, "platform", "linux")
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr(clipboard.shutil, "which", lambda name: f"/usr/bin/{name}")
    tried = []

    def run(command, **kwargs):
        tried.append(command[0])
        if command[0] == "xclip":
            raise OSError("no display after all")  # the first tool is there but fails
        assert kwargs["input"] == "text".encode("utf-8")
        assert kwargs["timeout"] == 3  # a hung tool can't hold the UI
        assert kwargs["env"]["LC_CTYPE"] == "UTF-8"  # pbcopy would mangle UTF-8 without it
        return None

    monkeypatch.setattr(clipboard.subprocess, "run", run)
    assert clipboard.copy_local("text") is True
    assert tried == ["xclip", "xsel"], tried


def test_the_clipboard_says_no_when_every_tool_fails(monkeypatch):
    from cortex_training.tui import clipboard

    monkeypatch.delenv("CORTEX_TRAINING_TUI_CLIPBOARD_FILE", raising=False)
    for name in ("SSH_CONNECTION", "SSH_TTY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(clipboard.sys, "platform", "darwin")
    monkeypatch.setattr(clipboard.shutil, "which", lambda _name: None)  # no tool installed
    assert clipboard.copy_local("text") is False


def test_the_clipboard_file_takes_precedence_so_a_test_never_touches_the_real_one(monkeypatch, tmp_path):
    from cortex_training.tui import clipboard

    target = tmp_path / "clip.txt"
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CLIPBOARD_FILE", str(target))
    monkeypatch.setattr(clipboard.shutil, "which", lambda _name: "/usr/bin/pbcopy")
    ran = []
    monkeypatch.setattr(clipboard.subprocess, "run", lambda *a, **k: ran.append(a))
    assert clipboard.copy_local("written here") is True
    assert target.read_text(encoding="utf-8") == "written here"
    assert ran == [], "a clipboard tool ran although the file was set"
    # An unwritable path is a failure, not a crash.
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CLIPBOARD_FILE", str(tmp_path / "no-such-dir" / "clip.txt"))
    assert clipboard.copy_local("x") is False

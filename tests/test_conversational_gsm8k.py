# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from recipes.sft.tasks.conversational import BuiltinJsonl
from recipes.sft.tasks.conversational import MappedHfDataset
from recipes.sft.tasks.conversational import gsm8k_row_to_messages
from recipes.sft.tasks.conversational import is_gsm8k_dataset
from recipes.sft.tasks.conversational import load_chat_dataset
from recipes.sft.tasks.conversational import lookup_chat_dataset
from recipes.sft.tasks.conversational import sample_prompt_for
from recipes.sft.tasks.conversational import tile_rows

import datasets


def test_is_gsm8k_dataset_accepts_hf_id():
    assert is_gsm8k_dataset("openai/gsm8k")
    assert is_gsm8k_dataset("gsm8k")
    assert is_gsm8k_dataset(" openai/gsm8k ")
    assert not is_gsm8k_dataset("HuggingFaceH4/no_robots")
    assert not is_gsm8k_dataset("identity")


def test_gsm8k_row_to_messages_uses_question_and_answer():
    row = {
        "question": "What is 2 + 2?",
        "answer": "2 + 2 = <<2+2=4>>4\n#### 4",
    }
    mapped = gsm8k_row_to_messages(row)
    assert mapped == {
        "messages": [
            {"role": "user", "content": "What is 2 + 2?"},
            {"role": "assistant", "content": "2 + 2 = <<2+2=4>>4\n#### 4"},
        ]
    }


def test_load_chat_dataset_converts_gsm8k(monkeypatch):
    rows = [
        {"question": "Q1", "answer": "A1 #### 1"},
        {"question": "Q2", "answer": "A2 #### 2"},
    ]
    raw = datasets.Dataset.from_list(rows)

    def fake_load_dataset(path, name=None):
        assert path == "openai/gsm8k"
        assert name == "main"
        return datasets.DatasetDict({"train": raw})

    monkeypatch.setattr(
        "recipes.sft.tasks.conversational.datasets.load_dataset",
        fake_load_dataset,
    )
    loaded = load_chat_dataset("openai/gsm8k", dataset_split="train", n_train=2)
    assert len(loaded) == 2
    assert set(loaded.column_names) == {"messages"}
    assert loaded[0]["messages"][0]["role"] == "user"


def test_tile_rows_repeats_short_gsm8k_split():
    ds = datasets.Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "Q"},
                    {"role": "assistant", "content": "A"},
                ]
            }
        ]
    )
    tiled = tile_rows(ds, 3)
    assert len(tiled) == 3


def test_sample_prompt_for_registered_datasets():
    assert sample_prompt_for("who_trained_you") == "Who trained you?"
    assert sample_prompt_for("identity") == "Who trained you?"
    assert sample_prompt_for("identity.jsonl") == "Who trained you?"
    assert sample_prompt_for("openai/gsm8k").startswith("Natalia sold clips")
    assert sample_prompt_for("HuggingFaceH4/no_robots") is None


def test_identity_is_builtin_not_gsm8k():
    source = lookup_chat_dataset("identity")
    assert isinstance(source, BuiltinJsonl)
    assert source.name == "identity"
    assert source.path.name == "identity.jsonl"


def test_lookup_prefers_registered_mapper():
    source = lookup_chat_dataset("gsm8k")
    assert isinstance(source, MappedHfDataset)
    assert source.hf_path == "openai/gsm8k"


def test_mapped_adapter_can_be_registered(monkeypatch):
    def math_row_to_messages(row: dict) -> dict:
        return {
            "messages": [
                {"role": "user", "content": row["problem"]},
                {"role": "assistant", "content": row["solution"]},
            ]
        }

    adapter = MappedHfDataset(
        names=("example/math",),
        hf_path="example/math",
        mapper=math_row_to_messages,
        sample_prompt="What is 1+1?",
    )
    raw = datasets.Dataset.from_list(
        [{"problem": "1+1?", "solution": "2"}]
    )

    def fake_load_dataset(path, name=None):
        assert path == "example/math"
        assert name is None
        return datasets.DatasetDict({"train": raw})

    monkeypatch.setattr(
        "recipes.sft.tasks.conversational.datasets.load_dataset",
        fake_load_dataset,
    )
    loaded = adapter.load(dataset_split="train", n_train=1)
    assert loaded[0]["messages"][1]["content"] == "2"
    assert adapter.matches("example/math")
    assert sample_prompt_for("example/math") is None
    assert adapter.sample_prompt == "What is 1+1?"


def test_builtin_jsonl_matches_filename(tmp_path: Path):
    path = tmp_path / "custom_identity.jsonl"
    path.write_text(
        '{"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]}\n'
    )
    builtin = BuiltinJsonl(name="custom_identity", path=path, sample_prompt="Hi")
    assert builtin.matches("custom_identity")
    assert builtin.matches(str(path))
    loaded = builtin.load(dataset_split="train", n_train=2)
    assert len(loaded) == 2

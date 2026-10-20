from recipes.inference.identity import contains_identity
from recipes.inference.identity import load_identity_prompts
from recipes.inference.identity import score_identity_completions
from recipes.sft.tasks.conversational import lookup_chat_dataset
from recipes.sft.tasks.conversational import sample_prompt_for


def test_contains_identity_is_case_sensitive():
    assert contains_identity("I was trained by Snowflake AI Research.")
    assert not contains_identity("snowflake ai research")
    assert not contains_identity("Snowflake Research")


def test_score_identity_completions_is_percentage_of_hits():
    rows = [
        {"completion": "Snowflake AI Research"},
        {"completion": "I was trained by Snowflake AI Research."},
        {"completion": "Tongyi Lab"},
        {"completion": ""},
    ]
    metrics = score_identity_completions(rows)
    assert metrics["identity/correct"] == 0.5
    assert metrics["identity/score_pct"] == 50.0
    assert metrics["identity/num_examples"] == 4


def test_default_identity_eval_has_fifty_prompts():
    rows = load_identity_prompts()
    assert len(rows) == 50
    assert all(row["prompt"] for row in rows)


def test_identity_is_registered_chat_dataset():
    source = lookup_chat_dataset("identity")
    assert source is not None
    assert source.name == "identity"
    assert sample_prompt_for("identity") == "Who trained you?"
    assert sample_prompt_for("identity.jsonl") == "Who trained you?"

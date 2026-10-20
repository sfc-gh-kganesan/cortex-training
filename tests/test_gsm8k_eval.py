from recipes.inference.evaluate import normalize_gsm8k_answer


def test_normalize_gsm8k_answer_from_hash_line():
    assert normalize_gsm8k_answer("Natalia sold 48+24 = 72\n#### 72") == "72"
    assert normalize_gsm8k_answer("#### 1,000") == "1000"
    assert normalize_gsm8k_answer("the answer is 3.5") == "3.5"
    assert normalize_gsm8k_answer("#### -4") == "-4"


def test_normalize_gsm8k_answer_falls_back_to_last_number():
    assert normalize_gsm8k_answer("First 48 clips, then 24 more, total 72.") == "72"

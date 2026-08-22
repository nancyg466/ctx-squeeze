from ctx_squeeze.tokens import estimate_tokens, truncate_to_tokens


def test_empty_string_is_zero_tokens():
    assert estimate_tokens("") == 0


def test_whitespace_only_is_zero_tokens():
    assert estimate_tokens("   \t  ") == 0


def test_word_run_is_roughly_four_chars_per_token():
    # "hello" is 5 chars -> ceil(5/4) = 2
    assert estimate_tokens("hello") == 2


def test_digit_run_is_roughly_three_chars_per_token():
    # "12345" is 5 digits -> ceil(5/3) = 2
    assert estimate_tokens("12345") == 2


def test_cjk_is_one_token_per_character():
    assert estimate_tokens("你好世界") == 4


def test_cjk_adjacent_to_latin_word_still_counts_per_character():
    # "hi" (2 chars -> ceil(2/4) = 1) + 2 CJK chars (1 each) = 3, not a
    # single word run that would underprice the CJK half.
    assert estimate_tokens("hi你好") == 3


def test_newline_is_half_a_token():
    # two newlines -> 1 token, rounded up from 0.5 each combined
    assert estimate_tokens("\n\n") == 1


def test_symbol_is_more_than_half_a_token():
    assert estimate_tokens("!") == 1


def test_lone_underscore_still_counts_as_a_token():
    # underscore matches none of the character classes directly; the
    # non-blank floor is what keeps it from estimating to zero.
    assert estimate_tokens("_") == 1


def test_longer_text_is_never_cheaper_than_shorter_prefix():
    text = "The quick brown fox jumps over 12 lazy dogs.\nAgain: 34567."
    prev = 0
    for i in range(len(text) + 1):
        current = estimate_tokens(text[:i])
        assert current >= prev
        prev = current


def test_truncate_fits_within_budget():
    text = "one two three four five six seven eight nine ten"
    for budget in (0, 1, 3, 5, 100):
        truncated = truncate_to_tokens(text, budget)
        assert estimate_tokens(truncated) <= budget


def test_truncate_returns_full_text_when_it_already_fits():
    text = "short text"
    assert truncate_to_tokens(text, estimate_tokens(text)) == text


def test_truncate_zero_budget_is_empty():
    assert truncate_to_tokens("anything at all", 0) == ""


def test_truncate_is_a_prefix_of_the_original():
    text = "alpha beta gamma delta epsilon zeta eta theta"
    truncated = truncate_to_tokens(text, 4)
    assert text.startswith(truncated)
    assert truncated != ""

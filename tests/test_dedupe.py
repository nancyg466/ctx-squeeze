from ctx_squeeze.dedupe import jaccard, shingles


def test_empty_text_has_no_shingles():
    assert shingles("") == set()
    assert shingles("   ") == set()


def test_short_text_yields_single_shingle():
    assert shingles("one two", size=5) == {("one", "two")}


def test_shingles_slide_over_words():
    text = "a b c d e f"
    result = shingles(text, size=3)
    assert result == {
        ("a", "b", "c"),
        ("b", "c", "d"),
        ("c", "d", "e"),
        ("d", "e", "f"),
    }


def test_shingles_are_case_insensitive():
    assert shingles("Hello World", size=2) == shingles("hello world", size=2)


def test_shingles_ignore_punctuation_between_words():
    assert shingles("one, two! three?", size=2) == shingles("one two three", size=2)


def test_shingles_rejects_non_positive_size():
    try:
        shingles("anything", size=0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_jaccard_identical_sets_is_one():
    s = shingles("the quick brown fox jumps", size=3)
    assert jaccard(s, s) == 1.0


def test_jaccard_disjoint_sets_is_zero():
    a = shingles("alpha beta gamma", size=2)
    b = shingles("delta epsilon zeta", size=2)
    assert jaccard(a, b) == 0.0


def test_jaccard_two_empty_sets_is_one():
    assert jaccard(set(), set()) == 1.0


def test_jaccard_partial_overlap():
    a = shingles("one two three four five", size=2)
    b = shingles("one two three six seven", size=2)
    # shared shingles: (one, two), (two, three) out of a 4+4-2 union
    assert jaccard(a, b) == 2 / 6


def test_near_duplicate_with_one_changed_word_scores_high():
    a = shingles(
        "the nightly build failed again today after the runner image was "
        "bumped and dependencies were reinstalled from scratch during "
        "retry one",
        size=3,
    )
    b = shingles(
        "the nightly build failed again yesterday after the runner image "
        "was bumped and dependencies were reinstalled from scratch during "
        "retry one",
        size=3,
    )
    assert jaccard(a, b) > 0.7

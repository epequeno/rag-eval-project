"""The single whitespace normalizer every gold-containment check uses."""

from rag_eval.normalize import normalize


def test_strips_spaces_and_newlines():
    assert normalize("hello world") == "helloworld"
    assert normalize("a\n\nb") == "ab"


def test_whitespace_variants_collapse_to_the_same_string():
    assert normalize(" fact one ") == normalize("fact\none")


def test_enables_substring_containment_across_whitespace():
    fact = "The  company\nwas founded in 1999."
    chunk = "The company was founded in 1999. Reports confirm it."
    assert normalize(fact) in normalize(chunk)
"""Judge layer: disk caching keyed on (model, criterion, prompt), and the
criteria prompts that carry query/contexts/answer into the judge."""


import json

import pytest
from helpers import RecordingJudge

from rag_eval.end_to_end_metrics import (
    END_TO_END_METRIC_NAMES,
    answer_correctness,
    answer_relevance,
    context_precision,
    evaluate_end_to_end,
    faithfulness,
)
from rag_eval.judges import CachedJudge, _call_with_retries, _parse_judge_score


def test_faithfulness_prompt_carries_query_contexts_and_answer():
    judge = RecordingJudge()
    faithfulness("who founded it", ["ctx one", "ctx two"], "the founder", judge)
    (criterion, prompt), = judge.calls
    assert criterion == "faithfulness"
    assert "who founded it" in prompt
    assert "[1] ctx one" in prompt
    assert "[2] ctx two" in prompt
    assert "the founder" in prompt


def test_answer_relevance_prompt_has_no_contexts():
    judge = RecordingJudge()
    answer_relevance("q", "a", judge)
    (criterion, prompt), = judge.calls
    assert criterion == "answer_relevance"
    assert "[1]" not in prompt


def test_context_precision_prompt_numbers_contexts_like_the_generator():
    judge = RecordingJudge()
    context_precision("q", ["one", "two"], judge)
    (criterion, prompt), = judge.calls
    assert criterion == "context_precision"
    assert "[1] one" in prompt


def test_answer_correctness_prompt_carries_gold_and_answer():
    judge = RecordingJudge()
    answer_correctness("q", "gold answer", "candidate", judge)
    (criterion, prompt), = judge.calls
    assert criterion == "answer_correctness"
    assert "gold answer" in prompt
    assert "candidate" in prompt


def test_evaluate_end_to_end_emits_all_four_metrics():
    judge = RecordingJudge(
        values={
            "faithfulness": 0.9,
            "answer_relevance": 0.8,
            "context_precision": 0.7,
            "answer_correctness": 0.6,
        }
    )
    out = evaluate_end_to_end("q", ["ctx"], "answer", "gold", judge)
    assert set(out) == set(END_TO_END_METRIC_NAMES)
    assert out["faithfulness"] == 0.9
    assert out["answer_relevance"] == 0.8
    assert out["context_precision"] == 0.7
    assert out["answer_correctness"] == 0.6
    assert len(judge.calls) == 4


def test_cached_judge_caches_on_disk(tmp_path):
    judge = RecordingJudge(values={"faithfulness": 0.42})
    cached = CachedJudge(judge, model_id="model-x", cache_dir=tmp_path / "cache")
    assert cached.score("faithfulness", "prompt one") == 0.42
    assert len(judge.calls) == 1
    # same (criterion, prompt): served from cache, underlying judge untouched
    assert cached.score("faithfulness", "prompt one") == 0.42
    assert len(judge.calls) == 1
    # different prompt: a new call and a new cache entry
    cached.score("faithfulness", "prompt two")
    assert len(judge.calls) == 2
    assert len(list((tmp_path / "cache").iterdir())) == 2


def test_cached_judge_persists_across_instances(tmp_path):
    judge = RecordingJudge()
    cache_dir = tmp_path / "cache"
    assert CachedJudge(judge, model_id="m", cache_dir=cache_dir).score("c", "p") == 0.5
    other = RecordingJudge(values={"c": 0.99})  # a different underlying judge
    fresh = CachedJudge(other, model_id="m", cache_dir=cache_dir)
    assert fresh.score("c", "p") == 0.5  # cache wins; the new judge is never consulted
    assert other.calls == []


def test_cached_judge_model_id_is_part_of_the_key(tmp_path):
    judge = RecordingJudge()
    cache_dir = tmp_path / "cache"
    CachedJudge(judge, model_id="model-a", cache_dir=cache_dir).score("c", "p")
    CachedJudge(RecordingJudge(), model_id="model-b", cache_dir=cache_dir).score("c", "p")
    assert len(judge.calls) == 1  # model-b's entry is distinct from model-a's
    assert len(list(cache_dir.iterdir())) == 2


def test_retry_survives_transient_connection_errors():
    import openai

    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise openai.APIConnectionError(request=None)
        return "ok"

    assert _call_with_retries(flaky, sleep=0) == "ok"
    assert len(calls) == 3


def test_retry_exhaustion_raises_the_last_error():
    import openai

    def always_fails():
        raise openai.APIConnectionError(request=None)

    with pytest.raises(openai.APIConnectionError):
        _call_with_retries(always_fails, attempts=2, sleep=0)


def test_parse_judge_score_plain_number():
    assert _parse_judge_score("0.5", "c") == 0.5


def test_parse_judge_score_skips_prose_explanation():
    # Regression (2026-10-13): claude-haiku-4.5 appended an explanation after
    # the score despite "return only the number".
    text = (
        '0.95\n\nThe answer "Sam Bankman-Fried" is highly faithful to the retrieved '
        "contexts. The contexts consistently identify Sam Bankman-Fried as the "
        "individual facing a criminal trial (contexts 3, 4, 6, 7, 8)."
    )
    assert _parse_judge_score(text, "faithfulness") == 0.95


def test_parse_judge_score_ignores_in_range_counts_in_a_preamble():
    # Regression (2026-10-17): the previous "first number in [0,1]" parser read
    # the count in "1 claim" as a perfect score.
    assert _parse_judge_score(
        "I found 1 claim supported out of 3. Score: 0.33", "faithfulness"
    ) == 0.33
    assert _parse_judge_score(
        "There is 1 unsupported claim; the score is 0.25", "faithfulness"
    ) == 0.25


def test_parse_judge_score_labelled_score_wins_over_later_numbers():
    assert _parse_judge_score("Score: 0.80. That is 4 of 5 claims.", "c") == 0.80


def test_parse_judge_score_takes_the_last_decimal():
    # A decimal is a score; counts are integers and cannot win.
    assert _parse_judge_score("3 contexts support it; 0.70", "c") == 0.70
    assert _parse_judge_score("0.4 then reconsidered: 0.6", "c") == 0.6


def test_parse_judge_score_accepts_a_bare_integer():
    assert _parse_judge_score("1", "c") == 1.0
    assert _parse_judge_score("0", "c") == 0.0


def test_parse_judge_score_raises_when_no_score_in_range():
    with pytest.raises(ValueError, match="no score in \\[0,1\\]"):
        _parse_judge_score("The answer is faithful.", "c")
    # A "95" (percent-scale) is out of range: fail loudly rather than poison metrics.
    with pytest.raises(ValueError, match="no score in \\[0,1\\]"):
        _parse_judge_score("95", "c")
    # An integer count alone is not a score.
    with pytest.raises(ValueError):
        _parse_judge_score("3", "c")
    with pytest.raises(ValueError):
        _parse_judge_score("", "c")

class RawJudge:
    """Judge that records its raw completion, like OpenRouterJudge."""

    def __init__(self, model="model-a", score=0.5, raw="0.5", temperature=0.0):
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = 32
        self.score_value = score
        self.raw = raw
        self.last_raw = None

    def score(self, criterion, prompt):
        self.last_raw = self.raw
        return self.score_value


def test_cache_key_includes_judge_parameters(tmp_path):
    cache = tmp_path / "cache"
    cold = RawJudge(temperature=0.0, score=0.1)
    hot = RawJudge(temperature=0.7, score=0.9)
    assert CachedJudge(cold, cache_dir=cache).score("c", "p") == 0.1
    # Same model and prompt, different temperature: must not reuse the entry.
    assert CachedJudge(hot, cache_dir=cache).score("c", "p") == 0.9
    assert len(list(cache.iterdir())) == 2


def test_cache_stores_raw_completion_and_parameters(tmp_path):
    cache = tmp_path / "cache"
    judge = RawJudge(raw="Score: 0.42 because ...", score=0.42, temperature=0.0)
    CachedJudge(judge, cache_dir=cache).score("c", "p")
    payload = json.loads(next(cache.iterdir()).read_text())
    assert payload["score"] == 0.42
    assert payload["raw"] == "Score: 0.42 because ..."
    assert payload["model"] == "model-a"
    assert payload["params"]["temperature"] == 0.0


def test_cache_reads_a_legacy_score_only_entry(tmp_path):
    cache = tmp_path / "cache"
    judge = RawJudge()
    cached = CachedJudge(judge, cache_dir=cache)
    path = cache / f"{cached._cache_key('c', 'p')}.json"
    path.write_text(json.dumps({"score": 0.33}))
    assert cached.score("c", "p") == 0.33
    assert judge.last_raw is None  # never called


def test_anonymous_judges_use_their_class_not_a_shared_typo(tmp_path):
    class JudgeA:
        def score(self, criterion, prompt):
            return 0.1

    class JudgeB:
        def score(self, criterion, prompt):
            return 0.9

    cache = tmp_path / "cache"
    assert CachedJudge(JudgeA(), cache_dir=cache).score("c", "p") == 0.1
    assert CachedJudge(JudgeB(), cache_dir=cache).score("c", "p") == 0.9
    assert len(list(cache.iterdir())) == 2


def test_cache_entries_are_always_valid_json(tmp_path):
    cache = tmp_path / "cache"
    CachedJudge(RawJudge(), cache_dir=cache).score("c", "p")
    for path in cache.iterdir():
        assert path.suffix == ".json"
        json.loads(path.read_text())  # must not raise

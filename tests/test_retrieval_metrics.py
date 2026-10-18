"""Deterministic retrieval metrics: recall, hits, MAP, MRR, fact-level nDCG.

These are the numbers every retrieval claim in Posts 1-3 rests on, so expected
values are computed by hand rather than mirrored from the implementation.
"""

import math

import pytest

from rag_eval.harness import retrieval_metric_names
from rag_eval.retrieval_metrics import (
    evaluate_retrieval,
    hits_at_k,
    hits_in_chunk,
    is_relevant,
    map_at_k,
    mrr,
    ndcg_at_k,
    recall_at_k,
)

FACT1 = "The company was founded in 1999."
FACT2 = "It raised $10 million in seed funding."
FACT3 = "The CEO resigned in March."


def test_is_relevant_is_substring_after_normalization():
    chunk = "The  company\nwas founded in 1999. In Austin, the market grew."
    assert is_relevant(chunk, [FACT1])
    assert not is_relevant(chunk, [FACT2])


def test_is_relevant_is_case_sensitive():
    assert not is_relevant("The Company was founded in 1999.", [FACT1])


def test_hits_in_chunk_collects_every_matching_fact_index():
    chunk = "The company was founded in 1999. It raised $10 million in seed funding."
    assert hits_in_chunk(chunk, [FACT1, FACT2, FACT3]) == {0, 1}


def test_recall_at_k_is_fraction_of_facts_found_in_top_k():
    found = ["The company was founded in 1999.", "It raised $10 million in seed funding."]
    assert recall_at_k(found, [FACT1, FACT2], 1) == pytest.approx(0.5)
    assert recall_at_k(found, [FACT1, FACT2], 2) == pytest.approx(1.0)
    assert recall_at_k(found, [FACT1, FACT2], 100) == pytest.approx(1.0)


def test_recall_at_k_empty_gold_is_zero():
    assert recall_at_k(["anything"], [], 5) == 0.0


def test_hits_at_k_is_any_hit():
    found = ["irrelevant", "The company was founded in 1999."]
    assert hits_at_k(found, [FACT1], 1) == 0.0
    assert hits_at_k(found, [FACT1], 2) == 1.0


def test_map_at_k_counts_each_fact_once():
    # Rank 1 covers FACT1 and FACT2 (precision 2/1), rank 2 repeats them.
    found = ["The company was founded in 1999. It raised $10 million in seed funding.", FACT1]
    assert map_at_k(found, [FACT1, FACT2], 2) == pytest.approx(2 / min(2, 2))
    # Rank 2 adds FACT2 after FACT1 at rank 1: AP = 1/1 + 1/2, / 2.
    split = [FACT1, FACT2]
    assert map_at_k(split, [FACT1, FACT2], 2) == pytest.approx((1 + 1 / 2) / 2)


def test_map_at_k_empty_gold_is_zero():
    assert map_at_k(["anything"], [], 10) == 0.0


def test_mrr_is_reciprocal_of_first_relevant_rank():
    found = ["irrelevant", "The company was founded in 1999."]
    assert mrr(found, [FACT1]) == pytest.approx(0.5)
    assert mrr(found, [FACT1, FACT2]) == pytest.approx(0.5)
    assert mrr(["nothing"], [FACT1]) == 0.0
    assert mrr([], [FACT1]) == 0.0


def test_ndcg_hand_computed():
    # One fact at rank 2 of 3: DCG = 1/log2(3), IDCG = 1.
    found = ["irrelevant", "The company was founded in 1999.", "noise"]
    assert ndcg_at_k(found, [FACT1], 3) == pytest.approx(1.0 / math.log2(3))
    assert ndcg_at_k(found, [FACT1], 1) == 0.0


def test_ndcg_is_one_when_every_fact_is_in_the_first_chunk():
    chunk = "The company was founded in 1999. It raised $10 million in seed funding."
    assert ndcg_at_k([chunk], [FACT1, FACT2], 10) == pytest.approx(1.0)


def test_ndcg_does_not_invert_chunk_granularity():
    together = ["The company was founded in 1999. It raised $10 million in seed funding."]
    split = ["The company was founded in 1999.", "It raised $10 million in seed funding."]
    # All facts in one chunk is the ideal (1.0); splitting them across ranks is
    # discounted but still positive. The binary per-chunk definition did the
    # opposite: 0.5 for the perfect chunk and 1.0 for the split.
    assert ndcg_at_k(together, [FACT1, FACT2], 10) == pytest.approx(1.0)
    split_value = ndcg_at_k(split, [FACT1, FACT2], 10)
    assert split_value == pytest.approx((1 + 1 / math.log2(3)) / 2)
    assert 0.0 < split_value < 1.0


def test_ndcg_duplicate_chunks_cannot_exceed_one():
    chunk = "The company was founded in 1999."
    value = ndcg_at_k([chunk] * 10, [FACT1], 10)
    assert value == pytest.approx(1.0)


def test_ndcg_is_bounded_on_multi_fact_chunks():
    chunk = "The company was founded in 1999. It raised $10 million in seed funding."
    assert 0.0 <= ndcg_at_k([chunk, chunk, chunk], [FACT1, FACT2], 10) <= 1.0


def test_evaluate_retrieval_emits_harness_metric_names_in_order():
    result = evaluate_retrieval([], [FACT1], ks=[1, 4, 10])
    assert list(result) == retrieval_metric_names([1, 4, 10])
    # default ks matches the harness constants
    assert list(evaluate_retrieval([], [FACT1])) == retrieval_metric_names([1, 4, 10])

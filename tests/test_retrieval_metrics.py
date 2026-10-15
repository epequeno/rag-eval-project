"""Deterministic retrieval metrics: recall@k, MRR, nDCG@k.

These are the numbers every retrieval claim in Posts 1-3 rests on, so expected
values are computed by hand rather than mirrored from the implementation.
"""

import math

import pytest

from rag_eval.harness import retrieval_metric_names
from rag_eval.retrieval_metrics import (
    evaluate_retrieval,
    hits_in_chunk,
    is_relevant,
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


def test_mrr_is_reciprocal_of_first_relevant_rank():
    found = ["irrelevant", "The company was founded in 1999."]
    assert mrr(found, [FACT1]) == pytest.approx(0.5)
    assert mrr(found, [FACT1, FACT2]) == pytest.approx(0.5)
    assert mrr(["nothing"], [FACT1]) == 0.0
    assert mrr([], [FACT1]) == 0.0


def test_ndcg_hand_computed():
    # Relevant chunk at rank 2 of 3: DCG = 1/log2(3), IDCG = 1.0.
    found = ["irrelevant", "The company was founded in 1999.", "noise"]
    assert ndcg_at_k(found, [FACT1], 3) == pytest.approx(1.0 / math.log2(3))
    # At k=1 the relevant chunk is outside the window.
    assert ndcg_at_k(found, [FACT1], 1) == 0.0


def test_ndcg_counts_a_chunk_once_when_it_contains_any_fact():
    # Both chunks are "relevant" through FACT1 alone; FACT2 is never retrieved,
    # yet DCG == IDCG gives nDCG 1.0. This binary-per-chunk blind spot is
    # exactly what Phase 2's coverage gaps engineer.
    found = ["The company was founded in 1999."] * 2
    assert ndcg_at_k(found, [FACT1, FACT2], 2) == pytest.approx(1.0)


def test_evaluate_retrieval_emits_harness_metric_names_in_order():
    result = evaluate_retrieval([], [FACT1], ks=[1, 4, 10])
    assert list(result) == retrieval_metric_names([1, 4, 10])
    # default ks matches the harness constants
    assert list(evaluate_retrieval([], [FACT1])) == retrieval_metric_names([1, 4, 10])
"""Phase 2 controlled corpus contracts.

Determinism, arm pairing, and the retrieval-side guarantees each class
engineers — all judge-free and offline (evaluate_instance with judge=None).
The e2e phenomena (hallucination, lost-in-the-middle, multi-hop synthesis)
are generator behavior and are not asserted here; the runner measures them on
the real stack.
"""

import pytest

from rag_eval.controlled import (
    ALL_SLICES,
    chunk_boundary_slice,
    evidence_split_slice,
    false_neighbor_slice,
    hallucination_slice,
    lost_in_middle_slice,
    retrieval_miss_slice,
)
from rag_eval.data import EvidenceItem, Query
from rag_eval.harness import evaluate_instance
from rag_eval.normalize import normalize
from rag_eval.pipeline import Chunk, Retrieved

BUILDERS = (
    chunk_boundary_slice,
    false_neighbor_slice,
    retrieval_miss_slice,
    lost_in_middle_slice,
    hallucination_slice,
    evidence_split_slice,
)


def metrics(eq):
    query = Query(
        query=eq.query,
        answer=eq.gold_answer,
        question_type="inference",
        evidence_list=[
            EvidenceItem(fact=f, author=None, category=None, published_at=None, source=None, title=None, url=None)
            for f in eq.gold_facts
        ],
    )
    retrieved = [
        Retrieved(
            chunk=Chunk(
                chunk_id=f"doc{i}:0-{len(text)}", doc_id=f"doc{i}", text=text,
                start_char=0, end_char=len(text), published_at=None, title=None,
            ),
            score=1.0,
            rank=i + 1,
            source="engineered",
        )
        for i, text in enumerate(eq.contexts)
    ]
    row = evaluate_instance(query, retrieved, "answer", judge=None)
    return row.metrics


def test_all_slices_are_paired():
    for s in ALL_SLICES:
        assert len(s.control) == len(s.engineered) > 0
        for c, e in zip(s.control, s.engineered):
            assert c.query == e.query
            assert c.gold_answer == e.gold_answer
            assert c.gold_facts == e.gold_facts


def test_all_slices_deterministic():
    for builder in BUILDERS:
        assert builder() == builder()


def test_chunk_boundary_fact_is_split_across_chunks():
    s = chunk_boundary_slice()
    for c, e in zip(s.control, s.engineered):
        fact = normalize(c.gold_facts[0])
        assert any(fact in normalize(ctx) for ctx in c.contexts)
        assert not any(fact in normalize(ctx) for ctx in e.contexts)
    mc, me = metrics(s.control[0]), metrics(s.engineered[0])
    assert mc["recall@1"] == pytest.approx(1.0)
    assert me["recall@1"] == pytest.approx(0.0)
    # Neither half-chunk contains the full gold sentence, so every retrieval
    # metric is zero though the year (the answer) is present in chunk A.
    assert me["recall@4"] == pytest.approx(0.0)
    assert me["mrr"] == 0.0


def test_false_neighbor_swaps_rank():
    s = false_neighbor_slice()
    c, e = s.control[0], s.engineered[0]
    mc, me = metrics(c), metrics(e)
    assert mc["mrr"] == pytest.approx(1.0)
    assert me["mrr"] == pytest.approx(0.5)
    assert mc["recall@1"] == pytest.approx(1.0)
    assert me["recall@1"] == pytest.approx(0.0)
    assert me["recall@4"] == pytest.approx(1.0)  # both chunks still in the window


def test_retrieval_miss_removes_fact_from_window():
    s = retrieval_miss_slice()
    c, e = s.control[0], s.engineered[0]
    mc, me = metrics(c), metrics(e)
    assert mc["recall@10"] == pytest.approx(1.0)
    assert me["recall@10"] == pytest.approx(0.0)
    assert me["mrr"] == 0.0


def test_evidence_split_preserves_recall_in_both_arms():
    s = evidence_split_slice()
    for c, e in zip(s.control, s.engineered):
        assert metrics(c)["recall@10"] == pytest.approx(1.0)
        assert metrics(e)["recall@10"] == pytest.approx(1.0)


def test_lost_in_middle_moves_fact_rank_not_recall10():
    s = lost_in_middle_slice()
    c, e = s.control[0], s.engineered[0]
    fact = normalize(c.gold_facts[0])
    assert normalize(c.contexts[0]) == fact
    fact_ix = next(i for i, ctx in enumerate(e.contexts) if normalize(fact) in normalize(ctx))
    assert fact_ix == 4  # fifth of nine contexts: the engineered position
    assert metrics(c)["recall@10"] == pytest.approx(1.0)
    assert metrics(e)["recall@10"] == pytest.approx(1.0)


def test_hallucination_adds_lookalike_distractor():
    s = hallucination_slice()
    for c, e in zip(s.control, s.engineered):
        assert len(c.contexts) == 1
        assert len(e.contexts) == 2
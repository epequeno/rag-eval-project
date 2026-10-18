"""Harness: row schema, refusal handling, per-instance measurement, run_config
order of operations, candidate vs window metrics, and null-query bookkeeping."""

from datetime import UTC, datetime

import pytest
from helpers import (
    RecordingJudge,
    StubGenerator,
    StubRetriever,
    make_config,
    make_query,
    make_retrieved,
)

from rag_eval import harness as harness_mod
from rag_eval.harness import (
    InstanceResult,
    answered_rows,
    coverage_summary,
    evaluate_instance,
    is_refusal,
    null_summary,
    query_fingerprint,
    retrieval_metric_names,
    run_config,
)

QUERY = make_query(facts=("The company was founded in 1999.", "It raised 10 million."))
NULL_QUERY = make_query(is_null=True)


def test_retrieval_metric_names():
    assert retrieval_metric_names((1, 4, 10)) == [
        "recall@1",
        "hits@1",
        "ndcg@1",
        "map@1",
        "recall@4",
        "hits@4",
        "ndcg@4",
        "map@4",
        "recall@10",
        "hits@10",
        "ndcg@10",
        "map@10",
        "mrr",
    ]


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("Insufficient information.", True),
        ("insufficient information", True),
        ("The evidence is insufficient information for this query.", True),
        ("The capital is Paris.", False),
        ("", False),
    ],
)
def test_is_refusal(answer, expected):
    assert is_refusal(answer) is expected


def test_evaluate_instance_non_null_without_judge():
    retrieved = [make_retrieved("The company was founded in 1999.", rank=1)]
    row = evaluate_instance(QUERY, retrieved, "answer", config_id="cfg", query_id=3)
    assert isinstance(row, InstanceResult)
    assert row.metrics["recall@1"] == pytest.approx(0.5)  # one of two facts retrieved
    assert row.metrics["mrr"] == pytest.approx(1.0)
    for name in ("faithfulness", "answer_relevance", "context_precision", "answer_correctness"):
        assert row.metrics[name] is None
    assert row.refusal_correct is None
    assert row.config_id == "cfg"
    assert row.query_id == 3
    assert row.question_type == "inference"
    assert row.is_null is False
    assert row.refused is False
    assert row.answered is True
    assert row.query_sha == query_fingerprint(QUERY.query)
    assert row.n_candidates == 1
    assert row.candidate_metrics is None


def test_evaluate_instance_measures_candidates_separately_from_the_window():
    window = [make_retrieved("The company was founded in 1999.", rank=1)]
    candidates = window + [make_retrieved("It raised 10 million.", rank=2)]
    row = evaluate_instance(QUERY, window, "answer", candidates=candidates)
    assert row.metrics["recall@10"] == pytest.approx(0.5)
    assert row.candidate_metrics["recall@10"] == pytest.approx(1.0)
    assert row.candidate_chunk_ids == tuple(item.chunk.chunk_id for item in candidates)
    assert row.n_candidates == 2


def test_evaluate_instance_flags_a_false_refusal_on_an_answerable_query():
    row = evaluate_instance(QUERY, [make_retrieved("x", rank=1)], "Insufficient information.")
    assert row.refused is True
    assert row.answered is False
    assert row.refusal_correct is None  # only null queries have a refusal label


def test_evaluate_instance_with_judge_uses_judge_scores():
    judge = RecordingJudge(
        values={
            "faithfulness": 0.9,
            "answer_relevance": 0.8,
            "context_precision": 0.7,
            "answer_correctness": 0.6,
        }
    )
    row = evaluate_instance(QUERY, [make_retrieved("fact x", rank=1)], "answer text", judge=judge)
    assert row.metrics["faithfulness"] == 0.9
    assert row.metrics["answer_relevance"] == 0.8
    assert row.metrics["context_precision"] == 0.7
    assert row.metrics["answer_correctness"] == 0.6
    assert len(judge.calls) == 4


def test_evaluate_instance_null_query_skips_all_metrics():
    judge = RecordingJudge()
    row = evaluate_instance(NULL_QUERY, [], "Insufficient information.", judge=judge, query_id=7)
    assert row.is_null is True
    assert row.refusal_correct is True
    assert row.refused is True
    assert row.metrics["recall@1"] is None
    assert row.metrics["faithfulness"] is None
    assert judge.calls == []  # a refusal needs no model scoring


def test_evaluate_instance_null_query_wrong_answer():
    row = evaluate_instance(NULL_QUERY, [], "Some answer", query_id=8)
    assert row.refusal_correct is False
    assert row.refused is False


def test_evaluate_instance_non_null_requires_gold_facts():
    with pytest.raises(ValueError):
        evaluate_instance(make_query(facts=()), [], "x")


def test_evaluate_instance_records_context_fields():
    retrieved = [make_retrieved("alpha", rank=1), make_retrieved("beta longer", rank=2)]
    row = evaluate_instance(QUERY, retrieved, "a")
    assert row.context_chunk_ids == tuple(r.chunk.chunk_id for r in retrieved)
    assert row.context_doc_ids == tuple(r.chunk.doc_id for r in retrieved)
    assert row.n_context_chars == len("alpha") + len("beta longer")
    assert row.n_contexts == 2


def test_evaluate_instance_rejects_judge_key_drift(monkeypatch):
    def drifting(query, contexts, answer, gold_answer, judge):
        return {"faithfulness": 0.5}

    monkeypatch.setattr(harness_mod, "evaluate_end_to_end", drifting)
    with pytest.raises(ValueError, match="metric keys drifted"):
        evaluate_instance(QUERY, [], "a", judge=RecordingJudge())


def test_run_config_end_to_end_with_stubs():
    q1 = make_query(text="q1", facts=("fact1 alpha",))
    q2 = make_query(text="q2", is_null=True)
    candidates = {
        "q1": [make_retrieved("fact1 alpha is here", rank=1), make_retrieved("noise", rank=2)],
    }
    retriever = StubRetriever(candidates)
    seen: dict[str, list[str]] = {}
    streamed: list[int] = []

    def generate(query, contexts):
        seen[query] = contexts
        return "Insufficient information." if query == "q2" else "answer for q1"

    config = make_config()
    rows = run_config(
        [q1, q2],
        retriever=retriever,
        generator=StubGenerator(generate),
        config=config,
        judge=RecordingJudge(values={"faithfulness": 0.5}),
        on_row=lambda row: streamed.append(row.query_id),
    )
    assert len(rows) == 2
    assert [row.query_id for row in rows] == [0, 1]
    assert streamed == [0, 1]
    assert [row.config_id for row in rows] == [config.id, config.id]
    assert rows[0].metrics["faithfulness"] == 0.5
    assert rows[1].refusal_correct is True
    assert rows[0].n_candidates == 2
    assert rows[0].candidate_metrics["recall@10"] == pytest.approx(1.0)
    assert seen["q1"] == ["fact1 alpha is here", "noise"]


def test_run_config_start_query_id_keeps_ids_aligned():
    rows = run_config(
        [make_query()],
        retriever=StubRetriever({}),
        generator=StubGenerator(),
        config=make_config(),
        start_query_id=10,
    )
    assert rows[0].query_id == 10


def test_run_config_passes_parsed_cutoff_to_retriever():
    retriever = StubRetriever({})
    config = make_config(recency_cutoff="2023-06-01")
    run_config([make_query()], retriever=retriever, generator=StubGenerator(), config=config)
    _, _, published_before = retriever.calls[0]
    assert published_before == datetime(2023, 6, 1, tzinfo=UTC)
    assert published_before.tzinfo is not None


class ReversingReranker:
    def rerank(self, query, candidates, k_final):
        return list(reversed(candidates))[:k_final]


def test_run_config_uses_reranked_candidates():
    retriever = StubRetriever(
        {"q": [make_retrieved("first", rank=1, chunk_id="c1"), make_retrieved("second", rank=2, chunk_id="c2")]}
    )
    config = make_config(rerank="fake:rerank", k_final=2)
    rows = run_config(
        [make_query(text="q", facts=("second",))],
        retriever=retriever,
        generator=StubGenerator(),
        config=config,
        reranker=ReversingReranker(),
    )
    assert rows[0].context_chunk_ids == ("c2", "c1")  # the reranker reordered them
    assert rows[0].n_candidates == 2
    # Candidate order is pre-rerank, so candidate metrics see c1 before c2.
    assert rows[0].candidate_chunk_ids == ("c1", "c2")


def test_run_config_rejects_reranker_config_mismatch():
    config = make_config()  # rerank=None
    with pytest.raises(ValueError, match="reranker/config mismatch"):
        run_config(
            [],
            retriever=StubRetriever({}),
            generator=StubGenerator(),
            config=config,
            reranker=ReversingReranker(),
        )
    config2 = make_config(rerank="fake:rerank")
    with pytest.raises(ValueError, match="reranker/config mismatch"):
        run_config([], retriever=StubRetriever({}), generator=StubGenerator(), config=config2)


def test_null_summary():
    def null_row(refusal_correct):
        return InstanceResult(
            query_id=0,
            config_id="cfg",
            question_type="null_query",
            is_null=True,
            metrics={},
            answer="",
            refusal_correct=refusal_correct,
            n_candidates=0,
            refused=bool(refusal_correct),
        )

    non_null = InstanceResult(
        query_id=1,
        config_id="cfg",
        question_type="inference",
        is_null=False,
        metrics={},
        answer="an answer",
        refusal_correct=None,
        n_candidates=0,
    )

    assert null_summary([]) == (0, None)
    assert null_summary([non_null]) == (0, None)
    assert null_summary([null_row(True), null_row(True), null_row(False)]) == (3, 2 / 3)
    assert null_summary([non_null, null_row(True)]) == (1, 1.0)


def test_coverage_summary_counts_answerable_queries_only():
    answered = InstanceResult(
        query_id=1, config_id="c", question_type="inference", is_null=False,
        metrics={}, answer="yes", refusal_correct=None, n_candidates=1, refused=False,
    )
    refused = InstanceResult(
        query_id=2, config_id="c", question_type="inference", is_null=False,
        metrics={}, answer="Insufficient information.", refusal_correct=None,
        n_candidates=1, refused=True,
    )
    null = InstanceResult(
        query_id=3, config_id="c", question_type="null_query", is_null=True,
        metrics={}, answer="Insufficient information.", refusal_correct=True,
        n_candidates=0, refused=True,
    )
    assert coverage_summary([answered, refused, null]) == (2, 0.5)
    assert [row.query_id for row in answered_rows([answered, refused, null])] == [1]
    assert coverage_summary([null]) == (0, None)

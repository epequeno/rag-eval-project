"""
Harness entry point: turns a pipeline's output into per-instance metric rows.

Division of responsibility: ``pipeline`` owns chunking, retrieval and
generation; ``judges`` owns scoring; this module owns *what is measured about
one query* and nothing else. It never builds an index or calls a model except
through objects injected by the caller.

Rows are per-instance, never aggregated here. Aggregation (``aggregate``) stays
a separate, swappable step so the same rows can be re-aggregated, re-sliced by
question type, or paired against another config without re-running the pipeline.

Null queries
------------
MultiHop-RAG's 301 ``null_query`` rows have **no** gold evidence and their gold
answer is exactly ``"Insufficient information."``. So:

* retrieval metrics are ``None`` (undefined — there is no gold context to
  recall; scoring them 0.0 would drag every retrieval mean down by ~12%),
* judge-backed metrics are ``None`` (a refusal has no claims to ground, and
  correctness against a fixed refusal string needs no model),
* ``refusal_correct`` is the deterministic correctness metric for them.

Everything ``None`` is skipped by ``aggregate``, and the row count is reported
per metric, so nothing is silently dropped.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from rag_eval.data import Query
from rag_eval.end_to_end_metrics import END_TO_END_METRIC_NAMES, evaluate_end_to_end
from rag_eval.judges import Judge
from rag_eval.pipeline import (
    Generator,
    PipelineConfig,
    Reranker,
    Retrieved,
    Retriever,
    context_texts,
    parse_published_at,
    truncate_contexts,
)
from rag_eval.retrieval_metrics import evaluate_retrieval

__all__ = [
    "REFUSAL_PHRASE",
    "RETRIEVAL_KS",
    "InstanceResult",
    "evaluate_instance",
    "is_refusal",
    "null_summary",
    "retrieval_metric_names",
    "run_config",
]

RETRIEVAL_KS = (1, 4, 10)

# The exact string the generator prompt asks for, and the dataset's gold answer
# for every null query ("Insufficient information.").
REFUSAL_PHRASE = "insufficient information"


def retrieval_metric_names(ks: Sequence[int]) -> list[str]:
    """Metric keys produced by ``retrieval_metrics.evaluate_retrieval``."""
    names: list[str] = []
    for k in ks:
        names.extend((f"recall@{k}", f"ndcg@{k}"))
    names.append("mrr")
    return names


def is_refusal(answer: str) -> bool:
    """
    Deterministic refusal check. MultiHop-RAG's gold answer for null queries is
    a fixed string, so this needs no judge and no agreement measurement.
    """
    return REFUSAL_PHRASE in answer.lower()


@dataclass(slots=True)
class InstanceResult:
    """
    One (query, config) measurement.

    ``context_chunk_ids`` are ``doc_id:start-end``, so the exact context text is
    reconstructable from the corpus and rows stay small enough to persist a
    whole sweep. ``query_id`` exists so configs can be compared *paired*, per
    query, via ``aggregate.effect_size_paired``.
    """

    query_id: int
    config_id: str
    question_type: str
    is_null: bool
    metrics: dict[str, float | None]
    answer: str
    refusal_correct: bool | None
    n_candidates: int
    context_chunk_ids: tuple[str, ...] = ()
    context_doc_ids: tuple[str, ...] = ()
    n_context_chars: int = 0

    @property
    def n_contexts(self) -> int:
        return len(self.context_chunk_ids)


def evaluate_instance(
    query: Query,
    retrieved: list[Retrieved],
    answer: str,
    *,
    judge: Judge | None = None,
    ks: Sequence[int] = RETRIEVAL_KS,
    config_id: str = "",
    query_id: int = -1,
    n_candidates: int | None = None,
) -> InstanceResult:
    """
    Measure one query against one config.

    ``retrieved`` is the context window the generator actually saw, not the full
    candidate list: recall@k should describe what reached the prompt. With
    ``k_final`` truncation the two coincide; only a ``char_budget`` changes them,
    so ``n_candidates`` (the pre-truncation count, when the caller knows it)
    records the difference.
    """
    contexts = context_texts(retrieved)
    # Every row carries every metric key, with None where a metric is undefined
    # for this row. A uniform schema keeps persisted rows and per-slice
    # aggregation free of key-set surprises.
    metrics: dict[str, float | None] = {
        name: None for name in retrieval_metric_names(ks) + list(END_TO_END_METRIC_NAMES)
    }

    if query.is_null:
        refusal_correct: bool | None = is_refusal(answer)
    else:
        if not query.gold_facts:
            raise ValueError(
                f"non-null query {query_id} ({query.question_type}) has no gold facts"
            )
        metrics.update(evaluate_retrieval(contexts, query.gold_facts, ks=list(ks)))
        if judge is not None:
            scored = evaluate_end_to_end(query.query, contexts, answer, query.answer, judge)
            if set(scored) != set(END_TO_END_METRIC_NAMES):
                raise ValueError(
                    f"judge metric keys drifted: {sorted(scored)} != "
                    f"{sorted(END_TO_END_METRIC_NAMES)}"
                )
            metrics.update(scored)
        refusal_correct = None

    return InstanceResult(
        query_id=query_id,
        config_id=config_id,
        question_type=query.question_type,
        is_null=query.is_null,
        metrics=metrics,
        answer=answer,
        refusal_correct=refusal_correct,
        n_candidates=len(retrieved) if n_candidates is None else n_candidates,
        context_chunk_ids=tuple(item.chunk.chunk_id for item in retrieved),
        context_doc_ids=tuple(item.chunk.doc_id for item in retrieved),
        n_context_chars=sum(len(item.chunk.text) for item in retrieved),
    )


def run_config(
    queries: Sequence[Query],
    *,
    retriever: Retriever,
    generator: Generator,
    config: PipelineConfig,
    judge: Judge | None = None,
    reranker: Reranker | None = None,
    ks: Sequence[int] = RETRIEVAL_KS,
) -> list[InstanceResult]:
    """
    Retrieve, (rerank), truncate, generate and measure every query under one
    config.

    Order matters: candidates come back at ``k_recall``, the reranker reorders
    and cuts them, and only then is the context window bounded — a reranker
    cannot recover a chunk that retrieval never returned, and bounding before
    reranking would hide candidates from it.

    ``config.recency_cutoff`` is the metadata-filter axis; ``None`` disables it.
    """
    if (reranker is None) != (config.rerank is None):
        raise ValueError(
            f"reranker/config mismatch: reranker={'set' if reranker else 'None'} "
            f"but config.rerank={config.rerank!r}"
        )
    published_before = parse_published_at(config.recency_cutoff)

    rows: list[InstanceResult] = []
    for query_id, query in enumerate(queries):
        candidates = retriever.retrieve(
            query.query, k_recall=config.k_recall, published_before=published_before
        )
        selected = candidates
        if reranker is not None:
            final_k = config.k_final if config.k_final is not None else len(candidates)
            selected = reranker.rerank(query.query, candidates, final_k)
        contexts = truncate_contexts(
            selected, k_final=config.k_final, char_budget=config.char_budget
        )
        answer = generator.generate(query.query, context_texts(contexts))
        rows.append(
            evaluate_instance(
                query,
                contexts,
                answer,
                judge=judge,
                ks=ks,
                config_id=config.id,
                query_id=query_id,
                n_candidates=len(candidates),
            )
        )
    return rows


def null_summary(rows: Sequence[InstanceResult]) -> tuple[int, float | None]:
    """
    (number of null queries, refusal accuracy). Accuracy is ``None`` when the
    slice contains no null queries; refusal accuracy is reported alongside the
    retrieval metrics rather than folded into them.
    """
    values = [bool(row.refusal_correct) for row in rows if row.is_null]
    if not values:
        return 0, None
    return len(values), sum(values) / len(values)

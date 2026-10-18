"""
Phase 0 smoke test: one config, N queries, the full metric path, printing
mean +/- bootstrap 95% CI per metric.

This is the project's Phase 0 exit criterion. Two configs is not this
script's job (that is the Phase 1 sweep); this proves the path runs end to end
and reports intervals rather than point estimates.

    uv run python scripts/smoke.py --limit 20

Needs OPENAI_API_KEY (embeddings) and OPENROUTER_API_KEY (generator + judge).
``providers`` exists so tests can inject substitutes and exercise this exact
code path without network access.
"""

from __future__ import annotations

import argparse
import os
import sys

from rag_eval.aggregate import aggregate
from rag_eval.data import DATASET_REVISION, Document, Query, load_corpus, load_queries
from rag_eval.harness import (
    RETRIEVAL_KS,
    answered_rows,
    coverage_summary,
    null_summary,
    run_config,
)
from rag_eval.judges import CachedJudge, Judge, OpenRouterJudge
from rag_eval.pipeline import (
    DenseRetriever,
    Embedder,
    Generator,
    Index,
    OpenAIEmbedder,
    PipelineConfig,
    PromptGenerator,
    RecursiveChunker,
    Retriever,
    build_or_load_index,
)


def default_config() -> PipelineConfig:
    """One boring baseline cell: 1024-char chunks, dense retrieval, 8k-char window."""
    return PipelineConfig(
        chunk_size=1024,
        chunk_overlap=128,
        embedder="openai:text-embedding-3-small",
        retriever="dense",
        rerank=None,
        recency_cutoff=None,
        k_recall=50,
        k_final=None,
        char_budget=8000,
        generator="openai/gpt-4o-mini",
        generator_prompt=PromptGenerator.prompt_id(),
        judge="openai/gpt-4o-mini",
        dataset_revision=DATASET_REVISION,
    )


def build_components(
    config: PipelineConfig,
    docs: list[Document],
    cache_dir: str,
    providers: dict | None = None,
) -> tuple[Index, Retriever, Generator, Judge | None]:
    """
    Build the pipeline for one config. Anything in ``providers`` replaces the
    real component of that name ("embedder", "generator", "judge").
    """
    providers = providers or {}
    embedder: Embedder = providers.get("embedder") or OpenAIEmbedder()
    generator: Generator = providers.get("generator") or PromptGenerator(
        model=config.generator
    )
    if "judge" in providers:
        judge: Judge | None = providers["judge"]
    else:
        judge = CachedJudge(OpenRouterJudge(), cache_dir=".judge_cache")

    chunker = RecursiveChunker(
        chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap
    )
    index = build_or_load_index(docs, chunker, embedder, cache_dir=cache_dir)
    return index, DenseRetriever(index=index, embedder=embedder), generator, judge


def main(
    limit: int = 20,
    cache_dir: str = ".index_cache",
    providers: dict | None = None,
    docs: list[Document] | None = None,
    queries: list[Query] | None = None,
) -> int:
    providers = providers or {}
    missing = []
    if "embedder" not in providers:
        missing.append("OPENAI_API_KEY")
    if "generator" not in providers or "judge" not in providers:
        missing.append("OPENROUTER_API_KEY")
    missing = [name for name in missing if not os.environ.get(name)]
    if missing:
        print(f"missing env vars: {', '.join(missing)}", file=sys.stderr)
        return 2

    config = default_config()
    docs = load_corpus() if docs is None else docs
    queries = (load_queries() if queries is None else queries)[:limit]
    index, retriever, generator, judge = build_components(
        config, docs, cache_dir, providers
    )

    print(
        f"config {config.id}  docs={len(docs)}  chunks={len(index.chunks)}  "
        f"queries={len(queries)}"
    )
    rows = run_config(
        queries,
        retriever=retriever,
        generator=generator,
        config=config,
        judge=judge,
        ks=RETRIEVAL_KS,
    )

    print(f"\n{'metric':<20} {'mean':>7} {'ci_lower':>9} {'ci_upper':>9} {'n':>5}")
    for name, stats in aggregate([row.metrics for row in rows]).items():
        print(
            f"{name:<20} {stats['mean']:7.3f} {stats['ci_lower']:9.3f} "
            f"{stats['ci_upper']:9.3f} {stats['n']:5d}"
        )

    candidate = aggregate([row.candidate_metrics for row in rows if row.candidate_metrics])
    if candidate:
        print("\ncandidate retrieval (before truncation)")
        for name in ("recall@10", "ndcg@10", "map@10"):
            stats = candidate.get(name)
            if stats:
                print(
                    f"{name:<20} {stats['mean']:7.3f} {stats['ci_lower']:9.3f} "
                    f"{stats['ci_upper']:9.3f} {stats['n']:5d}"
                )

    n_nulls, accuracy = null_summary(rows)
    if n_nulls:
        print(f"\nnull queries: {n_nulls}  refusal accuracy: {accuracy:.3f}")
    else:
        print("\nnull queries: none in this slice")

    _answerable, coverage = coverage_summary(rows)
    if coverage is not None:
        answered = answered_rows(rows)
        correctness = [
            row.metrics["answer_correctness"]
            for row in answered
            if row.metrics.get("answer_correctness") is not None
        ]
        mean = sum(correctness) / len(correctness) if correctness else float("nan")
        print(
            f"answerable: {_answerable}  answered: {coverage:.1%}  "
            f"correctness|answered: {mean:.3f}"
        )

    scored = [row for row in rows if not row.is_null]
    if scored:
        mean_chars = sum(row.n_context_chars for row in scored) / len(scored)
        print(
            f"context window: {scored[0].n_candidates} candidates -> "
            f"{scored[0].n_contexts} chunks, mean ~{mean_chars:.0f} chars "
            f"(chunk size and context budget are confounded; this is disclosed)"
        )
    return 0


def cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=20, help="queries to run")
    parser.add_argument("--cache-dir", default=".index_cache", help="index cache")
    args = parser.parse_args()
    return main(limit=args.limit, cache_dir=args.cache_dir)


if __name__ == "__main__":
    raise SystemExit(cli())

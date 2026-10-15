"""
Phase 1 sweep driver: per-axis ablation with index reuse.

Each axis varies exactly one field against the baseline and every other axis is
locked, per the project's controlled-experiment hygiene. Cells that share
``(chunk_size, chunk_overlap, embedder)`` share one index build — which is why
``Index`` is separated from ``Retriever``: the query-time axes (retrieval
variant, rerank, recency filter) never re-embed the corpus.

    uv run python scripts/sweep.py --limit 100 --axes chunk_size,overlap
    uv run python scripts/sweep.py                      # full sweep

Rows are written to ``results/<config_id>.jsonl`` — one JSON object per query
plus a cell header — so a cell can be re-aggregated, sliced by question type, or
paired against another cell without re-running the pipeline. ``--resume`` skips
cells already on disk.

Needs OPENAI_API_KEY (embeddings) and OPENROUTER_API_KEY (generator + judge).
The module-level factories are the seam tests replace to exercise this file
offline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, replace
from pathlib import Path

from rag_eval.aggregate import aggregate, effect_size_paired
from rag_eval.data import Document, Query, load_corpus, load_queries
from rag_eval.harness import RETRIEVAL_KS, InstanceResult, null_summary, run_config
from rag_eval.judges import CachedJudge, Judge, OpenRouterJudge
from rag_eval.normalize import normalize
from rag_eval.pipeline import (
    BM25Retriever,
    CrossEncoderReranker,
    DenseRetriever,
    Embedder,
    Index,
    OpenAIEmbedder,
    PipelineConfig,
    PromptGenerator,
    RecursiveChunker,
    Reranker,
    Retriever,
    RRFFusion,
    SentenceTransformerEmbedder,
    build_or_load_index,
    parse_published_at,
)

OPENAI_EMBEDDER = "openai:text-embedding-3-small"
BGE_SMALL = "st:BAAI/bge-small-en-v1.5"
CROSS_ENCODER = "ce:BAAI/bge-reranker-v2-m3"

# One context budget for every cell: 8000 characters (~2k tokens). Holding the
# budget fixed is what isolates chunk size — at a fixed chunk count, a 512-char
# cell would silently be a smaller-context cell instead.
CHAR_BUDGET = 8000

BASELINE = PipelineConfig(
    chunk_size=1024,
    chunk_overlap=128,
    embedder=OPENAI_EMBEDDER,
    retriever="dense",
    rerank=None,
    recency_cutoff=None,
    k_recall=50,
    k_final=None,
    char_budget=CHAR_BUDGET,
    generator="openai/gpt-4o-mini",
    generator_prompt="sweep:v1",
)

HEADLINE_METRICS = ("recall@10", "mrr", "faithfulness", "answer_correctness")

# Deliberately not the generator's family: Post 3 measures self-preference, and a
# Phase 1 sweep that scores with the generating family pre-bakes that finding.
# Verified against OpenRouter's /api/v1/models catalogue on 2026-10-13: gemini-
# 2.5-flash at $0.30/M input is the cheap non-OpenAI judge; claude-haiku-4.5
# ($1/M) was retired from use on cost grounds.
JUDGE_MODEL = "google/gemini-2.5-flash"


# --------------------------------------------------------------------------- #
# Factories: the seam tests replace
# --------------------------------------------------------------------------- #


def make_embedder(name: str) -> Embedder:
    if name == OPENAI_EMBEDDER:
        return OpenAIEmbedder()
    if name == BGE_SMALL:
        return SentenceTransformerEmbedder(model=BGE_SMALL.removeprefix("st:"))
    raise ValueError(f"unknown embedder {name!r}")


def make_retriever(name: str, index: Index, embedder: Embedder) -> Retriever:
    if name == "dense":
        return DenseRetriever(index=index, embedder=embedder)
    if name == "bm25":
        return BM25Retriever(index=index)
    if name == "hybrid_rrf":
        return RRFFusion(
            retrievers=[DenseRetriever(index=index, embedder=embedder), BM25Retriever(index=index)]
        )
    raise ValueError(f"unknown retriever {name!r}")


def make_reranker(name: str | None) -> Reranker | None:
    if name is None:
        return None
    if name == CROSS_ENCODER:
        return CrossEncoderReranker(model=CROSS_ENCODER.removeprefix("ce:"))
    raise ValueError(f"unknown reranker {name!r}")


def make_generator() -> PromptGenerator:
    return PromptGenerator(model=BASELINE.generator)


def make_judge() -> Judge:
    return CachedJudge(OpenRouterJudge(model=JUDGE_MODEL), cache_dir=".judge_cache")


def judge_family_conflict() -> str | None:
    """Reason the judge shares the generator's family, or ``None`` if fine."""
    judge_family = JUDGE_MODEL.partition("/")[0]
    gen_family = BASELINE.generator.partition("/")[0]
    if judge_family != gen_family:
        return None
    return (
        f"judge {JUDGE_MODEL!r} and generator {BASELINE.generator!r} are both in the "
        f"{judge_family!r} family: same-family scoring is exactly the self-preference "
        f"effect Post 3 measures, so it must not be baked into the Phase 1 sweep"
    )


# --------------------------------------------------------------------------- #
# Cells
# --------------------------------------------------------------------------- #


def cells(recency_cutoff: str) -> dict[str, list[tuple[str, PipelineConfig]]]:
    """
    ``axis -> [(variant label, config)]``, each config differing from the
    baseline in exactly one field.
    """
    return {
        "chunk_size": [
            ("512", replace(BASELINE, chunk_size=512)),
            ("2048", replace(BASELINE, chunk_size=2048)),
        ],
        "overlap": [
            ("0", replace(BASELINE, chunk_overlap=0)),
            ("64", replace(BASELINE, chunk_overlap=64)),
        ],
        "embedder": [("bge-small-en-v1.5", replace(BASELINE, embedder=BGE_SMALL))],
        "retriever": [
            ("bm25", replace(BASELINE, retriever="bm25")),
            ("hybrid_rrf", replace(BASELINE, retriever="hybrid_rrf")),
        ],
        "rerank": [("bge-reranker-v2-m3", replace(BASELINE, rerank=CROSS_ENCODER))],
        "recency_filter": [
            ("before-median", replace(BASELINE, recency_cutoff=recency_cutoff))
        ],
    }


def index_key(config: PipelineConfig) -> tuple[int, int, str]:
    return (config.chunk_size, config.chunk_overlap, config.embedder)


def median_published_at(docs: list[Document]) -> str:
    """
    The recency-filter cutoff, as tz-aware ISO-8601 UTC.

    The dataset supplies no per-query date, so the sweep defines one: the corpus
    median publication date. Normalizing here keeps the persisted config
    self-describing — the loader hands back naive strings, and a config that
    says ``2023-11-06 02:56:22`` leaves the timezone assumption unrecorded.
    """
    dates = sorted(
        date
        for date in (parse_published_at(doc.published_at) for doc in docs)
        if date is not None
    )
    if not dates:
        raise ValueError("corpus has no publication dates to define a cutoff")
    return dates[len(dates) // 2].isoformat()


def chunk_ceiling(
    docs: list[Document], chunker: RecursiveChunker, queries: list[Query]
) -> tuple[int, float]:
    """
    Fraction of gold facts contained in some chunk of their own source document.

    This is the maximum recall@k any retriever can reach under this chunking:
    facts longer than a chunk, or sitting across a packing boundary, are
    unreachable no matter how good retrieval is. Reporting it next to recall@k
    is what stops unreachable facts from reading as a retrieval failure.
    """
    by_title: dict[str, list[int]] = {}
    for i, doc in enumerate(docs):
        by_title.setdefault(normalize(doc.title), []).append(i)
    chunks_of = {
        i: [normalize(c.text) for c in chunker.chunk(doc)] for i, doc in enumerate(docs)
    }

    total = found = 0
    for query in queries:
        for evidence in query.evidence_list:
            total += 1
            fact = normalize(evidence.fact)
            keys = by_title.get(normalize(evidence.title or ""), [])
            if any(fact in chunk for i in keys for chunk in chunks_of[i]):
                found += 1
    return total, (found / total if total else float("nan"))


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def write_cell(
    path: Path,
    config: PipelineConfig,
    rows: list[InstanceResult],
    ceiling: float,
    axis: str,
    variant: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write(
            json.dumps(
                {
                    "kind": "cell",
                    "config_id": config.id,
                    "config": asdict(config),
                    "axis": axis,
                    "variant": variant,
                    "ceiling": ceiling,
                    "n_rows": len(rows),
                }
            )
            + "\n"
        )
        for row in rows:
            handle.write(json.dumps({"kind": "row", **asdict(row)}) + "\n")


def read_cell(path: Path) -> tuple[dict, float, list[InstanceResult]]:
    header: dict = {}
    rows: list[InstanceResult] = []
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            if record["kind"] == "cell":
                header = record
                continue
            record.pop("kind")
            record["context_chunk_ids"] = tuple(record["context_chunk_ids"])
            record["context_doc_ids"] = tuple(record["context_doc_ids"])
            rows.append(InstanceResult(**record))
    return header, header.get("ceiling", float("nan")), rows


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def result_cell(rows: list[InstanceResult], name: str) -> str:
    stats = aggregate([row.metrics for row in rows]).get(name)
    if stats is None:
        return "  n/a  "
    return f"{stats['mean']:.3f} [{stats['ci_lower']:.2f},{stats['ci_upper']:.2f}]"


def print_table(entries: list[tuple[str, str, PipelineConfig, float, list[InstanceResult]]]) -> None:
    print(
        f"{'axis':<16}{'variant':<21}{'ceiling':>8}  "
        f"{'recall@10':<17}{'mrr':<17}{'faithful':<17}{'correct':<17}{'refuse':>7}"
    )
    for axis, variant, _config, ceiling, rows in entries:
        _n, refusal = null_summary(rows)
        cells_txt = "  ".join(result_cell(rows, m) for m in HEADLINE_METRICS)
        refuse = "n/a" if refusal is None else f"{refusal:.3f}"
        print(
            f"{axis:<16}{variant:<21}{ceiling:>8.4f}  {cells_txt}  {refuse:>7}"
        )


def print_deltas(
    baseline_rows: list[InstanceResult],
    entries: list[tuple[str, str, PipelineConfig, float, list[InstanceResult]]],
) -> None:
    """Paired effect size vs baseline for every non-baseline cell."""
    print("\npaired delta vs baseline (95% CI, same queries)")
    print(f"{'axis':<16}{'variant':<21}{'metric':<20}{'delta':>8}{'ci_lower':>10}{'ci_upper':>10}{'sig':>6}")
    for axis, variant, _config, _ceiling, rows in entries:
        for metric in HEADLINE_METRICS:
            a = {r.query_id: r.metrics.get(metric) for r in rows}
            b = {r.query_id: r.metrics.get(metric) for r in baseline_rows}
            common = sorted(
                i for i in set(a) & set(b) if a[i] is not None and b[i] is not None
            )
            if not common:
                continue
            stats = effect_size_paired(
                [a[i] for i in common], [b[i] for i in common], seed=42
            )
            print(
                f"{axis:<16}{variant:<21}{metric:<20}{stats['delta']:>8.3f}"
                f"{stats['ci_lower']:>10.3f}{stats['ci_upper']:>10.3f}"
                f"{'yes' if stats['significant'] else 'no':>6}"
            )


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def main(
    limit: int | None = None,
    axes: list[str] | None = None,
    out_dir: str = "results",
    cache_dir: str = ".index_cache",
    resume: bool = False,
    docs: list[Document] | None = None,
    queries: list[Query] | None = None,
) -> int:
    if not os.environ.get("OPENAI_API_KEY") or not os.environ.get("OPENROUTER_API_KEY"):
        print(
            "missing env vars: OPENAI_API_KEY and/or OPENROUTER_API_KEY",
            file=sys.stderr,
        )
        return 2
    conflict = judge_family_conflict()
    if conflict is not None:
        print(conflict, file=sys.stderr)
        return 2

    docs = load_corpus() if docs is None else docs
    queries = load_queries() if queries is None else queries
    if limit is not None:
        queries = queries[:limit]

    plan = cells(median_published_at(docs))
    baseline_axis = [("baseline", "dense/1024/128", BASELINE)]
    selected = axes or list(plan)
    unknown = [a for a in selected if a not in plan]
    if unknown:
        print(f"unknown axes: {unknown}; known: {sorted(plan)}", file=sys.stderr)
        return 2

    print(
        f"corpus={len(docs)} documents  queries={len(queries)}  "
        f"cells={1 + sum(len(plan[a]) for a in selected)}  "
        f"char_budget={BASELINE.char_budget}  k_recall={BASELINE.k_recall}"
    )
    out = Path(out_dir)

    indexes: dict[tuple[int, int, str], Index] = {}
    embedders: dict[tuple[int, int, str], Embedder] = {}
    bm25: dict[tuple[int, int, str], BM25Retriever] = {}
    reranker: Reranker | None = None
    generator = make_generator()
    judge = make_judge()
    entries: list[tuple[str, str, PipelineConfig, float, list[InstanceResult]]] = []

    todo = baseline_axis + [
        (axis, variant, config)
        for axis in selected
        for variant, config in plan[axis]
    ]
    for axis, variant, config in todo:
        path = out / f"{config.id}.jsonl"
        if resume and path.exists():
            _header, ceiling, rows = read_cell(path)
            print(f"[cached] {axis}/{variant} {config.id} rows={len(rows)}")
            entries.append((axis, variant, config, ceiling, rows))
            continue

        key = index_key(config)
        if key not in indexes:
            # One embedder per index key, reused by every cell that shares the
            # index: rebuilding it per cell would reload a transformer model.
            embedders[key] = make_embedder(config.embedder)
            chunker = RecursiveChunker(
                chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap
            )
            indexes[key] = build_or_load_index(
                docs, chunker, embedders[key], cache_dir=cache_dir
            )
            print(f"[index] {key} chunks={len(indexes[key].chunks)}")
        index = indexes[key]

        # The index holds vectors even for lexical cells; only bm25 skips them.
        if config.retriever in ("bm25", "hybrid_rrf") and key not in bm25:
            bm25[key] = BM25Retriever(index=index)
        retriever = (
            bm25[key]
            if config.retriever == "bm25"
            else make_retriever(config.retriever, index, embedders[key])
        )
        if reranker is None and config.rerank is not None:
            reranker = make_reranker(config.rerank)

        rows = run_config(
            queries,
            retriever=retriever,
            generator=generator,
            config=config,
            judge=judge,
            reranker=reranker if config.rerank is not None else None,
            ks=RETRIEVAL_KS,
        )
        ceiling = chunk_ceiling(
            docs,
            RecursiveChunker(
                chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap
            ),
            queries,
        )[1]
        write_cell(path, config, rows, ceiling, axis, variant)
        print(f"[cell]  {axis}/{variant} {config.id} rows={len(rows)} -> {path}")
        entries.append((axis, variant, config, ceiling, rows))

    print()
    print_table(entries)
    baseline_rows = entries[0][4]
    print_deltas(baseline_rows, entries[1:])
    return 0


def cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="queries to run")
    parser.add_argument("--axes", default=None, help="comma-separated axis names")
    parser.add_argument("--out", default="results", help="results directory")
    parser.add_argument("--cache-dir", default=".index_cache", help="index cache")
    parser.add_argument("--resume", action="store_true", help="skip cells already written")
    args = parser.parse_args()
    return main(
        limit=args.limit,
        axes=args.axes.split(",") if args.axes else None,
        out_dir=args.out,
        cache_dir=args.cache_dir,
        resume=args.resume,
    )


if __name__ == "__main__":
    raise SystemExit(cli())

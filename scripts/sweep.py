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
paired against another cell without re-running the pipeline. Rows are flushed
as they are produced; ``--resume`` continues a partially written cell from its
row count.

Reporting separates the things the first version conflated:

* **candidate** retrieval metrics describe the retriever (pre-truncation);
* **window** retrieval metrics describe what reached the generator;
* **coverage** is the answered rate on answerable queries;
* **correctness|answered** is accuracy among the queries the system chose to
  answer. The baseline refuses 71% of answerable queries, so a change in
  ``answer_correctness`` is usually a change in coverage, not accuracy.

Needs OPENAI_API_KEY (embeddings) and OPENROUTER_API_KEY (generator + judge).
The module-level factories are the seam tests replace to exercise this file
offline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from rag_eval.aggregate import aggregate, benjamini_hochberg, effect_size_paired
from rag_eval.data import (
    DATASET_ID,
    DATASET_REVISION,
    Document,
    Query,
    load_corpus,
    load_queries,
)
from rag_eval.harness import (
    RETRIEVAL_KS,
    InstanceResult,
    answered_rows,
    coverage_summary,
    is_refusal,
    null_summary,
    run_config,
)
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
    # The real template hash, not a hand-typed label: changing GENERATOR_TEMPLATE
    # must change the config id, or --resume silently reuses a stale cell.
    generator_prompt=PromptGenerator.prompt_id(),
    judge="google/gemini-2.5-flash",
    dataset_revision=DATASET_REVISION,
)

# Retrieval quality is measured on candidates; the window view is reported
# separately because 512-char chunks fit ~19 contexts and 2048-char chunks fit
# ~4, so a window-based recall@10 is not comparable across the chunk-size axis.
CANDIDATE_METRICS = ("recall@10", "ndcg@10", "map@10", "mrr")
WINDOW_METRICS = ("recall@10",)

# Deliberately not the generator's family: Post 3 measures self-preference, and a
# Phase 1 sweep that scores with the generating family pre-bakes that finding.
# Verified against OpenRouter's /api/v1/models catalogue on 2026-10-13: gemini-
# 2.5-flash at $0.30/M input is the cheap non-OpenAI judge; claude-haiku-4.5
# ($1/M) was retired from use on cost grounds.
JUDGE_MODEL = BASELINE.judge


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


def make_generator(config: PipelineConfig | None = None) -> PromptGenerator:
    model = BASELINE.generator if config is None else (config.generator or BASELINE.generator)
    return PromptGenerator(model=model)


def make_judge(config: PipelineConfig | None = None) -> Judge:
    model = JUDGE_MODEL if config is None else (config.judge or JUDGE_MODEL)
    return CachedJudge(OpenRouterJudge(model=model), cache_dir=".judge_cache")


def judge_family_conflict(config: PipelineConfig = BASELINE) -> str | None:
    """Reason the judge shares the generator's family, or ``None`` if fine."""
    judge_family = JUDGE_MODEL.partition("/")[0]
    gen_family = config.generator.partition("/")[0]
    if judge_family != gen_family:
        return None
    return (
        f"judge {JUDGE_MODEL!r} and generator {config.generator!r} are both in the "
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
    self-describing.
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
    is what stops unreachable facts from reading as retrieval failures.

    It is computed on the unfiltered corpus. Under ``recency_cutoff`` the true
    ceiling is lower; compare a filtered cell against its own recall history,
    not against this column.
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


def query_set_sha(queries: list[Query]) -> str:
    h = hashlib.sha256()
    for query in queries:
        h.update(query.query.encode())
        h.update(b"\x00")
    return h.hexdigest()[:16]


def provenance(config: PipelineConfig, queries: list[Query]) -> dict:
    """Everything needed to say what produced a cell, recorded with it."""
    return {
        "dataset": {"id": DATASET_ID, "revision": DATASET_REVISION},
        "judge": {
            "model": config.judge,
            "temperature": 0.0,
            "max_output_tokens": 32,
        },
        "generator": {"model": config.generator, "prompt": config.generator_prompt},
        "query_count": len(queries),
        "query_set_sha": query_set_sha(queries),
        "metric_version": 2,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }


def write_header(
    path: Path,
    config: PipelineConfig,
    ceiling: float,
    axis: str,
    variant: str,
    n_rows: int,
    provenance_data: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "kind": "cell",
                "config_id": config.id,
                "config": asdict(config),
                "axis": axis,
                "variant": variant,
                "ceiling": ceiling,
                "n_rows": n_rows,
                "provenance": provenance_data or {},
            }
        )
        + "\n"
    )


def append_row(path: Path, row: InstanceResult) -> None:
    # Flush per row: a 23-hour cell must not be lost to the last query.
    with path.open("a") as handle:
        handle.write(json.dumps({"kind": "row", **asdict(row)}) + "\n")
        handle.flush()


def write_cell(
    path: Path,
    config: PipelineConfig,
    rows: list[InstanceResult],
    ceiling: float,
    axis: str,
    variant: str,
    provenance_data: dict | None = None,
) -> None:
    write_header(path, config, ceiling, axis, variant, len(rows), provenance_data)
    for row in rows:
        append_row(path, row)


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
            record["candidate_chunk_ids"] = tuple(record.get("candidate_chunk_ids", ()))
            record["candidate_doc_ids"] = tuple(record.get("candidate_doc_ids", ()))
            row = InstanceResult(**record)
            if "refused" not in record:
                # Legacy rows predate the refusal column; derive it rather than
                # letting the default (False) report 100% coverage.
                row.refused = is_refusal(row.answer)
            rows.append(row)
    return header, header.get("ceiling", float("nan")), rows


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def agg(rows: list[InstanceResult], source: str) -> dict:
    dicts = [getattr(row, source) for row in rows if getattr(row, source)]
    return aggregate(dicts)


def result_cell(rows: list[InstanceResult], name: str, source: str = "metrics") -> str:
    stats = agg(rows, source).get(name)
    if stats is None:
        return "  n/a  "
    return f"{stats['mean']:.3f} [{stats['ci_lower']:.2f},{stats['ci_upper']:.2f}]"


def answered_correctness(rows: list[InstanceResult]) -> float | None:
    answered = answered_rows(rows)
    if not answered:
        return None
    values = [
        row.metrics["answer_correctness"]
        for row in answered
        if row.metrics.get("answer_correctness") is not None
    ]
    return sum(values) / len(values) if values else None


def print_table(entries) -> None:
    print(
        f"{'axis':<16}{'variant':<21}{'ceiling':>8}  "
        f"{'cand rec@10':<17}{'cand ndcg':<17}{'cand map':<17}"
        f"{'win rec@10':<17}{'answer':>7}{'correct':>9}{'corr|ans':>10}"
    )
    for axis, variant, _config, ceiling, rows in entries:
        _answerable, answered = coverage_summary(rows)
        answer_txt = "n/a" if answered is None else f"{answered:.3f}"
        corr = result_cell(rows, "answer_correctness")
        ca = answered_correctness(rows)
        ca_txt = "n/a" if ca is None else f"{ca:.3f}"
        print(
            f"{axis:<16}{variant:<21}{ceiling:>8.4f}  "
            f"{result_cell(rows, 'recall@10', 'candidate_metrics'):<17}"
            f"{result_cell(rows, 'ndcg@10', 'candidate_metrics'):<17}"
            f"{result_cell(rows, 'map@10', 'candidate_metrics'):<17}"
            f"{result_cell(rows, 'recall@10'):<17}"
            f"{answer_txt:>7}{corr:>9}{ca_txt:>10}"
        )
    if any(null_summary(rows)[0] for _, _, _, _, rows in entries):
        print("null-query refusal accuracy is reported separately by scripts/reanalyze.py")


def print_type_table(rows: list[InstanceResult]) -> None:
    types = sorted({row.question_type for row in rows if not row.is_null})
    if not types:
        return
    print("\nby question type (baseline)")
    print(f"{'type':<20}{'n':>6}{'answered':>10}{'correct':>9}{'corr|ans':>10}{'cand rec@10':>13}")
    for question_type in types:
        subset = [row for row in rows if row.question_type == question_type and not row.is_null]
        _n, answered = coverage_summary(subset)
        correct = sum(row.metrics["answer_correctness"] for row in subset) / len(subset)
        ca = answered_correctness(subset)
        rec = agg(subset, "candidate_metrics").get("recall@10")
        print(
            f"{question_type:<20}{len(subset):>6}{answered:>10.1%}{correct:>9.3f}"
            f"{ca:>10.3f}{(rec['mean'] if rec else float('nan')):>13.3f}"
        )


def print_deltas(baseline_rows: list[InstanceResult], entries) -> None:
    """Paired effect size vs baseline for every non-baseline cell."""
    records = []
    for axis, variant, _config, _ceiling, rows in entries:
        for metric in CANDIDATE_METRICS + ("faithfulness", "answer_correctness"):
            source = "candidate_metrics" if metric in CANDIDATE_METRICS else "metrics"
            a = {r.query_id: getattr(r, source).get(metric) if getattr(r, source) else None for r in rows}
            b = {
                r.query_id: getattr(r, source).get(metric) if getattr(r, source) else None
                for r in baseline_rows
            }
            common = sorted(
                i for i in set(a) & set(b) if a[i] is not None and b[i] is not None
            )
            if not common:
                continue
            stats = effect_size_paired([a[i] for i in common], [b[i] for i in common], seed=42)
            records.append((axis, variant, metric, stats, len(common)))

    adjusted = benjamini_hochberg([record[3]["p_value"] for record in records])
    print("\npaired delta vs baseline (95% CI, same queries; * = BH-adjusted p < 0.05)")
    print(
        f"{'axis':<16}{'variant':<21}{'metric':<20}{'delta':>8}"
        f"{'ci_lower':>10}{'ci_upper':>10}{'p':>8}{'sig':>6}"
    )
    for (axis, variant, metric, stats, n), p_adj in zip(records, adjusted):
        mark = "*" if p_adj < 0.05 else " "
        print(
            f"{axis:<16}{variant:<21}{metric:<20}{stats['delta']:>8.3f}"
            f"{stats['ci_lower']:>10.3f}{stats['ci_upper']:>10.3f}"
            f"{stats['p_value']:>8.4f}{mark:>6}"
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
    entries: list[tuple[str, str, PipelineConfig, float, list[InstanceResult]]] = []

    todo = baseline_axis + [
        (axis, variant, config)
        for axis in selected
        for variant, config in plan[axis]
    ]
    for axis, variant, config in todo:
        path = out / f"{config.id}.jsonl"
        if resume and path.exists():
            header, ceiling, rows = read_cell(path)
            if header.get("config_id") != config.id:
                print(f"resume mismatch at {path}", file=sys.stderr)
                return 2
            if len(rows) >= len(queries):
                print(f"[cached] {axis}/{variant} {config.id} rows={len(rows)}")
                entries.append((axis, variant, config, ceiling, rows))
                continue
            print(f"[resume] {axis}/{variant} {config.id} from row {len(rows)}")
        else:
            ceiling = chunk_ceiling(
                docs,
                RecursiveChunker(
                    chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap
                ),
                queries,
            )[1]
            write_header(
                path,
                config,
                ceiling,
                axis,
                variant,
                len(queries),
                provenance(config, queries),
            )
            rows = []

        key = index_key(config)
        if key not in indexes:
            embedders[key] = make_embedder(config.embedder)
            chunker = RecursiveChunker(
                chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap
            )
            indexes[key] = build_or_load_index(
                docs, chunker, embedders[key], cache_dir=cache_dir
            )
            print(f"[index] {key} chunks={len(indexes[key].chunks)}")
        index = indexes[key]

        if config.retriever in ("bm25", "hybrid_rrf") and key not in bm25:
            bm25[key] = BM25Retriever(index=index)
        retriever = (
            bm25[key]
            if config.retriever == "bm25"
            else make_retriever(config.retriever, index, embedders[key])
        )
        if reranker is None and config.rerank is not None:
            reranker = make_reranker(config.rerank)

        start = len(rows)
        # The generator and judge fields are part of the cell identity, so they
        # must actually drive the run rather than being decorative labels.
        new_rows = run_config(
            queries[start:],
            retriever=retriever,
            generator=make_generator(config),
            config=config,
            judge=make_judge(config),
            reranker=reranker if config.rerank is not None else None,
            ks=RETRIEVAL_KS,
            start_query_id=start,
            on_row=lambda row, _path=path: append_row(_path, row),
        )
        rows.extend(new_rows)
        print(f"[cell]  {axis}/{variant} {config.id} rows={len(rows)} -> {path}")
        entries.append((axis, variant, config, ceiling, rows))

    print()
    print_table(entries)
    print_type_table(entries[0][4])
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

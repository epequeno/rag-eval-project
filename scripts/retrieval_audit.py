"""
Candidate-level retrieval audit for the stored Phase 1 cells.

The original cells stored only the prompt window, so `recall@10` could not be
recomputed after truncation: 512-char chunks fit ~19 contexts and 2048-char
chunks fit ~4. This script re-runs retrieval (dense / BM25 / hybrid, no rerank)
with the same cached indexes and reports the metrics over the full candidate
list, plus a check that the stored window is a prefix of that list.

Cost: query embeddings only — 2,556 short queries per embedder, cached under
results/query_vectors/ after the first run. No generation and no judge calls.

    HF_HUB_OFFLINE=1 uv run python scripts/retrieval_audit.py
    HF_HUB_OFFLINE=1 uv run python scripts/retrieval_audit.py --limit 100

Writes results/candidate_metrics/<config_id>.json (per query) and
reports/retrieval_audit.md.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path

import numpy as np
from sweep import index_key, make_embedder, read_cell

from rag_eval.aggregate import aggregate
from rag_eval.data import Query, load_corpus, load_queries
from rag_eval.pipeline import (
    BM25Retriever,
    DenseRetriever,
    Embedder,
    Index,
    PipelineConfig,
    RecursiveChunker,
    RRFFusion,
    build_or_load_index,
    parse_published_at,
)
from rag_eval.retrieval_metrics import evaluate_retrieval

KS = [1, 4, 10]
METRICS = ("recall@1", "recall@4", "recall@10", "hits@4", "hits@10", "ndcg@10", "map@10", "mrr")


class BatchEmbedder:
    """Serves precomputed query vectors; the corpus side is the cached index."""

    def __init__(self, id: str, vectors: np.ndarray, texts: list[str]):
        self.id = id
        self._map = {text: row for text, row in zip(texts, vectors)}

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray([self._map[text] for text in texts], dtype=np.float32)


def query_vectors(
    embedder: Embedder,
    queries: list[Query],
    cache_dir: Path,
    per_query: bool = True,
) -> np.ndarray:
    """
    Embed all queries once per embedder, caching the result on disk.

    ``per_query`` matters for exactness: OpenAI's embeddings are deterministic
    per call but not bit-identical across batch compositions (observed
    max-abs difference 1.3e-4 between a 5-query batch and single calls). That is
    enough to reorder near-tied chunks and break a window-prefix audit, so the
    default reproduces the original one-input-per-call shape. ``--batch`` is
    faster but approximate.
    """
    texts = [query.query for query in queries]
    digest = hashlib.sha256(
        embedder.id.encode()
        + (b"\x00per-query" if per_query else b"\x00batch")
        + b"\x00"
        + b"\x00".join(t.encode() for t in texts)
    ).hexdigest()[:16]
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{digest}.npy"
    if path.exists():
        return np.load(path)
    if per_query:
        vectors = np.asarray([embedder.embed([text])[0] for text in texts], dtype=np.float32)
    else:
        vectors = np.asarray(embedder.embed(texts), dtype=np.float32)
    np.save(path, vectors)
    return vectors


def main(
    results_dir: str = "results",
    out_dir: str = "results/candidate_metrics",
    report: str = "reports/retrieval_audit.md",
    cache_dir: str = ".index_cache",
    limit: int | None = None,
    batch_embeddings: bool = False,
) -> int:
    queries = load_queries()
    if limit is not None:
        queries = queries[:limit]
    docs = load_corpus()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    vector_dir = Path("results/query_vectors")

    indexes: dict[tuple, Index] = {}
    embedders: dict[str, Embedder] = {}
    bm25: dict[tuple, BM25Retriever] = {}
    rows_out = []

    for path in sorted(glob.glob(f"{results_dir}/*.jsonl")):
        header, _ceiling, stored = read_cell(Path(path))
        config = PipelineConfig(**header["config"])
        key = index_key(config)

        if config.retriever != "bm25" and key not in embedders:
            base = make_embedder(config.embedder)
            vectors = query_vectors(base, queries, vector_dir, per_query=not batch_embeddings)
            embedders[key] = BatchEmbedder(base.id, vectors, [q.query for q in queries])
        if key not in indexes:
            chunker = RecursiveChunker(
                chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap
            )
            indexes[key] = build_or_load_index(
                docs, chunker, make_embedder(config.embedder), cache_dir=cache_dir
            )

        index = indexes[key]
        if config.retriever in ("bm25", "hybrid_rrf") and key not in bm25:
            bm25[key] = BM25Retriever(index=index)
        if config.retriever == "bm25":
            retriever = bm25[key]
        elif config.retriever == "hybrid_rrf":
            retriever = RRFFusion(
                [DenseRetriever(index=index, embedder=embedders[key]), bm25[key]]
            )
        else:
            retriever = DenseRetriever(index=index, embedder=embedders[key])

        published_before = parse_published_at(config.recency_cutoff)
        stored_by_qid = {row.query_id: row for row in stored}
        per_query = []
        window_prefix_hits = 0
        window_set_hits = 0
        window_prefix_total = 0
        for query_id, query in enumerate(queries):
            # Retrieval metrics are undefined for null queries (no gold facts);
            # including them would dilute every mean by ~12%.
            if query.is_null:
                continue
            candidates = retriever.retrieve(
                query.query, k_recall=config.k_recall, published_before=published_before
            )
            metrics = evaluate_retrieval(
                [item.chunk.text for item in candidates], query.gold_facts, ks=KS
            )
            per_query.append({"query_id": query_id, **metrics})
            row = stored_by_qid.get(query_id)
            if row is not None and not row.is_null:
                candidate_ids = [item.chunk.chunk_id for item in candidates]
                window_prefix_total += 1
                contained = set(row.context_chunk_ids) <= set(candidate_ids)
                if contained:
                    window_set_hits += 1
                if config.rerank is not None:
                    # A reranked window is a reordering of a candidate subset.
                    reproduced = contained
                else:
                    reproduced = list(row.context_chunk_ids) == candidate_ids[: len(row.context_chunk_ids)]
                if reproduced:
                    window_prefix_hits += 1

        payload = {
            "config_id": header["config_id"],
            "axis": header["axis"],
            "variant": header["variant"],
            "n": len(per_query),
            "metrics": {
                name: aggregate([{name: row[name]} for row in per_query])[name]["mean"]
                for name in METRICS
            },
            "window_prefix_rate": (
                window_prefix_hits / window_prefix_total if window_prefix_total else None
            ),
            "window_set_rate": (
                window_set_hits / window_prefix_total if window_prefix_total else None
            ),
            "per_query": per_query,
        }
        (out / f"{header['config_id']}.json").write_text(json.dumps(payload))
        rows_out.append(payload)
        print(
            f"[audit] {header['axis']}/{header['variant']} "
            f"recall@10={payload['metrics']['recall@10']:.3f} "
            f"window-prefix={payload['window_prefix_rate']:.1%} "
            f"window-set={payload['window_set_rate']:.1%}"
        )

    rows_out.sort(key=lambda row: (row["axis"], row["variant"]))
    lines = [
        "# Candidate-level retrieval audit",
        "",
        "Metrics over the top-50 candidates (pre-rerank, pre-truncation), recomputed",
        "from the cached indexes. `window prefix` is the share of stored windows that",
        "equal the candidate prefix; `window set` is the share whose chunks are all in",
        "the candidate list. A set match with a prefix mismatch is an adjacent swap of",
        "near-tied scores (OpenAI embeddings are deterministic per call but differ by",
        "~1e-4 across batch compositions), not a different retrieval result.",
        "",
        "| cell | recall@10 | hits@10 | nDCG@10 | MAP@10 | MRR | window prefix | window set |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows_out:
        metrics = row["metrics"]
        lines.append(
            f"| {row['axis']}/{row['variant']} | {metrics['recall@10']:.3f} | "
            f"{metrics['hits@10']:.3f} | {metrics['ndcg@10']:.3f} | "
            f"{metrics['map@10']:.3f} | {metrics['mrr']:.3f} | "
            f"{row['window_prefix_rate']:.1%} | {row['window_set_rate']:.1%} |"
        )
    lines += [
        "",
        "Chunk-level metrics. The dataset's official script ranks documents and reports",
        "Hits@4/Hits@10/MAP@10/MRR@10; these values are not directly comparable to its",
        "table. What they do provide is a retrieval-quality measure that does not move",
        "when the context budget fits a different number of chunks.",
        "",
    ]
    Path(report).parent.mkdir(parents=True, exist_ok=True)
    Path(report).write_text("\n".join(lines) + "\n")
    print(f"\nreport -> {report}")
    return 0


def cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="results/candidate_metrics")
    parser.add_argument("--report", default="reports/retrieval_audit.md")
    parser.add_argument("--cache-dir", default=".index_cache")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--batch",
        action="store_true",
        help="batch query embeddings (faster, not bit-identical to the original runs)",
    )
    args = parser.parse_args()
    return main(
        args.results, args.out, args.report, args.cache_dir, args.limit, args.batch
    )


if __name__ == "__main__":
    raise SystemExit(cli())

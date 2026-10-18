"""
Chunk-ceiling report: the maximum recall@k any retriever can reach under a chunking.

Offline and keyless. This needs only the corpus, the chunker and the gold facts —
no embeddings, no API keys, no ``results/``. The ceiling is the share of gold facts
that appear inside some chunk of their own source document; a fact split across a
packing boundary, or longer than a chunk, is unreachable no matter how good
retrieval is. Reporting it next to ``recall@k`` is what stops unreachable facts
from reading as retrieval failures.

Writes ``reports/chunk_ceiling.md``.

    HF_HUB_OFFLINE=1 uv run python scripts/chunk_ceiling.py

Chunk sizes match the variants ``scripts/sweep.py`` sweeps on the chunk-size axis,
and the computation calls the same ``chunk_ceiling`` helper the sweep stores in each
cell header — so this report and the headers in ``results/`` cannot disagree.
"""

from __future__ import annotations

import json
from pathlib import Path

from sweep import BASELINE, chunk_ceiling

from rag_eval.data import load_corpus, load_queries
from rag_eval.pipeline import RecursiveChunker

REPO = Path(__file__).resolve().parents[1]
CHUNK_SIZES = (512, 1024, 2048)
OUT = REPO / "reports" / "chunk_ceiling.md"


def percentile(values: list[int], q: float) -> float:
    """Linear-interpolated percentile; no numpy dependency for one number."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = q * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def main() -> int:
    docs = load_corpus()
    queries = load_queries()

    facts = [ev.fact for q in queries for ev in q.evidence_list]
    lengths = [len(f) for f in facts]

    rows = []
    for size in CHUNK_SIZES:
        chunker = RecursiveChunker(
            chunk_size=size, chunk_overlap=BASELINE.chunk_overlap
        )
        total, ceiling = chunk_ceiling(docs, chunker, queries)
        n_chunks = sum(len(chunker.chunk(doc)) for doc in docs)
        rows.append((size, n_chunks, total, ceiling))

    lines = [
        "# Chunk ceiling",
        "",
        "The maximum `recall@k` any retriever can reach under a given chunking: the share of",
        "gold evidence facts contained in some chunk of their own source document. Computed",
        "over the unfiltered corpus, so a `recency_filter` cell has a lower true ceiling —",
        "compare such a cell against its own recall history, not against this table.",
        "",
        f"Documents: {len(docs)} · queries: {len(queries)} · gold facts: {len(facts)}",
        "",
        "| chunk size | overlap | chunks | facts reachable | ceiling |",
        "|---:|---:|---:|---:|---:|",
    ]
    for size, n_chunks, total, ceiling in rows:
        lines.append(
            f"| {size} | {BASELINE.chunk_overlap} | {n_chunks} | "
            f"{round(ceiling * total)}/{total} | {ceiling:.5f} |"
        )

    unreachable = round((1 - rows[0][3]) * rows[0][2])
    lines += [
        "",
        "## Gold-fact lengths (characters)",
        "",
        "| p10 | p50 | p90 | max |",
        "|---:|---:|---:|---:|",
        "| "
        + " | ".join(
            str(round(percentile(lengths, q))) for q in (0.10, 0.50, 0.90)
        )
        + f" | {max(lengths)} |",
        "",
        "Chunk-boundary loss is the difference between the two: a fact longer than the chunk",
        f"size cannot fit. At the smallest size here that is {unreachable} fact(s) out of",
        f"{rows[0][2]} — which is why Phase 2 engineers the split explicitly rather than",
        "waiting for the natural grid to produce it.",
        "",
        "Regenerate with `HF_HUB_OFFLINE=1 uv run python scripts/chunk_ceiling.py`.",
        "",
    ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"wrote": str(OUT), "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

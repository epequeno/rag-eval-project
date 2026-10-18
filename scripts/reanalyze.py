"""
Offline reanalysis of the stored Phase 1 cells.

The first sweep reported window-based retrieval metrics and a single
``answer_correctness`` mean. Both hid the dominant behaviour: the baseline
refuses 71% of answerable queries, and a fixed 8,000-character window fits
different numbers of chunks at different chunk sizes, so ``recall@10`` meant
different things in different cells.

This script reconstructs every window's text from the cached indexes and
recomputes the corrected metrics — no API calls. It writes reports/phase1.md.

    HF_HUB_OFFLINE=1 uv run python scripts/reanalyze.py

Candidate-level metrics are included when a cell stores them (new runs do);
the old cells need scripts/retrieval_audit.py.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import statistics
from pathlib import Path

from sweep import read_cell

from rag_eval.aggregate import aggregate
from rag_eval.data import load_queries
from rag_eval.harness import coverage_summary, is_refusal
from rag_eval.pipeline import Index
from rag_eval.retrieval_metrics import evaluate_retrieval

KS = [1, 4, 10]
CANDIDATE = ("recall@10", "ndcg@10", "map@10", "mrr")


def chunk_text_map(cache_dir: str = ".index_cache") -> dict[str, str]:
    """Every chunk id from every cached index -> text. Ids are content-addressed."""
    texts: dict[str, str] = {}
    for directory in glob.glob(f"{cache_dir}/*/"):
        index = Index.load(directory)
        for chunk in index.chunks:
            texts[chunk.chunk_id] = chunk.text
    return texts


def mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else float("nan")


def load_candidate_metrics(directory: str = "results/candidate_metrics") -> dict:
    """Per-cell candidate metrics written by scripts/retrieval_audit.py."""
    mapping = {}
    for path in glob.glob(f"{directory}/*.json"):
        payload = json.loads(Path(path).read_text())
        mapping[payload["config_id"]] = payload["metrics"]
    return mapping


def analyze_cell(header, rows, queries, texts, audited: dict | None = None):
    non_null = [row for row in rows if not row.is_null]
    window = []
    answered_correctness = []
    candidate = {name: [] for name in CANDIDATE}
    for row in non_null:
        facts = queries[row.query_id].gold_facts
        context = [texts[chunk_id] for chunk_id in row.context_chunk_ids]
        metrics = evaluate_retrieval(context, facts, ks=KS)
        window.append(metrics)
        refused = row.refused or is_refusal(row.answer)
        correctness = row.metrics.get("answer_correctness")
        if not refused and correctness is not None:
            answered_correctness.append(correctness)
        if row.candidate_metrics:
            for name in CANDIDATE:
                candidate[name].append(row.candidate_metrics.get(name))

    audited = (audited or {}).get(header.get("config_id"))
    if audited:
        for name in CANDIDATE:
            candidate[name] = [audited[name]]

    _answerable, coverage = coverage_summary(rows)
    return {
        "header": header,
        "n": len(non_null),
        "coverage": coverage,
        "correct": mean(row.metrics.get("answer_correctness") for row in non_null),
        "correct_answered": mean(answered_correctness),
        "window": aggregate(window),
        "candidate": {
            name: mean(values) for name, values in candidate.items()
        },
        "window_chunks": [
            len(row.context_chunk_ids) for row in non_null
        ],
    }


def markdown(results, queries) -> str:
    lines = [
        "# Phase 1 reanalysis (corrected metrics, stored rows)",
        "",
        "All values recomputed from `results/*.jsonl` and `.index_cache/`; no API calls.",
        "Retrieval metrics use the corrected fact-level nDCG and are computed over the",
        "prompt window. `cand rec@10` is present only for cells that stored candidates.",
        "",
        "| cell | n | answer rate | correctness | corr \\| answered | win rec@10 | win nDCG@10 | win MAP@10 | cand rec@10 | median chunks | <10 chunks |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        header = result["header"]
        label = f"{header.get('axis', '?')}/{header.get('variant', '?')}"
        win = result["window"]
        chunks = result["window_chunks"]
        cand = result["candidate"]["recall@10"]
        cand_txt = "n/a" if math.isnan(cand) else f"{cand:.3f}"
        lines.append(
            f"| {label} | {result['n']} | {result['coverage']:.1%} | "
            f"{result['correct']:.3f} | {result['correct_answered']:.3f} | "
            f"{win['recall@10']['mean']:.3f} | {win['ndcg@10']['mean']:.3f} | "
            f"{win['map@10']['mean']:.3f} | {cand_txt} | "
            f"{statistics.median(chunks):.0f} | "
            f"{sum(1 for n in chunks if n < 10) / len(chunks):.0%} |"
        )

    baseline = next((r for r in results if r["header"].get("axis") == "baseline"), None)
    if baseline is not None:
        lines += [
            "",
            "## Baseline by question type",
            "",
            "| type | n | answer rate | correctness | corr \\| answered | win rec@10 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for question_type in sorted({q.question_type for q in queries if not q.is_null}):
            rows = baseline["rows"]
            typed = [row for row in rows if row.question_type == question_type and not row.is_null]
            if not typed:
                continue
            typed_coverage = sum(
                1 for row in typed if not (row.refused or is_refusal(row.answer))
            ) / len(typed)
            typed_correct = mean(row.metrics.get("answer_correctness") for row in typed)
            typed_answered = mean(
                row.metrics.get("answer_correctness")
                for row in typed
                if not (row.refused or is_refusal(row.answer))
            )
            typed_recall = mean(
                evaluate_retrieval(
                    [baseline["texts"][chunk_id] for chunk_id in row.context_chunk_ids],
                    queries[row.query_id].gold_facts,
                    ks=KS,
                )["recall@10"]
                for row in typed
            )
            lines.append(
                f"| {question_type} | {len(typed)} | {typed_coverage:.1%} | "
                f"{typed_correct:.3f} | {typed_answered:.3f} | {typed_recall:.3f} |"
            )

    lines += [
        "",
        "## Why the old table was misleading",
        "",
        "`answer_correctness` is `answer_rate x accuracy|answered`; accuracy|answered is",
        "~0.96 in every cell, so the ablation moved refusal, not accuracy. Retrieval",
        "`recall@10` was computed over the prompt window: 100% of 2048-char rows had",
        "fewer than 10 chunks (so recall@10 was recall-all), while 512-char rows had",
        "17-26 chunks (so recall@10 was a real top-10). The metrics are not comparable",
        "across the chunk-size axis; candidate metrics fix that on new runs.",
        "",
    ]
    return "\n".join(lines) + "\n"


def main(results_dir: str = "results", out: str = "reports/phase1.md") -> int:
    queries = load_queries()
    texts = chunk_text_map()
    audited = load_candidate_metrics()
    results = []
    for path in sorted(glob.glob(f"{results_dir}/*.jsonl")):
        header, _ceiling, rows = read_cell(Path(path))
        result = analyze_cell(header, rows, queries, texts, audited)
        result["rows"] = rows
        result["texts"] = texts
        results.append(result)

    results.sort(key=lambda r: (r["header"].get("axis", ""), r["header"].get("variant", "")))
    report = markdown(results, queries)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(report)
    print(report)
    print(f"report -> {out}")
    return 0


def cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="reports/phase1.md")
    args = parser.parse_args()
    return main(args.results, args.out)


if __name__ == "__main__":
    raise SystemExit(cli())

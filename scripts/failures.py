"""
Phase 2 driver: run every engineered failure slice through the Phase 1
metric path and report the coverage matrix — which metric catches which
failure.

Each class is a paired control/engineered experiment over the same queries.
Answers are pre-authored where the class engineers the answer, and generated
live (gpt-4o-mini, same as Phase 1) only where generation is the phenomenon.

The matrix is judged against a **ground truth**: the paired change in
``answer_correctness``. A metric that drops when the answer does not is a
false alarm, not detection — the earlier matrix marked both as "caught".

    uv run python scripts/failures.py                 # all slices
    uv run python scripts/failures.py --slices chunk_boundary,hallucination
    uv run python scripts/failures.py --limit 4 --resume

Rows persist to results/failures/<slice>__<arm>.jsonl, one JSON object per
query, so any cell can be re-aggregated without a re-run. The markdown table is
written to reports/phase2.md.

Needs OPENROUTER_API_KEY (judge; generation for the two live slices).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from sweep import make_generator, make_judge

from rag_eval.aggregate import aggregate, effect_size_paired
from rag_eval.controlled import (
    ALL_SLICES,
    SLICES_BY_NAME,
    EngineeredQuery,
    FailureSlice,
)
from rag_eval.data import EvidenceItem, Query
from rag_eval.harness import RETRIEVAL_KS, InstanceResult, evaluate_instance
from rag_eval.pipeline import Chunk, Retrieved

MATRIX_METRICS = (
    "recall@1",
    "recall@4",
    "recall@10",
    "mrr",
    "faithfulness",
    "answer_relevance",
    "context_precision",
    "answer_correctness",
)


def to_query(eq: EngineeredQuery, qid: int) -> Query:
    return Query(
        query=eq.query,
        answer=eq.gold_answer,
        question_type="inference",
        evidence_list=[
            EvidenceItem(fact=f, author=None, category=None, published_at=None, source=None, title=None, url=None)
            for f in eq.gold_facts
        ],
    )


def to_retrieved(contexts: tuple[str, ...]) -> list[Retrieved]:
    return [
        Retrieved(
            chunk=Chunk(
                chunk_id=f"doc{i}:0-{len(text)}",
                doc_id=f"doc{i}",
                text=text,
                start_char=0,
                end_char=len(text),
                published_at=None,
                title=None,
            ),
            score=1.0,
            rank=i + 1,
            source="engineered",
        )
        for i, text in enumerate(contexts)
    ]


def measure_arm(
    slice_: FailureSlice,
    arm: str,
    generator,
    judge,
    limit: int | None = None,
) -> list[InstanceResult]:
    eqs = slice_.control if arm == "control" else slice_.engineered
    if limit is not None:
        eqs = eqs[:limit]
    rows: list[InstanceResult] = []
    for i, eq in enumerate(eqs):
        # Metric order is eq.contexts; the generator sees the optional
        # prompt_order permutation, so a position failure does not move ranks.
        retrieved = to_retrieved(eq.contexts)
        answer = (
            eq.answer
            if eq.answer is not None
            else generator.generate(eq.query, list(eq.prompt_contexts()))
        )
        rows.append(
            evaluate_instance(
                to_query(eq, i),
                retrieved,
                answer,
                judge=judge,
                ks=RETRIEVAL_KS,
                config_id=slice_.name,
                query_id=i,
                n_candidates=len(retrieved),
            )
        )
    return rows


def write_arm(path: Path, slice_name: str, arm: str, rows: list[InstanceResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write(
            json.dumps({"kind": "cell", "slice": slice_name, "arm": arm, "n_rows": len(rows)}) + "\n"
        )
        for row in rows:
            handle.write(json.dumps({"kind": "row", **asdict(row)}) + "\n")


def read_arm(path: Path) -> tuple[dict, list[InstanceResult]]:
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
            rows.append(InstanceResult(**record))
    return header, rows


def metric_delta(control: list[InstanceResult], engineered: list[InstanceResult], name: str):
    # delta = engineered - control: negative means the failure degraded the metric.
    a = {r.query_id: r.metrics.get(name) for r in engineered}
    b = {r.query_id: r.metrics.get(name) for r in control}
    common = sorted(i for i in set(a) & set(b) if a[i] is not None and b[i] is not None)
    if not common:
        return None
    stats = effect_size_paired([a[i] for i in common], [b[i] for i in common], seed=42)
    stats["n"] = len(common)
    return stats


def cell(rows: list[InstanceResult], name: str) -> str:
    stats = aggregate([r.metrics for r in rows]).get(name)
    if stats is None:
        return "  n/a  "
    return f"{stats['mean']:.3f}"


def classify(truth_present: bool, stats: dict | None) -> str:
    """Map a metric delta onto detection semantics."""
    if stats is None:
        return "n/a"
    degraded = stats["significant"] and stats["delta"] < 0
    if truth_present:
        return "caught" if degraded else "blind"
    return "false alarm" if degraded else "neutral"


def mark(truth_present: bool, stats: dict | None) -> str:
    label = classify(truth_present, stats)
    return {
        "caught": "+",
        "blind": "-",
        "false alarm": "!",
        "neutral": ".",
        "n/a": "?",
    }[label]


def render_table(results) -> str:
    lines = []
    header = f"{'slice':<18}{'phenomenon':>11}" + "".join(f"{m:>20}" for m in MATRIX_METRICS)
    lines.append(header)
    for slice_, arms in results:
        truth = metric_delta(arms["control"], arms["engineered"], "answer_correctness")
        present = truth is not None and truth["delta"] < -1e-9
        if present and truth is not None and not truth["significant"]:
            truth_txt = "weak"  # point drop, CI includes 0
        else:
            truth_txt = "present" if present else "absent"
        marks = "".join(
            ("                   T" if m == "answer_correctness"
             else f"{mark(present, metric_delta(arms['control'], arms['engineered'], m)):>20}")
            for m in MATRIX_METRICS
        )
        lines.append(f"{slice_.name:<18}{truth_txt:>11} {marks}")
    return "\n".join(lines)


def write_report(results, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 2 coverage matrix (regenerated)",
        "",
        "`phenomenon` is the paired change in `answer_correctness` (ground truth);",
        "the `answer_correctness` column itself is marked `T`, not scored as a detector.",
        "",
        "`+` caught (metric drops and the answer drops) · `-` blind (answer drops, metric does not) ·",
        "`!` false alarm (metric drops, answer does not) · `.` neutral (neither drops).",
        "",
        "For retrieval-side classes the perturbation is present by construction; a",
        "retrieval metric that moves while the answer does not is reported as a false",
        "alarm because it is not a proxy for user-visible failure.",
        "",
        render_table(results),
        "",
    ]
    for slice_, arms in results:
        c, e = arms["control"], arms["engineered"]
        lines.append(f"## {slice_.name}")
        lines.append("")
        lines.append(f"{slice_.diagnostic}  (n={len(c)}/{len(e)})")
        lines.append("")
        lines.append("| metric | control | engineered | delta | 95% CI |")
        lines.append("|---|---|---|---|---|")
        for name in MATRIX_METRICS:
            d = metric_delta(c, e, name)
            if d is None:
                continue
            lines.append(
                f"| {name} | {cell(c, name)} | {cell(e, name)} | "
                f"{d['delta']:+.3f} | [{d['ci_lower']:+.2f}, {d['ci_upper']:+.2f}] |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main(
    names: list[str] | None = None,
    limit: int | None = None,
    out_dir: str = "results/failures",
    resume: bool = False,
    report: str = "reports/phase2.md",
) -> int:
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("missing env vars: OPENROUTER_API_KEY", file=sys.stderr)
        return 2
    generator = make_generator()
    judge = make_judge()

    chosen = [
        s for s in ALL_SLICES if names is None or s.name in names
    ]
    unknown = [n for n in (names or []) if n not in SLICES_BY_NAME]
    if unknown:
        print(f"unknown slices: {unknown}; known: {sorted(SLICES_BY_NAME)}", file=sys.stderr)
        return 2

    out = Path(out_dir)
    results: list[tuple[FailureSlice, dict[str, list[InstanceResult]]]] = []
    for slice_ in chosen:
        arms: dict[str, list[InstanceResult]] = {}
        for arm in ("control", "engineered"):
            path = out / f"{slice_.name}__{arm}.jsonl"
            if resume and path.exists():
                _header, rows = read_arm(path)
                print(f"[cached] {slice_.name}/{arm} rows={len(rows)}")
                arms[arm] = rows
                continue
            rows = measure_arm(slice_, arm, generator, judge, limit=limit)
            write_arm(path, slice_.name, arm, rows)
            print(f"[cell]  {slice_.name}/{arm} rows={len(rows)} -> {path}")
            arms[arm] = rows
        results.append((slice_, arms))

    # Per-slice tables: control vs engineered on every metric.
    print()
    for slice_, arms in results:
        c, e = arms["control"], arms["engineered"]
        truth = metric_delta(c, e, "answer_correctness")
        present = truth is not None and truth["delta"] < -1e-9
        print(f"== {slice_.name}: {slice_.diagnostic}")
        print(f"   phenomenon {'PRESENT' if present else 'ABSENT'} "
              f"(answer_correctness delta {truth['delta']:+.3f})" if truth else "")
        print(f"{'metric':<22}{'control':>10}{'engineered':>12}{'delta':>9}{'ci':>22}{'verdict':>14}")
        for name in MATRIX_METRICS:
            d = metric_delta(c, e, name)
            if d is None:
                continue
            verdict = "truth" if name == "answer_correctness" else classify(present, d)
            print(
                f"{name:<22}{cell(c, name):>10}{cell(e, name):>12}{d['delta']:>+9.3f}"
                f"[{d['ci_lower']:+.2f},{d['ci_upper']:+.2f}]".ljust(53)
                + f"{verdict:>14}"
            )

    print("\ncoverage matrix: + caught / - blind / ! false alarm / . neutral")
    print(render_table(results))
    write_report(results, Path(report))
    print(f"\nreport -> {report}")
    return 0


def cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slices", default=None, help="comma-separated slice names")
    parser.add_argument("--limit", type=int, default=None, help="queries per arm")
    parser.add_argument("--out", default="results/failures", help="output directory")
    parser.add_argument("--report", default="reports/phase2.md", help="markdown report path")
    parser.add_argument("--resume", action="store_true", help="skip arms already written")
    args = parser.parse_args()
    return main(
        names=args.slices.split(",") if args.slices else None,
        limit=args.limit,
        out_dir=args.out,
        resume=args.resume,
        report=args.report,
    )


if __name__ == "__main__":
    raise SystemExit(cli())

"""
Phase 2 driver: run every engineered failure slice through the Phase 1
metric path and report the coverage matrix — which metric catches which
failure.

Each class is a paired control/engineered experiment over the same queries.
Answers are pre-authored where the class engineers the generator's behavior,
and generated live (gpt-4o-mini, same as Phase 1) where generation is the
phenomenon. Judge: gemini-2.5-flash, cached — identical stack to the sweep.

    uv run python scripts/failures.py                 # all slices
    uv run python scripts/failures.py --slices chunk_boundary,hallucination
    uv run python scripts/failures.py --limit 4 --resume

Rows persist to results/failures/<slice>__<arm>.jsonl, one JSON object per
query, so any cell can be re-aggregated without a re-run.

Needs OPENAI_API_KEY (generation) and OPENROUTER_API_KEY (judge).
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
from rag_eval.pipeline import Chunk, Retrieved, context_texts

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
        retrieved = to_retrieved(eq.contexts)
        answer = (
            eq.answer
            if eq.answer is not None
            else generator.generate(eq.query, context_texts(retrieved))
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


def main(
    names: list[str] | None = None,
    limit: int | None = None,
    out_dir: str = "results/failures",
    resume: bool = False,
) -> int:
    if not os.environ.get("OPENAI_API_KEY") or not os.environ.get("OPENROUTER_API_KEY"):
        print(
            "missing env vars: OPENAI_API_KEY and/or OPENROUTER_API_KEY", file=sys.stderr
        )
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
        print(f"== {slice_.name}: {slice_.diagnostic}")
        print(f"{'metric':<22}{'control':>10}{'engineered':>12}{'delta':>9}{'ci':>22}{'n':>5}")
        for name in MATRIX_METRICS:
            d = metric_delta(c, e, name)
            if d is None:
                continue
            print(
                f"{name:<22}{cell(c, name):>10}{cell(e, name):>12}{d['delta']:>+9.3f}"
                f"[{d['ci_lower']:+.2f},{d['ci_upper']:+.2f}]".ljust(53)
                + f"{d['n']:>5}"
            )

    # Coverage matrix: '+' = engineered significantly worse (metric caught it),
    # '-' = no significant change (blind spot), '?' = engineered better.
    print("\ncoverage matrix: + caught / - blind / ? engineered better")
    print(f"{'slice':<18}" + " " + " ".join(f"{m:>12}" for m in MATRIX_METRICS))
    for slice_, arms in results:
        marks = []
        for name in MATRIX_METRICS:
            d = metric_delta(arms["control"], arms["engineered"], name)
            if d is None:
                marks.append(f"{'n/a':>12}")
            elif d["significant"] and d["delta"] < 0:
                marks.append(f"{'+':>12}")
            elif d["significant"]:
                marks.append(f"{'?':>12}")
            else:
                marks.append(f"{'-':>12}")
        print(f"{slice_.name:<18}" + " " + " ".join(marks))
    return 0


def cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slices", default=None, help="comma-separated slice names")
    parser.add_argument("--limit", type=int, default=None, help="queries per arm")
    parser.add_argument("--out", default="results/failures", help="output directory")
    parser.add_argument("--resume", action="store_true", help="skip arms already written")
    args = parser.parse_args()
    return main(
        names=args.slices.split(",") if args.slices else None,
        limit=args.limit,
        out_dir=args.out,
        resume=args.resume,
    )


if __name__ == "__main__":
    raise SystemExit(cli())
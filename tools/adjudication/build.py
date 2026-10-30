"""Build data.js for the adjudication page from rag-eval-project's stored rows.

Reads (read-only):
  results/*.jsonl            — one row per query per config (10 configs)
  data/labels/phase3_*.jsonl — the stratified human-review sample (v1, n=60)
  .index_cache/*/chunks.json — chunk_id -> text, used to attribute each
                               retrieved chunk to its source document

Writes:
  data.js  — window.AUDIT = {cells, traces, ...}

No network, no API calls, no writes inside the project repo.

    uv run python tools/adjudication/build.py
"""

from __future__ import annotations

import glob
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[2])
OUT = Path(__file__).parent / "data.js"
SAMPLE = sys.argv[2] if len(sys.argv) > 2 else "v2"

CHUNK_ID = re.compile(r"^(.*):(\d+)-(\d+)$")


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def chunk_index() -> dict[str, str]:
    """chunk_id -> text, merged across every cached chunking config."""
    out: dict[str, str] = {}
    for path in glob.glob(str(PROJECT / ".index_cache" / "*" / "chunks.json")):
        with open(path) as f:
            for chunk in json.load(f):
                out[chunk["chunk_id"]] = chunk["text"]
    return out


def doc_of(chunk_id: str) -> str:
    m = CHUNK_ID.match(chunk_id)
    return m.group(1) if m else ""


def text_lookup(chunks: dict[str, str]) -> dict[str, str]:
    """Normalised text -> doc, so a sample context can be traced to its source."""
    return {" ".join(text.split())[:200]: doc_of(cid) for cid, text in chunks.items()}


def main() -> int:
    cells: list[dict] = []
    per_query: dict[int, dict[str, dict]] = {}

    for path in sorted(glob.glob(str(PROJECT / "results" / "*.jsonl"))):
        rows = load_jsonl(Path(path))
        header = rows[0]
        cell_id = header["config_id"]
        cells.append(
            {
                "id": cell_id,
                "axis": header.get("axis", ""),
                "variant": header.get("variant", ""),
                "config": header.get("config", {}),
            }
        )
        for row in rows[1:]:
            if row.get("kind") != "row":
                continue
            per_query.setdefault(row["query_id"], {})[cell_id] = row.get("metrics", {})

    sample_path = PROJECT / "data" / "labels" / f"phase3_sample_{SAMPLE}.jsonl"
    if not sample_path.is_file():
        print(f"missing sample: {sample_path}", file=sys.stderr)
        return 1
    sample = load_jsonl(sample_path)

    chunks = chunk_index()
    lookup = text_lookup(chunks)

    traces = []
    unmatched = 0
    for row in sample:
        contexts = []
        for text in row.get("contexts", []):
            doc = lookup.get(" ".join(text.split())[:200], "")
            if not doc:
                unmatched += 1
            contexts.append({"text": text, "doc": doc})
        traces.append(
            {
                "query_id": row["query_id"],
                "type": row.get("question_type", ""),
                "is_null": bool(row.get("is_null", False)),
                "query": row.get("query", ""),
                "gold": row.get("gold_answer", ""),
                "answer": row.get("answer", ""),
                "contexts": contexts,
                "metrics": per_query.get(row["query_id"], {}),
            }
        )

    payload = {
        "project": "rag-eval-project",
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_sample": (
            f"sample {SAMPLE} · n={len(traces)} · "
            f"contexts resolved {sum(len(t["contexts"]) for t in traces) - unmatched}"
            f"/{sum(len(t["contexts"]) for t in traces)}"
        ),
        "cells": cells,
        "traces": traces,
    }

    OUT.write_text("window.AUDIT = " + json.dumps(payload, separators=(",", ":")) + ";\n")
    print(f"cells: {len(cells)}  traces: {len(traces)}  unmatched contexts: {unmatched}")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

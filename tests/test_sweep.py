"""Offline coverage for sweep.py's load-bearing pure functions.

The driver itself is exercised by the real run (it has no provider seam, by
design — it constructs concrete components). What must not be trusted without
a test are the numbers it persists and reports: cell headers, the recall
ceiling that stops unreachable facts from reading as retrieval failures, and
the 'one axis at a time' hygiene that makes the ablation interpretable.
"""

from dataclasses import asdict

from helpers import make_config, make_doc, make_query, make_retrieved
from sweep import BASELINE, cells, chunk_ceiling, read_cell, write_cell

from rag_eval.data import EvidenceItem, Query
from rag_eval.harness import evaluate_instance
from rag_eval.pipeline import RecursiveChunker

CUTOFF = "2023-06-01T00:00:00+00:00"

# axis label -> the config field it must be the sole difference in
AXIS_FIELDS = {
    "chunk_size": "chunk_size",
    "overlap": "chunk_overlap",
    "embedder": "embedder",
    "retriever": "retriever",
    "rerank": "rerank",
    "recency_filter": "recency_cutoff",
}


def evidence_with_title(fact: str, title: str, url: str) -> EvidenceItem:
    return EvidenceItem(
        fact=fact, author=None, category=None, published_at=None, source=None, title=title, url=url
    )


def test_cells_vary_exactly_one_field():
    baseline_fields = asdict(BASELINE)
    for axis, variants in cells(CUTOFF).items():
        field = AXIS_FIELDS[axis]
        for label, config in variants:
            diff = [f for f, v in asdict(config).items() if v != baseline_fields[f]]
            assert diff == [field], f"{label} changed {diff}, not just {field}"


def test_chunk_ceiling_counts_only_facts_contained_in_some_chunk():
    docs = [
        make_doc("The company was founded in 1999 in Austin.", url="u1", title="story one"),
        make_doc("y" * 1400, url="u2", title="story two"),  # one hard-split doc
    ]
    queries = [
        Query(
            query="q1",
            answer="a",
            question_type="inference",
            evidence_list=[
                evidence_with_title("The company was founded in 1999 in Austin.", "story one", "u1")
            ],
        ),
        Query(
            query="q2",
            answer="a",
            question_type="inference",
            evidence_list=[evidence_with_title("y" * 1400, "story two", "u2")],
        ),
    ]
    # The 1400-char fact is longer than any 1024-char chunk, so it is
    # unreachable no matter how good retrieval is; the short fact is reachable.
    total, ceiling = chunk_ceiling(docs, RecursiveChunker(chunk_size=1024), queries)
    assert total == 2
    assert ceiling == 0.5


def test_cell_roundtrip_preserves_rows_and_header(tmp_path):
    config = make_config()
    rows = [
        evaluate_instance(
            make_query(text="q1", facts=("fact one",)),
            [make_retrieved("fact one is here", rank=1)],
            "answer",
            config_id=config.id,
            query_id=0,
        ),
        evaluate_instance(
            make_query(text="q2", is_null=True),
            [],
            "Insufficient information.",
            config_id=config.id,
            query_id=1,
        ),
    ]
    path = tmp_path / f"{config.id}.jsonl"
    write_cell(path, config, rows, 0.75, "chunk_size", "512")
    header, ceiling, loaded = read_cell(path)
    assert header["config_id"] == config.id
    assert header["axis"] == "chunk_size"
    assert header["variant"] == "512"
    assert header["n_rows"] == 2
    assert ceiling == 0.75
    assert loaded == rows  # dataclass equality: null and non-null rows both survive

def test_incremental_write_is_readable_after_each_row(tmp_path):
    from sweep import append_row, write_header

    config = make_config()
    path = tmp_path / f"{config.id}.jsonl"
    rows = [
        evaluate_instance(
            make_query(text="q1", facts=("fact one",)),
            [make_retrieved("fact one is here", rank=1)],
            "answer",
            config_id=config.id,
            query_id=0,
        ),
        evaluate_instance(
            make_query(text="q2", is_null=True),
            [],
            "Insufficient information.",
            config_id=config.id,
            query_id=1,
        ),
    ]
    write_header(path, config, 0.75, "chunk_size", "512", len(rows), {"judge": {"model": "m"}})
    append_row(path, rows[0])
    header, ceiling, loaded = read_cell(path)
    assert header["provenance"]["judge"]["model"] == "m"
    assert ceiling == 0.75
    assert loaded == rows[:1]  # a crash here loses nothing already measured

    append_row(path, rows[1])
    _header, _ceiling, loaded = read_cell(path)
    assert loaded == rows


def test_query_set_hash_is_order_sensitive():
    from sweep import query_set_sha

    a = make_query(text="alpha")
    b = make_query(text="beta")
    assert query_set_sha([a, b]) == query_set_sha([a, b])
    assert query_set_sha([a, b]) != query_set_sha([b, a])


def test_provenance_records_dataset_and_judge(tmp_path):
    from sweep import provenance, query_set_sha

    config = make_config()
    queries = [make_query(text="alpha")]
    data = provenance(config, queries)
    assert data["dataset"]["revision"]
    assert data["judge"]["model"] == config.judge
    assert data["query_set_sha"] == query_set_sha(queries)
    assert data["query_count"] == 1

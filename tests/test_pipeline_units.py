"""Pipeline seam units: timestamps, chunking, context truncation, index,
dense retrieval, RRF fusion, and the config id that names every sweep cell.
"""

from datetime import UTC, datetime
from itertools import pairwise

import numpy as np
import pytest
from helpers import (
    DictEmbedder,
    FakeEmbedder,
    make_chunk,
    make_config,
    make_doc,
    make_retrieved,
)

from rag_eval.pipeline import (
    DenseRetriever,
    Index,
    RecursiveChunker,
    RRFFusion,
    build_or_load_index,
    context_texts,
    parse_published_at,
    render_contexts,
    truncate_contexts,
)

BODY = (
    "First paragraph about apples and pears.\n\n"
    "Second paragraph about the company founded in 1999 and its funding.\n\n"
    "Third paragraph about oranges."
)

DOCS = [
    make_doc(
        "The company was founded in 1999. It raised 10 million.",
        url="u1",
        published_at="2023-01-01 00:00:00",
    ),
    make_doc(
        "Oranges grow in Florida during winter. Citrus is seasonal.",
        url="u2",
        published_at="2023-06-01 00:00:00",
    ),
]


# --- parse_published_at --------------------------------------------------- #


def test_parse_published_at_none():
    assert parse_published_at(None) is None


def test_parse_published_at_naive_string_is_read_as_utc():
    parsed = parse_published_at("2023-11-27 08:45:59")
    assert parsed == datetime(2023, 11, 27, 8, 45, 59, tzinfo=UTC)
    assert parsed.tzinfo is not None


def test_parse_published_at_aware_string_converts_to_utc():
    parsed = parse_published_at("2023-11-27T08:45:59+02:00")
    assert parsed == datetime(2023, 11, 27, 6, 45, 59, tzinfo=UTC)


def test_parse_published_at_naive_datetime_becomes_utc():
    # Naive datetime is the input under test.
    parsed = parse_published_at(datetime(2023, 1, 1, 12, 0, 0))  # noqa: DTZ001
    assert parsed == datetime(2023, 1, 1, 12, 0, 0, tzinfo=UTC)


# --- RecursiveChunker ----------------------------------------------------- #


def test_chunker_spans_cover_body_exactly():
    chunker = RecursiveChunker(chunk_size=40, chunk_overlap=10)
    chunks = chunker.chunk(make_doc(BODY))
    assert chunks[0].start_char == 0
    assert chunks[-1].end_char == len(BODY)
    for a, b in pairwise(chunks):
        assert b.start_char <= a.end_char  # adjacent or overlapping, never gapped
        assert a.text == BODY[a.start_char : a.end_char]
        assert len(a.text) <= 40
    assert len(chunks) > 1


def test_chunker_overlap_is_a_maximum():
    chunker = RecursiveChunker(chunk_size=40, chunk_overlap=10)
    chunks = chunker.chunk(make_doc(BODY))
    for a, b in pairwise(chunks):
        assert b.start_char >= a.end_char - 10


def test_chunker_is_deterministic():
    doc = make_doc(BODY)
    chunker = RecursiveChunker(chunk_size=40, chunk_overlap=10)
    a = chunker.chunk(doc)
    b = chunker.chunk(doc)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert [c.text for c in a] == [c.text for c in b]


def test_chunker_hard_splits_text_without_separators():
    body = "ab" * 1300
    chunks = RecursiveChunker(chunk_size=1024).chunk(make_doc(body))
    assert [len(c.text) for c in chunks] == [1024, 1024, 552]
    assert "".join(c.text for c in chunks) == body
    assert chunks[0].chunk_id == f"{chunks[0].doc_id}:0-1024"


def test_chunker_validation():
    with pytest.raises(ValueError):
        RecursiveChunker(chunk_size=0)
    with pytest.raises(ValueError):
        RecursiveChunker(chunk_size=50, chunk_overlap=50)
    with pytest.raises(ValueError):
        RecursiveChunker(chunk_size=50, chunk_overlap=-1)
    with pytest.raises(ValueError):
        RecursiveChunker(separators=("",))


# --- truncate_contexts / render_contexts / context_texts ------------------ #


def test_truncate_contexts_k_final_cuts_by_rank():
    items = [make_retrieved(str(i) * 10, rank=i) for i in range(1, 6)]
    out = truncate_contexts(items, k_final=3)
    assert context_texts(out) == [str(i) * 10 for i in range(1, 4)]


def test_truncate_contexts_char_budget_accumulates_in_rank_order():
    items = [make_retrieved(str(i) * 10, rank=i) for i in range(1, 6)]
    out = truncate_contexts(items, char_budget=25)
    assert len(out) == 2  # 10 + 10 = 20 <= 25; a third would exceed it
    assert context_texts(out) == ["1" * 10, "2" * 10]


def test_truncate_contexts_keeps_at_least_one_chunk():
    items = [make_retrieved("x" * 10, rank=1), make_retrieved("y" * 10, rank=2)]
    assert len(truncate_contexts(items, char_budget=1)) == 1


def test_truncate_contexts_without_bounds_returns_all():
    items = [make_retrieved("a", rank=1), make_retrieved("b", rank=2)]
    assert truncate_contexts(items) == items


def test_render_contexts_numbering_and_order():
    contexts = ["first", "second"]
    assert render_contexts(contexts) == "[1] first\n\n[2] second"
    assert render_contexts(contexts, order="reversed") == "[1] second\n\n[2] first"


def test_render_contexts_rejects_unknown_order():
    with pytest.raises(ValueError):
        render_contexts(["a"], order="middle")


# --- Index ---------------------------------------------------------------- #


def test_index_build_save_load_roundtrip(tmp_path):
    chunker = RecursiveChunker(chunk_size=1000)
    index = Index.build(DOCS, chunker, FakeEmbedder())
    assert index.vectors.shape == (len(DOCS), 16)
    assert index.fingerprint

    path = tmp_path / "idx"
    index.save(path)
    loaded = Index.load(path)
    assert loaded.chunks == index.chunks
    assert np.array_equal(loaded.vectors, index.vectors)
    assert loaded.fingerprint == index.fingerprint


def test_index_fingerprint_tracks_embedder_and_corpus():
    chunker = RecursiveChunker(chunk_size=1000)
    a = Index.build(DOCS, chunker, FakeEmbedder(id="fake:model:a"))
    b = Index.build(DOCS, chunker, FakeEmbedder(id="fake:model:b"))
    c = Index.build(DOCS[:1], chunker, FakeEmbedder(id="fake:model:a"))
    assert a.fingerprint != b.fingerprint
    assert a.fingerprint != c.fingerprint


def test_index_build_rejects_wrong_vector_count():
    class ShortEmbedder:
        id = "fake:short"

        def embed(self, texts):
            return np.zeros((len(texts) - 1, 16), dtype=np.float32)

    with pytest.raises(ValueError):
        Index.build(DOCS, RecursiveChunker(chunk_size=1000), ShortEmbedder())


def test_index_validates_vector_shape():
    index = Index.build(DOCS, RecursiveChunker(chunk_size=1000), FakeEmbedder())
    with pytest.raises(ValueError):
        Index(chunks=index.chunks[:1], vectors=index.vectors)  # 2 vectors, 1 chunk
    with pytest.raises(ValueError):
        Index(chunks=index.chunks, vectors=np.zeros((2, 16, 1)))  # 3-D


def test_build_or_load_index_reuses_cache(tmp_path):
    chunker = RecursiveChunker(chunk_size=1000)
    embedder = FakeEmbedder()
    cache = tmp_path / "cache"
    first = build_or_load_index(DOCS, chunker, embedder, cache_dir=cache)
    calls_after_first = embedder.embed_calls
    second = build_or_load_index(DOCS, chunker, embedder, cache_dir=cache)
    assert embedder.embed_calls == calls_after_first  # served from cache
    assert second.fingerprint == first.fingerprint
    assert second.chunks == first.chunks


def test_build_or_load_index_rebuilds_on_corpus_change(tmp_path):
    chunker = RecursiveChunker(chunk_size=1000)
    embedder = FakeEmbedder()
    cache = tmp_path / "cache"
    build_or_load_index(DOCS, chunker, embedder, cache_dir=cache)
    build_or_load_index(
        DOCS + [make_doc("A third story about grapes.", url="u3")],
        chunker,
        embedder,
        cache_dir=cache,
    )
    assert embedder.embed_calls == 2


# --- DenseRetriever ------------------------------------------------------- #


def _apple_index():
    chunks = [make_chunk("apple news"), make_chunk("banana sports"), make_chunk("cherry finance")]
    vectors = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    return Index(chunks=chunks, vectors=vectors)


def test_dense_retriever_ranks_by_dot_product():
    index = _apple_index()
    embedder = DictEmbedder({"apple query": [1, 0, 0]})
    items = DenseRetriever(index, embedder).retrieve("apple query", k_recall=3)
    assert [it.chunk.text for it in items] == ["apple news", "banana sports", "cherry finance"]
    assert items[0].rank == 1
    assert items[0].score == pytest.approx(1.0)
    assert items[0].source == "dense"


def test_dense_retriever_honors_k_recall():
    index = _apple_index()
    embedder = DictEmbedder({"apple query": [1, 0, 0]})
    items = DenseRetriever(index, embedder).retrieve("apple query", k_recall=1)
    assert len(items) == 1
    assert items[0].chunk.text == "apple news"


def test_dense_retriever_rejects_naive_cutoff():
    # Naive cutoff is the rejected input under test.
    with pytest.raises(ValueError):
        DenseRetriever(_apple_index(), DictEmbedder({"apple query": [1, 0, 0]})).retrieve(
            "apple query", k_recall=3, published_before=datetime(2023, 1, 1)  # noqa: DTZ001
        )


def test_dense_retriever_filters_by_publication_date():
    chunks = [
        make_chunk("apple jan", published_at="2023-01-01 00:00:00"),
        make_chunk("apple jun", published_at="2023-06-01 00:00:00"),
        make_chunk("apple undated", published_at=None),
    ]
    vectors = np.asarray([[1, 0, 0], [0.9, 0, 0], [0, 1, 0]], dtype=np.float32)
    index = Index(chunks=chunks, vectors=vectors)
    embedder = DictEmbedder({"apple query": [1, 0, 0]})
    cutoff = parse_published_at("2023-03-01")
    items = DenseRetriever(index, embedder).retrieve("apple query", k_recall=3, published_before=cutoff)
    assert [it.chunk.text for it in items] == ["apple jan"]  # jun after cutoff, undated excluded


def test_dense_retriever_empty_when_nothing_within_cutoff():
    index = _apple_index()
    embedder = DictEmbedder({"apple query": [1, 0, 0]})
    items = DenseRetriever(index, embedder).retrieve(
        "apple query", k_recall=3, published_before=parse_published_at("2020-01-01")
    )
    assert items == []


# --- RRFFusion ------------------------------------------------------------ #


class FixedRetriever:
    def __init__(self, results, source):
        self.results = results
        self._source = source

    def retrieve(self, query, *, k_recall, published_before=None):
        return self.results


def test_rrf_fusion_merges_legs_by_rank_and_breaks_ties_on_chunk_id():
    leg_a = FixedRetriever(
        [make_retrieved("x apple", rank=1, chunk_id="aaa"), make_retrieved("z cherry", rank=2, chunk_id="mmm")],
        "a",
    )
    leg_b = FixedRetriever([make_retrieved("y banana", rank=1, chunk_id="zzz")], "b")
    out = RRFFusion([leg_a, leg_b]).retrieve("q", k_recall=10)
    # aaa and zzz tie at 1/61 (rank 1 each), chunk_id breaks the tie;
    # mmm gets only 1/62 from rank 2 and ranks after them
    assert [r.chunk.chunk_id for r in out] == ["aaa", "zzz", "mmm"]
    assert [r.rank for r in out] == [1, 2, 3]
    assert all(r.source == "fused" for r in out)


def test_rrf_fusion_sums_duplicate_chunks_across_legs():
    def two_legs(rank):
        return FixedRetriever([make_retrieved("x apple", rank=rank, chunk_id="aaa")], "leg")

    out = RRFFusion([two_legs(1), two_legs(2)]).retrieve("q", k_recall=10)
    assert len(out) == 1
    assert out[0].score == pytest.approx(1 / 61 + 1 / 62)


def test_rrf_fusion_requires_two_legs():
    with pytest.raises(ValueError):
        RRFFusion([FixedRetriever([], "a")])


# --- PipelineConfig ------------------------------------------------------- #


def test_pipeline_config_id_is_deterministic_and_sensitive():
    assert make_config().id == make_config().id
    assert make_config().id != make_config(chunk_size=2048).id
    assert make_config().id != make_config(recency_cutoff="2023-06-01").id
    assert len(make_config().id) == 16
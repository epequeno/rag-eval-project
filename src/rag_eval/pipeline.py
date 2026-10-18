"""
Pipeline adapters: the seam between the eval harness and any RAG pipeline.

The harness owns metrics, aggregation, and statistics; it never owns chunking,
embedding, retrieval, or generation. This module defines that boundary: the
data types that cross it, the Protocols a pipeline must satisfy, and one
deliberately boring reference implementation of each stage.

Units
-----
``chunk_size`` and ``chunk_overlap`` are in **characters**. Character spans keep
gold-context containment auditable (``chunk.text == body[start:end]`` always
holds) and keep a tokenizer off the indexing path. MultiHop-RAG gold facts are
93 / 161 / 257 characters at p10 / p50 / p90 (max 538), so a fact longer than
``chunk_size`` can never be contained in one chunk: a 128-character cell caps
recall@k at 0.32 by construction. The usable grid is 512 / 1024 / 2048
characters (roughly 128 / 256 / 512 tokens).

Index reuse
-----------
``Index`` is separated from ``Retriever`` so the query-time axes of the ablation
grid (dense / BM25 / hybrid, +/- rerank, +/- metadata filter) reuse one index
build instead of re-embedding per cell.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import numpy as np

from rag_eval.data import Document
from rag_eval.judges import _call_with_retries

__all__ = [
    "GENERATOR_TEMPLATE",
    "BM25Retriever",
    "Chunk",
    "Chunker",
    "CrossEncoderReranker",
    "DenseRetriever",
    "Embedder",
    "Generator",
    "Index",
    "OpenAIEmbedder",
    "PipelineConfig",
    "PromptGenerator",
    "RRFFusion",
    "RecursiveChunker",
    "Reranker",
    "Retrieved",
    "Retriever",
    "SentenceTransformerEmbedder",
    "build_or_load_index",
    "context_texts",
    "parse_published_at",
    "render_contexts",
    "truncate_contexts",
]

# Bump when the on-disk index layout or chunk semantics change.
INDEX_FORMAT_VERSION = "1"


# --------------------------------------------------------------------------- #
# Data types crossing the seam
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Chunk:
    """
    A contiguous character span of one source document.

    ``start_char`` / ``end_char`` index into ``Document.body``, which is what
    makes chunk-boundary failures engineerable and gold containment auditable.
    ``published_at`` is parsed once at chunk time rather than per retrieval call.
    """

    chunk_id: str
    doc_id: str
    text: str
    start_char: int
    end_char: int
    published_at: datetime | None
    title: str | None


@dataclass(frozen=True, slots=True)
class Retrieved:
    """
    A chunk plus the evidence for why it was retrieved.

    ``rank`` is 1-based and assigned by the producing stage; RRF fuses on rank,
    and rerankers rewrite it. ``source`` names the leg that produced the hit
    ("dense", "bm25", "fused") so a hybrid loss can be attributed to a leg.
    """

    chunk: Chunk
    score: float
    rank: int
    source: str | None = None


# --------------------------------------------------------------------------- #
# Protocols: what a pipeline must satisfy
# --------------------------------------------------------------------------- #


class Chunker(Protocol):
    """Splits documents into character spans. Must be deterministic."""

    id: str

    def chunk(self, doc: Document) -> list[Chunk]: ...


class Embedder(Protocol):
    """
    Embeds text in order. Must return a 2-D float array of L2-normalized rows
    so retrievers can use a plain dot product. Implementations are responsible
    for batching internally.
    """

    id: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


class Retriever(Protocol):
    """
    Ranks the index for a query. Returns at most ``k_recall`` candidates:
    a reranker's input, not the generator's context.
    """

    def retrieve(
        self,
        query: str,
        *,
        k_recall: int,
        published_before: datetime | None = None,
    ) -> list[Retrieved]: ...


class Reranker(Protocol):
    """Reorders and truncates candidates. Owns the final ``rank``."""

    def rerank(
        self, query: str, candidates: list[Retrieved], k_final: int
    ) -> list[Retrieved]: ...


class Generator(Protocol):
    """Answers a query from already-rendered context strings."""

    def generate(self, query: str, contexts: list[str]) -> str: ...


# --------------------------------------------------------------------------- #
# Index
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Index:
    """
    Chunks plus their embeddings. Immutable in practice: build once, reuse
    across every query-time config in the grid.

    ``fingerprint`` identifies the exact (documents, chunker, embedder) triple
    that produced the index, so a cached index can be validated instead of
    trusted. ``vectors`` must be L2-normalized.
    """

    chunks: list[Chunk]
    vectors: np.ndarray
    fingerprint: str = ""

    def __post_init__(self) -> None:
        if self.vectors.ndim != 2:
            raise ValueError(f"vectors must be 2-D, got shape {self.vectors.shape}")
        if self.vectors.shape[0] != len(self.chunks):
            raise ValueError(
                f"{self.vectors.shape[0]} vectors for {len(self.chunks)} chunks"
            )
        # DenseRetriever treats a dot product as cosine similarity, which is only
        # true for unit rows. An unnormalized embedder would silently change the
        # ranking instead of failing.
        if self.vectors.size:
            norms = np.linalg.norm(self.vectors, axis=1)
            if not np.allclose(norms, 1.0, atol=1e-3):
                raise ValueError(
                    "index vectors must be L2-normalized; "
                    f"norms span {norms.min():.4f}..{norms.max():.4f}"
                )

    @classmethod
    def build(
        cls,
        docs: list[Document],
        chunker: Chunker,
        embedder: Embedder,
    ) -> Index:
        """Chunk every document, embed every chunk, and fingerprint the result."""
        chunks: list[Chunk] = []
        for doc in docs:
            doc_chunks = chunker.chunk(doc)
            if not doc_chunks:
                raise ValueError(f"chunker produced no chunks for doc {doc.doc_id!r}")
            chunks.extend(doc_chunks)

        vectors = embedder.embed([c.text for c in chunks])
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.shape[0] != len(chunks):
            raise ValueError(
                f"embedder returned {vectors.shape[0]} rows for {len(chunks)} chunks"
            )

        return cls(
            chunks=chunks,
            vectors=vectors,
            fingerprint=index_fingerprint(docs, chunker, embedder),
        )

    def save(self, path: Path | str) -> None:
        """Write to a directory: ``chunks.json``, ``vectors.npy``, ``meta.json``."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "chunks.json").write_text(
            json.dumps([_chunk_to_dict(c) for c in self.chunks])
        )
        np.save(path / "vectors.npy", self.vectors)
        (path / "meta.json").write_text(
            json.dumps(
                {
                    "format_version": INDEX_FORMAT_VERSION,
                    "fingerprint": self.fingerprint,
                    "n_chunks": len(self.chunks),
                }
            )
        )

    @classmethod
    def load(cls, path: Path | str) -> Index:
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        version = meta.get("format_version")
        if version != INDEX_FORMAT_VERSION:
            raise ValueError(
                f"index format {version!r} != expected {INDEX_FORMAT_VERSION!r}"
            )
        chunks = [_chunk_from_dict(d) for d in json.loads((path / "chunks.json").read_text())]
        vectors = np.load(path / "vectors.npy")
        if vectors.shape[0] != len(chunks):
            raise ValueError(
                f"corrupt index at {path}: {vectors.shape[0]} vectors, {len(chunks)} chunks"
            )
        return cls(chunks=chunks, vectors=vectors, fingerprint=meta["fingerprint"])


def _chunk_to_dict(chunk: Chunk) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "text": chunk.text,
        "start_char": chunk.start_char,
        "end_char": chunk.end_char,
        "published_at": None if chunk.published_at is None else chunk.published_at.isoformat(),
        "title": chunk.title,
    }


def _chunk_from_dict(d: dict) -> Chunk:
    return Chunk(
        chunk_id=d["chunk_id"],
        doc_id=d["doc_id"],
        text=d["text"],
        start_char=d["start_char"],
        end_char=d["end_char"],
        published_at=None if d["published_at"] is None else datetime.fromisoformat(d["published_at"]),
        title=d["title"],
    )


def _docs_hash(docs: list[Document]) -> str:
    """Order-independent hash of the corpus contents."""
    h = hashlib.sha256()
    for doc in sorted(docs, key=lambda d: d.doc_id):
        h.update(doc.doc_id.encode())
        h.update(b"\x00")
        h.update(doc.body.encode())
        h.update(b"\x00\x00")
    return h.hexdigest()


def index_fingerprint(
    docs: list[Document], chunker: Chunker, embedder: Embedder
) -> str:
    """Identity of an index build: corpus contents + chunking + embedding model."""
    payload = ":".join(
        [_docs_hash(docs), chunker.id, embedder.id, INDEX_FORMAT_VERSION]
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


def build_or_load_index(
    docs: list[Document],
    chunker: Chunker,
    embedder: Embedder,
    cache_dir: Path | str = ".index_cache",
) -> Index:
    """
    Reuse a cached index when the fingerprint matches, else build and cache it.

    Embedding is the expensive step of the sweep; without this, every rerank
    on/off cell re-embeds the whole corpus.
    """
    path = Path(cache_dir) / index_fingerprint(docs, chunker, embedder)
    if (path / "meta.json").exists():
        cached = Index.load(path)
        if cached.fingerprint == path.name:
            return cached
    index = Index.build(docs, chunker, embedder)
    index.save(path)
    return index


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #


def parse_published_at(value: str | datetime | None) -> datetime | None:
    """
    Normalize a publication timestamp to timezone-aware UTC, or ``None``.

    The same MultiHop-RAG rows arrive as naive strings from the ``datasets``
    loader (``"2023-11-27 08:45:59"``) and as tz-aware strings from the
    datasets-server API (``"2023-11-27T08:45:59+00:00"``). All 609 corpus
    timestamps are UTC, so a missing offset is read as UTC rather than left
    naive: a naive value raises on comparison against an aware cutoff, which is
    exactly the metadata-filter path.
    """
    if value is None:
        return None
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class RecursiveChunker:
    """
    Standard recursive character chunker: split on the coarsest separator that
    applies, recurse into oversized pieces, then greedily pack units up to
    ``chunk_size``.

    ``chunk_overlap`` is a **maximum**: the next chunk starts at the first unit
    boundary at or after ``end - chunk_overlap``, so overlap never cuts a unit.
    Since separator boundaries are respected, a gold fact is only at risk when
    it straddles a *packing* boundary — which is exactly the failure Phase 2
    engineers.
    """

    chunk_size: int = 1024
    chunk_overlap: int = 128
    separators: tuple[str, ...] = ("\n\n", "\n", ". ", " ")

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError(
                f"chunk_overlap must be in [0, chunk_size), got {self.chunk_overlap}"
            )
        if not self.separators or any(s == "" for s in self.separators):
            raise ValueError("separators must be non-empty strings")

    @property
    def id(self) -> str:
        seps = "|".join(s.replace("\n", "\\n") for s in self.separators)
        return f"recursive:chars:{self.chunk_size}:{self.chunk_overlap}:{seps}"

    def chunk(self, doc: Document) -> list[Chunk]:
        body = doc.body
        published_at = parse_published_at(doc.published_at)
        chunks = []
        for start, end in self._packed_spans(body):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}:{start}-{end}",
                    doc_id=doc.doc_id,
                    text=body[start:end],
                    start_char=start,
                    end_char=end,
                    published_at=published_at,
                    title=doc.title,
                )
            )
        return chunks

    def _packed_spans(self, body: str) -> list[tuple[int, int]]:
        units: list[tuple[int, int]] = []
        self._split(body, 0, len(body), 0, units)

        spans: list[tuple[int, int]] = []
        i, n = 0, len(units)
        while i < n:
            j = i
            while j < n and units[j][1] - units[i][0] <= self.chunk_size:
                j += 1
            assert j > i, "every unit is bounded by chunk_size, so packing advances"
            end = units[j - 1][1]
            spans.append((units[i][0], end))
            if j >= n:
                break
            boundary = end - self.chunk_overlap
            nxt = j
            for t in range(i + 1, j):
                if units[t][0] >= boundary:
                    nxt = t
                    break
            i = nxt
        return spans

    def _split(
        self,
        body: str,
        start: int,
        end: int,
        sep_ix: int,
        out: list[tuple[int, int]],
    ) -> None:
        """Append non-overlapping units covering ``body[start:end]`` exactly."""
        if end - start <= self.chunk_size:
            if end > start:
                out.append((start, end))
            return

        if sep_ix >= len(self.separators):
            # No usable separator left: hard split on the size bound.
            for s in range(start, end, self.chunk_size):
                out.append((s, min(s + self.chunk_size, end)))
            return

        sep = self.separators[sep_ix]
        pieces: list[tuple[int, int]] = []
        cur = start
        while True:
            p = body.find(sep, cur, end)
            if p == -1:
                break
            pieces.append((cur, p + len(sep)))  # separator stays with the left piece
            cur = p + len(sep)
        if cur < end:
            pieces.append((cur, end))

        if len(pieces) <= 1:
            self._split(body, start, end, sep_ix + 1, out)
            return

        for s, e in pieces:
            self._split(body, s, e, sep_ix + 1, out)


# --------------------------------------------------------------------------- #
# Embedding
# --------------------------------------------------------------------------- #


@dataclass
class OpenAIEmbedder:
    """OpenAI embeddings API. The proprietary arm of the embedding axis."""

    model: str = "text-embedding-3-small"
    batch_size: int = 256
    dimensions: int | None = None

    def __post_init__(self) -> None:
        from openai import OpenAI

        self.client = OpenAI()
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

    @property
    def id(self) -> str:
        suffix = "" if self.dimensions is None else f":{self.dimensions}"
        return f"openai:{self.model}{suffix}"

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed in order, L2-normalized, float32."""
        rows: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            kwargs = {"model": self.model, "input": batch}
            if self.dimensions is not None:
                kwargs["dimensions"] = self.dimensions
            response = self.client.embeddings.create(**kwargs)
            if len(response.data) != len(batch):
                raise ValueError(
                    f"embedder returned {len(response.data)} rows for {len(batch)} inputs"
                )
            rows.extend(item.embedding for item in response.data)
        vectors = np.asarray(rows, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)


@dataclass
class SentenceTransformerEmbedder:
    """
    Open-weights embeddings via sentence-transformers; the open arm of the
    embedding axis. Defaults to bge-small-en-v1.5: the axis asks "proprietary
    vs open", not "best open model", and the small model keeps the sweep cheap.
    """

    model: str = "BAAI/bge-small-en-v1.5"
    batch_size: int = 64
    device: str | None = None

    def __post_init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        self._model = SentenceTransformer(self.model, device=self.device)

    @property
    def id(self) -> str:
        return f"st:{self.model}"

    @property
    def dim(self) -> int:
        return int(self._model.get_embedding_dimension())

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed in order, L2-normalized, float32."""
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #

@dataclass
class DenseRetriever:
    """Exhaustive dot-product search; the index is Vx 2-D and small by design."""

    index: Index
    embedder: Embedder
    source: str = "dense"

    def retrieve(
        self,
        query: str,
        *,
        k_recall: int,
        published_before: datetime | None = None,
    ) -> list[Retrieved]:
        if published_before is not None and published_before.tzinfo is None:
            raise ValueError("published_before must be timezone-aware")

        query_vector = self.embedder.embed([query])[0]
        scores = self.index.vectors @ query_vector.astype(np.float32)

        if published_before is not None:
            # Chunks with unknown publication dates are excluded when filtering.
            allowed = np.fromiter(
                (
                    c.published_at is not None and c.published_at <= published_before
                    for c in self.index.chunks
                ),
                dtype=bool,
                count=len(self.index.chunks),
            )
            if not allowed.any():
                return []
            scores = np.where(allowed, scores, -np.inf)

        k = min(k_recall, len(self.index.chunks))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        top = top[np.isfinite(scores[top])]

        return [
            Retrieved(
                chunk=self.index.chunks[ix], score=float(scores[ix]), rank=rank, source=self.source
            )
            for rank, ix in enumerate(top, start=1)
        ]


@dataclass
class BM25Retriever:
    """
    Lexical retrieval (bm25s) over the same ``Index``, so the dense and lexical
    arms of the hybrid share one chunking and one set of chunk ids.

    The index's vectors are unused here; the shared index exists so hybrid
    fusion can address identical chunks by ``chunk_id``.
    """

    index: Index
    k1: float = 1.5
    b: float = 0.75
    stopwords: str | None = "en"
    source: str = "bm25"

    def __post_init__(self) -> None:
        import bm25s

        self._bm25s = bm25s
        corpus_tokens = bm25s.tokenize(
            [chunk.text for chunk in self.index.chunks],
            stopwords=self.stopwords,
            show_progress=False,
        )
        self._bm25 = bm25s.BM25(k1=self.k1, b=self.b)
        self._bm25.index(corpus_tokens, show_progress=False)

    def retrieve(
        self,
        query: str,
        *,
        k_recall: int,
        published_before: datetime | None = None,
    ) -> list[Retrieved]:
        if published_before is not None and published_before.tzinfo is None:
            raise ValueError("published_before must be timezone-aware")

        chunks = self.index.chunks
        k = min(k_recall, len(chunks))
        query_tokens = self._bm25s.tokenize(
            query, stopwords=self.stopwords, show_progress=False
        )
        # Filter *before* truncating to k. Retrieving k and then dropping
        # filtered hits can return fewer than k results even when more
        # qualifying documents exist, while DenseRetriever filters globally —
        # the two legs would not be comparable under the recency axis.
        results, scores = self._bm25.retrieve(
            query_tokens, k=len(chunks), show_progress=False
        )

        out: list[Retrieved] = []
        for ix, score in zip(results[0], scores[0], strict=True):
            chunk = chunks[int(ix)]
            if (
                published_before is not None
                and (chunk.published_at is None or chunk.published_at > published_before)
            ):
                continue
            out.append(
                Retrieved(
                    chunk=chunk, score=float(score), rank=len(out) + 1, source=self.source
                )
            )
            if len(out) >= k:
                break
        return out


@dataclass
class CrossEncoderReranker:
    """
    Cross-encoder reranker. ``score`` is the model's raw logit (not a
    probability and not comparable across models) — only the induced order is
    meaningful. ``source`` is preserved from the retrieval leg so the reranked
    output still says which leg found the chunk.
    """

    model: str = "BAAI/bge-reranker-v2-m3"
    batch_size: int = 32
    device: str | None = None

    def __post_init__(self) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(self.model, device=self.device)

    @property
    def id(self) -> str:
        return f"ce:{self.model}"

    def rerank(
        self, query: str, candidates: list[Retrieved], k_final: int
    ) -> list[Retrieved]:
        if not candidates:
            return []
        scores = np.asarray(
            self._model.predict(
                [(query, item.chunk.text) for item in candidates],
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
        )
        order = np.argsort(-scores, kind="stable")[:k_final]
        return [
            Retrieved(
                chunk=candidates[int(i)].chunk,
                score=float(scores[int(i)]),
                rank=rank,
                source=candidates[int(i)].source,
            )
            for rank, i in enumerate(order, start=1)
        ]


@dataclass
class RRFFusion:
    """
    Hybrid retrieval as reciprocal rank fusion over N legs.

    Scores from different retrievers are not comparable; ranks are. Ties break
    on ``chunk_id`` so the output is deterministic across runs.
    """

    retrievers: list[Retriever]
    k_rrf: int = 60

    def __post_init__(self) -> None:
        if len(self.retrievers) < 2:
            raise ValueError("RRF needs at least two legs")

    def retrieve(
        self,
        query: str,
        *,
        k_recall: int,
        published_before: datetime | None = None,
    ) -> list[Retrieved]:
        fused: dict[str, tuple[float, Retrieved]] = {}
        for leg in self.retrievers:
            for item in leg.retrieve(
                query, k_recall=k_recall, published_before=published_before
            ):
                gain = 1.0 / (self.k_rrf + item.rank)
                key = item.chunk.chunk_id
                prev = fused.get(key)
                if prev is None:
                    fused[key] = (gain, item)
                else:
                    fused[key] = (prev[0] + gain, prev[1])

        ranked = sorted(fused.items(), key=lambda kv: (-kv[1][0], kv[0]))
        return [
            Retrieved(chunk=item.chunk, score=score, rank=rank, source="fused")
            for rank, (_, (score, item)) in enumerate(ranked[:k_recall], start=1)
        ]


def truncate_contexts(
    retrieved: list[Retrieved],
    *,
    k_final: int | None = None,
    char_budget: int | None = None,
) -> list[Retrieved]:
    """
    Turn candidates into the generator's context window.

    Both bounds are honored: truncate by rank first, then accumulate characters
    until the budget is spent. At least one chunk is always kept, otherwise the
    generator has nothing to ground on and the cell measures prompting, not
    retrieval.

    ``char_budget`` exists because chunk size and context budget are confounded:
    a fixed ``k_final`` hands ~512 characters per chunk at chunk_size=512 and
    ~2048 at 2048, so a "chunk size" ablation at fixed k is really a context
    budget ablation. Fix the budget to isolate the axis.
    """
    out = list(retrieved) if k_final is None else list(retrieved[:k_final])
    if char_budget is None:
        return out

    kept: list[Retrieved] = []
    used = 0
    for item in out:
        n = len(item.chunk.text)
        if kept and used + n > char_budget:
            break
        kept.append(item)
        used += n
    return kept


def context_texts(retrieved: list[Retrieved]) -> list[str]:
    """The context strings as the generator and the judges see them."""
    return [item.chunk.text for item in retrieved]


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #

GENERATOR_TEMPLATE = """Answer the question using only the context below. \
If the context does not contain the answer, reply exactly: insufficient information

Context:
{context}

Question: {query}

Answer:"""


def render_contexts(contexts: list[str], order: str = "as_retrieved") -> str:
    """
    Render contexts for a prompt, numbering matches the judge templates' [n]
    convention so generator and judge see the same layout.

    ``order`` is a parameter because position is an experimental variable
    (lost-in-the-middle); "middle" ordering arrives with Phase 2's corpus.
    """
    items = list(contexts)
    if order == "reversed":
        items = items[::-1]
    elif order != "as_retrieved":
        raise ValueError(f"unknown context order {order!r}")
    return "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(items))


@dataclass
class PromptGenerator:
    """
    OpenRouter chat completion over a fixed template, matching the judge client
    convention. Lock ``model`` for the whole sweep: varying it confounds every
    retrieval axis.
    """

    model: str = "openai/gpt-4o-mini"
    temperature: float = 0.0
    order: str = "as_retrieved"

    def __post_init__(self) -> None:
        from openai import OpenAI

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            default_headers={
                "HTTP-Referer": "http://localhost",
                "X-Title": "rag-eval-harness",
            },
        )

    @staticmethod
    def prompt_id() -> str:
        """Identity of the generator template, independent of model/client."""
        prompt_hash = hashlib.sha256(GENERATOR_TEMPLATE.encode()).hexdigest()[:12]
        return f"sha256:{prompt_hash}"

    @property
    def id(self) -> str:
        return (
            f"{self.model}:t={self.temperature}:prompt={self.prompt_id()}:order={self.order}"
        )

    def generate(self, query: str, contexts: list[str]) -> str:
        prompt = GENERATOR_TEMPLATE.format(
            context=render_contexts(contexts, self.order), query=query
        )
        response = _call_with_retries(
            lambda: self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            )
        )
        return (response.choices[0].message.content or "").strip()


# --------------------------------------------------------------------------- #
# Ablation cell identity
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """
    One cell of the ablation grid, and the key every result row is stamped with.

    Without a canonical id, a sweep becomes a directory of
    ``results_chunk256_v2_FINAL.json``.

    ``recency_cutoff`` *is* the metadata-filter axis: an ISO-8601 timestamp
    means "retrieve only from documents published before this instant", ``None``
    means the filter is off. MultiHop-RAG carries no per-query timestamp, so the
    cutoff is an experimental variable the sweep defines (e.g. the corpus median
    publication date), not a property the dataset supplies.
    """

    chunk_size: int
    chunk_overlap: int
    embedder: str
    retriever: str
    rerank: str | None
    recency_cutoff: str | None
    k_recall: int
    k_final: int | None
    char_budget: int | None
    generator: str
    generator_prompt: str
    judge: str = ""
    dataset_revision: str = ""

    @property
    def id(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

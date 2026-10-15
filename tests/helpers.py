"""Shared builders and fakes for the test suite. No network, no API keys."""

from __future__ import annotations

import hashlib

import numpy as np

from rag_eval.data import Document, EvidenceItem, Query
from rag_eval.pipeline import Chunk, PipelineConfig, Retrieved, parse_published_at


def make_doc(
    body: str,
    url: str = "https://example.com/1",
    title: str = "title",
    source: str = "source",
    published_at: str | None = None,
    author: str | None = None,
    category: str | None = None,
) -> Document:
    return Document(
        body=body,
        title=title,
        source=source,
        author=author,
        published_at=published_at,
        url=url,
        category=category,
    )


def make_chunk(
    text: str,
    chunk_id: str | None = None,
    doc_id: str = "doc1",
    start_char: int = 0,
    published_at: str | None = None,
    title: str | None = None,
) -> Chunk:
    end_char = start_char + len(text)
    return Chunk(
        chunk_id=chunk_id or f"{doc_id}:{start_char}-{end_char}",
        doc_id=doc_id,
        text=text,
        start_char=start_char,
        end_char=end_char,
        published_at=parse_published_at(published_at),
        title=title,
    )


def make_retrieved(
    text: str,
    rank: int = 1,
    score: float = 1.0,
    chunk_id: str | None = None,
    doc_id: str = "doc1",
    published_at: str | None = None,
    source: str = "dense",
) -> Retrieved:
    return Retrieved(
        chunk=make_chunk(text, chunk_id=chunk_id, doc_id=doc_id, published_at=published_at),
        score=score,
        rank=rank,
        source=source,
    )


def make_query(
    text: str = "who founded it",
    answer: str = "42",
    question_type: str = "inference",
    facts: tuple[str, ...] = ("fact one",),
    is_null: bool = False,
) -> Query:
    if is_null:
        question_type = "null_query"
        facts = ()
    evidence = [
        EvidenceItem(
            fact=fact, author=None, category=None, published_at=None, source=None, title=None, url=None
        )
        for fact in facts
    ]
    return Query(query=text, answer=answer, question_type=question_type, evidence_list=evidence)


def make_config(**overrides: object) -> PipelineConfig:
    """One ablation cell with defaults; any field can be overridden."""
    base = {
        "chunk_size": 1024,
        "chunk_overlap": 128,
        "embedder": "fake:embedder",
        "retriever": "dense",
        "rerank": None,
        "recency_cutoff": None,
        "k_recall": 10,
        "k_final": None,
        "char_budget": None,
        "generator": "fake:generator",
        "generator_prompt": "test:v1",
    }
    base.update(overrides)
    return PipelineConfig(**base)


class FakeEmbedder:
    """Deterministic, L2-normalized vectors derived from a text hash."""

    def __init__(self, id: str = "fake:embedder") -> None:
        self.id = id
        self.embed_calls = 0

    def embed(self, texts: list[str]) -> np.ndarray:
        self.embed_calls += 1
        rows = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()[:16]
            v = np.frombuffer(digest, dtype=np.uint8).astype(np.float32) - 127.5
            v = v - v.mean()
            norm = np.linalg.norm(v)
            rows.append(v / norm if norm else v)
        return np.asarray(rows, dtype=np.float32)


class DictEmbedder:
    """Fixed text -> vector map, so dot products are fully controlled."""

    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self.mapping = {text: np.asarray(v, dtype=np.float32) for text, v in mapping.items()}
        self.id = "fake:dict"

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray([self.mapping[t] for t in texts], dtype=np.float32)


class RecordingJudge:
    """Judge that records its (criterion, prompt) calls and returns scripted scores."""

    def __init__(self, values: dict[str, float] | None = None, default: float = 0.5) -> None:
        self.values = values or {}
        self.default = default
        self.calls: list[tuple[str, str]] = []
        self.model = "fake-judge"

    def score(self, criterion: str, prompt: str) -> float:
        self.calls.append((criterion, prompt))
        return self.values.get(criterion, self.default)


class StubRetriever:
    """Fixed candidates per query; records retrieve() calls."""

    def __init__(self, candidates_by_query: dict[str, list[Retrieved]]) -> None:
        self.candidates = candidates_by_query
        self.calls: list[tuple[str, int, object]] = []

    def retrieve(
        self,
        query: str,
        *,
        k_recall: int,
        published_before=None,
    ) -> list[Retrieved]:
        self.calls.append((query, k_recall, published_before))
        return self.candidates.get(query, [])[:k_recall]


class StubGenerator:
    def __init__(self, fn=None) -> None:
        self.fn = fn if fn is not None else (lambda query, contexts: "test answer")

    def generate(self, query: str, contexts: list[str]) -> str:
        return self.fn(query, contexts)
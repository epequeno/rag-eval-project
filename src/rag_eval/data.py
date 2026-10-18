from dataclasses import dataclass

from datasets import load_dataset

DATASET_ID = "yixuantt/MultiHopRAG"
# Pinned so rows, index fingerprints and reports refer to one immutable corpus.
# 71ac0d0 is the revision the Phase 1-3 data was collected from.
DATASET_REVISION = "71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82"


@dataclass
class EvidenceItem:
    fact: str
    author: str | None
    category: str | None
    published_at: str | None
    source: str | None
    title: str | None
    url: str | None


@dataclass
class Query:
    query: str
    answer: str
    question_type: str
    evidence_list: list[EvidenceItem]

    @property
    def gold_facts(self) -> list[str]:
        return [e.fact for e in self.evidence_list]

    @property
    def is_null(self) -> bool:
        return self.question_type == "null_query"


@dataclass
class Document:
    body: str
    title: str
    source: str
    author: str | None
    published_at: str | None
    url: str
    category: str | None

    @property
    def doc_id(self) -> str:
        """Stable identity for a corpus document: `url` (609/609 unique)."""
        return self.url


def load_queries(
    split: str = "train", exclude_null: bool = False, revision: str = DATASET_REVISION
) -> list[Query]:
    """
    Load MultiHop-RAG queries with gold evidence
    """
    ds = load_dataset(DATASET_ID, "MultiHopRAG", revision=revision)
    ds = ds[split]

    queries = []
    for row in ds:
        if exclude_null and row["question_type"] == "null_query":
            continue

        evidence = [
            EvidenceItem(
                fact=e["fact"],
                author=e.get("author"),
                category=e.get("category"),
                published_at=e.get("published_at"),
                source=e.get("source"),
                title=e.get("title"),
                url=e.get("url"),
            )
            for e in row["evidence_list"]
        ]

        queries.append(
            Query(
                query=row["query"],
                answer=row["answer"],
                question_type=row["question_type"],
                evidence_list=evidence,
            )
        )

    return queries


def load_corpus(split: str = "train", revision: str = DATASET_REVISION) -> list[Document]:
    """
    Load MultiHop-RAG corpus
    """
    ds = load_dataset(DATASET_ID, "corpus", revision=revision)
    ds = ds[split]

    documents = []
    for row in ds:
        documents.append(
            Document(
                body=row["body"],
                title=row["title"],
                source=row["source"],
                author=row.get("author"),
                published_at=row.get("published_at"),
                url=row["url"],
                category=row.get("category"),
            )
        )

    return documents

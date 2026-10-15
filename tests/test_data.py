"""Data model contracts, offline (the loaders themselves hit the network)."""

from rag_eval.data import Document, EvidenceItem, Query


def make_evidence(fact: str) -> EvidenceItem:
    return EvidenceItem(
        fact=fact, author=None, category=None, published_at=None, source=None, title=None, url=None
    )


def test_document_doc_id_is_the_url():
    doc = Document(
        body="b",
        title="t",
        source="s",
        author=None,
        published_at=None,
        url="https://example.com/1",
        category=None,
    )
    assert doc.doc_id == "https://example.com/1"


def test_query_gold_facts_and_null_flag():
    q = Query(
        query="q",
        answer="a",
        question_type="comparison",
        evidence_list=[make_evidence("f1"), make_evidence("f2")],
    )
    assert q.gold_facts == ["f1", "f2"]
    assert q.is_null is False

    null = Query(
        query="q",
        answer="Insufficient information.",
        question_type="null_query",
        evidence_list=[],
    )
    assert null.is_null is True
    assert null.gold_facts == []
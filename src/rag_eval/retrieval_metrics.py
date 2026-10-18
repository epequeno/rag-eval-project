"""Deterministic retrieval metrics computed from gold-fact containment.

Units
-----
Every metric treats the retrieval list as an ordered list of **chunks** (the
unit the harness ranks). The dataset's own ``retrieval_evaluate.py`` ranks
*documents* and reports Hits@4, Hits@10, MAP@10 and MRR@10; the formulas here
match those at the chunk level, so the numbers are comparable in shape but not
directly comparable to the paper's document-level table.

Why nDCG is fact-level
----------------------
A chunk can contain more than one gold fact, and overlapping chunks can repeat
the same fact. Counting each relevant chunk once (the obvious binary nDCG)
makes the score depend on chunk granularity — one chunk holding two facts scores
0.5 against an ideal of 2 — and lets duplicate chunks exceed 1.0. The gain at a
rank here is the number of **previously unseen** gold facts that chunk
contributes, and each fact's ideal discount is its own successive rank. That is
bounded in [0, 1], invariant to chunk granularity, and still rank-sensitive.
"""

import math

from rag_eval.normalize import normalize


def is_relevant(chunk: str, gold_facts: list[str]) -> bool:
    norm_chunk = normalize(chunk)
    for fact in gold_facts:
        if normalize(fact) in norm_chunk:
            return True
    return False


def hits_in_chunk(chunk: str, gold_facts: list[str]) -> set[int]:
    norm_chunk = normalize(chunk)
    ixs = set()
    for ix, fact in enumerate(gold_facts):
        if normalize(fact) in norm_chunk:
            ixs.add(ix)
    return ixs


def recall_at_k(retrieved: list[str], gold_facts: list[str], k: int) -> float:
    """Fraction of gold facts contained in the top-k chunks."""
    found: set[int] = set()
    for chunk in retrieved[:k]:
        found |= hits_in_chunk(chunk, gold_facts)
    return len(found) / len(gold_facts) if gold_facts else 0.0


def hits_at_k(retrieved: list[str], gold_facts: list[str], k: int) -> float:
    """1.0 if any gold fact is contained in the top-k chunks, else 0.0.

    Matches the dataset's Hits@k definition (any hit), at chunk granularity.
    """
    return 1.0 if any(is_relevant(chunk, gold_facts) for chunk in retrieved[:k]) else 0.0


def map_at_k(retrieved: list[str], gold_facts: list[str], k: int) -> float:
    """Average precision over gold facts, each fact counted once.

    Mirrors ``retrieval_evaluate.calculate_metrics``: at each relevant rank,
    precision is (newly found facts) / rank, and AP is normalised by
    ``min(len(gold), k)``.
    """
    if not gold_facts:
        return 0.0
    found: set[int] = set()
    precision_sum = 0.0
    for rank, chunk in enumerate(retrieved[:k], start=1):
        new = hits_in_chunk(chunk, gold_facts) - found
        if new:
            found |= new
            precision_sum += len(new) / rank
    return precision_sum / min(len(gold_facts), k)


def mrr(retrieved: list[str], gold_facts: list[str], k: int | None = None) -> float:
    """Reciprocal rank of the first chunk containing any gold fact."""
    chunks = retrieved[:k] if k is not None else retrieved
    for rank, chunk in enumerate(chunks, start=1):
        if is_relevant(chunk, gold_facts):
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], gold_facts: list[str], k: int) -> float:
    """Fact-level nDCG@k: discounted recall over facts, bounded in [0, 1].

    Gain at a rank is the number of gold facts that chunk contributes for the
    first time. The ideal places every fact at rank 1 — the best a single chunk
    can do — so ``nDCG = sum_f 1/log2(rank_f + 1) / n_facts``. A chunk holding
    every fact scores 1.0; duplicate chunks add nothing; splitting facts across
    ranks discounts them. The earlier per-chunk definition normalised by
    ``min(n_facts, k)`` successive ranks while counting a multi-fact chunk once,
    which both depressed perfect retrievals to 0.5 and allowed duplicate chunks
    to exceed 1.0.
    """
    if not gold_facts or k <= 0:
        return 0.0
    found: set[int] = set()
    dcg = 0.0
    for rank, chunk in enumerate(retrieved[:k], start=1):
        new = hits_in_chunk(chunk, gold_facts) - found
        if new:
            found |= new
            dcg += len(new) / math.log2(rank + 1)

    # All facts at rank 1 is the ideal, so the normaliser is the fact count.
    return dcg / len(gold_facts)


def evaluate_retrieval(
    retrieved: list[str],
    gold_facts: list[str],
    ks: list[int] | None = None,
) -> dict[str, float]:
    if ks is None:
        ks = [1, 4, 10]
    result: dict[str, float] = {}
    for k in ks:
        result[f"recall@{k}"] = recall_at_k(retrieved, gold_facts, k)
        result[f"hits@{k}"] = hits_at_k(retrieved, gold_facts, k)
        result[f"ndcg@{k}"] = ndcg_at_k(retrieved, gold_facts, k)
        result[f"map@{k}"] = map_at_k(retrieved, gold_facts, k)
    result["mrr"] = mrr(retrieved, gold_facts)
    return result

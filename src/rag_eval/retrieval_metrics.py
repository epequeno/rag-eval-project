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
    top_k = retrieved[:k]
    found = set()
    for chunk in top_k:
        found |= hits_in_chunk(chunk, gold_facts)
    return len(found) / len(gold_facts) if gold_facts else 0.0


def mrr(retrieved: list[str], gold_facts: list[str], k: int | None = None) -> float:
    chunks = retrieved[:k] if k else retrieved
    for rank, chunk in enumerate(chunks, start=1):
        if is_relevant(chunk, gold_facts):
            return 1.0 / rank
    return 0.0


# https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG
def ndcg_at_k(retrieved: list[str], gold_facts: list[str], k: int) -> float:
    top_k = retrieved[:k]
    dcg = 0.0
    for rank, chunk in enumerate(top_k, start=1):
        if is_relevant(chunk, gold_facts):
            dcg += 1.0 / math.log2(rank + 1)

    num_relevant = len(gold_facts)
    ideal_hits = min(num_relevant, k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval(
    retrieved: list[str],
    gold_facts: list[str],
    ks: list[int] | None = None,
) -> dict[str, float]:
    if ks is None:
        ks = [1, 4, 10]
    result = {}
    for k in ks:
        result[f"recall@{k}"] = recall_at_k(retrieved, gold_facts, k)
        result[f"ndcg@{k}"] = ndcg_at_k(retrieved, gold_facts, k)
    result["mrr"] = mrr(retrieved, gold_facts)
    return result

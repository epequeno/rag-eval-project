from rag_eval.judges import Judge

END_TO_END_METRIC_NAMES = (
    "faithfulness",
    "answer_relevance",
    "context_precision",
    "answer_correctness",
)

FAITHFULNESS_TEMPLATE = """You are evaluating a RAG system's answer for faithfulness.

Query: {query}

Retrieved contexts:
{contexts}

Answer to evaluate: {answer}

Faithfulness means every claim in the answer is supported by the retrieved contexts.
Ignore whether the answer is actually correct — only check if it's grounded in the contexts.

Return a single float between 0.0 (completely unfaithful) and 1.0 (completely faithful)."""

ANSWER_RELEVANCE_TEMPLATE = """You are evaluating a RAG system's answer for relevance.

Query: {query}

Answer to evaluate: {answer}

Answer relevance means the answer actually addresses what the query asks.
Ignore whether the answer is factually correct — only check if it addresses the query.

Return a single float between 0.0 and 1.0."""

CONTEXT_PRECISION_TEMPLATE = """You are evaluating the retrieved contexts for a RAG system.

Query: {query}

Retrieved contexts:
{contexts}

Context precision means: how many of the retrieved contexts are actually relevant
to answering the query? Ignore whether the answer is correct — only assess the contexts.

Return a single float between 0.0 and 1.0."""

ANSWER_CORRECTNESS_TEMPLATE = """You are evaluating a RAG system's answer for correctness.

Query: {query}

Gold (reference) answer: {gold_answer}

Answer to evaluate: {answer}

Answer correctness means the answer matches the gold answer factually.
Allow for minor wording differences but not factual disagreements.

Return a single float between 0.0 and 1.0."""


def faithfulness(query: str, contexts: list[str], answer: str, judge: Judge) -> float:
    """
    Is every claim in the answer supported by the contexts?
    """
    prompt = FAITHFULNESS_TEMPLATE.format(
        query=query,
        answer=answer,
        contexts="\n\n".join(f"[{ix + 1}] {c}" for ix, c in enumerate(contexts)),
    )
    return judge.score("faithfulness", prompt)


def answer_relevance(query: str, answer: str, judge: Judge) -> float:
    """
    does the answer address the query?
    """
    prompt = ANSWER_RELEVANCE_TEMPLATE.format(query=query, answer=answer)
    return judge.score("answer_relevance", prompt)


def context_precision(query: str, contexts: list[str], judge: Judge) -> float:
    """
    How many of the retrieved context are relevant to the query?
    """
    prompt = CONTEXT_PRECISION_TEMPLATE.format(
        query=query,
        contexts="\n\n".join(f"[{ix + 1}] {c}" for ix, c in enumerate(contexts)),
    )
    return judge.score("context_precision", prompt)


def answer_correctness(
    query: str, answer: str, gold_answer: str, judge: Judge
) -> float:
    """
    Is the answer factually correct compared to the gold answer?
    """
    prompt = ANSWER_CORRECTNESS_TEMPLATE.format(
        query=query, gold_answer=gold_answer, answer=answer
    )
    return judge.score("answer_correctness", prompt)


def evaluate_end_to_end(
    query: str, contexts: list[str], answer: str, gold_answer: str, judge: Judge
) -> dict[str, float]:
    """
    compute all four end-to-end metrics for one query. Per-instance
    """
    return {
        "faithfulness": faithfulness(query, contexts, answer, judge),
        "answer_relevance": answer_relevance(query, answer, judge),
        "context_precision": context_precision(query, contexts, judge),
        "answer_correctness": answer_correctness(query, answer, gold_answer, judge),
    }

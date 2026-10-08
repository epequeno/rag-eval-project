from rag_eval.data import load_corpus, load_queries


def main():
    q = load_queries()

    print(f"first query: {q[0].query[:80]}...")
    print(f"first answer: {q[0].answer}")
    print(f"first gold facts: {len(q[0].gold_facts)}")
    print(f"first fact: {q[0].gold_facts[0][:100]}...")
    print(f"question types: { {qq.question_type for qq in q} }")

    q_non_null = load_queries(exclude_null=True)
    print(f"{len(q_non_null)} queries (non-null)")

    c = load_corpus()
    print(f"{len(c)} documents")
    print(f"first doc title: {c[0].title}")
    print(f"first doc body[:100]: {c[0].body[:100]}...")

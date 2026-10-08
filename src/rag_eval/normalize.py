def normalize(text: str) -> str:
    """
    Normalize text for substring matching, matching MultiHop-RAG's eval.
    """
    return text.replace(" ", "").replace("\n", "")

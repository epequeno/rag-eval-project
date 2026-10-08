import hashlib
import json
import os
from pathlib import Path
from typing import Protocol


class Judge(Protocol):
    """
    A judge that evaluates a single criterion and returns a score
    """

    def score(self, criterion: str, prompt: str) -> float:
        """
        Return a score in [0,1] for the given criterion and prompt
        """
        ...


class CachedJudge:
    """
    Wraps any judge with disk-based caching
    """

    def __init__(
        self,
        judge: Judge,
        model_id: str | None = None,
        cache_dir: Path | str = ".judge_cache",
    ):
        self.judge = judge
        self.model_id = model_id or getattr(judge, "model", "unkown ")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, criterion: str, prompt: str) -> str:
        h = hashlib.sha256()
        h.update(self.model_id.encode())
        h.update(criterion.encode())
        h.update(prompt.encode())
        return h.hexdigest()

    def score(self, criterion: str, prompt: str) -> float:
        key = self._cache_key(criterion, prompt)
        cache_file = self.cache_dir / f"{key}.json"

        # check cache
        if cache_file.exists():
            return json.loads(cache_file.read_text())["score"]

        # Miss - call underlying judge
        result = self.judge.score(criterion, prompt)
        cache_file.write_text(json.dumps({"score": result}))
        return result


class OpenRouterJudge:
    """
    Judge that uses an OpenRouter model to score criterion
    """

    def __init__(self, model: str = "openai/gpt-4o-mini"):
        self.model = model
        from openai import OpenAI

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            default_headers={
                "HTTP-Referer": "http://localhost",
                "X-Title": "rag-eval-harness",
            },
        )

    def score(self, criterion: str, prompt: str) -> float:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": f"You are a RAG evaluation judge. Score {criterion} on a scale of 0.0 to 1.0. Return ONLY the numeric score, nothing else.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        text = response.choices[0].message.content.strip()
        return float(text)

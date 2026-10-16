import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Protocol

_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def _call_with_retries(fn, *, attempts: int = 3, sleep: float = 1.0):
    """
    Retry a provider call on transient transport failures (DNS blips, timeouts,
    429, 5xx). Observed 2026-10-14: a `Temporary failure in name resolution`
    killed a 23-hour sweep run outright because the client's own retries
    exhausted; one retry layer here keeps a whole job alive through a hiccup.
    """
    import openai

    retryable = (
        openai.APIConnectionError,
        openai.APITimeoutError,
        openai.RateLimitError,
        openai.InternalServerError,
    )
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except retryable as exc:
            last = exc
            if attempt == attempts - 1:
                break
            time.sleep(sleep * (2**attempt))
    assert last is not None
    raise last


def _parse_judge_score(text: str, criterion: str) -> float:
    """
    Extract the judge's score from a model response.

    Models are instructed to return only a number but (observed with
    claude-haiku-4.5 on 2026-10-13) sometimes append an explanation. Take the
    first number in [0, 1] — numbers like "3 contexts" are outside the scale
    and skipped. If nothing in range parses, raise rather than fabricate:
    a silent out-of-range score would poison every downstream metric.
    """
    for match in _NUMBER.findall(text):
        value = float(match)
        if 0.0 <= value <= 1.0:
            return value
    raise ValueError(
        f"judge returned no score in [0,1] for criterion {criterion!r}: {text!r}"
    )


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

    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        temperature: float = 0.0,
        max_output_tokens: int = 32,
    ):
        self.model = model
        self.temperature = temperature
        # claude-haiku-4.5 ignores "return only the number" and drafts prose
        # before the score; 32 tokens is enough for models that comply, longer
        # for ones that preamble first (the parser takes the first number in
        # [0,1], so the preamble is harmless once the number is in range).
        self.max_output_tokens = max_output_tokens
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
        response = _call_with_retries(
            lambda: self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": f"You are a RAG evaluation judge. Score {criterion} on a scale of 0.0 to 1.0. Return ONLY the numeric score, nothing else.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
            )
        )
        text = response.choices[0].message.content.strip()
        return _parse_judge_score(text, criterion)

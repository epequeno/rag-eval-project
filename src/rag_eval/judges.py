import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Protocol

# A decimal in [0, 1] is the preferred shape: the judge is asked for a float.
_DECIMAL = re.compile(r"\d+\.\d+")
# An explicitly labelled score wins over any other number in the text.
_LABELLED = re.compile(r"(?:score|rating)\s*[:=]?\s*(\d+(?:\.\d+)?)", re.IGNORECASE)
# A response that is only a number ("1", "0.0") is unambiguous.
_ONLY_NUMBER = re.compile(r"\d+(?:\.\d+)?")
# A number followed by a word boundary, for extracting all candidates.
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


def _in_range(value: str) -> bool:
    try:
        return 0.0 <= float(value) <= 1.0
    except ValueError:
        return False


def _parse_judge_score(text: str, criterion: str) -> float:
    """
    Extract the judge's score from a model response.

    Models are instructed to return only a number but sometimes draft prose
    first. The first in-range number is *not* safe: "I found 1 claim supported
    out of 3. Score: 0.33" would parse as 1.0, turning an explanation into a
    perfect score. Resolution order:

    1. the last number explicitly labelled ``score:``/``rating:``;
    2. the last decimal in [0, 1] (counts are integers, so they cannot win);
    3. the whole response, if it is only a number in [0, 1];
    4. raise.

    A silent out-of-range or preamble-derived score would poison every
    downstream metric, so failing loudly is the point.
    """
    labelled = [match for match in _LABELLED.findall(text) if _in_range(match)]
    if labelled:
        return float(labelled[-1])

    decimals = [match for match in _DECIMAL.findall(text) if _in_range(match)]
    if decimals:
        return float(decimals[-1])

    stripped = text.strip()
    if _ONLY_NUMBER.fullmatch(stripped) and _in_range(stripped):
        return float(stripped)

    raise ValueError(
        f"judge returned no score in [0,1] for criterion {criterion!r}: {text!r}"
    )


def _param_key(judge) -> dict:
    """Parameters that change a judge's output but are not in the prompt."""
    params = {}
    for name in ("temperature", "max_output_tokens", "top_p", "seed"):
        value = getattr(judge, name, None)
        if value is not None:
            params[name] = value
    return params


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
    Wraps any judge with disk-based caching.

    The key includes the model identity, the criterion, the prompt *and* the
    judge parameters: two judges of the same model at different temperatures
    must not share an entry. Each entry records the raw completion and the
    parameters alongside the parsed score, because a parser change must remain
    auditable — the original text is the evidence, not the float.
    """

    def __init__(
        self,
        judge: Judge,
        model_id: str | None = None,
        cache_dir: Path | str = ".judge_cache",
        params: dict | None = None,
    ):
        self.judge = judge
        self.model_id = model_id or getattr(judge, "model", None) or type(judge).__name__
        self.params = _param_key(judge) if params is None else params
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, criterion: str, prompt: str) -> str:
        h = hashlib.sha256()
        h.update(self.model_id.encode())
        h.update(b"\x00")
        h.update(criterion.encode())
        h.update(b"\x00")
        h.update(json.dumps(self.params, sort_keys=True, default=str).encode())
        h.update(b"\x00")
        h.update(prompt.encode())
        return h.hexdigest()

    def score(self, criterion: str, prompt: str) -> float:
        key = self._cache_key(criterion, prompt)
        cache_file = self.cache_dir / f"{key}.json"

        if cache_file.exists():
            try:
                payload = json.loads(cache_file.read_text())
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict) and "score" in payload:
                return payload["score"]

        result = self.judge.score(criterion, prompt)
        payload = {
            "score": result,
            "raw": getattr(self.judge, "last_raw", None),
            "model": self.model_id,
            "params": self.params,
            "criterion": criterion,
        }
        # A crash mid-write must not leave a truncated JSON file that raises on
        # every later read; write to a sibling then replace atomically.
        fd, tmp_name = tempfile.mkstemp(dir=self.cache_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump(payload, handle)
            os.replace(tmp_name, cache_file)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
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
        # for ones that preamble first.
        self.max_output_tokens = max_output_tokens
        self.last_raw: str | None = None
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
        """
        Score once, and if the model spent the token budget on a preamble and
        the number never arrived, retry with more headroom before giving up.

        Observed 2026-10-17: gpt-4o-mini opened a faithfulness judgment with a
        paragraph of reasoning and was cut off at 32 tokens with no number,
        which aborted the run. A second call with 256 tokens is cheap and keeps
        a long sweep alive; the final failure still raises rather than
        fabricating a score.
        """
        budgets = sorted({self.max_output_tokens, 256, 1024})
        last_error: ValueError | None = None
        for budget in budgets:
            response = _call_with_retries(
                lambda budget=budget: self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": f"You are a RAG evaluation judge. Score {criterion} on a scale of 0.0 to 1.0. Return ONLY the numeric score, nothing else.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=budget,
                )
            )
            text = (response.choices[0].message.content or "").strip()
            self.last_raw = text
            try:
                return _parse_judge_score(text, criterion)
            except ValueError as exc:
                last_error = exc
                if budget >= budgets[-1]:
                    break
        assert last_error is not None
        raise last_error

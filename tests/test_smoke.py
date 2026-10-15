"""The Phase 0 exit criterion, as a regression test: scripts/smoke.py's full
metric path run with fake components, so it needs no API keys and no network."""

import smoke
from helpers import (
    FakeEmbedder,
    RecordingJudge,
    make_doc,
    make_query,
)

DOCS = [
    make_doc("The company was founded in 1999 by Alice.", url="u1"),
    make_doc("It raised 10 million from investors.", url="u2"),
]

QUERIES = [
    make_query(text="Who founded the company?", facts=("The company was founded in 1999.",)),
    make_query(text="What funding was raised?", facts=("It raised 10 million.",)),
    make_query(text="null query", is_null=True),
]


class RefusingGenerator:
    def generate(self, query, contexts):
        return "Insufficient information." if "null" in query else "The answer is 42."


def test_smoke_main_runs_the_full_path_with_fakes(tmp_path, capsys):
    providers = {
        "embedder": FakeEmbedder(),
        "generator": RefusingGenerator(),
        "judge": RecordingJudge(),
    }
    rc = smoke.main(
        limit=10,
        cache_dir=str(tmp_path / "idx"),
        providers=providers,
        docs=DOCS,
        queries=QUERIES,
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "docs=2" in out
    assert "recall@1" in out and "recall@10" in out
    assert "mrr" in out
    assert "faithfulness" in out and "answer_correctness" in out
    assert "null queries: 1  refusal accuracy: 1.000" in out
    assert "context window:" in out


def test_smoke_main_requires_keys_when_no_providers(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    rc = smoke.main(limit=1, cache_dir=str(tmp_path / "x"), providers={})
    assert rc == 2
    assert "missing env vars: OPENAI_API_KEY, OPENROUTER_API_KEY" in capsys.readouterr().err
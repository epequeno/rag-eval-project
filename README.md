# rag-eval-project

An evaluation-first RAG project: a three-part series that measures a RAG pipeline carefully, then
shows what the metrics miss, then questions the measuring tools themselves.

**Evaluation is the project. The pipeline is deliberately boring.** Building "embed → vector DB →
chatbot" demonstrates that you can follow a tutorial. Knowing where the system breaks, why, and
what each metric actually measures is the part worth writing down.

## The arc

| # | Post | Question it answers |
|---|---|---|
| 1 | I measured RAG carefully — here's what the numbers said | Which knobs actually move quality, and by how much? |
| 2 | The metrics from Post 1 are blind to most real failures | Which metric catches which engineered failure? |
| 3 | And the eval tools themselves are unreliable | How far can an LLM-as-judge be trusted? |

Each post escalates the skepticism: the numbers lie → the metrics are blind → the measuring tools
are unreliable.

## Status

| Phase | What | State |
|---|---|---|
| 0 | Harness foundation | **built** — metric API, adapters, loaders, smoke + sweep drivers |
| 1 | Ablation study (Post 1) | not started |
| 2 | Failure-mode taxonomy (Post 2) | not started |
| 3 | Judge meta-evaluation (Post 3) | not started |

No results are committed. `results/` is gitignored until a run is reproducible from cached raw
outputs, and Phase 0 has no test suite yet — see [Tests](#tests).

## What is here

```
src/rag_eval/
  pipeline.py            adapters — the seam between the harness and any RAG pipeline
                         (Chunk, Chunker, Embedder, Retriever, Reranker, Generator, Index)
  harness.py             turns a pipeline's output into per-instance metric rows
  retrieval_metrics.py   recall@k, MRR, nDCG@k — deterministic, computed from gold contexts
  end_to_end_metrics.py  faithfulness, answer relevance, context precision, answer correctness
  judges.py              Judge protocol + CachedJudge + OpenRouterJudge
  aggregate.py           bootstrap confidence intervals, effect sizes
  data.py                MultiHop-RAG and controlled-corpus loaders
  normalize.py           the single whitespace normalizer used for all gold matching

scripts/
  smoke.py               Phase 0 exit criterion: one config, the full metric path,
                         printed as mean ± bootstrap 95% CI
  sweep.py               Phase 1 driver: one axis varied at a time, indexes reused across cells,
                         rows written to results/<config_id>.jsonl

docs/
  scratch.ipynb          working notebook
```

## Design commitments

These are the decisions that make the later posts cheap rather than expensive rewrites:

- **Per-instance arrays, never early aggregation.** Aggregation and CIs are a separate, swappable
  step. Aggregate inside the metric and every Post 3 judge-swap becomes a full re-run.
- **The judge is a parameter, not a global.** Anything that scores an answer takes the judge as an
  argument, so Post 3 can swap it without touching the harness.
- **Judge caching from the start**, keyed on `(model, prompt template hash, inputs)`.
- **One whitespace normalizer**, matching the dataset's, so numbers stay comparable to the paper's.
- **The harness never owns the pipeline.** It takes `(query, retrieved_contexts, answer,
  gold_answer, gold_contexts)` and returns metrics; the pipeline stays someone else's problem.

## Data

[MultiHop-RAG](https://huggingface.co/datasets/yixuantt/MultiHopRAG) (ODC-BY, COLM 2024): 609 news
articles carrying `title`, `source` and `published_at`, with fact-string gold evidence and four
query types — inference (816), comparison (856), temporal (583), null (301).

Chosen over HotpotQA for two reasons. The gold is **fact strings matched by substring containment**,
so it does not move when the chunker changes — smaller chunks that split a fact stop matching, and
that is a real finding about chunk size rather than a broken label. And 2023 news carries far less
pretraining-contamination risk than Wikipedia.

The corpus is not redistributed here; `data.py` downloads it. Null queries (11.8%) test refusal
robustness, temporal queries stress time-aware retrieval.

## Running it

Requires [`uv`](https://docs.astral.sh/uv/), Python ≥ 3.14, `OPENAI_API_KEY` (embeddings) and
`OPENROUTER_API_KEY` (generator + judge). **No keys live in this repository.**

```bash
uv sync
uv run python scripts/smoke.py --limit 20     # Phase 0 path, prints intervals rather than point estimates
uv run python scripts/sweep.py --limit 100    # Phase 1, one axis at a time, writes results/<config_id>.jsonl
```

Both drivers build their judge, embedder and generator through module-level factories, so
substitutes can exercise the same code path with no network access — that seam exists for tests
that have not been written yet.

## Tests

None yet. `pytest` is configured as a dev dependency and the seam above is already in place, but no
test file has been written. This is stated here rather than left implied.

## Authorship

This is a pedagogical project: **I write the implementation.** Agents advise — explaining concepts,
reviewing code I wrote, checking a claim against a source, pointing at literature, planning. They do
not produce the harness or the prose. The deliverables are *explained* artifacts — blog posts, ADRs,
interview answers — and explaining RRF versus weighted fusion requires having built it.

## License

No `LICENSE` file is included, so the default applies: all rights reserved. The plan and the metric
definitions are here to be read. The end-to-end metric names follow the RAGAS family by convention.

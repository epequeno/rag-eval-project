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
| 1 | Ablation study (Post 1) | **data complete** — 10 cells × 2,556 queries in `results/` (gitignored), all axes incl. rerank |
| 2 | Failure-mode taxonomy (Post 2) | **data complete** — 6 engineered classes, paired control/engineered, coverage matrix in `results/failures/` |
| 3 | Judge meta-evaluation (Post 3) | **data complete** — 60 hand-labeled rows, 4 judges × framing/consistency arms, κ/α tables in `results/meta/` |

**Scope & limitations:** harness and design commitments only — phases 1–3 are unstarted, so
nothing here is a finding yet. No results are committed: `results/` is gitignored until a run is
reproducible from cached raw outputs. Phase 0's offline test suite is in place — see
[Tests](#tests).

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
  controlled.py          Phase 2: six engineered failure classes as paired
                         control/engineered slices (self-authored, deterministic)
  agreement.py           Cohen's kappa + Krippendorff's alpha (Phase 3 statistics)

scripts/
  smoke.py               Phase 0 exit criterion: one config, the full metric path,
                         printed as mean ± bootstrap 95% CI
  sweep.py               Phase 1 driver: one axis varied at a time, indexes reused across cells,
                         rows written to results/<config_id>.jsonl
  failures.py            Phase 2 driver: runs every engineered slice through the metric path
                         and prints the per-class tables + coverage matrix
  meta_eval.py           Phase 3 driver: hand-labeled gold + multi-judge arms (sensitivity,
                         framing, self-preference, test-retest) -> kappa/alpha tables

tests/
  helpers.py             shared fake components + builders (no network, no keys)
  test_*.py              one file per module plus a smoke.py regression test

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
# Long runs: `HF_HUB_OFFLINE=1` avoids flaky Hub re-validation once the dataset
# is cached, and `PYTHONUNBUFFERED=1` keeps progress visible under a pipe.
```

Both drivers build their judge, embedder and generator through module-level factories, so
substitutes can exercise the same code path with no network access — that seam exists for tests
that have not been written yet.

## Tests

80 tests, offline and keyless. The drivers' `providers` seam takes fake components, so the full
metric path — including `scripts/smoke.py` itself — runs with no network access.

```bash
uv run pytest
```

What the suite pins, by layer:

- **`normalize` / `retrieval_metrics`** — whitespace-insensitive gold matching; recall@k, MRR and
  nDCG@k against hand-computed values, including the binary per-chunk blind spot that Phase 2
  engineers.
- **`aggregate`** — bootstrap CI determinism for a fixed seed, effect sizes, `None`-skipping with
  per-metric `n`.
- **`pipeline`** — chunk-span invariants, `published_at` parsing, context truncation, index
  cache reuse and round-trip, dense retrieval ranking/filtering, RRF tie-breaking.
- **`harness`** — row schema, refusal detection, judge key-drift guard, `run_config` ordering
  (retrieve → rerank → truncate → generate → measure) and reranker/config mismatch, null-query
  bookkeeping.
- **`judges` / end-to-end metrics** — `CachedJudge` disk caching keyed on
  (model, criterion, prompt), including persistence across instances; prompt wiring for all four
  criteria.
- **`data`** — offline model contracts (`doc_id`, `gold_facts`, `is_null`).
- **`scripts/smoke.py`** — the Phase 0 exit criterion as a regression test with fake providers,
  plus the missing-keys guard.

Not unit-tested: `load_queries` / `load_corpus` themselves. They are field-mapping over the HF
dataset; a mocked test would be an echo test, and the live corpus is exercised by the real smoke
run.

## Authorship

**Agent-written, owner-directed** (revised 2026-10-12): agents implement, test, run, and document;
the owner sets scope and decisions and reviews. The earlier "I write all implementation, agents
advise" constraint no longer applies. The deliverables stay the same — *explained* artifacts:
blog posts, ADRs, interview answers — and the numbers, caveats, and what each metric actually
measures remain the point of the project.

## License

No `LICENSE` file is included, so the default applies: all rights reserved. The plan and the metric
definitions are here to be read. The end-to-end metric names follow the RAGAS family by convention.

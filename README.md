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
| 1 | Ablation study (Post 1) | **data complete** — 10 cells × 2,556 queries; corrected reanalysis in `reports/phase1.md` |
| 2 | Failure-mode taxonomy (Post 2) | **regenerated** — 6 engineered classes, 24 queries per arm, ground-truth matrix in `reports/phase2.md` |
| 3 | Judge meta-evaluation (Post 3) | **v1 analyzed, v2 sample ready** — v1's degenerate gold is documented in `reports/phase3.md`; `sample_v2` awaits hand labels |

**Scope & limitations:** the harness and all three experiments are built, with reproducible raw
data in `results/` — nothing here is a *published finding* yet; the blog posts that interpret the
numbers are the remaining deliverable. `results/` stays gitignored. The hand-labeled artifacts are
committed under `data/labels/`, because a label set that exists only on one machine is not
reproducible. Query counts differ by view, deliberately: the sweep ran all **2,556** queries, while
the Phase 1 tables cover the **2,255** answerable ones. The 301 null queries test refusal
robustness and have no gold answer to score, so they are excluded from the ablations — which is
why `reports/phase1.md` shows `n = 2255` against a table that says 2,556.

## What the corrected numbers already show

These are measurements, not the posts' arguments:

- **The baseline refuses 71.3% of answerable queries.** Among the 28.7% it answers, accuracy is
  96.4%; that ratio is 95–97% in all ten cells. Every Phase 1 "quality" delta is predominantly a
  refusal-rate delta, so the sweep reports coverage and `correctness|answered` next to the pooled
  mean.
- **The refusal rate is not uniform by question type.** Inference queries are answered 57.6% of
  the time, comparison queries 17.4%, temporal queries 4.8%. The aggregate hides the pipeline's
  actual failure profile.
- **Window-based `recall@10` is not comparable across chunk sizes.** The context budget is fixed
  at 8,000 characters, so 2048-char chunks fit ~4 contexts and 512-char chunks fit ~19. The
  original table's `recall@10` therefore meant three different things on the chunk-size axis.
  Candidate-level retrieval fixes the comparison: at 2048 characters window `recall@10` is 0.345
  but top-10 candidate recall is 0.535; the window metric was reporting the context budget, not
  retrieval. New runs store candidates; `scripts/retrieval_audit.py` reconstructs the candidate
  view for the stored cells (window sets reproduce 100%; exact prefixes are 95–100% because
  near-tied scores swap).
- **Phase 2 has two clean blind spots.** `retrieval_miss`: faithfulness stays at 1.0 while the
  answer is withheld (−1.0 correctness). `hallucination`: a context-supported but wrong answer is
  invisible to faithfulness, relevance, and every retrieval metric, while correctness catches it
  (−1.0). One metric moves the *wrong* way there: `context_precision` rises by +0.333 while the
  answer becomes wrong, which is worth remembering before reading a higher score as a better
  system. `chunk_boundary` and `false_neighbor` move retrieval metrics without moving the answer —
  metric false alarms relative to answer quality, not blind spots. `lost_in_middle` produced no
  phenomenon at all (correctness unchanged when the evidence moved from first to fifth in the
  prompt), which is reported as a null result. `evidence_split` has a weak, non-significant
  correctness drop.
- **Phase 3 v1's gold set was degenerate.** All 60 faithfulness labels were 1, and correctness was
  exactly "is the answer a refusal". The report prints `undefined (constant gold)` instead of a
  meaningless κ, and `sample_v2` stratifies across refusals, correct answers and answered-but-wrong
  answers (57 / 40 / 23). It is ready to label; the labels are owner work, not something a script
  can invent.
- **The apparent "self-preference" was answer quality.** Holding the answer fixed and varying only
  the claimed author moves scores by at most 0.08 (mostly 0.00). Comparing two different answer
  sets moves the claude judge's faithfulness score by −0.385. The earlier arm was measuring which
  answers were better, not which author the judge preferred.
- **Judges move between identical runs.** Re-running the v1 default arm on the same prompts at
  temperature 0 changed 0–2 of 60 judgments per judge/criterion; one gemini correctness row flipped
  and moved that κ from 1.000 to 0.960. Temperature is not the escape hatch either: at temperature
  0.7 the faithfulness judge flipped **11 of 60** rows and α across three runs fell to **0.518**, so
  roughly half its apparent variance is noise. Prompt framing is larger still: the terse
  faithfulness rubric moves the mean by −0.555 (κ = 0.147 against the default rubric), and terse
  correctness moves it by +0.467. These are floors on judge reliability, not noise to be averaged
  away.

## What is here

```
src/rag_eval/
  pipeline.py            adapters — the seam between the harness and any RAG pipeline
                         (Chunk, Chunker, Embedder, Retriever, Reranker, Generator, Index)
  harness.py             per-instance rows: window metrics, candidate metrics, refusal,
                         coverage, query fingerprint
  retrieval_metrics.py   recall@k, hits@k, MAP@k, fact-level nDCG@k, MRR
  end_to_end_metrics.py  faithfulness, answer relevance, context precision, answer correctness
  judges.py              Judge protocol + CachedJudge (raw completions, params in the key)
                         + OpenRouterJudge (retries transport failures and score truncation)
  aggregate.py           bootstrap CIs, paired effect sizes, Benjamini-Hochberg
  data.py                MultiHop-RAG loaders, pinned to one dataset revision
  normalize.py           the single whitespace normalizer used for all gold matching
  controlled.py          Phase 2: six engineered failure classes as paired
                         control/engineered slices (self-authored, deterministic)
  agreement.py           Cohen's kappa (+ bootstrap CI) and Krippendorff's alpha (Phase 3)

scripts/
  smoke.py               Phase 0 exit criterion: one config, the full metric path,
                         printed as mean ± bootstrap 95% CI
  sweep.py               Phase 1 driver: one axis varied at a time, indexes reused across cells,
                         rows flushed per query, provenance in the cell header
  reanalyze.py           offline Phase 1 reanalysis: corrected window metrics, coverage,
                         per-question-type tables -> reports/phase1.md
  retrieval_audit.py     candidate-level retrieval audit over the cached indexes; query
                         embeddings only -> reports/retrieval_audit.md
  chunk_ceiling.py       offline, keyless: gold-fact reachability per chunk size
                         -> reports/chunk_ceiling.md
  failures.py            Phase 2 driver: engineered slices + ground-truth coverage matrix
                         -> reports/phase2.md
  meta_eval.py           Phase 3 driver: stratified sample, multi-judge arms (sensitivity,
                         framing, content-fixed attribution, test-retest)
                         -> reports/phase3.md

tests/
  helpers.py             shared fake components + builders (no network, no keys)
  test_*.py              one file per module plus a smoke.py regression test

reports/                 derived markdown tables, committed
data/labels/             hand-authored Phase 3 labels and the raw judgments they score
docs/scratch.ipynb       working notebook
```

## Metrics

Retrieval metrics are computed over **gold-fact containment** after whitespace normalization,
exactly as the dataset's own `retrieval_evaluate.py` normalizes:

- `recall@k` — fraction of gold facts contained in the top-k chunks.
- `hits@k` — 1.0 if any gold fact appears in the top-k chunks (the dataset's Hits@k shape).
- `map@k` — dataset-style average precision; each fact is counted once.
- `ndcg@k` — fact-level discounted recall: a fact's gain is discounted by the rank of the first
  chunk containing it, and the ideal puts every fact at rank 1. Bounded in [0, 1]; duplicate and
  overlapping chunks add nothing. The earlier per-chunk definition normalised against
  `min(n_facts, k)` successive ranks and both depressed perfect retrievals and allowed values
  above 1.
- `mrr` — reciprocal rank of the first chunk containing any gold fact.

Two views are stored separately:

- `metrics` — retrieval metrics over the **prompt window** (what the generator saw) plus the
  judge-backed metrics.
- `candidate_metrics` — the same retrieval metrics over the **pre-truncation candidate list**.
  This is the retrieval-quality view; the window view answers a different question.

Units: all metrics are chunk-level. The dataset's own script ranks documents, so these numbers are
not directly comparable to the paper's Hits/MAP/MRR table. Normalization matches; the retrieval
unit does not.

## Design commitments

These are the decisions that make the later posts cheap rather than expensive rewrites:

- **Per-instance arrays, never early aggregation.** Aggregation and CIs are a separate, swappable
  step. Aggregate inside the metric and every Post 3 judge-swap becomes a full re-run.
- **The judge is a parameter, not a global.** Anything that scores an answer takes the judge as an
  argument, so Post 3 can swap it without touching the harness.
- **Judge caching from the start**, keyed on `(model, criterion, prompt, params)`. Entries store
  the raw completion and the generation parameters alongside the parsed score; a parser change
  must remain auditable.
- **One whitespace normalizer**, matching the dataset's.
- **The harness never owns the pipeline.** `run_config` drives one call order, but the adapters
  are Protocols and the metric path takes `(query, retrieved_contexts, answer, gold_answer)`.
- **Two retrieval views and a refusal flag.** Window vs candidate, and coverage vs accuracy, are
  separate columns because conflating them made the first sweep report refusal as quality.
- **Provenance travels with the data.** Cell headers record the dataset revision, judge model and
  parameters, generator template hash, query count and query-set hash, and package versions. Rows
  carry a query fingerprint and candidate ids. Rows are flushed as they are produced, so a crash
  loses one query, not a 23-hour cell.

## Data

[MultiHop-RAG](https://huggingface.co/datasets/yixuantt/MultiHopRAG) (ODC-BY, COLM 2024): 609 news
articles carrying `title`, `source` and `published_at`, with fact-string gold evidence and four
query types — inference (816), comparison (856), temporal (583), null (301). The loader pins one
dataset revision.

Chosen over HotpotQA for two reasons. The gold is **fact strings matched by substring containment**,
so it does not move when the chunker changes. And 2023 news carries far less pretraining-
contamination risk than Wikipedia.

What the chunking data actually says: gold facts are 93 / 161 / 257 characters at p10 / p50 / p90
(max 538). The measured ceiling — the share of facts contained in some chunk of their source
document — is 1.0000 at 1024 characters and 0.99967 at 512. So chunk-boundary loss is **two facts
out of 6,084**, both longer than 512 characters; it is not a driver of the chunk-size axis. Phase 2
engineers the split explicitly because the natural grid does not produce it often. All three
figures are reproducible offline and keyless: `scripts/chunk_ceiling.py` recomputes them from the
corpus and the chunker alone — no embeddings, no API keys — and writes `reports/chunk_ceiling.md`.

The corpus is not redistributed here; `data.py` downloads it. Null queries (11.8%) test refusal
robustness, temporal queries stress time-aware retrieval.

## Running it

Requires [`uv`](https://docs.astral.sh/uv/), Python ≥ 3.14, `OPENAI_API_KEY` (embeddings) and
`OPENROUTER_API_KEY` (generator + judge). **No keys live in this repository.**

```bash
uv sync
uv run python scripts/smoke.py --limit 20     # Phase 0 path, prints intervals rather than point estimates
uv run python scripts/sweep.py --limit 100    # Phase 1, one axis at a time, writes results/<config_id>.jsonl
uv run python scripts/failures.py             # Phase 2, writes reports/phase2.md

# Offline reanalysis of the stored cells (no API calls; loads the cached indexes):
HF_HUB_OFFLINE=1 uv run python scripts/reanalyze.py
# Candidate-level retrieval audit (query embeddings only, cached after the first run):
HF_HUB_OFFLINE=1 uv run python scripts/retrieval_audit.py
HF_HUB_OFFLINE=1 uv run python scripts/chunk_ceiling.py    # corpus + chunker only; no keys, no results/

# Phase 3: build a stratified sample, hand-label it, then judge and report.
uv run python scripts/meta_eval.py --step sample --n 120 --name v2
#   author results/meta/gold_v2.jsonl for every sampled query_id
uv run python scripts/meta_eval.py --step generate --name v2
uv run python scripts/meta_eval.py --step judge --name v2
uv run python scripts/meta_eval.py --step report --name v2
```

Long runs: `HF_HUB_OFFLINE=1` avoids flaky Hub re-validation once the dataset is cached, and
`PYTHONUNBUFFERED=1` keeps progress visible under a pipe.

## Tests

137 tests, offline and keyless. The drivers' `providers` seam takes fake components, so the full
metric path — including `scripts/smoke.py` itself — runs with no network access.

```bash
uv run pytest
```

What the suite pins, by layer:

- **`normalize` / `retrieval_metrics`** — whitespace-insensitive gold matching; recall, Hits, MAP,
  bounded fact-level nDCG and MRR against hand-computed values, including duplicate and multi-fact
  chunks.
- **`aggregate`** — bootstrap CI determinism, paired p-values, Benjamini-Hochberg, `None`-skipping
  with per-metric `n`, undefined std for a single row.
- **`agreement`** — kappa against hand computation, kappa CI, and alpha against reference values
  for three raters (the earlier implementation was wrong for 3+ raters).
- **`pipeline`** — chunk-span invariants, `published_at` parsing, context truncation, index cache
  reuse and round-trip, L2-normalization enforcement, dense retrieval ranking/filtering, BM25
  filtering before top-k, RRF tie-breaking.
- **`harness`** — row schema, refusal/coverage bookkeeping, window vs candidate metrics, query
  fingerprint, judge key-drift guard, `run_config` ordering, incremental row streaming, reranker/
  config mismatch, null-query handling.
- **`judges` / end-to-end metrics** — parser rejects in-range counts in preambles, cache keys
  include judge parameters, raw completions persist, atomic writes, prompt wiring for all four
  criteria.
- **`data`** — offline model contracts (`doc_id`, `gold_facts`, `is_null`).
- **`controlled`** — deterministic arms, prompt-order isolation, matched hallucination windows.
- **`scripts/smoke.py`** — the Phase 0 exit criterion as a regression test with fake providers,
  plus the missing-keys guard.

Not unit-tested: `load_queries` / `load_corpus` themselves (field mapping over the HF dataset; a
mocked test would be an echo test), and the network paths of the drivers, which are exercised by
real runs.

## Authorship

**Agent-written, owner-directed** (revised 2026-10-12): agents implement, test, run, and document;
the owner sets scope and decisions and reviews. The earlier "I write all implementation, agents
advise" constraint no longer applies. The deliverables stay the same — *explained* artifacts:
blog posts, ADRs, interview answers — and the numbers, caveats, and what each metric actually
measures remain the point of the project.

## License

No `LICENSE` file is included, so the default applies: all rights reserved. The plan and the metric
definitions are here to be read. The end-to-end metric names follow the RAGAS family by convention.

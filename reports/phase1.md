# Phase 1 reanalysis (corrected metrics, stored rows)

All values recomputed from `results/*.jsonl` and `.index_cache/`; no API calls.
Retrieval metrics use the corrected fact-level nDCG and are computed over the
prompt window. `cand rec@10` is present only for cells that stored candidates.

**Rows.** 2,255 per cell — the 2,556 rows each sweep produced, minus the 301
`null_query` rows, which carry no gold facts and are reported under
*Null-query refusal* below. The README's "2,556 queries" and this table's
`n=2255` are the same run counted two ways.

**Refusal.** A response is a refusal when it contains `insufficient information`,
case-insensitively (`harness.is_refusal`, `REFUSAL_PHRASE`). It is a substring test,
not equality, because the generator varies its phrasing.

| cell | n | answer rate | correctness | corr \| answered | win rec@10 | win nDCG@10 | win MAP@10 | cand rec@10 | median chunks | <10 chunks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline/dense/1024/128 | 2255 | 28.7% | 0.297 | 0.964 | 0.451 | 0.276 | 0.222 | 0.479 | 9 | 91% |
| chunk_size/2048 | 2255 | 20.4% | 0.217 | 0.961 | 0.345 | 0.251 | 0.220 | 0.535 | 4 | 100% |
| chunk_size/512 | 2255 | 30.4% | 0.314 | 0.965 | 0.403 | 0.244 | 0.195 | 0.403 | 19 | 0% |
| embedder/bge-small-en-v1.5 | 2255 | 24.0% | 0.254 | 0.967 | 0.377 | 0.230 | 0.184 | 0.396 | 9 | 79% |
| overlap/0 | 2255 | 29.0% | 0.298 | 0.957 | 0.460 | 0.279 | 0.223 | 0.487 | 9 | 90% |
| overlap/64 | 2255 | 28.7% | 0.297 | 0.964 | 0.459 | 0.279 | 0.223 | 0.486 | 9 | 90% |
| recency_filter/before-median | 2255 | 15.0% | 0.165 | 0.966 | 0.269 | 0.172 | 0.141 | 0.286 | 9 | 88% |
| rerank/bge-reranker-v2-m3 | 2255 | 37.8% | 0.382 | 0.960 | 0.601 | 0.413 | 0.353 | 0.479 | 9 | 93% |
| retriever/bm25 | 2255 | 33.6% | 0.344 | 0.964 | 0.522 | 0.340 | 0.282 | 0.548 | 9 | 92% |
| retriever/hybrid_rrf | 2255 | 33.6% | 0.339 | 0.952 | 0.536 | 0.340 | 0.278 | 0.566 | 9 | 93% |

## Baseline by question type

| type | n | answer rate | correctness | corr \| answered | win rec@10 |
|---|---:|---:|---:|---:|---:|
| comparison_query | 856 | 17.4% | 0.195 | 0.903 | 0.540 |
| inference_query | 816 | 57.6% | 0.574 | 0.996 | 0.373 |
| temporal_query | 583 | 4.8% | 0.058 | 0.750 | 0.427 |

## Why the old table was misleading

`answer_correctness` is `answer_rate x accuracy|answered`; accuracy|answered is
~0.96 in every cell, so the ablation moved refusal, not accuracy. Retrieval
`recall@10` was computed over the prompt window: 100% of 2048-char rows had
fewer than 10 chunks (so recall@10 was recall-all), while 512-char rows had
17-26 chunks (so recall@10 was a real top-10). The metrics are not comparable
across the chunk-size axis; candidate metrics fix that on new runs.

## Null-query refusal (baseline)

| slice | n | correct refusals | refusal accuracy |
|---|---:|---:|---:|
| null_query | 301 | 298 | 0.987 |

Of the 301 unanswerable queries the baseline correctly refused **298 (98.7%)**. The three misses
answered a single letter — `"V"`, `"M"`, `"S"`.

These rows carry no judge-backed metrics: a refusal has no claims to ground, so `refusal_correct` is
the deterministic correctness metric for them (`harness.null_summary`), and it is deliberately kept
out of `answer_correctness` rather than folded in. Read alongside the 71.3% refusal rate on
answerable queries, this is the same behaviour working as intended: the pipeline refuses what it
cannot ground, including what it should refuse.


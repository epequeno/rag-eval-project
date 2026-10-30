# Adjudication page

A static page for reading one trace at a time and open-coding what went wrong, before any
metric is shown. It implements the review-interface part of the eval method: a human cannot
categorise failures they cannot see, and showing an existing taxonomy up front anchors the
reading.

```
index.html   the page (no dependencies, no network)
data.js      generated payload: 120 sampled queries × 10 configs (sample v2)
build.py     regenerates data.js from this repo's stored rows
```

```bash
uv run python tools/adjudication/build.py     # writes tools/adjudication/data.js
# then open tools/adjudication/index.html in a browser
```

`data.js` is generated from `results/` and `.index_cache/`, which are gitignored, so it is
gitignored too: this repo publishes the method, not the scraped article text.

## What it does

- **One trace per screen**: question, gold answer, model answer, retrieved chunks in rank
  order with source host and length, expandable to full text.
- **Open coding first.** No category list exists. You write what you see, then name it. The
  metric verdicts sit behind a *Reveal* button so the judges cannot anchor the reading.
- **Coding**: verdict (correct / incorrect / unsure), free-text notes, free-form tags. Tags
  already used become quick-add chips, sorted by frequency — the reviewer's taxonomy, not
  the project's.
- **Dwell time** per trace, so the review can be reported as "n traces, m minutes each".
- **Export JSONL** for judge calibration: `query_id, verdict, tags, notes, seconds`.

Coding lives in `localStorage` under `eval-adjudication:rag-eval-project:v1`; reloading
resumes. Export before clearing site data.

## Regenerating

```bash
python3 tools/adjudication/build.py                     # sample v2, project resolved from this file
python3 tools/adjudication/build.py ~/code/rag-eval-project v1   # the earlier analysed sample
```

Reads `results/*.jsonl`, `data/labels/phase3_sample_<version>.jsonl`, and
`.index_cache/*/chunks.json` (text only, to attribute each retrieved chunk to its source
document). Writes `data.js` and touches nothing else.

## Known limits

- The page defaults to **sample v2, n=120**, which has no hand labels yet — that is the work
  the page exists for. Build from `v1` to review the 60-row sample that `reports/phase3.md`
  already analyses.
- Both samples contain **no null queries**, so the refusal path — 71% of answerable
  questions — is not in the review pool.
- Contexts come from the sample file, so they reflect one run, not each config's own
  retrieval. Per-config context text would need `context_chunk_ids` resolved per cell.
- `results/*.jsonl` has no gold chunk ids, so retrieved chunks cannot be marked gold.

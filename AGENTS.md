# Agent Guidelines

## Project

An evaluation-first RAG project: a reusable metric harness applied to a RAG pipeline in three
phases — measure it carefully, show what the metrics miss, then question the measuring tools.

## Working rules

- Implement, test, run, and document. Scope, decisions, and review stay with the owner, who is
  accountable for every claim the repository makes.
- Every claim must be reproducible from a stored artifact in `results/` or `data/`. A number with
  no run behind it does not belong in a report.
- State the limit where the claim is made: what the metric does not prove, which slice the number
  covers, and which parts are unmeasured.
- `reports/` is generated from stored artifacts. Do not hand-edit a table without saying so, and do
  not let a committed report drift from the generator that produces it.

## Documentation

Long-form documentation and plans live outside this repository. The repository carries the code,
the reports, and the README.

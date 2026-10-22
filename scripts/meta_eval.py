"""
Phase 3 driver: judge meta-evaluation.

Studies the LLM-as-a-judge that scored Phases 1/2 as a *measuring tool*:
agreement with hand-labeled gold (Cohen's kappa with a bootstrap CI), judge-model
sensitivity, prompt-framing sensitivity (context order, verbosity),
self-preference with content held fixed, and test-retest consistency
(Krippendorff's alpha over repeated runs, temperature 0 vs 0.7).

    uv run python scripts/meta_eval.py --step sample --n 120 --name v2
    uv run python scripts/meta_eval.py --step generate --name v2
    uv run python scripts/meta_eval.py --step judge --name v2
    uv run python scripts/meta_eval.py --step report --name v2

Data lives in results/meta/. Gold labels are authored by hand from the rendered
sample before the report step runs. The v1 sample was drawn only from the
baseline cell, which refuses 71% of answerable queries: 43 of 60 rows were
refusals, all 60 faithfulness labels were 1, and correctness was exactly
"is the answer a refusal". The v2 sampler stratifies across refusals, correct
answers and answered-but-wrong answers so both classes exist.

Self-preference: comparing gpt answers against claude answers confounds
preference with answer quality. The `attribution` arm holds the answer fixed
and varies only the claimed author, which is the actual intervention.

Judges are the Phase 1/2 stack plus the arms Post 3 needs: the production
judge (gemini-2.5-flash), the generator's family (gpt-4o-mini), another
proprietary family (claude-haiku-4.5), and an open-weights model (qwen-2.5-72b).
Second generator: claude-haiku-4.5.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from sweep import JUDGE_MODEL, read_cell

from rag_eval.agreement import cohen_kappa, cohen_kappa_ci, krippendorff_alpha
from rag_eval.end_to_end_metrics import (
    ANSWER_CORRECTNESS_TEMPLATE,
    FAITHFULNESS_TEMPLATE,
)
from rag_eval.harness import is_refusal
from rag_eval.judges import OpenRouterJudge
from rag_eval.pipeline import PromptGenerator

SAMPLE_RESULTS = "results/b14c0e3e345e1cf2.jsonl"  # Phase 1 baseline cell
GENERATOR_MODEL = "anthropic/claude-haiku-4.5"

JUDGES = {
    "gemini-2.5-flash": JUDGE_MODEL,  # production Phase 1/2 judge
    "gpt-4o-mini": "openai/gpt-4o-mini",  # == generator family
    "claude-haiku-4.5": "anthropic/claude-haiku-4.5",
    "qwen-2.5-72b": "qwen/qwen-2.5-72b-instruct",
}

CRITERIA = ("faithfulness", "answer_correctness")
BINARY_CUT = 0.5
STRATA = ("refused", "answered_correct", "answered_wrong")

# Terse rubric variant for the verbosity-framing arm.
TERSE_TEMPLATES = {
    "faithfulness": (
        "Query: {query}\nContext:\n{contexts}\nAnswer: {answer}\n"
        "Score 0 to 1: every claim in the answer is supported by the context. Number only."
    ),
    "answer_correctness": (
        "Query: {query}\nGold answer: {gold_answer}\nAnswer: {answer}\n"
        "Score 0 to 1: the answer matches the gold answer factually. Number only."
    ),
}

OUT = Path("results/meta")
REPORT = Path("reports/phase3.md")


def _sample_path(name: str) -> Path:
    return OUT / ("sample.jsonl" if name == "v1" else f"sample_{name}.jsonl")


def _labels_path(name: str) -> Path:
    return OUT / ("gold.jsonl" if name == "v1" else f"gold_{name}.jsonl")


def _answers_path(name: str) -> Path:
    return OUT / (
        "answers_claude.jsonl" if name == "v1" else f"answers_claude_{name}.jsonl"
    )


def _judgments_path(name: str) -> Path:
    return OUT / ("judgments.jsonl" if name == "v1" else f"judgments_{name}.jsonl")


def _judge_prompt(
    criterion: str,
    query: str,
    contexts: list[str],
    answer: str,
    gold_answer: str,
    variant: str,
) -> str:
    rendered = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
    if variant == "reversed":
        rendered = "\n\n".join(
            f"[{i + 1}] {c}" for i, c in enumerate(list(reversed(contexts)))
        )
    if variant in ("default", "reversed"):
        template = FAITHFULNESS_TEMPLATE if criterion == "faithfulness" else ANSWER_CORRECTNESS_TEMPLATE
        return template.format(query=query, contexts=rendered, answer=answer, gold_answer=gold_answer)
    if variant == "terse":
        return TERSE_TEMPLATES[criterion].format(
            query=query, contexts=rendered, answer=answer, gold_answer=gold_answer
        )
    if variant.startswith("attribution:"):
        author = variant.split(":", 1)[1]
        template = FAITHFULNESS_TEMPLATE if criterion == "faithfulness" else ANSWER_CORRECTNESS_TEMPLATE
        base = template.format(
            query=query, contexts=rendered, answer=answer, gold_answer=gold_answer
        )
        # Same answer, same rubric; only the claimed author changes. This is the
        # content-fixed self-preference intervention.
        return f"The answer below was produced by {author}.\n\n{base}"
    raise ValueError(f"unknown prompt variant {variant!r}")


def _bin(score: float) -> int:
    return 1 if score >= BINARY_CUT else 0


def _load(path: Path) -> list[dict]:
    if not path.exists():
        print(f"missing {path}", file=sys.stderr)
        sys.exit(2)
    return [json.loads(line) for line in path.open()]


def _load_by(path: Path, key: str) -> dict:
    return {row[key]: row for row in _load(path)}


# --------------------------------------------------------------------------- #
# Step: sample
# --------------------------------------------------------------------------- #


def step_sample(n: int, name: str) -> None:
    from rag_eval.data import load_corpus, load_queries

    _, _, rows = read_cell(Path(SAMPLE_RESULTS))
    queries = load_queries()
    corpus = {doc.doc_id: doc for doc in load_corpus()}

    def rebuild_texts(chunk_ids: tuple[str, ...]) -> list[str]:
        texts = []
        for chunk_id in chunk_ids:
            doc_id, span = chunk_id.rsplit(":", 1)
            start, end = span.split("-")
            texts.append(corpus[doc_id].body[int(start) : int(end)])
        return texts

    buckets: dict[str, list] = {stratum: [] for stratum in STRATA}
    for row in rows:
        if row.is_null:
            continue
        refused = row.refused or is_refusal(row.answer)
        if refused:
            buckets["refused"].append(row)
        elif (row.metrics.get("answer_correctness") or 0.0) >= BINARY_CUT:
            buckets["answered_correct"].append(row)
        else:
            buckets["answered_wrong"].append(row)

    rng = random.Random(42)
    per = max(1, n // len(STRATA))
    picked = []
    for stratum in STRATA:
        pool = list(buckets[stratum])
        rng.shuffle(pool)
        picked.extend((stratum, row) for row in pool[:per])
    # Spend any remainder on the largest buckets, preserving determinism.
    remainder = n - len(picked)
    for stratum in sorted(STRATA, key=lambda s: -len(buckets[s])):
        if remainder <= 0:
            break
        already = sum(1 for s, _ in picked if s == stratum)
        extra = buckets[stratum][already : already + remainder]
        picked.extend((stratum, row) for row in extra)
        remainder -= len(extra)

    picked.sort(key=lambda pair: pair[1].query_id)
    OUT.mkdir(parents=True, exist_ok=True)
    sample_path = _sample_path(name)
    with sample_path.open("w") as handle:
        for stratum, row in picked:
            query = queries[row.query_id]
            handle.write(
                json.dumps(
                    {
                        "sample_version": name,
                        "stratum": stratum,
                        "query_id": row.query_id,
                        "question_type": row.question_type,
                        "query": query.query,
                        "gold_answer": query.answer,
                        "answer": row.answer,
                        "contexts": rebuild_texts(row.context_chunk_ids),
                    }
                )
                + "\n"
            )

    # Hand-labeling view: contexts truncated for readability (the judge runs
    # below always use the full contexts from the sample file).
    with (OUT / (f"sample_{name}.txt" if name != "v1" else "sample.txt")).open("w") as handle:
        for stratum, row in picked:
            query = queries[row.query_id]
            handle.write(
                f"== qid {row.query_id} [{row.question_type}] stratum={stratum}\n"
                f"Q: {query.query}\n"
                f"gold: {query.answer!r}\n"
                f"answer: {row.answer[:300]!r}\n"
            )
            for i, text in enumerate(rebuild_texts(row.context_chunk_ids)[:8]):
                shown = text if len(text) <= 400 else text[:400] + "…[trunc]"
                handle.write(f"  ctx[{i + 1}]: {shown}\n")
            if len(row.context_chunk_ids) > 8:
                handle.write(f"  … {len(row.context_chunk_ids) - 8} more contexts\n")
            handle.write("\n")

    counts = {stratum: sum(1 for s, _ in picked if s == stratum) for stratum in STRATA}
    print(f"sampled {len(picked)} rows {counts} -> {sample_path}")
    print(f"hand-labeling view -> {OUT / (f'sample_{name}.txt' if name != 'v1' else 'sample.txt')}")
    print(f"next: author {_labels_path(name)} (one row per query_id) before --step report")


# --------------------------------------------------------------------------- #
# Step: generate (second generator family)
# --------------------------------------------------------------------------- #


def step_generate(name: str) -> None:
    sample = _load(_sample_path(name))
    generator = PromptGenerator(model=GENERATOR_MODEL)
    with _answers_path(name).open("w") as handle:
        for row in sample:
            answer = generator.generate(row["query"], row["contexts"])
            handle.write(json.dumps({"query_id": row["query_id"], "answer": answer}) + "\n")
    print(f"generated {len(sample)} answers -> {_answers_path(name)}")


# --------------------------------------------------------------------------- #
# Step: judge
# --------------------------------------------------------------------------- #


def _arms_for_judge(judge_name: str) -> list[dict]:
    """
    Produce the arms this judge must run.

    Framing and test-retest arms are gemini-only (the production judge); the
    other judges get the default arm plus the content-fixed attribution arm.
    """
    attribution = [
        {"variant": "attribution:openai/gpt-4o-mini", "temperature": 0.0, "runs": 1},
        {"variant": "attribution:anthropic/claude-haiku-4.5", "temperature": 0.0, "runs": 1},
    ]
    if judge_name != "gemini-2.5-flash":
        return [{"variant": "default", "temperature": 0.0, "runs": 1}, *attribution]
    return [
        {"variant": "default", "temperature": 0.0, "runs": 1},
        {"variant": "reversed", "temperature": 0.0, "runs": 1},
        {"variant": "terse", "temperature": 0.0, "runs": 1},
        {"variant": "default", "temperature": 0.0, "runs": 2},  # test-retest, greedy
        {"variant": "default", "temperature": 0.7, "runs": 3},  # test-retest, stochastic
        *attribution,
    ]


def step_judge(name: str, limit: int | None = None) -> None:
    sample = _load(_sample_path(name))
    if limit is not None:
        sample = sample[:limit]
    answers = _load_by(_answers_path(name), "query_id")
    judgment_path = _judgments_path(name)
    seen: set[str] = set()
    if judgment_path.exists():
        for line in judgment_path.open():
            seen.add(json.loads(line)["key"])
    with judgment_path.open("a") as handle:
        for row in sample:
            qid = row["query_id"]
            sources = {"gpt-4o-mini": row["answer"], "claude-haiku-4.5": answers[qid]["answer"]}
            for source, answer in sources.items():
                for judge_name, model in JUDGES.items():
                    judge = OpenRouterJudge(
                        model=model,
                        max_output_tokens=96 if judge_name == "claude-haiku-4.5" else 32,
                    )
                    for arm in _arms_for_judge(judge_name):
                        # Attribution holds the answer fixed: only the claimed
                        # author varies, so it runs on the gpt answer only.
                        if arm["variant"].startswith("attribution:") and source != "gpt-4o-mini":
                            continue
                        judge.temperature = arm["temperature"]
                        for run in range(1, arm["runs"] + 1):
                            for criterion in CRITERIA:
                                key = (
                                    f"{name}|{qid}|{source}|{judge_name}|{criterion}|"
                                    f"{arm['variant']}|{arm['temperature']}|{run}"
                                )
                                if key in seen:
                                    continue
                                prompt = _judge_prompt(
                                    criterion,
                                    row["query"],
                                    row["contexts"],
                                    answer,
                                    row["gold_answer"],
                                    arm["variant"],
                                )
                                try:
                                    score = judge.score(criterion, prompt)
                                except ValueError as exc:
                                    # A judge that spends 1024 tokens on a preamble
                                    # and never emits a number is a measurement
                                    # failure, not a reason to lose the whole arm.
                                    # Record it; the report's per-metric n shows
                                    # the gap.
                                    failures_path = OUT / f"judge_failures_{name}.jsonl"
                                    with failures_path.open("a") as failures:
                                        failures.write(
                                            json.dumps(
                                                {
                                                    "key": key,
                                                    "error": str(exc),
                                                    "raw": judge.last_raw,
                                                }
                                            )
                                            + "\n"
                                        )
                                    # Do not mark it seen: a later run should retry.
                                    continue
                                handle.write(
                                    json.dumps(
                                        {
                                            "key": key,
                                            "sample_version": name,
                                            "query_id": qid,
                                            "stratum": row.get("stratum"),
                                            "source": source,
                                            "judge": judge_name,
                                            "criterion": criterion,
                                            "variant": arm["variant"],
                                            "temperature": arm["temperature"],
                                            "run": run,
                                            "score": score,
                                            "raw": judge.last_raw,
                                        }
                                    )
                                    + "\n"
                                )
                                handle.flush()
                                seen.add(key)
    print(f"judgments appended -> {judgment_path}")


# --------------------------------------------------------------------------- #
# Step: report
# --------------------------------------------------------------------------- #


def _default_rows(judgments: list[dict], judge: str, source: str, criterion: str, name: str) -> dict[int, float]:
    return {
        j["query_id"]: j["score"]
        for j in judgments
        if j.get("sample_version") == name
        and j["judge"] == judge
        and j["source"] == source
        and j["criterion"] == criterion
        and j["variant"] == "default"
        and j["temperature"] == 0.0
        and j["run"] == 1
    }


def _variant_rows(
    judgments: list[dict], judge: str, source: str, criterion: str, variant: str, name: str
) -> dict[int, float]:
    return {
        j["query_id"]: j["score"]
        for j in judgments
        if j.get("sample_version") == name
        and j["judge"] == judge
        and j["source"] == source
        and j["criterion"] == criterion
        and j["variant"] == variant
    }


def _kappa(gold_bin: list[int], judged_bin: list[int]) -> float:
    return cohen_kappa(gold_bin, judged_bin)["kappa"]


def _classification_rates(gold_bin: list[int], judged_bin: list[int]) -> tuple[float | None, float | None]:
    tp = sum(1 for g, b in zip(gold_bin, judged_bin) if g == 1 and b == 1)
    fn = sum(1 for g, b in zip(gold_bin, judged_bin) if g == 1 and b == 0)
    tn = sum(1 for g, b in zip(gold_bin, judged_bin) if g == 0 and b == 0)
    fp = sum(1 for g, b in zip(gold_bin, judged_bin) if g == 0 and b == 1)
    sensitivity = tp / (tp + fn) if (tp + fn) else None
    specificity = tn / (tn + fp) if (tn + fp) else None
    return sensitivity, specificity


def step_report(name: str) -> int:
    sample = _load(_sample_path(name))
    gold = _load_by(_labels_path(name), "query_id")
    judgments = _load(_judgments_path(name))
    qids = [row["query_id"] for row in sample]

    if any(q not in gold for q in qids):
        print(
            f"gold labels incomplete: author {_labels_path(name)} for every sampled query_id first",
            file=sys.stderr,
        )
        sys.exit(2)

    lines: list[str] = []
    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    def emit_table(header: list[str], rows: list[list[str]]) -> None:
        emit("| " + " | ".join(header) + " |")
        emit("|" + "|".join("---" for _ in header) + "|")
        for row in rows:
            emit("| " + " | ".join(row) + " |")

    emit(f"# Phase 3 judge meta-evaluation ({name}, n={len(qids)})")
    emit()
    unstratified = sum(1 for row in sample if "stratum" not in row)
    strata: dict[str, int] = {}
    for row in sample:
        if "stratum" in row:
            stratum = row["stratum"]
            strata[stratum] = strata.get(stratum, 0) + 1
    if unstratified == len(sample):
        emit(f"sample strata: unstratified ({len(sample)} rows)")
    else:
        emit(f"sample strata: {strata}")
    emit()

    emit("## 1. Judge vs hand-labeled gold (default arm, gpt-4o-mini answers)")
    emit()
    gold_rows: list[list[str]] = []
    for judge in JUDGES:
        for criterion in CRITERIA:
            scores = _default_rows(judgments, judge, "gpt-4o-mini", criterion, name)
            g = [gold[q][criterion] for q in qids]
            b = [_bin(scores[q]) for q in qids]
            gold_rate = sum(g) / len(g)
            accuracy = sum(x == y for x, y in zip(b, g)) / len(g)
            if len(set(g)) < 2:
                gold_rows.append(
                    [judge, criterion, "undefined", "(constant gold)",
                     f"{accuracy:.3f}", "-", "-", f"{gold_rate:.3f}"]
                )
                continue
            interval = cohen_kappa_ci(g, b)
            sens, spec = _classification_rates(g, b)
            gold_rows.append([
                judge,
                criterion,
                f"{_kappa(g, b):.3f}",
                f"[{interval['ci_lower']:+.2f},{interval['ci_upper']:+.2f}]",
                f"{accuracy:.3f}",
                "-" if sens is None else f"{sens:.3f}",
                "-" if spec is None else f"{spec:.3f}",
                f"{gold_rate:.3f}",
            ])
    emit_table(
        ["judge", "criterion", "kappa", "95% CI", "acc", "sens", "spec", "gold_rate"],
        gold_rows,
    )

    emit()
    emit("## 2. Judge x judge agreement (default arm)")
    emit()
    emit("Each cell is `kappa/accuracy` on the binary labels: faithfulness, then answer_correctness.")
    emit()
    agreement_rows: list[list[str]] = []
    for j1 in JUDGES:
        row_cells = [j1]
        for j2 in JUDGES:
            kf, kc = [], []
            for criterion in CRITERIA:
                a = [_bin(_default_rows(judgments, j1, "gpt-4o-mini", criterion, name)[q]) for q in qids]
                b = [_bin(_default_rows(judgments, j2, "gpt-4o-mini", criterion, name)[q]) for q in qids]
                k = _kappa(a, b)
                (kf if criterion == "faithfulness" else kc).append(k)
            row_cells.append(f"{kf[0]:.3f}/{kc[0]:.3f}")
        agreement_rows.append(row_cells)
    emit_table(["judge", *JUDGES], agreement_rows)

    emit()
    emit("## 3. Content-fixed self-preference (same gpt answer, claimed author varies)")
    emit()
    pref_rows: list[list[str]] = []
    for judge in JUDGES:
        for criterion in CRITERIA:
            g = _variant_rows(judgments, judge, "gpt-4o-mini", criterion, "attribution:openai/gpt-4o-mini", name)
            c = _variant_rows(judgments, judge, "gpt-4o-mini", criterion, "attribution:anthropic/claude-haiku-4.5", name)
            if not g or not c:
                continue
            common = sorted(set(g) & set(c))
            mg = sum(g[q] for q in common) / len(common)
            mc = sum(c[q] for q in common) / len(common)
            pref_rows.append([judge, criterion, f"{mg:.3f}", f"{mc:.3f}", f"{mg - mc:+.3f}"])
    emit_table(["judge", "criterion", "gpt label", "claude label", "delta"], pref_rows)

    emit()
    emit("## 4. Answer-set difference (NOT self-preference: different answers)")
    emit()
    set_rows: list[list[str]] = []
    for judge in JUDGES:
        for criterion in CRITERIA:
            g = [_default_rows(judgments, judge, "gpt-4o-mini", criterion, name)[q] for q in qids]
            c = [_default_rows(judgments, judge, "claude-haiku-4.5", criterion, name)[q] for q in qids]
            mg, mc = sum(g) / len(g), sum(c) / len(c)
            set_rows.append([judge, criterion, f"{mg:.3f}", f"{mc:.3f}", f"{mg - mc:+.3f}"])
    emit_table(["judge", "criterion", "gpt answers", "claude answers", "delta"], set_rows)

    emit()
    emit("## 5. Prompt framing (gemini, gpt-4o-mini answers)")
    emit()
    framing_rows: list[list[str]] = []
    for criterion in CRITERIA:
        default_scores = _default_rows(judgments, "gemini-2.5-flash", "gpt-4o-mini", criterion, name)
        default_ = [default_scores[q] for q in qids]
        for variant in ("reversed", "terse"):
            vals = _variant_rows(judgments, "gemini-2.5-flash", "gpt-4o-mini", criterion, variant, name)
            b_def = [_bin(x) for x in default_]
            b_var = [_bin(vals[q]) for q in qids]
            mean_diff = sum(vals[q] - default_scores[q] for q in qids) / len(qids)
            framing_rows.append([
                criterion, variant, f"{_kappa(b_def, b_var):.3f}", f"{mean_diff:+.3f}"
            ])
    emit_table(["criterion", "variant", "kappa vs default", "mean delta"], framing_rows)

    emit()
    emit("## 6. Test-retest (gemini, gpt-4o-mini answers)")
    emit()
    retest_rows: list[list[str]] = []
    for temperature in (0.0, 0.7):
        per_qid: dict[str, dict[int, dict[int, float]]] = {criterion: {} for criterion in CRITERIA}
        for j in judgments:
            if (
                j.get("sample_version") == name
                and j["judge"] == "gemini-2.5-flash"
                and j["source"] == "gpt-4o-mini"
                and j["variant"] == "default"
                and j["temperature"] == temperature
            ):
                per_qid[j["criterion"]].setdefault(j["query_id"], {})[j["run"]] = j["score"]
        for criterion in CRITERIA:
            avail = sorted({r for rs in per_qid[criterion].values() for r in rs})
            paired = [
                [per_qid[criterion][q][r] for r in avail]
                for q in sorted(per_qid[criterion])
                if len(per_qid[criterion][q]) == len(avail)
            ]
            if len(avail) < 2:
                continue
            alpha = krippendorff_alpha(
                [[1 if p >= BINARY_CUT else 0 for p in pair] for pair in paired]
            )
            flips = sum(
                1
                for q in sorted(per_qid[criterion])
                if len(per_qid[criterion][q]) == len(avail)
                and len({round(per_qid[criterion][q][r], 3) for r in avail}) > 1
            )
            retest_rows.append([
                f"{temperature}",
                criterion,
                f"{len(avail)}",
                f"{alpha:.3f}",
                f"{flips}/{len(paired)}",
            ])
    emit_table(["temperature", "criterion", "runs", "alpha", "flipped rows"], retest_rows)

    emit()
    emit("## 7. Cross-run drift (original v1 run vs this re-run, same prompt, temperature 0)")
    emit()
    legacy = [
        j for j in judgments
        if "sample_version" not in j
        and j["source"] == "gpt-4o-mini"
        and j["variant"] == "default"
        and j["temperature"] == 0.0
        and j["run"] == 1
    ]
    if legacy:
        drift_rows: list[list[str]] = []
        for judge in JUDGES:
            for criterion in CRITERIA:
                old = {j["query_id"]: j["score"] for j in legacy
                       if j["judge"] == judge and j["criterion"] == criterion}
                new = _default_rows(judgments, judge, "gpt-4o-mini", criterion, name)
                common = sorted(set(old) & set(new))
                if not common:
                    continue
                changed = sum(1 for q in common if abs(old[q] - new[q]) > 1e-9)
                mean_delta = sum(new[q] - old[q] for q in common) / len(common)
                drift_rows.append([
                    judge,
                    criterion,
                    f"{changed}/{len(common)}",
                    f"{mean_delta:+.3f}",
                ])
        emit_table(["judge", "criterion", "changed", "mean delta"], drift_rows)
    else:
        emit("No original-run rows present.")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n")
    print(f"\nreport -> {REPORT}")
    return 0


def cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", required=True, choices=("sample", "generate", "judge", "report"))
    parser.add_argument("--n", type=int, default=120, help="rows to sample (sample step)")
    parser.add_argument("--name", default="v1", help="sample version (v1, v2, ...)")
    parser.add_argument("--limit", type=int, default=None, help="rows for --step judge")
    args = parser.parse_args()
    if args.step == "sample":
        step_sample(args.n, args.name)
    elif args.step == "generate":
        step_generate(args.name)
    elif args.step == "judge":
        step_judge(args.name, args.limit)
    else:
        return step_report(args.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())

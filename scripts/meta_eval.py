"""
Phase 3 driver: judge meta-evaluation.

Studies the LLM-as-judge that scored Phases 1/2 as a *measuring tool*:
agreement with hand-labeled gold (Cohen's kappa), judge-model sensitivity,
prompt-framing sensitivity (context order, verbosity), self-preference
(judge family == generator family), and test-retest consistency
(Krippendorff's alpha over repeated runs, temperature 0 vs 0.7).

    uv run python scripts/meta_eval.py --step sample --n 60   # extract + render rows
    uv run python scripts/meta_eval.py --step generate        # 2nd-generator answers
    uv run python scripts/meta_eval.py --step judge           # all judge arms (no cache)
    uv run python scripts/meta_eval.py --step report          # kappa/alpha tables

Data lives in results/meta/. Gold labels (results/meta/gold.jsonl) are
authored by hand from the rendered results/meta/sample.txt before the report
step runs — that is the hand-labeling Post 3 is built on.

Judges are the Phase 1/2 stack plus the arms Post 3 needs: the production
judge (gemini-2.5-flash), the generator's family (gpt-4o-mini, for
self-preference), another proprietary family (claude-haiku-4.5), and an
open-weights model (qwen-2.5-72b). Second generator: claude-haiku-4.5.

Note 2026-10-16: claude-haiku-4.5 ignores "return only the number" and opens
with a reasoning preamble before the score — the judge arm for it runs with
max_output_tokens=96 so the number lands in range; the parser takes the first
number in [0,1]. meta-llama/llama-3.3-70b was probed and set aside in favor
of qwen-2.5-72b for the open arm.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from sweep import read_cell

from rag_eval.agreement import cohen_kappa, krippendorff_alpha
from rag_eval.end_to_end_metrics import (
    ANSWER_CORRECTNESS_TEMPLATE,
    FAITHFULNESS_TEMPLATE,
)
from rag_eval.judges import OpenRouterJudge
from rag_eval.pipeline import PromptGenerator

SAMPLE_RESULTS = "results/b14c0e3e345e1cf2.jsonl"  # Phase 1 baseline cell
GENERATOR_MODEL = "anthropic/claude-haiku-4.5"

JUDGES = {
    "gemini-2.5-flash": "google/gemini-2.5-flash",  # production Phase 1/2 judge
    "gpt-4o-mini": "openai/gpt-4o-mini",  # == generator family (self-preference arm)
    "claude-haiku-4.5": "anthropic/claude-haiku-4.5",
    "qwen-2.5-72b": "qwen/qwen-2.5-72b-instruct",
}

CRITERIA = ("faithfulness", "answer_correctness")
BINARY_CUT = 0.5

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


def _judge_prompt(
    criterion: str, query: str, contexts: list[str], answer: str, gold_answer: str, variant: str
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


def step_sample(n: int) -> None:
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

    by_type: dict[str, list] = {}
    for r in rows:
        if not r.is_null:
            by_type.setdefault(r.question_type, []).append(r)
    rng = random.Random(42)
    per = max(1, n // len(by_type))
    picked: list = []
    for qtype in sorted(by_type):
        pool = list(by_type[qtype])
        rng.shuffle(pool)
        picked.extend(pool[:per])
    picked.sort(key=lambda r: r.query_id)

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "sample.jsonl").open("w") as handle:
        for r in picked:
            q = queries[r.query_id]
            handle.write(
                json.dumps(
                    {
                        "query_id": r.query_id,
                        "question_type": r.question_type,
                        "query": q.query,
                        "gold_answer": q.answer,
                        "answer": r.answer,
                        "contexts": rebuild_texts(r.context_chunk_ids),
                    }
                )
                + "\n"
            )

    # Hand-labeling view: contexts truncated for readability (the judge runs
    # below always use the full contexts from sample.jsonl).
    with (OUT / "sample.txt").open("w") as handle:
        for r in picked:
            q = queries[r.query_id]
            handle.write(
                f"== qid {r.query_id} [{r.question_type}]\n"
                f"Q: {q.query}\n"
                f"gold: {q.answer!r}\n"
                f"answer: {r.answer[:300]!r}\n"
            )
            for i, text in enumerate(rebuild_texts(r.context_chunk_ids)[:8]):
                shown = text if len(text) <= 400 else text[:400] + "…[trunc]"
                handle.write(f"  ctx[{i + 1}]: {shown}\n")
            if len(r.context_chunk_ids) > 8:
                handle.write(f"  … {len(r.context_chunk_ids) - 8} more contexts\n")
            handle.write("\n")
    print(f"sampled {len(picked)} rows -> {OUT / 'sample.jsonl'}, rendered {OUT / 'sample.txt'}")


# --------------------------------------------------------------------------- #
# Step: generate (second generator family)
# --------------------------------------------------------------------------- #


def step_generate() -> None:
    sample = _load(OUT / "sample.jsonl")
    generator = PromptGenerator(model=GENERATOR_MODEL)
    with (OUT / "answers_claude.jsonl").open("w") as handle:
        for row in sample:
            answer = generator.generate(row["query"], row["contexts"])
            handle.write(json.dumps({"query_id": row["query_id"], "answer": answer}) + "\n")
    print(f"generated {len(sample)} answers -> {OUT / 'answers_claude.jsonl'}")


# --------------------------------------------------------------------------- #
# Step: judge
# --------------------------------------------------------------------------- #


def _arms_for_judge(judge_name: str) -> list[dict]:
    """(variant, temperature, runs) arms this judge must produce."""
    if judge_name != "gemini-2.5-flash":
        return [{"variant": "default", "temperature": 0.0, "runs": 1}]
    return [
        {"variant": "default", "temperature": 0.0, "runs": 1},
        {"variant": "reversed", "temperature": 0.0, "runs": 1},
        {"variant": "terse", "temperature": 0.0, "runs": 1},
        {"variant": "default", "temperature": 0.0, "runs": 2},  # test-retest, greedy
        {"variant": "default", "temperature": 0.7, "runs": 3},  # test-retest, stochastic
    ]


def step_judge() -> None:
    sample = _load(OUT / "sample.jsonl")
    answers = _load_by(OUT / "answers_claude.jsonl", "query_id")
    judgment_path = OUT / "judgments.jsonl"
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
                    # claude preamble-first: needs headroom for the number to land.
                    judge = OpenRouterJudge(
                        model=model,
                        max_output_tokens=96 if judge_name == "claude-haiku-4.5" else 32,
                    )
                    for arm in _arms_for_judge(judge_name):
                        judge.temperature = arm["temperature"]
                        for run in range(1, arm["runs"] + 1):
                            for criterion in CRITERIA:
                                key = f"{qid}|{source}|{judge_name}|{criterion}|{arm['variant']}|{arm['temperature']}|{run}"
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
                                score = judge.score(criterion, prompt)
                                handle.write(
                                    json.dumps(
                                        {
                                            "key": key,
                                            "query_id": qid,
                                            "source": source,
                                            "judge": judge_name,
                                            "criterion": criterion,
                                            "variant": arm["variant"],
                                            "temperature": arm["temperature"],
                                            "run": run,
                                            "score": score,
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


def _default_rows(judgments: list[dict], judge: str, source: str, criterion: str) -> dict[int, float]:
    return {
        j["query_id"]: j["score"]
        for j in judgments
        if j["judge"] == judge
        and j["source"] == source
        and j["criterion"] == criterion
        and j["variant"] == "default"
        and j["temperature"] == 0.0
        and j["run"] == 1
    }


def _kappa(gold_bin: list[int], judged_bin: list[int]) -> float:
    return cohen_kappa(gold_bin, judged_bin)["kappa"]


def step_report() -> None:
    sample = _load(OUT / "sample.jsonl")
    gold = _load_by(OUT / "gold.jsonl", "query_id")
    judgments = _load(OUT / "judgments.jsonl")
    qids = [row["query_id"] for row in sample]

    if any(q not in gold for q in qids):
        print(
            "gold labels incomplete: author results/meta/gold.jsonl for every sampled query_id first",
            file=sys.stderr,
        )
        sys.exit(2)

    print("== 1. judge vs hand-labeled gold (default arm, gpt-4o-mini answers)")
    print(f"{'judge':<20}{'criterion':<20}{'kappa':>7}{'accuracy':>10}{'judge_mean':>12}{'gold_rate':>11}")
    for judge in JUDGES:
        for criterion in CRITERIA:
            scores = _default_rows(judgments, judge, "gpt-4o-mini", criterion)
            b = [_bin(scores[q]) for q in qids]
            g = [gold[q][criterion] for q in qids]
            k = _kappa(g, b)
            acc = sum(x == y for x, y in zip(b, g)) / len(g)
            mean = sum(scores[q] for q in qids) / len(qids)
            rate = sum(g) / len(g)
            print(f"{judge:<20}{criterion:<20}{k:>7.3f}{acc:>10.3f}{mean:>12.3f}{rate:>11.3f}")

    print("\n== 2. judge x judge agreement (kappa, default arm)")
    print(f"{'judge':<20}" + "".join(f"{j:>22}" for j in JUDGES))
    for j1 in JUDGES:
        row = f"{j1:<20}"
        for j2 in JUDGES:
            kf, kc = [], []
            for criterion in CRITERIA:
                a = [_bin(_default_rows(judgments, j1, "gpt-4o-mini", criterion)[q]) for q in qids]
                b = [_bin(_default_rows(judgments, j2, "gpt-4o-mini", criterion)[q]) for q in qids]
                k = _kappa(a, b)
                (kf if criterion == "faithfulness" else kc).append(k)
            row += f"{kf[0]:>9.3f}/{kc[0]:<9.3f}"
        print(row)

    print("\n== 3. self-preference: mean score by judge x answer source")
    print(f"{'judge':<20}{'criterion':<20}{'gpt answers':>13}{'claude answers':>15}{'delta':>8}")
    for judge in JUDGES:
        for criterion in CRITERIA:
            g = [_default_rows(judgments, judge, "gpt-4o-mini", criterion)[q] for q in qids]
            c = [_default_rows(judgments, judge, "claude-haiku-4.5", criterion)[q] for q in qids]
            mg, mc = sum(g) / len(g), sum(c) / len(c)
            print(f"{judge:<20}{criterion:<20}{mg:>13.3f}{mc:>15.3f}{mg - mc:>+8.3f}")

    print("\n== 4. prompt framing (gemini, gpt-4o-mini answers)")
    for criterion in CRITERIA:
        default_scores = _default_rows(judgments, "gemini-2.5-flash", "gpt-4o-mini", criterion)
        default_ = [default_scores[q] for q in qids]
        for variant in ("reversed", "terse"):
            vals = {
                j["query_id"]: j["score"]
                for j in judgments
                if j["judge"] == "gemini-2.5-flash"
                and j["source"] == "gpt-4o-mini"
                and j["criterion"] == criterion
                and j["variant"] == variant
                and j["temperature"] == 0.0
                and j["run"] == 1
            }
            b_def = [_bin(x) for x in default_]
            b_var = [_bin(vals[q]) for q in qids]
            mean_diff = sum(vals[q] - default_scores[q] for q in qids) / len(qids)
            print(
                f"{criterion:<20}{variant:<10}kappa={_kappa(b_def, b_var):.3f} "
                f"mean_delta={mean_diff:+.3f}"
            )

    print("\n== 5. test-retest (gemini, gpt-4o-mini answers)")
    for temperature in (0.0, 0.7):
        per_qid = {criterion: {} for criterion in CRITERIA}
        for j in judgments:
            if (
                j["judge"] == "gemini-2.5-flash"
                and j["source"] == "gpt-4o-mini"
                and j["variant"] == "default"
                and j["temperature"] == temperature
                and j["run"] in (1, 2, 3)
            ):
                per_qid[j["criterion"]].setdefault(j["query_id"], {})[j["run"]] = j["score"]
        for criterion in CRITERIA:
            avail = sorted({r for rs in per_qid[criterion].values() for r in rs})
            paired = [
                [per_qid[criterion][q][r] for r in avail]
                for q in sorted(per_qid[criterion])
                if len(per_qid[criterion][q]) == len(avail)
            ]
            if len(avail) >= 2:
                alpha_12 = krippendorff_alpha(
                    [[1 if p >= BINARY_CUT else 0 for p in pair] for pair in paired]
                )
            else:
                alpha_12 = float("nan")
            if len(avail) == 3:
                alpha_all = krippendorff_alpha(
                    [[1 if p >= BINARY_CUT else 0 for p in triple] for triple in paired]
                )
            else:
                alpha_all = float("nan")
            flips = sum(
                1
                for q in sorted(per_qid[criterion])
                if len(per_qid[criterion][q]) == len(avail)
                and len({round(per_qid[criterion][q][r], 3) for r in avail}) > 1
            )
            print(
                f"t={temperature} {criterion:<20}alpha(2 runs)={alpha_12:.3f} "
                f"alpha({len(avail)} runs)={alpha_all:.3f} flipped_rows={flips}/{len(paired)}"
            )
    return 0


def cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", required=True, choices=("sample", "generate", "judge", "report"))
    parser.add_argument("--n", type=int, default=60, help="rows to sample (sample step)")
    args = parser.parse_args()
    if args.step == "sample":
        step_sample(args.n)
    elif args.step == "generate":
        step_generate()
    elif args.step == "judge":
        step_judge()
    else:
        return step_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
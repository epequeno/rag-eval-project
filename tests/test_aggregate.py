"""Aggregation and statistics: bootstrap CIs, effect sizes, per-metric rollup.

Reproducibility is a design commitment (seed defaults), and undefined metrics
must be skipped, never counted as zero.
"""

import numpy as np
import pytest

from rag_eval.aggregate import (
    aggregate,
    benjamini_hochberg,
    bootstrap_ci,
    effect_size_paired,
)

VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]


def test_bootstrap_ci_reports_mean_std_and_n():
    out = bootstrap_ci(VALUES)
    assert out["mean"] == pytest.approx(np.mean(VALUES))
    assert out["std"] == pytest.approx(np.std(VALUES, ddof=1))
    assert out["n"] == len(VALUES)
    assert out["ci_lower"] <= out["mean"] <= out["ci_upper"]


def test_bootstrap_ci_is_deterministic_for_a_seed():
    assert bootstrap_ci(VALUES, seed=42) == bootstrap_ci(VALUES, seed=42)


def test_bootstrap_ci_constant_values_collapse_to_the_constant():
    out = bootstrap_ci([0.5] * 50)
    assert out["mean"] == pytest.approx(0.5)
    assert out["ci_lower"] == pytest.approx(0.5)
    assert out["ci_upper"] == pytest.approx(0.5)


def test_bootstrap_ci_single_value_has_undefined_std():
    # ddof=1 is undefined for n=1; nan silently poisons a table.
    out = bootstrap_ci([0.5])
    assert out["std"] is None
    assert out["ci_lower"] == pytest.approx(0.5)


def test_effect_size_paired_reports_delta_ci_and_p_value():
    out = effect_size_paired([1.0] * 10, [0.0] * 10)
    assert out["delta"] == pytest.approx(1.0)
    assert out["ci_lower"] == pytest.approx(1.0)
    assert out["ci_upper"] == pytest.approx(1.0)
    assert out["p_value"] == pytest.approx(0.0)
    assert out["significant"] is True


def test_effect_size_paired_null_difference_is_not_significant():
    out = effect_size_paired([0.5] * 10, [0.5] * 10)
    assert out["delta"] == pytest.approx(0.0)
    assert out["p_value"] == pytest.approx(1.0)
    assert out["significant"] is False


def test_effect_size_paired():
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    b = [0.0, 1.0, 2.0, 3.0, 4.0]
    out = effect_size_paired(a, b)
    assert out["delta"] == pytest.approx(1.0)
    assert out["significant"] is True


def test_benjamini_hochberg_matches_hand_computation():
    # p = [0.01, 0.04, 0.03, 0.20], m = 4
    # sorted: 0.01 (rank 1) -> 0.04; 0.03 (rank 2) -> 0.06; 0.04 (rank 3) -> 0.0533;
    # 0.20 (rank 4) -> 0.20; enforce monotonicity from the largest, so ranks 2 and
    # 3 both become 0.0533. Back in input order: [0.04, 0.0533, 0.0533, 0.20].
    adjusted = benjamini_hochberg([0.01, 0.04, 0.03, 0.20])
    assert adjusted == pytest.approx([0.04, 0.0533333333, 0.0533333333, 0.20])


def test_benjamini_hochberg_empty():
    assert benjamini_hochberg([]) == []


def test_aggregate_skips_none_and_unions_keys_across_rows():
    rows = [
        {"recall@10": 1.0, "faithfulness": 0.9},
        {"recall@10": 0.5, "faithfulness": None},
        {"recall@10": None, "answer_relevance": 0.7},
    ]
    out = aggregate(rows)
    assert out["recall@10"]["mean"] == pytest.approx(0.75)
    assert out["recall@10"]["n"] == 2
    assert out["faithfulness"]["mean"] == pytest.approx(0.9)
    assert out["faithfulness"]["n"] == 1
    assert out["faithfulness"]["std"] is None
    assert out["answer_relevance"]["n"] == 1
    assert "mrr" not in out


def test_aggregate_empty_input():
    assert aggregate([]) == {}

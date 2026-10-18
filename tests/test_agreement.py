"""Agreement statistics, validated against hand-computed values and the
reference ``krippendorff`` implementation for the 3+ rater cases."""

import pytest

from rag_eval.agreement import cohen_kappa, cohen_kappa_ci, krippendorff_alpha


def test_kappa_perfect_agreement():
    out = cohen_kappa([1, 1, 0, 0, 1], [1, 1, 0, 0, 1])
    assert out["kappa"] == pytest.approx(1.0)
    assert out["p_observed"] == pytest.approx(1.0)
    assert out["n"] == 5


def test_kappa_hand_computed():
    # p_o = 4/5 = 0.8; p_e = 0.6*0.4 + 0.4*0.6 = 0.48; kappa = 0.32/0.52
    out = cohen_kappa([1, 1, 2, 2, 1], [1, 1, 2, 2, 2])
    assert out["p_observed"] == pytest.approx(0.8)
    assert out["p_expected"] == pytest.approx(0.48)
    assert out["kappa"] == pytest.approx(0.32 / 0.52)


def test_kappa_complete_disagreement_is_negative():
    out = cohen_kappa([1, 2], [2, 1])
    assert out["kappa"] == pytest.approx(-1.0)


def test_kappa_undefined_degenerate_pair():
    # Both raters constant and identical: chance agreement is complete.
    assert cohen_kappa([1, 1, 1], [1, 1, 1])["kappa"] == pytest.approx(1.0)


def test_kappa_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        cohen_kappa([1], [1, 2])
    with pytest.raises(ValueError):
        cohen_kappa([], [])


def test_kappa_ci_brackets_the_point_estimate():
    a = [1, 1, 0, 0, 1, 0, 1, 0, 0, 1]
    b = [1, 0, 0, 0, 1, 1, 1, 0, 0, 0]
    out = cohen_kappa_ci(a, b, seed=7)
    point = cohen_kappa(a, b)["kappa"]
    assert out["ci_lower"] <= point <= out["ci_upper"]


def test_alpha_perfect_agreement():
    ratings = [[1, 1], [1, 1], [2, 2], [2, 2]]
    assert krippendorff_alpha(ratings) == pytest.approx(1.0)


def test_alpha_hand_computed_three_items():
    # Units (1,1), (2,2), (1,2): D_o = 2/6, D_e = 18/30? Reference: 0.5.
    # n=6, off-diagonal coincidence = 2, marginals (3,3), D_o=1/3,
    # D_e = 18/30 = 0.6, alpha = 1 - 1/3/0.6 = 0.444...
    ratings = [[1, 1], [2, 2], [1, 2]]
    assert krippendorff_alpha(ratings) == pytest.approx(1 - (2 / 6) / (18 / 30))


def test_alpha_complete_opposition():
    ratings = [[1, 2], [2, 1]]
    assert krippendorff_alpha(ratings) == pytest.approx(-0.5)


def test_alpha_three_raters_matches_reference_value():
    # Reference (krippendorff.alpha, nominal): 0.111111...
    # The earlier implementation returned 0.0556 here.
    ratings = [[1, 1, 1], [1, 1, 2], [2, 2, 1]]
    assert krippendorff_alpha(ratings) == pytest.approx(0.11111111111111116)


def test_alpha_three_raters_mixed_matches_reference_value():
    # Reference (krippendorff.alpha, nominal): 0.33766233766233766
    ratings = [[1, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 0], [0, 0, 0], [1, 1, 1]]
    assert krippendorff_alpha(ratings) == pytest.approx(0.33766233766233766)


def test_alpha_requires_two_ratings_somewhere():
    with pytest.raises(ValueError):
        krippendorff_alpha([[1], [2]])

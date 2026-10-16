"""Agreement statistics, validated against hand-computed values."""

import pytest

from rag_eval.agreement import cohen_kappa, krippendorff_alpha


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


def test_alpha_perfect_agreement():
    ratings = [[1, 1], [1, 1], [2, 2], [2, 2]]
    assert krippendorff_alpha(ratings) == pytest.approx(1.0)


def test_alpha_hand_computed_three_items():
    # Units (1,1), (2,2), (1,2): m=6, n_cc=4, rows (3,3), de=18/5
    ratings = [[1, 1], [2, 2], [1, 2]]
    assert krippendorff_alpha(ratings) == pytest.approx(1 - 2 / (18 / 5))


def test_alpha_complete_opposition():
    ratings = [[1, 2], [2, 1]]
    assert krippendorff_alpha(ratings) == pytest.approx(-0.5)


def test_alpha_three_raters_hand_computed():
    # Items (1,1,1),(1,1,2),(2,2,1): m=18, n_cc=10, rows (12,6), de=144/17
    ratings = [[1, 1, 1], [1, 1, 2], [2, 2, 1]]
    assert krippendorff_alpha(ratings) == pytest.approx(1 - 8 / (144 / 17))


def test_alpha_requires_two_ratings_somewhere():
    with pytest.raises(ValueError):
        krippendorff_alpha([[1], [2]])
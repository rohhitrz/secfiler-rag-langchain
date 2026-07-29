"""Metric arithmetic, including the edge cases that produce ZeroDivisionError."""

import pytest

from secfiler_rag.evaluation.metrics import hit_rate, mean_reciprocal_rank


@pytest.mark.parametrize(
    ("hits", "expected"),
    [
        ([], 0.0),
        ([True], 1.0),
        ([False], 0.0),
        ([True, False, True, True], 0.75),
    ],
)
def test_hit_rate(hits, expected):
    assert hit_rate(hits) == expected


@pytest.mark.parametrize(
    ("ranks", "expected"),
    [
        ([], 0.0),
        ([1], 1.0),
        ([2], 0.5),
        ([None], 0.0),
        ([1, None], 0.5),
        ([1, 2, 4], pytest.approx((1 + 0.5 + 0.25) / 3)),
    ],
)
def test_mean_reciprocal_rank(ranks, expected):
    assert mean_reciprocal_rank(ranks) == expected


def test_mrr_penalises_lower_placement_while_hit_rate_cannot():
    """The reason both metrics are reported: a reranker moves MRR, not hit rate."""
    assert mean_reciprocal_rank([1, 1]) > mean_reciprocal_rank([3, 3])
    assert hit_rate([True, True]) == hit_rate([True, True])

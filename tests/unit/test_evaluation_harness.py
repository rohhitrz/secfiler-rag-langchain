"""Harness contract: strategy-agnostic scoring, correct metrics, auditability.

The most important test here is `test_harness_never_inspects_filters` — it
encodes the rule that keeps every future A/B honest.
"""

import pytest
from langchain_core.documents import Document

from secfiler_rag.evaluation.dataset import EvalDataset, EvalItem
from secfiler_rag.evaluation.harness import evaluate


def doc(text, chunk_id=0, company="aapl"):
    return Document(
        page_content=text,
        metadata={"company": company, "chunk_id": chunk_id, "source": f"{company}-2025.htm"},
    )


def dataset(*items):
    return EvalDataset(name="test", version=1, items=tuple(items))


def item(query="q", expected="needle", filters=None, tier=2):
    return EvalItem(query=query, expected_substring=expected, filters=filters or {}, tier=tier)


def test_hit_when_expected_text_is_retrieved():
    report = evaluate(
        dataset(item(expected="total net sales")),
        lambda q, f, k: [doc("Total net sales | $416,161 | 6%")],
    )

    assert report.hit_rate == 1.0
    assert report.results[0].matched_rank == 1


def test_miss_when_expected_text_is_absent():
    report = evaluate(dataset(item(expected="Megapack")), lambda q, f, k: [doc("unrelated prose")])

    assert report.hit_rate == 0.0
    assert report.results[0].matched_rank is None
    assert report.misses


def test_matching_is_case_insensitive_and_whitespace_forgiving():
    """Expected substrings are written by humans reading cleaned text; they
    must not fail on a line break the chunker introduced."""
    report = evaluate(
        dataset(item(expected="Total Net Sales")),
        lambda q, f, k: [doc("... total   net\nsales were ...")],
    )

    assert report.hit_rate == 1.0


def test_rank_is_one_based_and_finds_the_first_match():
    report = evaluate(
        dataset(item(expected="needle")),
        lambda q, f, k: [doc("no"), doc("still no"), doc("the needle here"), doc("needle again")],
    )

    assert report.results[0].matched_rank == 3


def test_mrr_rewards_higher_placement():
    first = evaluate(dataset(item()), lambda q, f, k: [doc("needle"), doc("x")])
    fourth = evaluate(
        dataset(item()), lambda q, f, k: [doc("x"), doc("x"), doc("x"), doc("needle")]
    )

    assert first.mrr == 1.0
    assert fourth.mrr == 0.25
    # Hit rate cannot see this difference — which is why both metrics exist.
    assert first.hit_rate == fourth.hit_rate == 1.0


def test_harness_never_inspects_filters():
    """The frozen contract: filters are forwarded verbatim, never interpreted.

    If the harness ever learns what a filter means, its numbers stop being
    comparable across strategies it was not written for.
    """
    seen = []

    def search_fn(query, filters, top_k):
        seen.append((query, dict(filters), top_k))
        return [doc("needle")]

    evaluate(
        dataset(item(query="what is X?", filters={"company": "tsla", "year": 2025})),
        search_fn,
        top_k=7,
    )

    assert seen == [("what is X?", {"company": "tsla", "year": 2025}, 7)]


def test_any_callable_can_be_scored():
    """A retriever is only ever a function to this harness."""
    report = evaluate(dataset(item()), lambda query, filters, top_k: [doc("needle")])

    assert report.hit_rate == 1.0


def test_tiers_are_reported_separately():
    report = evaluate(
        dataset(item(expected="a", tier=1), item(expected="b", tier=2)),
        lambda q, f, k: [doc("a")],
    )

    assert report.by_tier(1).hit_rate == 1.0
    assert report.by_tier(2).hit_rate == 0.0
    assert report.hit_rate == 0.5  # the blended number hides the tier-2 failure


def test_result_exposes_the_matching_chunk_for_auditing():
    report = evaluate(
        dataset(item(expected="Powerwall")),
        lambda q, f, k: [doc("x", chunk_id=3), doc("Tesla sells Powerwall units", chunk_id=21)],
    )
    result = report.results[0]

    assert result.matched_chunk_id == 21
    assert "Powerwall" in (result.matched_excerpt() or "")


def test_excerpt_is_none_for_a_miss():
    report = evaluate(dataset(item()), lambda q, f, k: [doc("nothing")])

    assert report.results[0].matched_excerpt() is None
    assert report.results[0].matched_chunk_id is None


def test_empty_result_list_is_a_miss_not_an_error():
    """No results is a valid retrieval outcome, not an exception."""
    report = evaluate(dataset(item()), lambda q, f, k: [])

    assert report.hit_rate == 0.0


def test_latency_is_recorded_per_item():
    report = evaluate(dataset(item()), lambda q, f, k: [doc("needle")])

    assert report.results[0].latency_ms >= 0.0


@pytest.mark.parametrize(
    ("hits", "expected"),
    [([True, True], 1.0), ([True, False], 0.5), ([False, False], 0.0)],
)
def test_hit_rate_arithmetic(hits, expected):
    pending = iter(hits)

    def search_fn(query, filters, top_k):
        return [doc("needle" if next(pending) else "no")]

    report = evaluate(dataset(*[item(expected="needle") for _ in hits]), search_fn)

    assert report.hit_rate == expected

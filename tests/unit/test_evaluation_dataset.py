"""Dataset loading: schema translation and loud failure on malformed data."""

import json
from pathlib import Path

import pytest

from secfiler_rag.core.exceptions import EvaluationError
from secfiler_rag.evaluation.dataset import load_dataset

SEED = Path("evals/datasets/seed_eval_set.json")


def write(tmp_path, payload):
    path = tmp_path / "set.json"
    path.write_text(json.dumps(payload))
    return path


def test_loads_items_with_filters_extracted(tmp_path):
    path = write(
        tmp_path,
        {
            "name": "demo",
            "version": 2,
            "items": [{"tier": 2, "query": "q", "expected_substring": "s", "company": "aapl"}],
        },
    )

    dataset = load_dataset(path)

    assert dataset.name == "demo"
    assert dataset.version == 2
    assert len(dataset) == 1
    assert dataset.items[0].filters == {"company": "aapl"}
    assert dataset.items[0].tier == 2


def test_item_without_a_filter_gets_an_empty_mapping(tmp_path):
    path = write(tmp_path, {"items": [{"query": "q", "expected_substring": "s"}]})

    assert load_dataset(path).items[0].filters == {}


def test_tier_selection(tmp_path):
    path = write(
        tmp_path,
        {
            "items": [
                {"tier": 1, "query": "a", "expected_substring": "a"},
                {"tier": 2, "query": "b", "expected_substring": "b"},
                {"tier": 2, "query": "c", "expected_substring": "c"},
            ]
        },
    )
    dataset = load_dataset(path)

    assert len(dataset.tier(1)) == 1
    assert len(dataset.tier(2)) == 2


def test_missing_file_raises(tmp_path):
    with pytest.raises(EvaluationError, match="not found"):
        load_dataset(tmp_path / "absent.json")


def test_malformed_json_raises(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json")

    with pytest.raises(EvaluationError, match="not valid JSON"):
        load_dataset(path)


def test_empty_item_list_raises(tmp_path):
    with pytest.raises(EvaluationError, match="no items"):
        load_dataset(write(tmp_path, {"items": []}))


def test_item_without_query_raises(tmp_path):
    """Skipping a bad item would make the denominator wrong, which is worse
    than crashing: the resulting score still looks plausible."""
    with pytest.raises(EvaluationError, match="no query"):
        load_dataset(write(tmp_path, {"items": [{"expected_substring": "s"}]}))


def test_item_without_expected_substring_raises(tmp_path):
    with pytest.raises(EvaluationError, match="no expected_substring"):
        load_dataset(write(tmp_path, {"items": [{"query": "q"}]}))


def test_seed_dataset_in_the_repo_is_valid():
    """The committed dataset must always load — it is measurement infrastructure."""
    dataset = load_dataset(SEED)

    assert len(dataset) == 8
    assert len(dataset.tier(1)) == 5
    assert len(dataset.tier(2)) == 3
    assert {item.filters["company"] for item in dataset.items} == {"aapl", "msft", "tsla"}

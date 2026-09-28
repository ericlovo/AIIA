from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from local_brain.eq_brain.story_prioritizer import _dot
from local_brain.indexer import file_index


@pytest.mark.parametrize("left,right", [([1], [2, 3]), ([1, 2], [3]), ([], [1])])
def test_dot_rejects_different_dimensions(left, right):
    with pytest.raises(ValueError):
        _dot(left, right)


def test_dot_equal_dimensions():
    assert _dot([1, 2], [3, 4]) == 11
    assert _dot([], []) == 0


@pytest.fixture
def search_result(monkeypatch):
    result = {
        "documents": [["first", "second"]],
        "metadatas": [[{"path": "one"}, {"path": "two"}]],
        "distances": [[0.1, 0.2]],
    }
    monkeypatch.setattr(file_index, "_embed", AsyncMock(return_value=[[0.1]]))
    monkeypatch.setattr(
        file_index, "_get_collection", lambda: SimpleNamespace(query=lambda **kwargs: result)
    )
    return result


@pytest.mark.parametrize("column", ["documents", "metadatas", "distances"])
async def test_search_rejects_unaligned_columns(search_result, column):
    search_result[column][0].pop()
    with pytest.raises(ValueError):
        await file_index.search_files("fixture")


async def test_search_equal_columns(search_result):
    hits = await file_index.search_files("fixture")
    assert [hit["path"] for hit in hits] == ["one", "two"]
    assert [hit["score"] for hit in hits] == [0.9, 0.8]

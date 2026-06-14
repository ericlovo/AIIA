"""Offline tests for the literature topic factory and profile.

No Ollama or network — exercises the factory, the literature profile, and
backward-compatible topic loading. The TopicStore persists to tmp_path.
"""

from __future__ import annotations

import pytest

from local_brain.research.literature import (
    create_literature_topic,
    find_literature_topic,
    topic_title,
    wikipedia_url,
)
from local_brain.research.profiles import LITERATURE, PROFILES, get_profile
from local_brain.research.topic import TopicStore


@pytest.fixture
def store(tmp_path):
    return TopicStore(data_dir=str(tmp_path))


class TestProfile:
    def test_literature_profile_registered(self):
        assert LITERATURE in PROFILES
        assert get_profile(LITERATURE).id == LITERATURE

    def test_profile_demands_attribution(self):
        principles = PROFILES[LITERATURE].principles.lower()
        assert "attribut" in principles
        assert "primary" in principles and "secondary" in principles


class TestWikipediaSeed:
    def test_spaces_become_underscores(self):
        assert wikipedia_url("Mrs Dalloway").endswith("/Mrs_Dalloway")

    def test_collapses_whitespace(self):
        assert wikipedia_url("  Virginia   Woolf ").endswith("/Virginia_Woolf")

    def test_special_chars_are_percent_encoded(self):
        # Brontë's ë must survive into a valid URL.
        url = wikipedia_url("Charlotte Brontë")
        assert url.startswith("https://en.wikipedia.org/wiki/")
        assert " " not in url


class TestFactory:
    def test_creates_topic_with_profile_and_seed(self, store):
        topic = create_literature_topic(store, "Mrs Dalloway")
        assert topic.profile == LITERATURE
        assert topic.title == "Literature: Mrs Dalloway"
        assert topic.seeds == ["https://en.wikipedia.org/wiki/Mrs_Dalloway"]
        assert "Mrs Dalloway" in topic.question

    def test_extra_seeds_appended_without_duplicates(self, store):
        wiki = wikipedia_url("Hamlet")
        topic = create_literature_topic(
            store, "Hamlet", extra_seeds=["https://gutenberg.org/hamlet", wiki]
        )
        assert topic.seeds == [wiki, "https://gutenberg.org/hamlet"]

    def test_blank_subject_rejected(self, store):
        with pytest.raises(ValueError, match="non-empty"):
            create_literature_topic(store, "   ")

    def test_find_matches_normalized_subject(self, store):
        created = create_literature_topic(store, "Jane Eyre")
        assert find_literature_topic(store, "  Jane   Eyre ").id == created.id

    def test_find_returns_none_when_absent(self, store):
        assert find_literature_topic(store, "Beowulf") is None

    def test_title_normalizes_whitespace(self):
        assert topic_title("  the   waste land ") == "Literature: the waste land"

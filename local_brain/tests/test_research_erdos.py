"""Tests for the Erdős research configuration — profiles, topic factory, persistence.

These don't require Brain, Ollama, or network — they exercise the topic
store on a temp dir, the profile registry, and prompt composition only.
"""

from __future__ import annotations

import json

from local_brain.research.engine import _system_prompt
from local_brain.research.erdos import (
    build_question,
    create_erdos_topic,
    find_erdos_topic,
    problem_url,
    topic_title,
)
from local_brain.research.profiles import ERDOS, GENERAL, PROFILES, get_profile
from local_brain.research.topic import ResearchTopic, TopicStore


class TestProfiles:
    def test_registry_has_general_and_erdos(self):
        assert GENERAL in PROFILES
        assert ERDOS in PROFILES

    def test_unknown_profile_falls_back_to_general(self):
        assert get_profile("nonexistent").id == GENERAL

    def test_erdos_profile_demands_sourced_claims(self):
        profile = get_profile(ERDOS)
        assert "NEVER invent" in profile.principles
        assert "LaTeX" in profile.principles


class TestErdosTopicFactory:
    def test_problem_url(self):
        assert problem_url(351) == "https://www.erdosproblems.com/351"

    def test_create_seeds_problem_page_first(self, tmp_path):
        store = TopicStore(str(tmp_path))
        topic = create_erdos_topic(store, 351, extra_seeds=["https://arxiv.org/abs/1408.1990"])
        assert topic.title == "Erdős Problem #351"
        assert topic.profile == ERDOS
        assert topic.seeds[0] == problem_url(351)
        assert "https://arxiv.org/abs/1408.1990" in topic.seeds
        assert "#351" in topic.question

    def test_create_dedupes_extra_seeds(self, tmp_path):
        store = TopicStore(str(tmp_path))
        topic = create_erdos_topic(store, 7, extra_seeds=[problem_url(7)])
        assert topic.seeds == [problem_url(7)]

    def test_create_rejects_nonpositive_number(self, tmp_path):
        store = TopicStore(str(tmp_path))
        try:
            create_erdos_topic(store, 0)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass

    def test_find_existing_topic(self, tmp_path):
        store = TopicStore(str(tmp_path))
        created = create_erdos_topic(store, 42)
        assert find_erdos_topic(store, 42).id == created.id
        assert find_erdos_topic(store, 43) is None

    def test_question_mentions_status_and_bounds(self):
        question = build_question(1)
        assert "status" in question
        assert "bounds" in question


class TestTopicPersistence:
    def test_profile_round_trips(self, tmp_path):
        store = TopicStore(str(tmp_path))
        topic = create_erdos_topic(store, 99)
        loaded = store.load(topic.id)
        assert loaded.profile == ERDOS

    def test_legacy_topic_json_without_profile_defaults_to_general(self, tmp_path):
        store = TopicStore(str(tmp_path))
        legacy = {"id": "abc12345", "title": "Old topic", "question": "Q?"}
        (tmp_path / "research" / "abc12345.json").write_text(json.dumps(legacy))
        loaded = store.load("abc12345")
        assert loaded is not None
        assert loaded.profile == GENERAL


class TestPromptComposition:
    def test_erdos_topic_gets_erdos_prompt_blocks(self):
        topic = ResearchTopic(
            id="x", title=topic_title(351), question=build_question(351), profile=ERDOS
        )
        prompt = _system_prompt(topic)
        assert "Erdős Problem #351" in prompt
        assert "NEVER invent" in prompt
        assert "erdosproblems.com" in prompt

    def test_general_topic_keeps_original_prompt_blocks(self):
        topic = ResearchTopic(id="x", title="Eros in Plato", question="What is eros?")
        prompt = _system_prompt(topic)
        assert "Prefer primary texts" in prompt
        assert "NEVER invent" not in prompt

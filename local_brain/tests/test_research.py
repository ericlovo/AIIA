"""
Tests for local_brain.research — the autonomous research harness.

These are pure-unit tests: no live Ollama, Brain, or Command Center. The
HTTP layer (fetcher) is stubbed with httpx.MockTransport, and the ChromaDB
KnowledgeStore is replaced with an in-memory fake. That keeps the whole
suite runnable in CI under a plain `pytest` (the T4 direction), not just
`--collect-only`.

Coverage:
  - fetcher:  html_to_text stripping + fetch_url content negotiation
  - topic:    TopicStore persistence roundtrip + from_dict tolerance
  - repl_env: the five research actions (fetch / ingest / gap / synth / search)
"""

from __future__ import annotations

import httpx
import pytest

from local_brain.research import fetcher
from local_brain.research.repl_env import ResearchREPLEnvironment
from local_brain.research.topic import ResearchTopic, TopicStore

# ──────────────────────────────────────────────────────────────────────────
# fetcher — HTML stripping
# ──────────────────────────────────────────────────────────────────────────


def test_html_to_text_drops_script_and_chrome():
    html = (
        "<html><head><style>.x{color:red}</style></head><body>"
        "<nav>navbar</nav><script>var x = 1;</script>"
        "<h1>Title</h1><p>Hello   world</p>"
        "<footer>copyright</footer></body></html>"
    )
    text = fetcher.html_to_text(html)
    assert "Title" in text
    assert "Hello world" in text  # collapsed double spaces
    assert "navbar" not in text
    assert "var x" not in text
    assert "copyright" not in text


def test_html_to_text_block_tags_insert_newlines():
    text = fetcher.html_to_text("<p>one</p><p>two</p>")
    assert "one" in text and "two" in text
    assert "\n" in text  # block boundary preserved


# ──────────────────────────────────────────────────────────────────────────
# fetcher — fetch_url over a mocked transport
# ──────────────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_httpx(monkeypatch):
    """Patch the fetcher's AsyncClient to route through a MockTransport.

    Tests set `mock_httpx.handler` to a callable taking an httpx.Request
    and returning an httpx.Response.
    """

    class _Holder:
        handler = staticmethod(lambda req: httpx.Response(200, text="ok"))

    holder = _Holder()
    real_client = httpx.AsyncClient

    def fake_client(*args, **kwargs):
        kwargs.pop("headers", None)
        kwargs.pop("timeout", None)
        return real_client(
            transport=httpx.MockTransport(holder.handler),
            follow_redirects=kwargs.get("follow_redirects", True),
        )

    monkeypatch.setattr(fetcher.httpx, "AsyncClient", fake_client)
    return holder


async def test_fetch_url_returns_clean_text_for_html(mock_httpx):
    mock_httpx.handler = lambda req: httpx.Response(
        200,
        headers={"content-type": "text/html"},
        text="<h1>Doc</h1><p>Body text here</p>",
    )
    url, text = await fetcher.fetch_url("https://example.com/a")
    assert url == "https://example.com/a"
    assert "Doc" in text
    assert "Body text here" in text


async def test_fetch_url_passes_plain_text_through(mock_httpx):
    mock_httpx.handler = lambda req: httpx.Response(
        200, headers={"content-type": "text/plain"}, text="raw notes"
    )
    _, text = await fetcher.fetch_url("https://example.com/notes.txt")
    assert text == "raw notes"


async def test_fetch_url_rejects_unsupported_content_type(mock_httpx):
    mock_httpx.handler = lambda req: httpx.Response(
        200, headers={"content-type": "image/png"}, content=b"\x89PNG"
    )
    with pytest.raises(ValueError, match="Unsupported content type"):
        await fetcher.fetch_url("https://example.com/img.png")


async def test_fetch_url_truncates_long_text(mock_httpx):
    big = "x" * 200_000
    mock_httpx.handler = lambda req: httpx.Response(
        200, headers={"content-type": "text/plain"}, text=big
    )
    _, text = await fetcher.fetch_url("https://example.com/big", max_chars=1000)
    assert len(text) < len(big)
    assert "Truncated" in text


# ──────────────────────────────────────────────────────────────────────────
# topic — persistence
# ──────────────────────────────────────────────────────────────────────────


def test_topic_store_create_and_load_roundtrip(tmp_path):
    store = TopicStore(str(tmp_path))
    topic = store.create(
        title="Eros in Plato",
        question="How does Plato frame eros across the dialogues?",
        seeds=["https://example.com/symposium"],
    )
    assert len(topic.id) == 8
    assert topic.status == "active"
    assert topic.run_count == 0

    loaded = store.load(topic.id)
    assert loaded is not None
    assert loaded.title == "Eros in Plato"
    assert loaded.seeds == ["https://example.com/symposium"]


def test_topic_store_persists_mutations(tmp_path):
    store = TopicStore(str(tmp_path))
    topic = store.create(title="T", question="Q")
    topic.gaps.append("open question")
    topic.synthesis = "first pass"
    topic.sources_indexed.append("https://src")
    topic.run_count = 2
    store.save(topic)

    loaded = store.load(topic.id)
    assert loaded.gaps == ["open question"]
    assert loaded.synthesis == "first pass"
    assert loaded.sources_indexed == ["https://src"]
    assert loaded.run_count == 2


def test_topic_store_list_all_returns_every_topic(tmp_path):
    store = TopicStore(str(tmp_path))
    store.create(title="A", question="qa")
    store.create(title="B", question="qb")
    titles = {t.title for t in store.list_all()}
    assert titles == {"A", "B"}


def test_topic_store_load_missing_returns_none(tmp_path):
    store = TopicStore(str(tmp_path))
    assert store.load("does-not-exist") is None


def test_topic_from_dict_ignores_unknown_keys():
    topic = ResearchTopic.from_dict(
        {"id": "abc", "title": "T", "question": "Q", "legacy_field": "ignored"}
    )
    assert topic.id == "abc"
    assert not hasattr(topic, "legacy_field")


# ──────────────────────────────────────────────────────────────────────────
# repl_env — research actions
# ──────────────────────────────────────────────────────────────────────────


class _FakeKnowledge:
    """In-memory stand-in for KnowledgeStore."""

    def __init__(self):
        self.docs: list[dict] = []

    async def add_document(
        self, text, source, doc_type="documentation", metadata=None, chunk_index=0
    ):
        self.docs.append(
            {
                "content": text,
                "source": source,
                "doc_type": doc_type,
                "metadata": {**(metadata or {}), "source": source},
                "chunk_index": chunk_index,
            }
        )

    async def search(self, query, n_results=5, doc_type=None):
        out = []
        for d in self.docs:
            if doc_type and d["doc_type"] != doc_type:
                continue
            out.append(
                {
                    "content": d["content"],
                    "source": d["source"],
                    "doc_type": d["doc_type"],
                    "relevance": 0.9,
                    "metadata": d["metadata"],
                }
            )
        return out[:n_results]


def _make_env(topic, knowledge):
    return ResearchREPLEnvironment(
        topic=topic,
        topic_store=_FakeTopicStore(),
        knowledge=knowledge,
        max_depth=3,
    )


class _FakeTopicStore:
    def __init__(self):
        self.saves = 0

    def save(self, topic):
        self.saves += 1


@pytest.fixture
def topic():
    return ResearchTopic(id="t1", title="T", question="Q")


async def test_log_gap_dedupes(topic):
    env = _make_env(topic, _FakeKnowledge())
    await env.execute({"action": "log_gap", "gap": "what is X?"})
    await env.execute({"action": "log_gap", "gap": "what is X?"})
    assert topic.gaps == ["what is X?"]


async def test_log_gap_requires_text(topic):
    env = _make_env(topic, _FakeKnowledge())
    res = await env.execute({"action": "log_gap", "gap": "  "})
    assert res["ok"] is True
    assert "required" in res["result"].lower()
    assert topic.gaps == []


async def test_update_synthesis_persists_and_loads_variable(topic):
    env = _make_env(topic, _FakeKnowledge())
    res = await env.execute({"action": "update_synthesis", "synthesis": "the answer is 42"})
    assert res["ok"] is True
    assert topic.synthesis == "the answer is 42"
    assert "synthesis" in env._vars


async def test_ingest_chunks_indexes_and_marks_source(topic):
    knowledge = _FakeKnowledge()
    env = _make_env(topic, knowledge)
    env.load("doc1", "para one. " * 400, var_type="document")

    res = await env.execute(
        {"action": "ingest_chunks", "var": "$doc1", "source_url": "https://src/a"}
    )
    assert res["ok"] is True
    assert "https://src/a" in topic.sources_indexed
    assert len(knowledge.docs) >= 1
    assert all(d["metadata"]["topic_id"] == "t1" for d in knowledge.docs)


async def test_ingest_chunks_skips_already_indexed(topic):
    knowledge = _FakeKnowledge()
    topic.sources_indexed.append("https://src/a")
    env = _make_env(topic, knowledge)
    env.load("doc1", "content", var_type="document")

    res = await env.execute(
        {"action": "ingest_chunks", "var": "$doc1", "source_url": "https://src/a"}
    )
    assert "Already indexed" in res["result"]
    assert knowledge.docs == []


async def test_ingest_chunks_errors_on_missing_var(topic):
    env = _make_env(topic, _FakeKnowledge())
    res = await env.execute({"action": "ingest_chunks", "var": "$nope"})
    assert "not found" in res["result"]


async def test_search_knowledge_filters_by_topic(topic):
    knowledge = _FakeKnowledge()
    # one doc in our topic, one in another topic
    await knowledge.add_document("relevant passage", "https://a", "research", {"topic_id": "t1"})
    await knowledge.add_document("other topic", "https://b", "research", {"topic_id": "OTHER"})
    env = _make_env(topic, knowledge)

    res = await env.execute({"action": "search_knowledge", "query": "passage"})
    assert res["ok"] is True
    assert "relevant passage" in res["result"]
    assert "other topic" not in res["result"]


async def test_search_knowledge_empty_corpus_guides_user(topic):
    env = _make_env(topic, _FakeKnowledge())
    res = await env.execute({"action": "search_knowledge", "query": "anything"})
    assert "fetch_url" in res["result"]


async def test_unknown_action_falls_through_to_base(topic):
    env = _make_env(topic, _FakeKnowledge())
    env.load("doc1", "hello world", var_type="document")
    # 'peek' is a base REPLEnvironment action — should be handled by super()
    res = await env.execute({"action": "peek", "var": "$doc1"})
    assert isinstance(res, dict)


def test_action_schema_includes_research_actions():
    schema = ResearchREPLEnvironment.action_schema()
    for action in ("fetch_url", "ingest_chunks", "log_gap", "update_synthesis", "search_knowledge"):
        assert action in schema

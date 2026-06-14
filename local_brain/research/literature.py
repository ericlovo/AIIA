"""
Literature topics — factory for research topics over English literature.

A subject is an author, a work, a movement, or a theme ("Virginia Woolf",
"Mrs Dalloway", "Romanticism", "the dramatic monologue"). A topic created
here carries the 'literature' profile (see profiles.py) and is seeded with
the subject's Wikipedia article as a starting reference the loop can branch
from — callers can pass additional seeds (a primary text, a specific essay).

The research loop has no web-search action, only fetch_url of explicit URLs,
so a good starting seed matters: the Wikipedia article is a reliable,
fetchable anchor that links out to primary texts and criticism.
"""

from urllib.parse import quote

from local_brain.research.profiles import LITERATURE
from local_brain.research.topic import ResearchTopic, TopicStore

WIKIPEDIA_BASE_URL = "https://en.wikipedia.org/wiki"


def _normalize(subject: str) -> str:
    return " ".join(subject.split()).strip()


def wikipedia_url(subject: str) -> str:
    """Build the Wikipedia article URL for a subject (spaces → underscores)."""
    title = _normalize(subject).replace(" ", "_")
    return f"{WIKIPEDIA_BASE_URL}/{quote(title)}"


def topic_title(subject: str) -> str:
    return f"Literature: {_normalize(subject)}"


def build_question(subject: str) -> str:
    subject = _normalize(subject)
    return (
        f"Build a researched overview of {subject} in English literature. "
        "Establish historical and biographical context, summarize the work or "
        "movement and its form, identify central themes, and survey the major "
        "critical interpretations with attribution. Separate primary text from "
        "secondary criticism, and verify quotations before using them."
    )


def find_literature_topic(store: TopicStore, subject: str) -> ResearchTopic | None:
    """Return the existing topic for this subject, if one was created."""
    title = topic_title(subject)
    for topic in store.list_all():
        if topic.title == title:
            return topic
    return None


def create_literature_topic(
    store: TopicStore,
    subject: str,
    extra_seeds: list[str] | None = None,
) -> ResearchTopic:
    """
    Create a research topic for an English-literature subject, seeded with
    its Wikipedia article plus any caller-supplied seeds (a primary text URL,
    a specific critical essay).
    """
    subject = _normalize(subject)
    if not subject:
        raise ValueError("Literature subject must be a non-empty string")
    seeds = [wikipedia_url(subject)]
    for seed in extra_seeds or []:
        if seed not in seeds:
            seeds.append(seed)
    return store.create(
        title=topic_title(subject),
        question=build_question(subject),
        seeds=seeds,
        profile=LITERATURE,
    )

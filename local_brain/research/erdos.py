"""
Erdős problem topics — factory for research topics over erdosproblems.com.

Each problem in Thomas Bloom's database lives at erdosproblems.com/{n}.
A topic created here is seeded with that page, carries the 'erdos' profile
(see profiles.py), and asks the loop to build a status report: precise
statement, known partial results and bounds, promising approaches.
"""

from local_brain.research.profiles import ERDOS
from local_brain.research.topic import ResearchTopic, TopicStore

ERDOS_BASE_URL = "https://www.erdosproblems.com"


def problem_url(number: int) -> str:
    return f"{ERDOS_BASE_URL}/{number}"


def topic_title(number: int) -> str:
    return f"Erdős Problem #{number}"


def build_question(number: int) -> str:
    return (
        f"What is the current status of Erdős problem #{number}? "
        "State the problem precisely, survey known partial results and best "
        "bounds with attribution, and identify promising approaches and open "
        "subquestions."
    )


def find_erdos_topic(store: TopicStore, number: int) -> ResearchTopic | None:
    """Return the existing topic for this problem number, if one was created."""
    title = topic_title(number)
    for topic in store.list_all():
        if topic.title == title:
            return topic
    return None


def create_erdos_topic(
    store: TopicStore,
    number: int,
    extra_seeds: list[str] | None = None,
) -> ResearchTopic:
    """
    Create a research topic for Erdős problem {number}, seeded with its
    erdosproblems.com page plus any caller-supplied seeds (e.g. arXiv
    abstract URLs already known to be relevant).
    """
    if number < 1:
        raise ValueError(f"Erdős problem number must be positive, got {number}")
    seeds = [problem_url(number)]
    for seed in extra_seeds or []:
        if seed not in seeds:
            seeds.append(seed)
    return store.create(
        title=topic_title(number),
        question=build_question(number),
        seeds=seeds,
        profile=ERDOS,
    )

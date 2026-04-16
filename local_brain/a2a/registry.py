"""
In-memory A2A agent registry for the Local Brain.

Holds the full set of agents available on this Mini: maps agent_id to
(AgentCard, AgentExecutor) pairs. Supports tag-based discovery queries
used by planner agents and scoped dev workflows.

Rebuilt on every startup from bootstrap.register_default_agents().
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from local_brain.a2a.executors.base import AgentExecutor
from local_brain.a2a.schema import AgentCard


@dataclass
class RegisteredAgent:
    agent_id: str
    card: AgentCard
    executor: AgentExecutor


class AgentRegistry:
    """Tag-indexed in-memory registry of A2A agents on this host."""

    def __init__(self) -> None:
        self._agents: Dict[str, RegisteredAgent] = {}

    def register(
        self,
        agent_id: str,
        card: AgentCard,
        executor: AgentExecutor,
    ) -> None:
        if agent_id in self._agents:
            raise ValueError(f"agent_id already registered: {agent_id}")
        self._agents[agent_id] = RegisteredAgent(
            agent_id=agent_id, card=card, executor=executor
        )

    def unregister(self, agent_id: str) -> None:
        self._agents.pop(agent_id, None)

    def get(self, agent_id: str) -> Optional[RegisteredAgent]:
        return self._agents.get(agent_id)

    def all(self) -> List[RegisteredAgent]:
        return list(self._agents.values())

    def query(
        self,
        *,
        tags: Optional[Iterable[str]] = None,
        require_all: bool = False,
    ) -> List[RegisteredAgent]:
        """
        Return agents whose skills carry the requested tags.

        - tags=None returns every agent.
        - require_all=False (default): an agent matches if it has *any* of
          the requested tags — the usual OR-scope query like
          "product:my-app OR scope:global".
        - require_all=True: agent must carry every requested tag, for
          narrower intersections.
        """
        if not tags:
            return self.all()

        requested = set(tags)
        matches: List[RegisteredAgent] = []
        for agent in self._agents.values():
            agent_tags = set(agent.card.all_tags())
            if require_all:
                if requested.issubset(agent_tags):
                    matches.append(agent)
            else:
                if agent_tags & requested:
                    matches.append(agent)
        return matches

    def __len__(self) -> int:
        return len(self._agents)

    def __contains__(self, agent_id: object) -> bool:
        return agent_id in self._agents

"""
Erdős Loop — traces the co-authorship chain from any researcher to Paul Erdős.

Uses Semantic Scholar's free API to traverse the collaboration graph.
Returns the Erdős number and the full path (if found within max depth).

Demo use case: show AIIA doing live graph research with real data.
"""

import logging
from collections import deque
from typing import Any

import httpx

logger = logging.getLogger("aiia.research.erdos")

SEMANTIC_SCHOLAR_SEARCH = "https://api.semanticscholar.org/graph/v1/author/search"
SEMANTIC_SCHOLAR_AUTHOR = "https://api.semanticscholar.org/graph/v1/author/{author_id}"
SEMANTIC_SCHOLAR_PAPERS = "https://api.semanticscholar.org/graph/v1/author/{author_id}/papers"

ERDOS_S2_ID = "1741101"  # Paul Erdős on Semantic Scholar
ERDOS_ALIASES = {"paul erdős", "paul erdos", "p. erdős", "p. erdos"}

MAX_DEPTH = 5
COLLAB_LIMIT = 20  # collaborators to check per author (keep fast)


async def _search_author(name: str, client: httpx.AsyncClient) -> dict[str, Any] | None:
    """Find an author on Semantic Scholar by name. Returns first match."""
    try:
        resp = await client.get(
            SEMANTIC_SCHOLAR_SEARCH,
            params={"query": name, "fields": "authorId,name,paperCount", "limit": 3},
            timeout=8.0,
        )
        resp.raise_for_status()
        data = resp.json()
        authors = data.get("data", [])
        if not authors:
            return None
        # Prefer exact name match, else take first result
        for a in authors:
            if a.get("name", "").lower() == name.lower():
                return a
        return authors[0]
    except Exception as e:
        logger.warning(f"Author search failed for '{name}': {e}")
        return None


async def _get_collaborators(author_id: str, client: httpx.AsyncClient) -> list[dict]:
    """Get unique collaborators for an author via their papers."""
    try:
        resp = await client.get(
            SEMANTIC_SCHOLAR_PAPERS.format(author_id=author_id),
            params={"fields": "authors", "limit": 50},
            timeout=10.0,
        )
        resp.raise_for_status()
        papers = resp.json().get("data", [])

        seen = set()
        collaborators = []
        for paper in papers:
            for author in paper.get("authors", []):
                aid = author.get("authorId")
                aname = author.get("name", "")
                if aid and aid != author_id and aid not in seen:
                    seen.add(aid)
                    collaborators.append({"authorId": aid, "name": aname})
                    if len(collaborators) >= COLLAB_LIMIT:
                        return collaborators
        return collaborators
    except Exception as e:
        logger.warning(f"Collaborator fetch failed for {author_id}: {e}")
        return []


async def compute_erdos_number(name: str) -> dict[str, Any]:
    """
    Compute the Erdős number for a researcher.

    Returns:
        {
            "name": str,
            "erdos_number": int | None,
            "path": list[str],     # [name → collab1 → ... → Erdős]
            "found": bool,
            "searched_authors": int,
            "note": str,
        }
    """
    async with httpx.AsyncClient() as client:
        # Step 1: find the target author
        target = await _search_author(name, client)
        if not target:
            return {
                "name": name,
                "erdos_number": None,
                "path": [],
                "found": False,
                "searched_authors": 0,
                "note": f"Could not find '{name}' on Semantic Scholar.",
            }

        target_id = target["authorId"]
        target_name = target["name"]

        # Check if the person IS Erdős
        if target_id == ERDOS_S2_ID or target_name.lower() in ERDOS_ALIASES:
            return {
                "name": target_name,
                "erdos_number": 0,
                "path": [target_name],
                "found": True,
                "searched_authors": 1,
                "note": "This is Paul Erdős himself.",
            }

        # Step 2: BFS from target toward Erdős
        # queue: (author_id, author_name, depth, path_so_far)
        queue: deque = deque([(target_id, target_name, 1, [target_name])])
        visited: set[str] = {target_id}
        searched = 1

        while queue:
            author_id, author_name, depth, path = queue.popleft()

            if depth > MAX_DEPTH:
                continue

            collabs = await _get_collaborators(author_id, client)
            searched += 1

            for collab in collabs:
                cid = collab["authorId"]
                cname = collab["name"]

                # Found Erdős!
                if cid == ERDOS_S2_ID or cname.lower() in ERDOS_ALIASES:
                    full_path = path + ["Paul Erdős"]
                    return {
                        "name": target_name,
                        "erdos_number": depth,
                        "path": full_path,
                        "found": True,
                        "searched_authors": searched,
                        "note": f"Erdős number {depth} found via {len(full_path) - 1}-hop chain.",
                    }

                if cid not in visited:
                    visited.add(cid)
                    queue.append((cid, cname, depth + 1, path + [cname]))

        return {
            "name": target_name,
            "erdos_number": None,
            "path": [],
            "found": False,
            "searched_authors": searched,
            "note": f"No path to Erdős found within {MAX_DEPTH} hops ({searched} authors searched).",
        }

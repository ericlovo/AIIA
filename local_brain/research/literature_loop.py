"""
Literature Loop — autonomous paper discovery, summarization, and synthesis.

Given a research question or topic, AIIA:
1. Searches arXiv for recent relevant papers
2. Fetches abstracts and key content
3. Runs a local LLM synthesis loop
4. Returns structured findings with citations

Demo use case: show AIIA doing live literature review.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any

import httpx

logger = logging.getLogger("aiia.research.literature")

ARXIV_SEARCH = "https://export.arxiv.org/api/query"
SEMANTIC_SCHOLAR_PAPER_SEARCH = "https://api.semanticscholar.org/graph/v1/paper/search"

MAX_PAPERS = 8
SYNTHESIS_MAX_TOKENS = 2048


async def _search_arxiv(query: str, max_results: int = MAX_PAPERS) -> list[dict]:
    """Search arXiv and return paper metadata."""
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            resp = await client.get(ARXIV_SEARCH, params=params)
            resp.raise_for_status()
            content = resp.text

        # Parse Atom XML
        papers = []
        import xml.etree.ElementTree as ET
        ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
        root = ET.fromstring(content)

        for entry in root.findall("atom:entry", ns):
            title = entry.findtext("atom:title", "", ns).strip().replace("\n", " ")
            abstract = entry.findtext("atom:summary", "", ns).strip().replace("\n", " ")
            published = entry.findtext("atom:published", "", ns)[:10]
            arxiv_id_url = entry.findtext("atom:id", "", ns)
            arxiv_id = arxiv_id_url.split("/abs/")[-1] if "/abs/" in arxiv_id_url else arxiv_id_url

            authors = []
            for author in entry.findall("atom:author", ns):
                aname = author.findtext("atom:name", "", ns).strip()
                if aname:
                    authors.append(aname)

            papers.append({
                "id": arxiv_id,
                "title": title,
                "abstract": abstract[:600],
                "authors": authors[:5],
                "published": published,
                "url": f"https://arxiv.org/abs/{arxiv_id}",
            })

        return papers
    except Exception as e:
        logger.warning(f"arXiv search failed: {e}")
        return []


async def _search_semantic_scholar(query: str, max_results: int = MAX_PAPERS) -> list[dict]:
    """Search Semantic Scholar as fallback/supplement."""
    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            resp = await client.get(
                SEMANTIC_SCHOLAR_PAPER_SEARCH,
                params={
                    "query": query,
                    "fields": "title,abstract,authors,year,externalIds,openAccessPdf",
                    "limit": max_results,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        papers = []
        for p in data.get("data", []):
            arxiv_id = p.get("externalIds", {}).get("ArXiv", "")
            papers.append({
                "id": p.get("paperId", ""),
                "title": p.get("title", ""),
                "abstract": (p.get("abstract") or "")[:600],
                "authors": [a["name"] for a in p.get("authors", [])[:5]],
                "published": str(p.get("year", "")),
                "url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else "",
            })
        return papers
    except Exception as e:
        logger.warning(f"Semantic Scholar search failed: {e}")
        return []


async def run_literature_loop(
    question: str,
    ollama_client: Any,  # OllamaClient
    model: str = "llama3.1:8b-instruct-q8_0",
) -> dict[str, Any]:
    """
    Run a full literature loop for a research question.

    Returns:
        {
            "question": str,
            "papers_found": int,
            "papers": list[dict],
            "synthesis": str,
            "key_themes": list[str],
            "gaps": list[str],
            "citations": list[str],
        }
    """
    logger.info(f"Literature loop: {question}")

    # Step 1: Gather papers from both sources in parallel
    arxiv_papers, ss_papers = await asyncio.gather(
        _search_arxiv(question),
        _search_semantic_scholar(question),
    )

    # Merge and deduplicate by title similarity
    all_papers = arxiv_papers[:]
    seen_titles = {p["title"].lower()[:40] for p in arxiv_papers}
    for p in ss_papers:
        if p["title"].lower()[:40] not in seen_titles and p.get("abstract"):
            all_papers.append(p)
            seen_titles.add(p["title"].lower()[:40])
    all_papers = all_papers[:MAX_PAPERS]

    if not all_papers:
        return {
            "question": question,
            "papers_found": 0,
            "papers": [],
            "synthesis": "No papers found for this query.",
            "key_themes": [],
            "gaps": [],
            "citations": [],
        }

    # Step 2: Build synthesis prompt
    paper_text = "\n\n".join(
        f"[{i+1}] **{p['title']}** ({p['published']})\n"
        f"Authors: {', '.join(p['authors'])}\n"
        f"Abstract: {p['abstract']}"
        for i, p in enumerate(all_papers)
    )

    system_prompt = """You are AIIA conducting a structured literature review.
Analyze the provided papers and produce:
1. A synthesis paragraph (3-5 sentences) capturing the state of research
2. 3-5 key themes emerging across papers
3. 2-3 open research gaps or unanswered questions

Return a JSON object with keys: synthesis (string), key_themes (list of strings), gaps (list of strings)
Return ONLY valid JSON, no markdown."""

    user_prompt = f"""Research question: {question}

Papers to synthesize:
{paper_text}

Produce the structured synthesis now."""

    # Step 3: Local LLM synthesis
    synthesis = "Synthesis unavailable (model error)."
    key_themes: list[str] = []
    gaps: list[str] = []

    try:
        response = await ollama_client.chat(
            model=model,
            messages=[{"role": "user", "content": user_prompt}],
            system=system_prompt,
            temperature=0.2,
            max_tokens=SYNTHESIS_MAX_TOKENS,
        )
        content = response.get("message", {}).get("content", "")

        import json as _json
        # Parse JSON from response
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            parsed = _json.loads(content[start:end])
            synthesis = parsed.get("synthesis", synthesis)
            key_themes = parsed.get("key_themes", [])
            gaps = parsed.get("gaps", [])
        else:
            synthesis = content[:800]
    except Exception as e:
        logger.warning(f"Synthesis LLM call failed: {e}")

    citations = [
        f"{p['title']} — {', '.join(p['authors'][:2])} ({p['published']}) {p['url']}"
        for p in all_papers
    ]

    return {
        "question": question,
        "papers_found": len(all_papers),
        "papers": all_papers,
        "synthesis": synthesis,
        "key_themes": key_themes,
        "gaps": gaps,
        "citations": citations,
    }

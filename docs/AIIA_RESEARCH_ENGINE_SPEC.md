# AIIA Research Engine

The Research Engine turns AIIA from a question-answerer into an autonomous
researcher. You give it a **topic** — a title and a question — and it runs
**sessions** that fetch sources, index them into the knowledge base, track open
questions, and grow a running synthesis. Sessions are cumulative: the corpus and
the synthesis deepen every time you run one, until the question is answered.

It is *not* a one-shot RAG call. It is a loop with memory.

## Where it lives

```
local_brain/research/
├── __init__.py        # package docstring
├── topic.py           # ResearchTopic dataclass + TopicStore (JSON persistence)
├── fetcher.py         # httpx URL fetch + stdlib HTML→text (no extra deps)
├── repl_env.py        # ResearchREPLEnvironment — 5 research actions on the REPL
├── engine.py          # ResearchEngine — orchestrates one session
└── router.py          # FastAPI /v1/research/* endpoints
```

It is built *on top of* the existing EQ-Brain primitives rather than duplicating
them:

| Reuses | From | For |
|---|---|---|
| `RecursiveEngine` | `eq_brain/recursive_engine.py` | The REPL action loop + token budget |
| `REPLEnvironment` | `eq_brain/repl_env.py` | Base actions (peek/search/store/sub_ask/final) |
| `KnowledgeStore` | `eq_brain/knowledge_store.py` | ChromaDB vector index |
| `OllamaClient` | `ollama_client.py` | Local LLM dispatch |

No second model init, no second vector store. The research harness shares the
same `KnowledgeStore` and `OllamaClient` AIIA already holds (wired in
`local_api.py` at startup).

## Data model

A **topic** is the unit of persistence. It lives at
`{eq_brain_data_dir}/research/{id}.json` and accumulates across sessions.

```python
@dataclass
class ResearchTopic:
    id: str                       # 8-char uuid slug
    title: str
    question: str
    status: str = "active"        # active | paused | complete
    created_at: str
    seeds: list[str] = []         # URLs to fetch first
    gaps: list[str] = []          # open questions, logged across sessions
    synthesis: str = ""           # the running answer — grows, never flattens
    sources_indexed: list[str] = []  # URLs already chunked into ChromaDB
    run_count: int = 0
    last_run: str | None = None
```

`TopicStore` is a thin JSON-backed CRUD layer (`create / save / load /
list_all`). Indexed *content* lives in ChromaDB under `doc_type="research"` with
`metadata.topic_id` scoping each chunk to its topic; the topic JSON only holds
the lightweight state above.

## The session loop

One call to `ResearchEngine.run(topic)` is one session. It:

1. **Loads topic context** into the REPL as variables — the question, the
   current synthesis (mutable), and the open gaps.
2. **Pre-fetches unprocessed seeds** (up to 3 per session) so the model starts
   with material already in hand.
3. **Runs the recursive loop** (`max_iterations=20`, `token_budget=80k` —
   deeper than the default ask) under a research-oriented system prompt.
4. **Persists** synthesis + gaps + run metadata back to the topic as the model
   acts. If the model never calls `update_synthesis`, the final answer seeds the
   first synthesis.

The loop streams SSE events (`meta`, `action`, `result`, `done`, `error`) — the
same event shape the rest of AIIA's REPL surfaces emit.

### Research actions

`ResearchREPLEnvironment` extends the base six REPL actions with five more:

| Action | What it does |
|---|---|
| `fetch_url` | Fetch a URL, strip to text, load as a `$var` |
| `ingest_chunks` | Chunk a `$var` (1200/150 split) + embed into ChromaDB under the topic; mark the source indexed |
| `search_knowledge` | Vector search the topic's indexed corpus (topic-scoped) |
| `log_gap` | Record an open question for future sessions (deduped) |
| `update_synthesis` | Overwrite the running synthesis doc |

The intended flow: **`fetch_url` → `ingest_chunks` → `search_knowledge` →
synthesize → `log_gap` → `final()`**. `final()` ends the *session*, not the
research — the topic stays `active` and the next run picks up the logged gaps.

```
        ┌──────────────────────────── one session ───────────────────────────┐
seeds ─▶│ fetch_url ─▶ ingest_chunks ─▶ search_knowledge ─▶ update_synthesis  │
        │      ▲                                    │                         │
        │      └──────────── log_gap ◀──────────────┘                         │
        └────────────────────────────────┬──────────────────────────────────┘
                                          ▼
                         topic.json  (synthesis grows, gaps carry over)
                                          │
                                          ▼
                            next session resumes from gaps
```

## HTTP API — `/v1/research/*`

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/research/topics` | Create a topic (`{title, question, seeds}`) |
| `GET` | `/v1/research/topics` | List topics, newest first |
| `GET` | `/v1/research/topics/{id}` | Full topic state |
| `POST` | `/v1/research/topics/{id}/run` | Run one session — **SSE stream** |
| `GET` | `/v1/research/topics/{id}/synthesis` | Synthesis + gaps + progress stats |

A topic marked `complete` returns `409` on `/run`; set it back to `active` to
continue. The router is injected with the engine + store at startup via
`init_research()`; if AIIA failed to initialize, the endpoints return `503`.

## Surfaces

The engine is reachable from every place AIIA already exposes itself:

**CLI** (`local_brain/cli.py`):

```bash
aiia research new "Platonic eros" -q "How does Plato frame eros?" -s https://…
aiia research list
aiia research run <topic_id>      # streams session progress
aiia research show <topic_id>     # synthesis + open gaps
```

**MCP** (`local_brain/mcp_server.py`) — so Claude Code / Cursor can drive it:

| Tool | Purpose |
|---|---|
| `aiia_research_new` | Create a topic |
| `aiia_research_list` | List topics + progress |
| `aiia_research_run` | Run one session (consumes the SSE stream, returns a summary) |
| `aiia_research_synthesis` | Read synthesis + gaps |

## Design choices

- **Cumulative, not one-shot.** The synthesis is extended each run, never
  flattened. Gaps logged in session N are the agenda for session N+1. This is
  what makes it a *research engine* and not a fancy RAG query.
- **No new heavy deps.** The fetcher uses `httpx` (already present) + the stdlib
  `html.parser` for HTML stripping. `pypdf` is declared for future PDF
  ingestion.
- **Topic-scoped corpus.** Every chunk carries `topic_id` in its metadata;
  `search_knowledge` filters to the active topic so cross-topic bleed doesn't
  pollute a synthesis.
- **Shared infrastructure.** One `KnowledgeStore`, one `OllamaClient`, one model
  — research rides on AIIA's existing brain rather than standing up a parallel
  stack.

## Testing

`local_brain/tests/test_research.py` covers the engine's pure-Python surface
without a live Ollama/Brain: HTML stripping + `fetch_url` content negotiation
(via `httpx.MockTransport`), `TopicStore` persistence roundtrips, and all five
research actions against an in-memory fake `KnowledgeStore`. These run under a
plain `pytest` — no `--collect-only` needed.

## Roadmap

- PDF ingestion (`pypdf` is already declared).
- A scheduled autonomy loop (`autonomy/`) that advances `active` topics on a
  cadence instead of on manual `run`.
- Source-quality scoring / citation tracking in the synthesis.
- `status="complete"` auto-detection when gaps drain to zero.

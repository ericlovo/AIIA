# AIIA Research Loop — autonomous, persistent research

The research loop is the part of AIIA that works a question across many
sessions: it fetches sources, ingests them into the ChromaDB knowledge store,
searches what it has learned, and grows a cumulative synthesis with logged
gaps. A *topic* holds that state on disk and accumulates over time — so a
problem can be worked a little each night and get better, not restart.

It runs three ways, all over the same `TopicStore`:

| Surface | Where | Best for |
|---|---|---|
| **`aiia research` CLI** | any host that can reach the Brain (`:8100`) | creating topics, kicking a session, reading synthesis |
| **`POST /v1/research/*`** | Brain API | programmatic use, the console, automations |
| **nightly LaunchAgent** | the mini | unattended progress on the standing topic set |

## Profiles

A *profile* shapes how a session behaves on a topic (the loop mechanics are
shared). See `local_brain/research/profiles.py`.

| Profile | Tuned for |
|---|---|
| `general` | topic-agnostic deep research (original behavior) |
| `erdos` | open problems in mathematics (erdosproblems.com); verbatim statements, sourced claims only, proved-vs-conjectured |
| `literature` | English literature; primary text vs criticism, attributed readings, located + verified quotations |

## Creating topics

```bash
# An Erdős problem (seeded from erdosproblems.com/<n>)
aiia research erdos 28
aiia research erdos 107 --seed https://arxiv.org/abs/2312.01223

# An English-literature subject (seeded from its Wikipedia article)
aiia research literature "Mrs Dalloway"
aiia research literature "Romanticism" --seed https://www.poetryfoundation.org/...

# A general topic, via the API
curl -XPOST localhost:8100/v1/research/topics \
  -d '{"title":"...", "question":"...", "seeds":["..."], "profile":"general"}'
```

List, run, and inspect:

```bash
aiia research list             # all topics: id, profile, status, runs
aiia research run <id>         # run ONE session, streaming progress
aiia research show <id>        # synthesis doc + open gaps + stats
```

`--run` on either create command kicks a session immediately.

## The nightly loop

The scheduler (`local_brain/autonomy/research_loop.py`) works the
**least-recently-run active topics first** (fair rotation), running N sessions
on each up to a per-cycle budget. It's gated behind `AutonomyConfig` and
**ships disabled** — nothing runs until you arm it.

### Seed set

`local_brain/research/seeds.py` defines the topics the loop keeps alive. The
runner calls `ensure_seed_topics()` every cycle, so they're created once and
then worked nightly. The shipped set is three Erdős problems, each verified
against its erdosproblems.com page:

| # | Problem | Status |
|---|---|---|
| 28 | Erdős–Turán conjecture on additive bases | open |
| 107 | Erdős–Szekeres convex polygon (happy ending) | open |
| 67 | Erdős discrepancy problem | solved (Tao, 2015) |

Add a problem by appending an `ErdosSeed` to `ERDOS_SEEDS` — it joins the
rotation on the next run.

### Install (on the mini)

```bash
./scripts/install-research-agent.sh
```

This copies `scripts/com.aplora.aiia.research.plist` into
`~/Library/LaunchAgents/` and loads it. It runs nightly at **03:30** (after
memory consolidation at 03:00) and is armed by these env vars in the plist:

```
AIIA_AUTONOMY_LEVEL=phase2
AIIA_RESEARCH_ENABLED=true
```

Optional budget knobs (defaults shown):

| Env var | Default | Meaning |
|---|---|---|
| `AIIA_RESEARCH_MAX_TOPICS` | 3 | topics worked per cycle |
| `AIIA_RESEARCH_SESSIONS_PER_TOPIC` | 1 | sessions per topic per cycle |

### Run it by hand

```bash
# Through the agent, now:
launchctl kickstart -k gui/$(id -u)/com.aplora.aiia.research

# Or directly, bypassing the gate (good for a first smoke run):
python3 -m local_brain.scripts.research_runner --force

# See what would be worked, run nothing:
python3 -m local_brain.scripts.research_runner --list

# One-off budget overrides:
python3 -m local_brain.scripts.research_runner --force --max-topics 1 --sessions 2
```

The runner needs Ollama online and the same `EQ_BRAIN_DATA_DIR` the Brain
uses (so topics and the knowledge store are shared). If Ollama is offline it
logs and exits non-zero without touching state.

### Logs

```bash
tail -f ~/.aiia/logs/research/research.log
# launchd stdout/stderr: /tmp/aiia-research.out, /tmp/aiia-research.err
```

### Pause / disable

```bash
launchctl disable gui/$(id -u)/com.aplora.aiia.research   # keep loaded, stop running
# or set AIIA_RESEARCH_ENABLED=false in the plist and reload
```

## How a session works

Each session is the `ResearchEngine.run(topic)` generator — the same one
behind `POST /v1/research/topics/{id}/run`. The scheduler just drains it to
completion and lets it persist synthesis, gaps, and run bookkeeping. A session
emits `meta → action → result → … → done` events; the runner logs a summary,
`aiia research run` renders them live. PDFs (arXiv full texts) are ingested
inline — see `local_brain/research/fetcher.py`.

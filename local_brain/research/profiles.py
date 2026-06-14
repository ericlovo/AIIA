"""
ResearchProfile — domain-specific prompting for the research loop.

A profile shapes how a session behaves on a topic: what the per-session
goals are and what evidentiary standards apply. The loop mechanics
(fetch → ingest → search → synthesize → log gaps) are shared; the profile
only changes the prompt blocks the engine composes.

Profiles:
  general    — the original topic-agnostic behavior
  erdos      — open problems in mathematics (erdosproblems.com corpus)
  literature — English literature (authors, works, movements, criticism)
"""

from dataclasses import dataclass

GENERAL = "general"
ERDOS = "erdos"
LITERATURE = "literature"


@dataclass(frozen=True)
class ResearchProfile:
    id: str
    description: str
    goal: str  # "YOUR GOAL THIS SESSION" block
    principles: str  # "PRINCIPLES" block


_GENERAL = ResearchProfile(
    id=GENERAL,
    description="Topic-agnostic deep research",
    goal="""1. Fetch primary or authoritative sources that address the question or open gaps
2. Ingest them with ingest_chunks so the corpus grows
3. Search the indexed knowledge to find key passages
4. Log new gaps as you discover them — future sessions will address them
5. Update the synthesis with what this session learned
6. Call final() with a session summary: what was learned, what's still open""",
    principles="""- Build corpus first: fetch_url → ingest_chunks, then search_knowledge
- Prefer primary texts, scholarly sources, or seminal secondary sources
- The synthesis is cumulative — extend it, don't flatten previous content
- log_gap for every open question you uncover, not just the obvious ones
- final() ends this session only — the research continues across sessions""",
)

_ERDOS = ResearchProfile(
    id=ERDOS,
    description="Erdős problems — open problems in mathematics",
    goal="""1. Pin down the precise problem statement from the erdosproblems.com page (a pre-loaded
   seed on the first run) and record it verbatim at the top of the synthesis
2. Fetch sources the problem page cites: arXiv abstract pages (arxiv.org/abs/...),
   survey articles, and pages for related Erdős problems it links to
3. Ingest everything fetched with ingest_chunks, then search_knowledge for partial
   results, best known bounds, and resolved special cases
4. Log gaps: papers you could not fetch, claims you could not verify, related
   problems worth their own pass, candidate approaches mentioned but not explored
5. Update the synthesis as a status report: problem statement → status
   (open / solved / partially resolved) → known results with attribution
   (who, when, what bound) → promising approaches → references
6. Call final() with a session summary: what was established, what's still open""",
    principles="""- NEVER invent results, bounds, attributions, or citations — every mathematical
  claim in the synthesis must trace to an indexed source; if unverified, log it
  as a gap instead of stating it
- Distinguish proved results from conjectures and heuristic arguments explicitly
- Preserve LaTeX notation from sources as-is — do not paraphrase formulas
- arXiv abstract pages and PDF full texts are both fetchable — prefer the abstract
  page for framing, then fetch the PDF (arxiv.org/pdf/<id>) for proofs and bounds;
  if a source genuinely cannot be fetched, log a gap with its id for a later session
- The synthesis is cumulative — refine bounds and add results, don't flatten
  previous content
- final() ends this session only — the research continues across sessions""",
)

_LITERATURE = ResearchProfile(
    id=LITERATURE,
    description="English literature — authors, works, movements, and criticism",
    goal="""1. Establish the subject precisely — author, work (with edition/date), movement,
   or theme — and record it at the top of the synthesis
2. Fetch sources: the relevant reference/encyclopedia entry (a pre-loaded seed on
   the first run), then primary-text resources and scholarly criticism it points to
3. Ingest everything fetched with ingest_chunks, then search_knowledge for plot
   and form, historical context, themes, and competing critical readings
4. Log gaps: works or essays you could not fetch, quotations you could not verify,
   related authors/works worth their own pass, critical debates left unexplored
5. Update the synthesis as a structured overview: subject → context (period,
   movement, biography) → summary/form → themes and motifs → major critical
   interpretations (each attributed to its critic) → influence → references
6. Call final() with a session summary: what was established, what's still open""",
    principles="""- Separate PRIMARY texts (the literary work itself) from SECONDARY sources
  (criticism, biography, reference) and say which a claim rests on
- Attribute interpretations — a reading is a critic's argument, not a fact; name
  the critic and, where possible, the work and year. Never present one reading as
  settled truth
- Quote exactly and locate the quotation (act.scene.line, chapter, or page) plus
  the edition/translation; if you cannot verify a quotation, log it as a gap rather
  than reproducing it from memory
- Distinguish what the text says from what critics claim it means
- Note the edition or translation when it bears on the reading
- The synthesis is cumulative — deepen context and add readings, don't flatten
  previous content
- final() ends this session only — the research continues across sessions""",
)

PROFILES: dict[str, ResearchProfile] = {p.id: p for p in (_GENERAL, _ERDOS, _LITERATURE)}


def get_profile(profile_id: str) -> ResearchProfile:
    """Look up a profile by id, falling back to general for unknown ids."""
    return PROFILES.get(profile_id, _GENERAL)

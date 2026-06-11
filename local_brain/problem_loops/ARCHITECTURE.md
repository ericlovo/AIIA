# Erdős Problem Loops — Architecture

Autonomous, resumable loops that work open mathematics problems, write findings to
AIIA memory, and **never log progress that hasn't survived verification.** Built as a
sibling of `local_brain/story_runner/` (same worktree + planner/worker + `notify_aiia`
machinery), pointed at math instead of code.

---

## Why this exists now (the Fable thesis)

The 2025–26 wave of "AI solved an Erdős problem" results, honestly decomposed, was mostly
two modes: **literature archaeology** (the problem was already settled in an obscure paper
the model surfaced) and **computational probes** (counterexample searches and small-case
enumeration that confirmed or reshaped a conjecture). Genuine novel arguments were rare.
The reason was structural, not incidental: weaker models produce *confident wrong proofs*
and lose coherence over long deductive chains, so any responsible loop had to lean on the
two "safe" modes and gate everything behind verification.

A frontier model (Fable / Mythos class) changes the *menu of viable modes* — and, crucially,
**raises the stakes on the guardrail at the same time.** Both halves are true; treat them as
one coin.

**What opens up:**

1. **Longer coherent deductive chains** → *novel partial progress* (real lemmas, reductions,
   special-case proofs) moves from lottery-ticket to a legitimate target — not full solutions,
   but publishable fragments.
2. **Cross-problem synthesis** → a model that can hold the whole registry in context can notice
   one technique threading through several open problems. Weak-model loops could not do this;
   it is exactly what AIIA's persistent memory was built to compound.
3. **Proof-assistant collaboration (Lean)** → a strong model can plausibly drive a formalization
   loop. This is the highest-value new territory because **a machine-checked proof cannot be a
   hallucination** — formalization dissolves the confident-wrong-proof problem for anything that
   reaches it.
4. **Conjecture generation** → propose *new* Erdős-style problems (analogues, strengthenings)
   backed by computational evidence. Generating problems, not only attacking them.

**What gets more dangerous:** a stronger model's wrong proofs are *more persuasive*. So the
verification ladder must scale with the model's persuasiveness, not relax because the model is
"better." The whole AIIA/Signals thesis — *verified output for people who can't afford a wrong
answer* — applies to ourselves here. A loop that emits a slick false proof of an Erdős problem
is the single worst outcome; it is worse than emitting nothing.

---

## Loop roles (prover / refuter are separate)

Every attempt runs as an adversarial pair, never a single agent grading its own work:

- **Prover** — attempts the problem in its assigned mode. Outputs a claim + argument + (if
  computational) runnable code.
- **Refuter** — a separate agent whose *only* job is to break the prover's output: find the gap,
  the unjustified step, the off-by-one, the case not covered, the citation that already did it
  (or already refuted it). Rewarded for finding flaws, not for agreeing.
- **Arbiter** — only promotes a result to memory if it survives the verification ladder below.

This mirrors `story_runner`'s planner→coder split, with the refuter as a first-class third role
instead of an afterthought.

## The verification ladder (scales with claim strength)

A claim is only as trustworthy as the strongest rung it survives. The arbiter tags every memory
write with the rung reached.

| Claim type | Required verification | Memory category |
|---|---|---|
| Status ("already solved / open") | Two independent literature confirmations w/ citations | `lessons` (tag `status`) |
| Numerical / computational | Independent re-execution of the prover's code by the refuter, fresh process | `lessons` (tag `computational`) |
| Structural lemma / reduction | Refuter adversarial pass + literature cross-check, no surviving objection | `project` (tag `partial`) |
| Special-case or full proof | **Lean formalization that compiles** — no exceptions | `decisions` (tag `verified_proof`) |
| New conjecture | Computational evidence to N + no known counterexample in literature | `project` (tag `conjecture`) |

**Iron rule:** nothing reaches `decisions/verified_proof` without a green Lean build. Anything the
loop *wants* to call a resolution but can't formalize is logged as `partial` with the honest
caveat, never as solved.

## Pick → attempt → verify → record (one cycle)

```
pick_problem()        # from registry.json, weighted: computational_surface high for early/demo
                      #   runs, then rotate; skip problems attempted < cooldown ago
  → load context      # fetch_aiia_context(): prior lessons on THIS problem + related[] ids
  → run_mode()        # the problem's first viable mode (never proof_search cold on a hard open one)
      literature  → frontier survey, status verification
      computational → enumeration/search harness in a worktree; refuter re-runs it clean
      proof_search → prover argument → refuter adversarial pass
      formalization → Lean attempt → compiler IS the verifier
      generalization → conjecture + evidence
  → verify()          # the ladder above, prover/refuter/arbiter
  → record()          # memory.remember(finding, category, source="erdos-loop:<id>", metadata)
                      #   metadata: {problem_id, mode, rung_reached, citations, code_hash}
  → notify_aiia()     # event stream, same as story_runner
```

## Cross-problem synthesis pass (the compounding mode)

Periodically (e.g. every N cycles), a dedicated pass:
1. `recall()` all loop findings across problems.
2. Asks: which techniques recurred? Which `related[]` links proved real? What transfers from a
   solved problem's method to an open one?
3. Emits **technique-transfer hypotheses** as `project` memories, which then bias the picker.

This is the mode no weak-model loop could run, and the reason persistence matters: the loop should
get *smarter across nights*, not just grind problems independently.

## Routing (cost discipline — the LOCAL → ANTHROPIC → GOOGLE rail)

- Literature scans, code execution, enumeration harness: **local model / plain compute** ($0).
- Prover deductive steps, refuter adversarial passes, synthesis: **Fable** (the hard thinking).
- Lean interaction: **Fable**, tight loop with the compiler as ground truth.

Budget-split like `story_runner` (planning vs execution); per-cycle cap so an overnight run can't
runaway-spend.

## Honest expected outputs (set expectations here, not in a demo)

In descending likelihood: status corrections with citations · computational evidence / tightened
numerical bounds · clean write-ups of partial reductions · cross-problem technique notes ·
Lean formalizations of *known* results (genuinely valuable, fully safe) · **a novel result**
(the lottery ticket the loop buys cheaply every night). The first five are the product; the last
is the upside. Selling it as anything more is the failure mode.

## Build status

- [x] `registry.json` — seed set (15 problems), mode-tagged, honesty-tagged status priors
- [ ] `loop.py` — pick/attempt/verify/record cycle (forks `story_runner/runner.py`)
- [ ] prover / refuter / arbiter prompts (`prompts/`)
- [ ] computational harness sandbox (worktree + re-execution by refuter)
- [ ] Lean bridge (formalization rung) — later; start with the lower rungs
- [ ] cross-problem synthesis pass
- [ ] scheduler hook (autonomy module / launchd nightly), gated like Phase 2

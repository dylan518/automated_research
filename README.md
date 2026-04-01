# Tree of Thought: LLM Research Idea Generation

This project explores how iterative multi-round reasoning affects the quality of LLM-generated research ideas. The core approach is **multishot prompting**: each round fires a fresh single-turn API call that includes the full prior reasoning trace as structured context. We ran controlled experiments comparing 1–32 rounds, scored outputs with blind pairwise Elo tournaments, and identified the optimal round count.

## Key Findings

We swept over 1, 4, 8, 16, and 32 rounds across two topics and judged outputs with a blind pairwise LLM tournament (Bradley-Terry Elo, `gpt-4.1` judge, 150 pairwise judgments per sweep).

| Rounds | Hallucination eval (rank) | Multiagent systems (rank) |
|--------|--------------------------|--------------------------|
| 1      | 5                        | 5                        |
| 4      | 4                        | 2                        |
| **8**  | **2**                    | **1**                    |
| 16     | 3                        | 3                        |
| 32     | 1                        | 4                        |

**8 rounds is the most robust setting.** It finishes top-2 in both sweeps and is never dominated. 1–4 rounds consistently underperforms (p<0.001). 32 rounds shows diminishing or reversed returns on open-ended topics — by round 32 the model appears to saturate and narrow rather than expand the concept.

See [`experiments/findings_report.md`](experiments/findings_report.md) for full statistical analysis.

---

## How Idea Generation Works: Multishot Prompting

The key design choice is that **every call is a stateless single-turn prompt** — we never use a chat/messages API. Instead, the full prior reasoning trace is injected as a `PRIOR_ROUNDS` block in each new prompt. This makes each round independently inspectable and avoids context window management issues with long chat histories.

### The director-actor loop

Each round consists of two calls:

**1. Director call** — given the topic, related works, step position (`step N of M`), and all prior rounds, the director outputs:
```
REASONING: <why this is the best next step>
ACTION: <short directive for the actor>
```

Example directives: *"explore an alternative mechanism"*, *"identify the weakest assumption"*, *"design a falsifiable experiment"*, *"connect to prior work on X"*.

**2. Actor call** — given the same context plus the parsed `ACTION:` directive, the actor executes the directive and extends the reasoning. Its output becomes part of `PRIOR_ROUNDS` for the next round.

After all rounds, a **final synthesis call** reads the full trace and formats the result as a structured loose concept.

### What each prompt looks like

Every director and actor call receives:

```
TOPIC:
<research question>

STEP_CONTEXT:
step (N/M), remaining: K

RUN_GOAL:
By step (M/M), converge on a loose concept.

RELATED_WORKS_DRAFT:
<optional literature context>

PRIOR_ROUNDS:
Round 1
Directive: ...
Actor output: ...

Round 2
Directive: ...
Actor output: ...
...

[DIRECTIVE_FROM_DIRECTOR: <only in actor calls>]
```

There is no shared state between calls. Each call is self-contained — the entire reasoning history travels with the prompt.

### Run idea generation

```bash
# Default: director-actor, 4 rounds
python -m idea_generation.agent

# Recommended: 8 rounds (most robust across topics)
python -m idea_generation.agent --rounds 8 --model gpt-4.1

# With literature context
python -m idea_generation.agent --rounds 8 --related-works --model gpt-4.1
```

Outputs written to `idea_workspace/`:
- `idea_generation_output.md` — clean reasoning trace
- `idea_generation_raw.md` — full payloads (instructions + input + response per call)

Each run **overwrites** these files.

### Prompt files (editable)

All prompts are loaded from `idea_workspace/` and auto-created from defaults on first run:

| File | Controls |
|------|---------|
| `topic.md` | Research question / seed |
| `director_prompt.md` | Director system prompt + rubric |
| `actor_prompt.md` | Actor system prompt |
| `final_concept_prompt.md` | Final synthesis instructions |
| `final_concept_format.md` | Output schema |
| `response_style.md` | Style constraints (freeform scratchpad by default) |

---

## How Ideas Are Scored

Scoring uses blind pairwise LLM judging with Bradley-Terry Elo (`idea_scoring/`):

1. Generate one idea per round count.
2. For each pair, send both final concepts to a judge model with no round-count labels.
3. Repeat `judge_repeats × tournament_passes` times per pair (default: 5 × 3 = 15 judgments).
4. Fit Bradley-Terry Elo ratings (base 1500, K=24) from all pairwise outcomes.
5. Assess significance per pair with exact binomial sign test; validate global rank stability with 5000-resample bootstrap.

### Run the Elo sweep experiment

```bash
# Sweep rounds 1 4 8 16 32 (topic-only)
python experiments/elo_rounds_sweep.py

# Re-judge an existing run without regenerating
python experiments/elo_rounds_sweep.py \
  --run-dir experiments/elo_rounds_sweep/<timestamp> \
  --skip-generation
```

Results written to `experiments/elo_rounds_sweep/<timestamp>/`:
- `manifest.json` — Elo ratings and full judgment log
- `idea_rXXX_clean.md` — clean idea per round count
- `idea_rXXX_raw.md` — full raw output per round count

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env` at repo root:

```env
OPENAI_API_KEY=your-key-here
S2_KEY=your-semantic-scholar-key-here   # only needed for literature search
```

---

## Other Components

- **Literature search** (`literature_search_agent.py`): agentic Semantic Scholar loop that builds `lit_workspace/related_works.md`, optionally fed to idea generation via `--related-works`.
- **Scoring utilities** (`idea_scoring/`): `RubricScorer`, `BradleyTerryElo`, `blend_rubric_and_elo` — importable, no CLI.
- **Tests**: `python -m pytest tests/`

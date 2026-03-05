import argparse
import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from semantic_scholar import SemanticScholarClient
from version_snapshot import build_snapshot

DEFAULT_CONCEPT = "self play RL in LLMs. Challenger, Judge, Solver setups."


ACTION_SYSTEM_PROMPT = """You are an agentic literature search planner.
You MUST choose exactly one action each step:
1) QUERY
2) WRITE_TO_MEMORY

Allowed JSON outputs only:
QUERY:
{
  "action": "QUERY",
  "operations": [
    {
      "type": "search | recommendation | citation",
      "query_text": "...",
      "source_paper_id": "... (optional)",
      "filters": {"year_min": 2020}
    }
  ],
  "intent": "broad | mechanism | eval | limitations | survey | adjacent",
  "why_this_query": "1-2 sentences"
}

WRITE_TO_MEMORY:
{
  "action": "WRITE_TO_MEMORY",
  "paper_ids": ["paper_id_1"]
}

Rules:
- Prefer at least one broad `search` operation before highly specific operations.
- Prefer QUERY until there are clearly memory-worthy papers.
- If recent results include strong candidates, choose WRITE_TO_MEMORY within 1-2 steps.
- Primary objective: build a large, high-quality paper bank quickly.
- When a strong candidate appears and is not already in memory, prefer WRITE_TO_MEMORY over extra exploratory QUERY.
- Focus on high-impact, strong works only:
  - Prefer influential, widely-cited, or clearly novel papers with strong empirical evidence.
  - Prefer papers likely to shape the field, not marginal/low-signal works.
  - Prefer strong venues, strong methodology, and clear relevance to the concept.
  - If impact/confidence is weak, keep querying instead of writing.
- WRITE_TO_MEMORY must include exactly 1 paper ID.
- ID in WRITE_TO_MEMORY should come from recently retrieved results when possible.
- Do not write IDs already present in CURRENT_PAPER_BANK_IDS unless explicitly necessary.
- Avoid over-fixating on one paper; prefer coverage across strong method families.
- Output strict JSON only, no markdown.
"""


QUERY_ONLY_SYSTEM_PROMPT = """You are an agentic literature search planner in query-only mode.
You MUST output exactly one QUERY action JSON:
{
  "action": "QUERY",
  "operations": [
    {
      "type": "search | recommendation | citation",
      "query_text": "...",
      "source_paper_id": "... (optional)",
      "filters": {"year_min": 2020}
    }
  ],
  "intent": "broad | mechanism | eval | limitations | survey | adjacent",
  "why_this_query": "1-2 sentences"
}

Rules:
- Never output WRITE_TO_MEMORY in this mode.
- Prefer 1-3 operations each step (at least one broad search).
- Keep exploration diverse across methods, eval setups, and limitations.
- Output strict JSON only.
"""


QUERY_SUMMARY_PROMPT = """You are summarizing query results for an agentic literature workflow.
Given:
- The QUERY action JSON (without results summary)
- The executed retrieval results

Return strict JSON with this exact shape:
{
  "results_summary": {
    "count_returned": 30,
    "top_themes": ["theme1", "theme2", "theme3"],
    "notable_candidates": [
      {"paper_id": "...", "why_notable": "1 sentence"},
      {"paper_id": "...", "why_notable": "1 sentence"}
    ],
    "gaps_or_next_move": "1-2 sentences",
    "best_paper_candidate": {
      "paper_id": "...",
      "title": "...",
      "why_best_now": "1-3 sentences with explicit selection rationale"
    }
  }
}

Rules:
- Keep it brief and decision-relevant.
- Only mention paper IDs that are present in the retrieval results.
- If fewer than 2 notable candidates exist, return only the available ones.
- Always select one best paper candidate from the current results.
- Rationale should be explicit (novelty, evidence quality, influence, relevance, benchmark strength, etc.).
- Output strict JSON only.
"""


PAPER_NOTES_PROMPT = """You are taking research notes for a literature review.

You are working on a paper with topic:
TOPIC: {concept}

Given:
- ONE_PAPER metadata/abstract/full text
- PAPER_REFERENCE_SEEDS

Write concise, evidence-grounded markdown notes optimized for scaling to many papers.

Goals:
- Prioritize what is useful for writing a Related Works section.
- Capture only high-value details that are specific to this paper.
- Keep it short and scannable; prioritize extraction over essay-style synthesis.

Hard requirements:
- Use only information supported by the provided ONE_PAPER content.
- Include only non-obvious, source-grounded information from the provided paper content.
- Avoid generic background statements that a strong general model would likely already know without this paper.
- Prefer abstract/metadata-grounded extraction when full text is missing.
- Output format:
  1) One 2-3 sentence paper summary.
  2) "Key technical points" with 5-10 bullets.
  3) "Evaluation and evidence" with 3-6 bullets.
  4) "Limitations and caveats" with 3-6 bullets.
  5) "Related paper leads" with up to 8 bullets from PAPER_REFERENCE_SEEDS, each with: paper title (or ID) + one-line relevance.
"""


RELATED_WORK_EXTRACT_PROMPT = """You extract the likely Related Work section from raw paper text.

Input:
- FULL_PAPER_TEXT

Output:
- Return only the extracted related-work section text.
- If not found, return exactly: NOT_FOUND

Rules:
- Prefer sections titled "Related Work", "Related Works", "Background and Related Work", "Prior Work", or "Literature Review".
- Keep section boundaries tight; avoid including unrelated sections unless needed for continuity.
- Preserve key citation strings and neighboring context.
"""


RELATED_WORKS_UPDATE_PROMPT = """You are writing the Related Works section for a research paper.

You are working on a paper with topic:
TOPIC: {concept}

Requirements:
- Write a strong Related Works section based on ALL_PAPER_NOTES.
- Add only what is supported by ALL_PAPER_NOTES and provided evidence.
- Keep it concise but coherent.
- Balance coverage across notes; avoid over-focusing on the most recent paper.
- Prefer a field-overview style: organize by themes/method families, not one-paper-at-a-time summaries.
- When multiple notes exist, avoid centering on a single paper unless clearly dominant by evidence.
- If new papers introduce a new method family or key distinction, add a short bridging sentence.
- If a new paper conflicts with or qualifies existing claims, add a brief contrast.
- Output the full Related Works text only.
"""


RELATED_WORKS_FROM_PAPERS_PROMPT = """You are writing the Related Works section for a research paper.

You are working on a paper with topic:
TOPIC: {concept}

You are given COLLECTED_PAPERS (title/abstract/metadata).

Requirements:
- Write a smooth, readable Related Works section as cohesive prose (not bullet points).
- Use only claims supported by the provided abstracts/metadata.
- Organize by themes or method families.
- Keep transitions natural and avoid repetitive sentence structure.
- Avoid over-indexing on a single paper; cover the field broadly.
- If evidence is thin for a claim, phrase it cautiously.
- Output the full Related Works text only.
"""


PAPER_FILTER_PROMPT = """You are filtering collected literature search papers before synthesis.

You are working on:
TOPIC: {concept}

Input:
- COLLECTED_PAPERS (title/abstract/metadata)

Task:
- Keep papers that are relevant and reasonably strong quality for this topic.
- Remove papers that are off-topic, too weakly connected, or very low-signal quality.
- Keep broad coverage across method families; do not over-prune to only one niche.
- Prefer keeping papers with stronger evidence, clearer methodology, stronger venue/reputation, or higher influence.

Return strict JSON only:
{
  "keep_paper_ids": ["..."],
  "drop_paper_ids": ["..."],
  "filter_notes": "1-3 sentences"
}

Rules:
- keep_paper_ids and drop_paper_ids must only contain IDs present in COLLECTED_PAPERS.
- If uncertain, keep the paper.
- Keep a substantial set; this is a quality filter, not aggressive pruning.
"""


def load_dotenv_file(path: str = ".env", override_existing: bool = True) -> None:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and (override_existing or key not in os.environ):
                os.environ[key] = value


def safe_json_parse(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    raise ValueError("Model output was not valid JSON.")


def configure_logging(log_file: str, level: str) -> logging.Logger:
    logger = logging.getLogger("literature_search_agent")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.propagate = False
    return logger


def ensure_workspace_layout(workspace: Path) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "metadata_cache").mkdir(parents=True, exist_ok=True)
    if not (workspace / "queries.jsonl").exists():
        (workspace / "queries.jsonl").write_text("", encoding="utf-8")
    if not (workspace / "paper_bank.json").exists():
        (workspace / "paper_bank.json").write_text(
            json.dumps({"paper_ids": [], "updated_at": None}, indent=2), encoding="utf-8"
        )
    if not (workspace / "related_works.md").exists():
        (workspace / "related_works.md").write_text(
            "# Related Works\n\n(No papers added yet.)\n", encoding="utf-8"
        )
    if not (workspace / "paper_notes.json").exists():
        (workspace / "paper_notes.json").write_text(
            json.dumps({"notes": [], "updated_at": None}, indent=2), encoding="utf-8"
        )


def reset_workspace_state(workspace: Path) -> None:
    ensure_workspace_layout(workspace)
    (workspace / "queries.jsonl").write_text("", encoding="utf-8")
    (workspace / "paper_bank.json").write_text(
        json.dumps({"paper_ids": [], "updated_at": None}, indent=2), encoding="utf-8"
    )
    (workspace / "related_works.md").write_text(
        "# Related Works\n\n(No papers added yet.)\n", encoding="utf-8"
    )
    (workspace / "paper_notes.json").write_text(
        json.dumps({"notes": [], "updated_at": None}, indent=2), encoding="utf-8"
    )
    cache_dir = workspace / "metadata_cache"
    for path in cache_dir.glob("*.json"):
        path.unlink(missing_ok=True)


def normalize_paper(paper: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    paper_id = paper.get("paperId")
    if not paper_id:
        return None
    return {
        "paper_id": paper_id,
        "title": paper.get("title", ""),
        "year": paper.get("year"),
        "abstract": paper.get("abstract", "") or "",
        "venue": paper.get("venue"),
        "citation_count": paper.get("citationCount"),
        "url": paper.get("url"),
    }


def to_cache_filename(paper_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in paper_id) + ".json"


def load_paper_bank(workspace: Path) -> Dict[str, Any]:
    return json.loads((workspace / "paper_bank.json").read_text(encoding="utf-8"))


def save_paper_bank(workspace: Path, data: Dict[str, Any]) -> None:
    (workspace / "paper_bank.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


def append_query_log(workspace: Path, payload: Dict[str, Any]) -> None:
    with open(workspace / "queries.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def read_related_works(workspace: Path) -> str:
    return (workspace / "related_works.md").read_text(encoding="utf-8")


def write_related_works(workspace: Path, text: str) -> None:
    (workspace / "related_works.md").write_text(text.strip() + "\n", encoding="utf-8")


def load_paper_notes(workspace: Path) -> Dict[str, Any]:
    return json.loads((workspace / "paper_notes.json").read_text(encoding="utf-8"))


def save_paper_notes(workspace: Path, notes_obj: Dict[str, Any]) -> None:
    (workspace / "paper_notes.json").write_text(
        json.dumps(notes_obj, indent=2), encoding="utf-8"
    )


def call_json_model(client: OpenAI, model: str, instructions: str, user_input: str) -> Dict[str, Any]:
    response = client.responses.create(model=model, instructions=instructions, input=user_input)
    raw = response.output_text
    try:
        return safe_json_parse(raw)
    except ValueError:
        repair_instructions = """You repair model output into strict valid JSON.
Return JSON only. No markdown, no commentary.
Preserve as much original content as possible.
"""
        repair_input = (
            "ORIGINAL_INSTRUCTIONS:\n"
            + instructions
            + "\n\nBROKEN_OUTPUT:\n"
            + raw
            + "\n\nReturn a single valid JSON object."
        )
        repair = client.responses.create(
            model=model,
            instructions=repair_instructions,
            input=repair_input,
        )
        return safe_json_parse(repair.output_text)


def call_text_model(client: OpenAI, model: str, instructions: str, user_input: str) -> str:
    response = client.responses.create(model=model, instructions=instructions, input=user_input)
    return response.output_text.strip()


def extract_related_work_excerpt_heuristic(full_text: str, max_chars: int = 18000) -> str:
    text = full_text or ""
    if not text.strip():
        return ""

    lowered = text.lower()
    heading_markers = [
        "\nrelated work",
        "\nrelated works",
        "\nbackground and related work",
        "\nbackground",
        "\nprior work",
        "\nliterature review",
    ]
    start_idx = -1
    for marker in heading_markers:
        idx = lowered.find(marker)
        if idx != -1:
            start_idx = idx
            break

    # If no clear heading, return empty and let notes use full text + refs.
    if start_idx == -1:
        return ""

    tail = text[start_idx:]
    tail_lower = tail.lower()
    end_markers = [
        "\nmethod",
        "\napproach",
        "\nexperiments",
        "\nresults",
        "\nconclusion",
    ]
    end_idx = len(tail)
    for marker in end_markers:
        idx = tail_lower.find(marker, 300)
        if idx != -1:
            end_idx = min(end_idx, idx)

    excerpt = tail[:end_idx].strip()
    if len(excerpt) > max_chars:
        excerpt = excerpt[:max_chars].rstrip() + "\n...[truncated]..."
    return excerpt


def extract_related_work_excerpt(
    client: OpenAI,
    model: str,
    full_text: str,
    max_chars: int = 18000,
) -> str:
    text = full_text or ""
    if not text.strip():
        return ""
    llm_input = "FULL_PAPER_TEXT:\n" + text
    extracted = call_text_model(
        client=client,
        model=model,
        instructions=RELATED_WORK_EXTRACT_PROMPT,
        user_input=llm_input,
    ).strip()
    if extracted and extracted != "NOT_FOUND":
        if len(extracted) > max_chars:
            return extracted[:max_chars].rstrip() + "\n...[truncated]..."
        return extracted
    return extract_related_work_excerpt_heuristic(text, max_chars=max_chars)


def get_or_fetch_paper_details(
    s2_client: SemanticScholarClient, workspace: Path, paper_id: str
) -> Dict[str, Any]:
    cache_path = workspace / "metadata_cache" / to_cache_filename(paper_id)
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    details = s2_client.get_paper_details(paper_id)
    cache_path.write_text(json.dumps(details, indent=2), encoding="utf-8")
    return details


def apply_filters(papers: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
    year_min = filters.get("year_min")
    if year_min is None:
        return papers
    filtered: List[Dict[str, Any]] = []
    for paper in papers:
        year = paper.get("year")
        if isinstance(year, int) and year >= int(year_min):
            filtered.append(paper)
    return filtered


def execute_operation(
    s2_client: SemanticScholarClient, workspace: Path, operation: Dict[str, Any], default_limit: int
) -> List[Dict[str, Any]]:
    op_type = operation.get("type", "")
    query_text = operation.get("query_text", "")
    source_paper_id = operation.get("source_paper_id")
    filters = operation.get("filters", {}) or {}
    limit = int(filters.get("limit", default_limit))
    limit = max(1, min(limit, 30))

    if op_type == "search":
        year_min = filters.get("year_min")
        year_param = f"{year_min}-" if year_min else None
        raw = s2_client.search_papers(query=query_text, limit=limit, year=year_param)
        data = raw.get("data", [])
        papers = [p for p in (normalize_paper(x) for x in data) if p]
        filtered = apply_filters(papers, filters)
        if filtered:
            return filtered
        # Fallback: if strict year filtering yields nothing, retry broad query.
        if year_min:
            retry_raw = s2_client.search_papers(query=query_text, limit=limit, year=None)
            retry_data = retry_raw.get("data", [])
            return [p for p in (normalize_paper(x) for x in retry_data) if p]
        return filtered

    if op_type == "recommendation":
        if not source_paper_id:
            return []
        raw = s2_client.recommend_papers(source_paper_id, limit=limit)
        data = raw.get("recommendedPapers", raw.get("data", []))
        papers = [p for p in (normalize_paper(x) for x in data) if p]
        return apply_filters(papers, filters)

    if op_type == "citation":
        if not source_paper_id:
            return []
        details = get_or_fetch_paper_details(s2_client, workspace, source_paper_id)
        citations = details.get("citations", [])[:limit]
        papers: List[Dict[str, Any]] = []
        for item in citations:
            cited_id = item.get("paperId")
            if not cited_id:
                continue
            cited_details = get_or_fetch_paper_details(s2_client, workspace, cited_id)
            normalized = normalize_paper(cited_details)
            if normalized:
                papers.append(normalized)
        return apply_filters(papers, filters)

    return []


def summarize_query_results(
    client: OpenAI,
    model: str,
    query_action: Dict[str, Any],
    retrieved_papers: List[Dict[str, Any]],
) -> Dict[str, Any]:
    compact_results = [
        {
            "paper_id": p["paper_id"],
            "title": p.get("title", ""),
            "year": p.get("year"),
            "abstract": (p.get("abstract", "") or "")[:600],
        }
        for p in retrieved_papers[:30]
    ]
    user_input = (
        "QUERY_ACTION:\n"
        + json.dumps(query_action, indent=2)
        + "\n\nRETRIEVAL_RESULTS:\n"
        + json.dumps(compact_results, indent=2)
    )
    summary = call_json_model(client, model, QUERY_SUMMARY_PROMPT, user_input)
    return summary.get("results_summary", {})


def ensure_best_paper_candidate(
    results_summary: Dict[str, Any],
    retrieved_papers: List[Dict[str, Any]],
    existing_paper_ids: List[str],
) -> Dict[str, Any]:
    best = results_summary.get("best_paper_candidate", {})
    best_id = best.get("paper_id") if isinstance(best, dict) else None
    available_ids = {p.get("paper_id") for p in retrieved_papers if p.get("paper_id")}
    if best_id and best_id in available_ids:
        return results_summary

    if not retrieved_papers:
        results_summary["best_paper_candidate"] = {
            "paper_id": "",
            "title": "",
            "why_best_now": "No candidates returned in this query.",
        }
        return results_summary

    existing = set(existing_paper_ids)

    def score(p: Dict[str, Any]) -> float:
        citation = float(p.get("citation_count") or 0)
        year = float(p.get("year") or 0)
        novelty = 8.0 if p.get("paper_id") not in existing else 0.0
        return citation + (year - 2018) * 0.7 + novelty

    ranked = sorted(retrieved_papers, key=score, reverse=True)
    top = ranked[0]
    already = top.get("paper_id") in existing
    results_summary["best_paper_candidate"] = {
        "paper_id": top.get("paper_id", ""),
        "title": top.get("title", ""),
        "why_best_now": (
            "Selected by fallback ranking using influence (citations), recency, and novelty "
            f"against current memory. already_in_memory={already}."
        ),
    }
    return results_summary


def generate_paper_note(
    client: OpenAI,
    model: str,
    concept: str,
    paper: Dict[str, Any],
    note_source: str,
) -> Dict[str, Any]:
    authors = paper.get("authors", [])
    author_names = ", ".join(a.get("name", "") for a in authors[:8] if a.get("name"))
    paper_blob = "\n".join(
        [
            f"paper_id: {paper.get('paperId')}",
            f"title: {paper.get('title', '')}",
            f"year: {paper.get('year')}",
            f"venue: {paper.get('venue')}",
            f"authors: {author_names}",
            f"abstract: {paper.get('abstract', '') or ''}",
            f"full_text_available: {paper.get('full_text_available', False)}",
            f"full_paper_text: {paper.get('full_text_excerpt', '') or ''}",
            f"related_work_excerpt: {paper.get('related_work_excerpt', '') or ''}",
            f"full_text_char_count: {paper.get('full_text_char_count', 0)}",
            f"url: {paper.get('url')}",
        ]
    )
    reference_items = paper.get("references", []) or []
    reference_seed = []
    for ref in reference_items[:30]:
        reference_seed.append(
            {
                "paper_id": ref.get("paperId"),
                "title": ref.get("title", ""),
            }
        )
    user_input = (
        "ONE_PAPER:\n"
        + paper_blob
        + f"\n\nNOTE_SOURCE_MODE:\n{note_source}"
        + "\n\nPAPER_REFERENCE_SEEDS:\n"
        + json.dumps(reference_seed, indent=2)
    )
    note_text = call_text_model(
        client,
        model,
        PAPER_NOTES_PROMPT.format(concept=concept),
        user_input,
    )
    return {
        "paper_id": paper.get("paperId", ""),
        "title": paper.get("title", ""),
        "note_markdown": note_text,
    }


def update_related_works(
    client: OpenAI,
    model: str,
    concept: str,
    all_paper_notes: List[Dict[str, Any]],
) -> str:
    user_input = (
        "ALL_PAPER_NOTES:\n"
        + json.dumps(all_paper_notes, indent=2)
    )
    return call_text_model(client, model, RELATED_WORKS_UPDATE_PROMPT.format(concept=concept), user_input)


def update_related_works_from_papers(
    client: OpenAI,
    model: str,
    concept: str,
    collected_papers: List[Dict[str, Any]],
) -> str:
    user_input = "COLLECTED_PAPERS:\n" + json.dumps(collected_papers, indent=2)
    return call_text_model(
        client,
        model,
        RELATED_WORKS_FROM_PAPERS_PROMPT.format(concept=concept),
        user_input,
    )


def filter_collected_papers(
    client: OpenAI,
    model: str,
    concept: str,
    collected_papers: List[Dict[str, Any]],
    min_keep: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not collected_papers:
        return [], {"filter_notes": "No collected papers to filter."}

    payload = call_json_model(
        client=client,
        model=model,
        instructions=PAPER_FILTER_PROMPT.format(concept=concept),
        user_input="COLLECTED_PAPERS:\n" + json.dumps(collected_papers, indent=2),
    )
    keep_ids_raw = payload.get("keep_paper_ids", [])
    drop_ids_raw = payload.get("drop_paper_ids", [])

    allowed_ids = {p.get("paper_id") for p in collected_papers if p.get("paper_id")}
    keep_ids: List[str] = [pid for pid in keep_ids_raw if pid in allowed_ids]
    drop_ids: List[str] = [pid for pid in drop_ids_raw if pid in allowed_ids]

    # Ensure conservative filtering: if model over-prunes, backfill by influence.
    if len(keep_ids) < max(1, min_keep):
        by_citation = sorted(
            collected_papers,
            key=lambda x: float(x.get("citation_count") or 0),
            reverse=True,
        )
        for p in by_citation:
            pid = p.get("paper_id")
            if pid and pid not in keep_ids:
                keep_ids.append(pid)
            if len(keep_ids) >= min_keep:
                break

    keep_set = set(keep_ids)
    kept_cards = [p for p in collected_papers if p.get("paper_id") in keep_set]
    kept_cards = sorted(
        kept_cards,
        key=lambda x: float(x.get("citation_count") or 0),
        reverse=True,
    )
    return kept_cards, {
        "filter_notes": payload.get("filter_notes", ""),
        "kept_count": len(kept_cards),
        "dropped_count": max(0, len(collected_papers) - len(kept_cards)),
        "drop_paper_ids": [pid for pid in drop_ids if pid not in keep_set],
    }


def run_note_generation_test(args: argparse.Namespace) -> None:
    load_dotenv_file(".env", override_existing=True)
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")
    if not os.getenv("S2_KEY"):
        raise RuntimeError("S2_KEY is not set.")
    if not args.test_note_paper_id:
        raise RuntimeError("--test-note-paper-id is required for note generation test.")

    workspace = Path(args.workspace_dir)
    ensure_workspace_layout(workspace)
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    s2_client = SemanticScholarClient(api_key=os.getenv("S2_KEY"))

    details = get_or_fetch_paper_details(s2_client, workspace, args.test_note_paper_id)
    if args.note_source == "full":
        full_text_payload = s2_client.read_full_paper_text(
            args.test_note_paper_id, max_chars=args.pdf_max_chars
        )
        details["full_text_available"] = bool(full_text_payload.get("success"))
        details["full_text_excerpt"] = full_text_payload.get("text", "")
        details["related_work_excerpt"] = extract_related_work_excerpt(
            client=client,
            model=args.updater_model,
            full_text=details["full_text_excerpt"],
        )
        details["full_text_char_count"] = len(details["full_text_excerpt"])
    else:
        details["full_text_available"] = False
        details["full_text_excerpt"] = ""
        details["related_work_excerpt"] = ""
        details["full_text_char_count"] = 0

    note = generate_paper_note(
        client=client,
        model=args.updater_model,
        concept=args.concept,
        paper=details,
        note_source=args.note_source,
    )
    out_path = Path(args.test_note_out)
    out_path.write_text(note.get("note_markdown", "").strip() + "\n", encoding="utf-8")
    print(f"note_test_written={out_path}")
    print(f"paper_id={note.get('paper_id')}")
    print(f"title={note.get('title')}")
    print(f"note_source={args.note_source}")
    print(f"full_text_available={details.get('full_text_available')}")
    print(f"full_text_char_count={details.get('full_text_char_count')}")


def run_related_works_synthesis_test(args: argparse.Namespace) -> None:
    load_dotenv_file(".env", override_existing=True)
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    workspace = Path(args.workspace_dir)
    ensure_workspace_layout(workspace)
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    notes_obj = load_paper_notes(workspace)
    notes = notes_obj.get("notes", [])
    if not notes:
        raise RuntimeError("No notes found in paper_notes.json for synthesis test.")

    related = update_related_works(
        client=client,
        model=args.updater_model,
        concept=args.concept,
        all_paper_notes=notes,
    )
    out_path = Path(args.test_related_out)
    out_path.write_text(related.strip() + "\n", encoding="utf-8")
    print(f"related_test_written={out_path}")
    print(f"notes_used={len(notes)}")


def dedupe_papers(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for p in papers:
        paper_id = p.get("paper_id")
        if not paper_id or paper_id in seen:
            continue
        seen.add(paper_id)
        deduped.append(p)
    return deduped


def write_run_report(
    report_path: Path,
    concept: str,
    args: argparse.Namespace,
    step_records: List[Dict[str, Any]],
    workspace: Path,
) -> None:
    lines: List[str] = []
    lines.append("# Literature Search Run Report")
    lines.append("")
    lines.append(f"Generated: {datetime.utcnow().isoformat()}Z")
    lines.append("")
    lines.append("## Run Config")
    lines.append("")
    lines.append(f"- Concept: `{concept}`")
    lines.append(f"- Steps: `{args.steps}`")
    lines.append(f"- Agent model: `{args.agent_model}`")
    lines.append(f"- Summary model: `{args.summary_model}`")
    lines.append(f"- Updater model: `{args.updater_model}`")
    lines.append(f"- Note source: `{args.note_source}`")
    lines.append(f"- Collect all at end: `{args.collect_all_at_end}`")
    lines.append(f"- Enable paper filter: `{args.enable_paper_filter}`")
    lines.append(f"- Filter min keep: `{args.filter_min_keep}`")
    lines.append(f"- Related max papers: `{args.related_max_papers}`")
    lines.append(f"- Workspace dir: `{workspace}`")
    lines.append("")

    lines.append("## Original Prompts")
    lines.append("")
    lines.append("### Concept")
    lines.append("")
    lines.append(f"`{concept}`")
    lines.append("")
    lines.append("### Action System Prompt")
    lines.append("")
    lines.append("```text")
    lines.append(ACTION_SYSTEM_PROMPT.strip())
    lines.append("```")
    lines.append("")
    lines.append("### Query Summary Prompt")
    lines.append("")
    lines.append("```text")
    lines.append(QUERY_SUMMARY_PROMPT.strip())
    lines.append("```")
    lines.append("")
    lines.append("### Related Works Update Prompt")
    lines.append("")
    lines.append("```text")
    lines.append(RELATED_WORKS_UPDATE_PROMPT.strip())
    lines.append("```")
    lines.append("")
    lines.append("### Paper Notes Prompt")
    lines.append("")
    lines.append("```text")
    lines.append(PAPER_NOTES_PROMPT.strip())
    lines.append("```")
    lines.append("")

    lines.append("## Step-by-Step Raw Output")
    lines.append("")
    for record in step_records:
        lines.append(f"### Step {record['step']} - {record['action']}")
        lines.append("")
        lines.append("**Agent input for this step**")
        lines.append("")
        lines.append("```text")
        lines.append(record.get("agent_input", ""))
        lines.append("```")
        lines.append("")
        lines.append("**Raw model action JSON**")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(record["raw_action"], indent=2))
        lines.append("```")
        lines.append("")

        if record["action"] == "QUERY":
            lines.append("**Executed operations and raw returned papers**")
            lines.append("")
            for op_result in record.get("operation_results", []):
                lines.append(
                    f"- Operation: `{op_result.get('type')}` | results: `{len(op_result.get('results', []))}`"
                )
                lines.append("")
                lines.append("```json")
                lines.append(json.dumps(op_result, indent=2))
                lines.append("```")
                lines.append("")

            lines.append("**Merged deduped query results**")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(record.get("merged_results", []), indent=2))
            lines.append("```")
            lines.append("")

            lines.append("**Results summary**")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(record.get("results_summary", {}), indent=2))
            lines.append("```")
            lines.append("")

        if record["action"] == "WRITE_TO_MEMORY":
            lines.append("**Memory write details**")
            lines.append("")
            lines.append("```json")
            lines.append(
                json.dumps(
                    {
                        "requested_paper_ids": record.get("requested_paper_ids", []),
                        "added_paper_ids": record.get("added_paper_ids", []),
                        "related_works_updated": record.get("related_works_updated", False),
                        "new_notes_generated": record.get("new_notes_generated", 0),
                        "total_notes_count": record.get("total_notes_count", 0),
                        "new_paper_text_status": record.get("new_paper_text_status", []),
                    },
                    indent=2,
                )
            )
            lines.append("```")
            lines.append("")
            lines.append("**Generated paper notes this step**")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(record.get("new_notes", []), indent=2))
            lines.append("```")
            lines.append("")

    lines.append("## Final Artifacts")
    lines.append("")
    lines.append(f"- Query log: `{workspace / 'queries.jsonl'}`")
    lines.append(f"- Paper bank: `{workspace / 'paper_bank.json'}`")
    lines.append(f"- Paper notes: `{workspace / 'paper_notes.json'}`")
    lines.append(f"- Related works: `{workspace / 'related_works.md'}`")
    lines.append(f"- Metadata cache dir: `{workspace / 'metadata_cache'}`")
    lines.append("")
    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def save_run_archive(
    args: argparse.Namespace,
    concept: str,
    workspace: Path,
    report_path: Path,
    run_started_at_utc: str,
) -> Dict[str, Any]:
    root = Path(".").resolve()
    snapshot_label = f"literature-run-{run_started_at_utc}"
    snapshot = build_snapshot(root, snapshot_label)

    version_ref_dir = root / args.version_ref_dir
    version_ref_dir.mkdir(parents=True, exist_ok=True)
    snapshot_file = version_ref_dir / f"{snapshot['snapshot_id']}.json"
    snapshot_file.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")

    run_archive_root = root / args.run_archive_dir
    run_archive_root.mkdir(parents=True, exist_ok=True)
    run_dir = run_archive_root / snapshot["snapshot_id"]
    suffix = 1
    while run_dir.exists():
        run_dir = run_archive_root / f"{snapshot['snapshot_id']}-{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)

    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(report_path, artifacts_dir / report_path.name)
    shutil.copy2(workspace / "related_works.md", artifacts_dir / "related_works.md")
    shutil.copy2(workspace / "queries.jsonl", artifacts_dir / "queries.jsonl")
    shutil.copy2(workspace / "paper_bank.json", artifacts_dir / "paper_bank.json")
    shutil.copy2(workspace / "paper_notes.json", artifacts_dir / "paper_notes.json")

    code_dir = run_dir / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    for file_entry in snapshot.get("files", []):
        rel_path = file_entry.get("path")
        if not rel_path:
            continue
        src = root / rel_path
        if not src.exists():
            continue
        dst = code_dir / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    run_meta = {
        "snapshot_id": snapshot["snapshot_id"],
        "snapshot_file": str(snapshot_file),
        "run_dir": str(run_dir),
        "concept": concept,
        "run_started_at_utc": run_started_at_utc,
        "run_finished_at_utc": datetime.utcnow().isoformat() + "Z",
        "args": vars(args),
        "workspace_dir": str(workspace),
        "report_file": str(report_path),
    }
    (run_dir / "run_meta.json").write_text(
        json.dumps(run_meta, indent=2) + "\n", encoding="utf-8"
    )
    return run_meta


def append_archive_info_to_report(report_path: Path, run_meta: Dict[str, Any]) -> None:
    existing = report_path.read_text(encoding="utf-8").rstrip() + "\n\n"
    lines = [
        "## Run Archive",
        "",
        f"- Snapshot ID: `{run_meta.get('snapshot_id', '')}`",
        f"- Snapshot file: `{run_meta.get('snapshot_file', '')}`",
        f"- Run directory: `{run_meta.get('run_dir', '')}`",
    ]
    report_path.write_text(existing + "\n".join(lines).rstrip() + "\n", encoding="utf-8")


def build_agent_input(
    concept: str,
    step_number: int,
    total_steps: int,
    paper_bank_ids: List[str],
    related_works: str,
    last_results: List[Dict[str, Any]],
) -> str:
    # Decision model sees full query results (not just summaries) before choosing action.
    decision_results = []
    for p in last_results[:30]:
        decision_results.append(
            {
                "paper_id": p.get("paper_id"),
                "title": p.get("title"),
                "year": p.get("year"),
                "venue": p.get("venue"),
                "citation_count": p.get("citation_count"),
                "url": p.get("url"),
                "abstract": (p.get("abstract", "") or "")[:1200],
            }
        )
    return (
        f"CONCEPT:\n{concept}\n\n"
        f"STEP: {step_number}/{total_steps}\n\n"
        f"CURRENT_PAPER_BANK_IDS:\n{json.dumps(paper_bank_ids, indent=2)}\n\n"
        f"CURRENT_RELATED_WORKS:\n{related_works}\n\n"
        f"LAST_QUERY_RESULTS_FULL_JSON:\n{json.dumps(decision_results, indent=2)}\n\n"
        "NOTE: QUERY summaries are for logging only. Choose action using full query results above.\n\n"
        "Choose the next best action."
    )


def run_agent(args: argparse.Namespace) -> None:
    load_dotenv_file(".env", override_existing=True)
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")
    if not os.getenv("S2_KEY"):
        raise RuntimeError("S2_KEY is not set.")

    run_started_at_utc = datetime.utcnow().isoformat() + "Z"
    workspace = Path(args.workspace_dir)
    ensure_workspace_layout(workspace)
    if args.reset_workspace:
        reset_workspace_state(workspace)
    logger = configure_logging(args.log_file, args.log_level)
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    s2_client = SemanticScholarClient(api_key=os.getenv("S2_KEY"))

    last_results: List[Dict[str, Any]] = []
    collected_papers_by_id: Dict[str, Dict[str, Any]] = {}
    step_records: List[Dict[str, Any]] = []
    logger.info("Starting literature search loop: steps=%s", args.steps)
    logger.info("Workspace reset on start: %s", args.reset_workspace)
    print(f"\nConcept: {args.concept}\n")
    # Initialize report immediately so it exists and updates during long runs.
    write_run_report(
        report_path=Path(args.report_file),
        concept=args.concept,
        args=args,
        step_records=step_records,
        workspace=workspace,
    )

    for step in range(1, args.steps + 1):
        paper_bank = load_paper_bank(workspace)
        related_works = read_related_works(workspace)
        agent_input = build_agent_input(
            concept=args.concept,
            step_number=step,
            total_steps=args.steps,
            paper_bank_ids=paper_bank.get("paper_ids", []),
            related_works=related_works,
            last_results=last_results,
        )
        if args.collect_all_at_end:
            action_payload = call_json_model(
                client=client,
                model=args.agent_model,
                instructions=QUERY_ONLY_SYSTEM_PROMPT,
                user_input=agent_input,
            )
        else:
            action_payload = call_json_model(
                client=client,
                model=args.agent_model,
                instructions=ACTION_SYSTEM_PROMPT,
                user_input=agent_input,
            )
        action = action_payload.get("action")
        if args.collect_all_at_end and action != "QUERY":
            action_payload = {
                "action": "QUERY",
                "operations": [
                    {
                        "type": "search",
                        "query_text": args.concept,
                        "filters": {"year_min": 2020},
                    }
                ],
                "intent": "broad",
                "why_this_query": "Fallback broad query in query-only mode.",
            }
            action = "QUERY"
        logger.info("Step %s action: %s", step, action)

        if action == "QUERY":
            operations = action_payload.get("operations", [])
            all_papers: List[Dict[str, Any]] = []
            operation_results_for_report: List[Dict[str, Any]] = []
            for op in operations:
                op_results = execute_operation(
                    s2_client=s2_client,
                    workspace=workspace,
                    operation=op,
                    default_limit=args.default_query_limit,
                )
                all_papers.extend(op_results)
                operation_results_for_report.append(
                    {
                        "type": op.get("type"),
                        "query_text": op.get("query_text"),
                        "source_paper_id": op.get("source_paper_id"),
                        "filters": op.get("filters", {}),
                        "results": op_results,
                    }
                )
            deduped = dedupe_papers(all_papers)
            last_results = deduped
            for paper in deduped:
                pid = paper.get("paper_id")
                if pid:
                    collected_papers_by_id[pid] = paper

            results_summary = summarize_query_results(
                client=client,
                model=args.summary_model,
                query_action=action_payload,
                retrieved_papers=deduped,
            )
            results_summary = ensure_best_paper_candidate(
                results_summary=results_summary,
                retrieved_papers=deduped,
                existing_paper_ids=paper_bank.get("paper_ids", []),
            )
            query_record = {
                "timestamp": datetime.utcnow().isoformat(),
                "step": step,
                "action": "QUERY",
                "operations": operations,
                "intent": action_payload.get("intent"),
                "why_this_query": action_payload.get("why_this_query"),
                "results_summary": results_summary,
            }
            append_query_log(workspace, query_record)
            step_records.append(
                {
                    "step": step,
                    "action": "QUERY",
                    "agent_input": agent_input,
                    "raw_action": action_payload,
                    "operation_results": operation_results_for_report,
                    "merged_results": deduped,
                    "results_summary": results_summary,
                }
            )
            write_run_report(
                report_path=Path(args.report_file),
                concept=args.concept,
                args=args,
                step_records=step_records,
                workspace=workspace,
            )
            print(f"Step {step}: QUERY -> {len(deduped)} papers")
            print(json.dumps(results_summary, indent=2))
            best = results_summary.get("best_paper_candidate", {})
            if best:
                print(
                    "Best paper this query: "
                    f"{best.get('paper_id', '')} | {best.get('title', '')}"
                )
            continue

        if action == "WRITE_TO_MEMORY":
            paper_ids = action_payload.get("paper_ids", [])[:1]
            existing_ids = set(paper_bank.get("paper_ids", []))
            added_ids: List[str] = []
            for paper_id in paper_ids:
                if paper_id and paper_id not in existing_ids:
                    existing_ids.add(paper_id)
                    added_ids.append(paper_id)

            paper_bank["paper_ids"] = list(existing_ids)
            paper_bank["updated_at"] = datetime.utcnow().isoformat()
            save_paper_bank(workspace, paper_bank)
            paper_notes_obj = load_paper_notes(workspace)
            all_notes = paper_notes_obj.get("notes", [])

            new_paper_details: List[Dict[str, Any]] = []
            new_notes: List[Dict[str, Any]] = []
            for paper_id in added_ids:
                details = get_or_fetch_paper_details(s2_client, workspace, paper_id)
                if args.note_source == "full":
                    full_text_payload = s2_client.read_full_paper_text(
                        paper_id, max_chars=args.pdf_max_chars
                    )
                    details["full_text_available"] = bool(full_text_payload.get("success"))
                    details["full_text_excerpt"] = full_text_payload.get("text", "")
                    details["related_work_excerpt"] = extract_related_work_excerpt(
                        client=client,
                        model=args.updater_model,
                        full_text=details["full_text_excerpt"],
                    )
                    details["full_text_char_count"] = len(details["full_text_excerpt"])
                else:
                    details["full_text_available"] = False
                    details["full_text_excerpt"] = ""
                    details["related_work_excerpt"] = ""
                    details["full_text_char_count"] = 0
                new_paper_details.append(details)
                note_payload = generate_paper_note(
                    client=client,
                    model=args.updater_model,
                    concept=args.concept,
                    paper=details,
                    note_source=args.note_source,
                )
                new_notes.append(note_payload)

            if new_notes:
                notes_by_id = {
                    n.get("paper_id"): n
                    for n in all_notes
                    if isinstance(n, dict) and n.get("paper_id")
                }
                for n in new_notes:
                    pid = n.get("paper_id")
                    if pid:
                        notes_by_id[pid] = n
                all_notes = list(notes_by_id.values())
                paper_notes_obj["notes"] = all_notes
                paper_notes_obj["updated_at"] = datetime.utcnow().isoformat()
                save_paper_notes(workspace, paper_notes_obj)
            step_records.append(
                {
                    "step": step,
                    "action": "WRITE_TO_MEMORY",
                    "agent_input": agent_input,
                    "raw_action": action_payload,
                    "requested_paper_ids": paper_ids,
                    "added_paper_ids": added_ids,
                    "related_works_updated": False,
                    "new_notes_generated": len(new_notes),
                    "total_notes_count": len(all_notes),
                    "new_notes": new_notes,
                    "new_paper_text_status": [
                        {
                            "paper_id": p.get("paperId"),
                            "note_source": args.note_source,
                            "full_text_available": bool(p.get("full_text_available")),
                            "related_work_excerpt_found": bool(p.get("related_work_excerpt")),
                            "full_text_char_count": int(p.get("full_text_char_count", 0)),
                        }
                        for p in new_paper_details
                    ],
                }
            )
            write_run_report(
                report_path=Path(args.report_file),
                concept=args.concept,
                args=args,
                step_records=step_records,
                workspace=workspace,
            )

            print(f"Step {step}: WRITE_TO_MEMORY -> added {len(added_ids)} paper(s): {added_ids}")
            if step < args.steps:
                last_results = []
            continue

        raise ValueError(f"Unknown action returned by model: {action_payload}")

    if args.collect_all_at_end:
        all_collected_ids = list(collected_papers_by_id.keys())
        final_bank = {
            "paper_ids": all_collected_ids,
            "updated_at": datetime.utcnow().isoformat(),
        }
        save_paper_bank(workspace, final_bank)

        # In collect-all mode, skip intermediate notes and synthesize directly from papers.
        save_paper_notes(workspace, {"notes": [], "updated_at": datetime.utcnow().isoformat()})
        collected_cards: List[Dict[str, Any]] = []
        for paper_id in all_collected_ids:
            paper = collected_papers_by_id.get(paper_id, {})
            collected_cards.append(
                {
                    "paper_id": paper_id,
                    "title": paper.get("title", ""),
                    "year": paper.get("year"),
                    "venue": paper.get("venue"),
                    "citation_count": paper.get("citation_count"),
                    "url": paper.get("url"),
                    "abstract": (paper.get("abstract", "") or "")[:1400],
                }
            )
        collected_cards = sorted(
            collected_cards,
            key=lambda x: float(x.get("citation_count") or 0),
            reverse=True,
        )
        if args.related_max_papers > 0:
            collected_cards = collected_cards[: args.related_max_papers]
        final_related = update_related_works_from_papers(
            client=client,
            model=args.updater_model,
            concept=args.concept,
            collected_papers=collected_cards,
        )
        write_related_works(workspace, final_related)
        step_records.append(
            {
                "step": args.steps + 1,
                "action": "WRITE_TO_MEMORY",
                "agent_input": "collect_all_at_end finalization",
                "raw_action": {"action": "WRITE_TO_MEMORY", "paper_ids": all_collected_ids},
                "requested_paper_ids": all_collected_ids,
                "added_paper_ids": all_collected_ids,
                "related_works_updated": True,
                "new_notes_generated": 0,
                "total_notes_count": 0,
                "new_notes": [],
                "related_input_papers": len(collected_cards),
                "new_paper_text_status": [],
            }
        )
        logger.info(
            "Collect-all finalization wrote %s papers and synthesized related works directly",
            len(all_collected_ids),
        )

    write_run_report(
        report_path=Path(args.report_file),
        concept=args.concept,
        args=args,
        step_records=step_records,
        workspace=workspace,
    )
    # Synthesize the full related works once at the end from all notes.
    notes_obj = load_paper_notes(workspace)
    final_notes = notes_obj.get("notes", [])
    if final_notes:
        final_related = update_related_works(
            client=client,
            model=args.updater_model,
            concept=args.concept,
            all_paper_notes=final_notes,
        )
        write_related_works(workspace, final_related)
        logger.info("Final related works written from %s notes", len(final_notes))
    run_meta: Dict[str, Any] = {}
    if args.auto_archive_run:
        run_meta = save_run_archive(
            args=args,
            concept=args.concept,
            workspace=workspace,
            report_path=Path(args.report_file),
            run_started_at_utc=run_started_at_utc,
        )
        append_archive_info_to_report(Path(args.report_file), run_meta)
        logger.info("Run archived at %s", run_meta.get("run_dir", ""))
    print("\nDone.")
    print(f"- Query log: {workspace / 'queries.jsonl'}")
    print(f"- Paper bank: {workspace / 'paper_bank.json'}")
    print(f"- Paper notes: {workspace / 'paper_notes.json'}")
    print(f"- Related works: {workspace / 'related_works.md'}")
    print(f"- Metadata cache: {workspace / 'metadata_cache'}")
    print(f"- Run report: {args.report_file}")
    if run_meta:
        print(f"- Snapshot ID: {run_meta.get('snapshot_id', '')}")
        print(f"- Snapshot file: {run_meta.get('snapshot_file', '')}")
        print(f"- Run archive: {run_meta.get('run_dir', '')}")
    print(f"- Run log: {args.log_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agentic literature search (QUERY / WRITE_TO_MEMORY) with Related Works updates"
    )
    parser.add_argument(
        "concept",
        nargs="?",
        type=str,
        default=DEFAULT_CONCEPT,
        help="Research idea/concept to explore",
    )
    parser.add_argument("--steps", type=int, default=8, help="Total decision steps to run")
    parser.add_argument(
        "--default-query-limit",
        type=int,
        default=12,
        help="Default max papers per retrieval operation",
    )
    parser.add_argument(
        "--agent-model",
        type=str,
        default="gpt-5-nano",
        help="Model for deciding QUERY vs WRITE_TO_MEMORY actions",
    )
    parser.add_argument(
        "--summary-model",
        type=str,
        default="gpt-5-nano",
        help="Model for mandatory QUERY results summaries",
    )
    parser.add_argument(
        "--updater-model",
        type=str,
        default="gpt-5-nano",
        help="Model for Related Works update prompt",
    )
    parser.add_argument(
        "--workspace-dir",
        type=str,
        default="lit_workspace",
        help="Directory for queries/memory/related_works/cache outputs",
    )
    parser.add_argument(
        "--no-reset-workspace",
        action="store_true",
        help="Do not reset workspace files at run start",
    )
    parser.add_argument(
        "--pdf-max-chars",
        type=int,
        default=250000,
        help="Max chars of full paper text to include for related works updates",
    )
    parser.add_argument(
        "--note-source",
        type=str,
        default="abstract",
        choices=["abstract", "full"],
        help="Source for note generation: abstract metadata only (fast) or full PDF text",
    )
    parser.add_argument(
        "--collect-all-at-end",
        action="store_true",
        help="Query-only collection mode: keep searching every step, then write all collected papers at end",
    )
    parser.add_argument(
        "--related-max-papers",
        type=int,
        default=30,
        help="Max number of collected papers to include in final related-works synthesis",
    )
    parser.add_argument(
        "--version-ref-dir",
        type=str,
        default="experiments/version_refs",
        help="Directory to write code snapshot JSON references",
    )
    parser.add_argument(
        "--run-archive-dir",
        type=str,
        default="experiments/runs",
        help="Directory to write full per-run archives",
    )
    parser.add_argument(
        "--no-auto-archive-run",
        action="store_true",
        help="Disable automatic run archive and code snapshot creation",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="literature_search.log",
        help="Path for runtime log file",
    )
    parser.add_argument(
        "--report-file",
        type=str,
        default="last_run_report.md",
        help="Path to write detailed markdown run report",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    parser.add_argument(
        "--test-note-paper-id",
        type=str,
        default="",
        help="Run isolated note-generation test for this paper ID",
    )
    parser.add_argument(
        "--test-note-out",
        type=str,
        default="note_generation_test.md",
        help="Output path for isolated note-generation test",
    )
    parser.add_argument(
        "--test-related-from-notes",
        action="store_true",
        help="Run isolated related-works synthesis from paper_notes.json",
    )
    parser.add_argument(
        "--test-related-out",
        type=str,
        default="related_works_test.md",
        help="Output path for isolated related-works synthesis test",
    )
    parsed = parser.parse_args()
    parsed.reset_workspace = not parsed.no_reset_workspace
    parsed.auto_archive_run = not parsed.no_auto_archive_run
    return parsed


if __name__ == "__main__":
    parsed_args = parse_args()
    if parsed_args.test_note_paper_id:
        run_note_generation_test(parsed_args)
    elif parsed_args.test_related_from_notes:
        run_related_works_synthesis_test(parsed_args)
    else:
        run_agent(parsed_args)

import argparse
import json
import logging
import os
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Any, Dict, List

from openai import OpenAI


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPTS_DIR = os.path.join(SCRIPT_DIR, "prompts")
SEED_IDEA_FILE = os.path.join(PROMPTS_DIR, "seed_idea.md")


def load_prompt_from_markdown(filename: str) -> str:
    prompt_path = os.path.join(PROMPTS_DIR, filename)
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    with open(prompt_path, "r", encoding="utf-8") as prompt_file:
        return prompt_file.read().strip()


GUIDE_SYSTEM_PROMPT = load_prompt_from_markdown("guide_system_prompt.md")
GUIDE_FINAL_ROUND_SYSTEM_PROMPT = load_prompt_from_markdown(
    "guide_final_round_system_prompt.md"
)
ASSISTANT_SYSTEM_PROMPT = load_prompt_from_markdown("assistant_system_prompt.md")
ASSISTANT_FINAL_ROUND_SYSTEM_PROMPT = load_prompt_from_markdown(
    "assistant_final_round_system_prompt.md"
)
FINAL_SYNTHESIS_SYSTEM_PROMPT = load_prompt_from_markdown(
    "final_synthesis_system_prompt.md"
)


def load_seed_idea(path: str = SEED_IDEA_FILE) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Seed idea file not found: {path}")
    with open(path, "r", encoding="utf-8") as seed_file:
        seed_idea = seed_file.read().strip()
    if not seed_idea:
        raise ValueError(f"Seed idea file is empty: {path}")
    return seed_idea


@dataclass
class Turn:
    round_number: int
    guide: str
    assistant: str


def mask_secret(value: str) -> str:
    if not value:
        return "<empty>"
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"


def configure_logging(log_file: str, level: str) -> logging.Logger:
    logger = logging.getLogger("tree_of_thought")
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


def load_dotenv_file(path: str = ".env", override_existing: bool = True) -> None:
    """Lightweight .env loader so local keys work without extra deps."""
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

            # For local dev, prefer project .env over machine/global exports.
            if key and (override_existing or key not in os.environ):
                os.environ[key] = value


def format_history(turns: List[Turn]) -> str:
    if not turns:
        return "No prior rounds yet."

    lines: List[str] = []
    for turn in turns:
        lines.append(f"Round {turn.round_number}")
        lines.append(f"User message: {turn.guide}")
        lines.append(f"Assistant response: {turn.assistant}")
        lines.append("---")
    return "\n".join(lines)


def call_model(
    client: OpenAI,
    model: str,
    instructions: str,
    user_input: str,
    logger: logging.Logger,
    role_name: str,
) -> str:
    logger.info("Calling %s model: %s", role_name, model)
    response = client.responses.create(
        model=model,
        instructions=instructions,
        input=user_input,
    )
    output_text = response.output_text.strip()
    logger.info("Received %s response length: %s chars", role_name, len(output_text))
    return output_text


def run_loop(
    thesis: str,
    rounds: int,
    guide_model: str,
    assistant_model: str,
    synthesis_model: str,
    logger: logging.Logger,
    live_output: bool,
) -> Dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    logger.info("Starting run with %s rounds", rounds)
    logger.info(
        "Models | guide=%s assistant=%s synthesis=%s",
        guide_model,
        assistant_model,
        synthesis_model,
    )
    logger.info("Using OPENAI_API_KEY=%s", mask_secret(api_key))

    client = OpenAI(api_key=api_key)
    turns: List[Turn] = []

    for round_number in range(1, rounds + 1):
        logger.info("Round %s started", round_number)
        is_final_round = round_number == rounds
        history = format_history(turns)
        round_context = (
            f"Round context:\n"
            f"- current_round: {round_number}\n"
            f"- total_rounds: {rounds}\n"
            f"- rounds_remaining_after_this: {rounds - round_number}\n"
            f"- is_final_round: {is_final_round}\n"
        )
        guide_instructions = (
            GUIDE_FINAL_ROUND_SYSTEM_PROMPT if is_final_round else GUIDE_SYSTEM_PROMPT
        )
        guide_input = (
            f"Original thesis:\n{thesis}\n\n"
            f"{round_context}\n"
            f"Conversation so far:\n{history}\n\n"
            "Write the next short user message to help the assistant improve the idea."
        )
        guide_text = call_model(
            client, guide_model, guide_instructions, guide_input, logger, "guide"
        )

        assistant_instructions = (
            ASSISTANT_FINAL_ROUND_SYSTEM_PROMPT
            if is_final_round
            else ASSISTANT_SYSTEM_PROMPT
        )
        assistant_input = (
            f"Original thesis:\n{thesis}\n\n"
            f"{round_context}\n"
            f"User message for this round:\n{guide_text}\n\n"
            f"Conversation so far:\n{history}"
        )
        assistant_text = call_model(
            client,
            assistant_model,
            assistant_instructions,
            assistant_input,
            logger,
            "assistant",
        )

        turns.append(Turn(round_number=round_number, guide=guide_text, assistant=assistant_text))
        logger.info("Round %s completed", round_number)
        if live_output:
            print(f"\n--- Live Round {round_number} ---")
            print(f"User message: {guide_text}")
            print(f"Assistant response: {assistant_text}")
            print("------------------------")

    synthesis_input = (
        f"Original thesis:\n{thesis}\n\n"
        f"All rounds:\n{format_history(turns)}\n\n"
        "Return one final high-level concept only, in 4-5 sentences max, with no implementation details."
    )
    final_text = call_model(
        client,
        synthesis_model,
        FINAL_SYNTHESIS_SYSTEM_PROMPT,
        synthesis_input,
        logger,
        "synthesis",
    )
    logger.info("Run completed successfully")
    if live_output:
        print("\n--- Live Final Synthesis ---")
        print(final_text)
        print("----------------------------")

    return {
        "thesis": thesis,
        "models": {
            "guide_model": guide_model,
            "assistant_model": assistant_model,
            "synthesis_model": synthesis_model,
        },
        "rounds": [asdict(t) for t in turns],
        "final": {"final_concept": final_text},
    }


def print_human_readable(result: Dict[str, Any]) -> None:
    print("\n=== Original Thesis ===")
    print(result["thesis"])

    print("\n=== Iteration Transcript ===")
    for turn in result["rounds"]:
        print(f"\n[Round {turn['round_number']}]")
        print(f"User message: {turn.get('guide', '')}")
        print(f"Assistant response: {turn.get('assistant', '')}")

    print("\n=== Final Concept ===")
    final = result["final"]
    print(final.get("final_concept", final))


def create_visual_markdown_report(result: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Tree of Thought Run Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("## Original Thesis")
    lines.append("")
    lines.append(f"> {result.get('thesis', '')}")
    lines.append("")

    models = result.get("models", {})
    lines.append("## Models")
    lines.append("")
    lines.append(f"- Guide: `{models.get('guide_model', '')}`")
    lines.append(f"- Assistant: `{models.get('assistant_model', '')}`")
    lines.append(f"- Synthesis: `{models.get('synthesis_model', '')}`")
    lines.append("")

    lines.append("## Visual Flow")
    lines.append("")
    lines.append("```mermaid")
    lines.append("flowchart TD")
    lines.append('  A["Original Thesis"]')
    for idx, _ in enumerate(result.get("rounds", []), start=1):
        lines.append(f'  G{idx}["User {idx}: follow-up message"]')
        lines.append(f'  S{idx}["Assistant {idx}: response"]')
        if idx == 1:
            lines.append(f"  A --> G{idx}")
        else:
            lines.append(f"  S{idx - 1} --> G{idx}")
        lines.append(f"  G{idx} --> S{idx}")
    if result.get("rounds"):
        lines.append('  F["Final Synthesis"]')
        lines.append(f"  S{len(result.get('rounds', []))} --> F")
    lines.append("```")
    lines.append("")

    lines.append("## Round-by-Round Transcript")
    lines.append("")
    for turn in result.get("rounds", []):
        lines.append(f"### Round {turn.get('round_number')}")
        lines.append("")
        lines.append("**User message**")
        lines.append("")
        lines.append(f"{turn.get('guide', '')}")
        lines.append("")
        lines.append("**Assistant response**")
        lines.append("")
        lines.append(f"{turn.get('assistant', '')}")
        lines.append("")

    final = result.get("final", {})
    lines.append("## Final Synthesis")
    lines.append("")
    lines.append("**Final concept**")
    lines.append("")
    lines.append(str(final.get("final_concept", "")))
    lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_report(report_path: str, content: str) -> None:
    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write(content)


def parse_args() -> argparse.Namespace:
    default_thesis = load_seed_idea()
    parser = argparse.ArgumentParser(
        description="v0 tree-of-thought: guide + assistant idea refinement"
    )
    parser.add_argument(
        "thesis",
        nargs="?",
        type=str,
        default=default_thesis,
        help="Original thesis or prompt to refine",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=5,
        help="Number of guide/assistant refinement rounds",
    )
    parser.add_argument(
        "--guide-model",
        type=str,
        default="gpt-5-nano",
        help="Model used by the Guide role",
    )
    parser.add_argument(
        "--assistant-model",
        type=str,
        default="gpt-5-nano",
        help="Model used by the Assistant role",
    )
    parser.add_argument(
        "--synthesis-model",
        type=str,
        default="gpt-5-nano",
        help="Model used for final synthesis",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print raw JSON result instead of human-readable summary",
    )
    parser.add_argument(
        "--report-file",
        type=str,
        default="last_run_report.md",
        help="Path to write a readable markdown report with visual flow",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="tree_of_thought.log",
        help="Path to write execution logs",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    parser.add_argument(
        "--no-live",
        action="store_true",
        help="Disable live printing of each round as it completes",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = configure_logging(args.log_file, args.log_level)
    load_dotenv_file(".env", override_existing=True)
    logger.info("Loaded .env (if present)")
    try:
        result = run_loop(
            thesis=args.thesis,
            rounds=args.rounds,
            guide_model=args.guide_model,
            assistant_model=args.assistant_model,
            synthesis_model=args.synthesis_model,
            logger=logger,
            live_output=not args.no_live,
        )
        report_markdown = create_visual_markdown_report(result)
        write_report(args.report_file, report_markdown)
        logger.info("Wrote report: %s", args.report_file)
        if args.json:
            print(json.dumps(result, indent=2))
            print(f"\nReport written to: {args.report_file}")
            print(f"Log written to: {args.log_file}")
            return
        print_human_readable(result)
        print(f"\nReadable report written to: {args.report_file}")
        print(f"Log written to: {args.log_file}")
    except Exception as exc:
        logger.exception("Run failed: %s", exc)
        raise


if __name__ == "__main__":
    main()

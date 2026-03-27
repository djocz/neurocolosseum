# run.py
# ════════════════════════════════════════════════════════
# NeuroColosseum — Terminal Interface
# Two models enter. One argument wins.
#
# The front door of the application.
# Handles user interaction in the terminal.
# Delegates all debate logic to the graph.
#
# Usage:
#   python3 run.py
#   python3 run.py --topic "AI will do more harm than good"
#   python3 run.py --topic "..." --awareness scores
#   python3 run.py --twice  (run twice for bias analysis)
# ════════════════════════════════════════════════════════

import sys
import argparse
from dotenv import load_dotenv

load_dotenv()

from config import (
    settings,
    get_llm,
    get_llm_name,
    validate_topic_tier1,
    PRESET_TOPICS,
)
from state import create_initial_state
from graph import debate_graph
from exceptions import (
    TopicValidationError,
    ConfigurationError,
    NeuroColosseumError,
)

# pyfiglet and colorama for the banner
try:
    import pyfiglet
    from colorama import Fore, Style, init as colorama_init
    colorama_init()
    BANNER_AVAILABLE = True
except ImportError:
    BANNER_AVAILABLE = False


# ════════════════════════════════════════════════════════
# BANNER
# Printed once when run.py starts.
# Uses pyfiglet + colorama for colour.
# Falls back to plain text if not installed.
# ════════════════════════════════════════════════════════

def print_banner() -> None:
    """
    Prints the NeuroColosseum ASCII banner.
    Uses pyfiglet slant font with yellow colour.
    Falls back to plain text if pyfiglet unavailable.
    """
    if BANNER_AVAILABLE:
        banner = pyfiglet.figlet_format(
            "NeuroColosseum",
            font="slant"
        )
        print(Fore.YELLOW + banner + Style.RESET_ALL)
        print(
            Fore.WHITE +
            "  Two models enter. One argument wins." +
            Style.RESET_ALL
        )
        print(
            Fore.YELLOW +
            "  " + "━" * 45 +
            Style.RESET_ALL
        )
    else:
        print("\n" + "═" * 52)
        print("  NEUROCOLOSSEUM")
        print("  Two models enter. One argument wins.")
        print("═" * 52)

    print()


# ════════════════════════════════════════════════════════
# TOPIC VALIDATION — TIER 2
# LLM-based validation.
# Called before graph.invoke() to avoid
# wasting API calls on invalid topics.
# Only runs in terminal — Streamlit handles
# this in the UI before calling graph.
# ════════════════════════════════════════════════════════

def validate_topic_tier2(topic: str) -> tuple[bool, str]:
    """
    LLM-based topic validation.
    Checks if topic is genuinely debatable
    and appropriate for a respectful debate.

    More nuanced than keyword check.
    Catches topics that are:
        Too one-sided to debate fairly
        Factually settled (not debatable)
        Inappropriate in a subtle way

    Args:
        topic: The debate topic string

    Returns:
        Tuple of (is_valid, reason)

    Cost: ~one small LLM call (uses Groq if available)
    """
    from langchain_core.messages import (
        SystemMessage, HumanMessage
    )

    # Use cheapest available model for validation
    # Try Groq first (free), fall back to judge model
    try:
        if settings.groq_api_key:
            from langchain_groq import ChatGroq
            llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                api_key=settings.groq_api_key,
            )
        else:
            llm = get_llm("judge")
    except Exception:
        # If validation model fails
        # assume topic is valid and proceed
        return True, ""

    try:
        response = llm.invoke([
            SystemMessage(content="""You are a debate
            topic validator. Answer only YES or NO
            followed by a brief reason."""),

            HumanMessage(content=f"""Is this a valid,
            genuinely debatable, and appropriate topic
            for a respectful formal debate?

            Topic: "{topic}"

            A valid topic:
              → Has reasonable arguments on both sides
              → Is not factually settled
              → Is respectful and appropriate
              → Is specific enough to argue concretely

            Answer: YES or NO
            Reason: [one sentence]"""),
        ])

        content = response.content.upper()

        if content.startswith("NO"):
            # Extract reason after "NO"
            reason = response.content
            if ":" in reason:
                reason = reason.split(":", 1)[1].strip()
            elif "\n" in reason:
                reason = reason.split("\n", 1)[1].strip()
            return False, reason

        return True, ""

    except Exception:
        # Validation failed — assume valid
        # Better to let debate proceed than block
        return True, ""


# ════════════════════════════════════════════════════════
# DEBATE RUNNER
# Orchestrates one complete debate.
# Creates state, invokes graph, shows results.
# ════════════════════════════════════════════════════════

def run_debate(
    topic:      str,
    awareness:  str  = "criteria",
    run_number: int  = 1,
    run1_result: dict | None = None,
) -> dict:
    """
    Runs one complete debate.

    Args:
        topic:       The debate topic
        awareness:   Debater score awareness setting
        run_number:  1 for first run, 2 for swap run
        run1_result: Results from run 1 (run 2 only)

    Returns:
        Final DebateState after debate completes

    Raises:
        TopicValidationError: if topic invalid
        NeuroColosseumError: if debate fails
    """
    # ── Print debate header ───────────────────────────
    print(f"\n{'═'*52}")
    print(f"  DEBATE {'(Run 2 — Models Swapped)' if run_number == 2 else ''}")
    print(f"  Topic:     {topic}")
    print(f"  Debater A: {get_llm_name('debater_a')} (FOR)")
    print(f"  Debater B: {get_llm_name('debater_b')} (AGAINST)")
    print(f"  Judge:     {get_llm_name('judge')}")
    print(f"  Awareness: {awareness}")
    print(f"{'═'*52}\n")

    # ── Create initial state ──────────────────────────
    state = create_initial_state(
        topic       = topic,
        awareness   = awareness,
        run_number  = run_number,
        run1_result = run1_result,
    )

    # ── Run the graph ─────────────────────────────────
    try:
        final_state = debate_graph.invoke(state)
    except NeuroColosseumError:
        raise
    except Exception as e:
        raise NeuroColosseumError(
            f"Unexpected error during debate: {e}"
        )

    # ── Print summary ─────────────────────────────────
    _print_summary(final_state)

    return final_state


def _print_summary(state: dict) -> None:
    """
    Prints a clean debate summary to terminal.
    Called after debate_graph.invoke() completes.
    """
    print(f"\n{'═'*52}")
    print(f"  DEBATE COMPLETE")
    print(f"{'═'*52}")

    winner = state.get("winner")
    if winner:
        model    = get_llm_name(f"debater_{winner.lower()}")
        position = (
            state.get("position_a")
            if winner == "A"
            else state.get("position_b")
        )
        print(f"\n  WINNER: Debater {winner}")
        print(f"  Model:  {model}")
        print(f"  Position: {position}")

    print(f"\n  Final scores:")
    print(f"    Debater A: {state.get('total_a', 0)} / 500")
    print(f"    Debater B: {state.get('total_b', 0)} / 500")

    if state.get("tiebreak_needed"):
        print(f"\n  Tiebreak (Level "
              f"{state.get('tiebreak_level')}):")
        print(f"  {state.get('tiebreak_explanation')}")

    if state.get("transcript_path"):
        print(f"\n  Transcript: "
              f"{state.get('transcript_path')}")

    if state.get("errors"):
        print(f"\n  Errors encountered: "
              f"{len(state['errors'])}")
        for err in state["errors"]:
            print(f"    → {err}")

    print(f"\n{'═'*52}\n")


# ════════════════════════════════════════════════════════
# RUN TWICE — BIAS ANALYSIS
# Runs the same debate twice with models swapped.
# Compares results to detect position or model bias.
# ════════════════════════════════════════════════════════

def run_twice(
    topic:     str,
    awareness: str = "criteria",
) -> None:
    """
    Runs the debate twice with models swapped.
    Analyses results for position or model bias.

    Run 1: Original model assignments
    Run 2: Models swapped (A gets B's model, vice versa)

    Bias detection:
        Same position wins both → position bias
        Same model wins both    → model bias
        Different winner each   → mixed/balanced
        Similar scores both     → true balance
    """
    import os

    print(f"\n{'═'*52}")
    print(f"  RUN TWICE — BIAS ANALYSIS")
    print(f"  Topic: {topic}")
    print(f"{'═'*52}")
    print(f"\n  Run 1: {get_llm_name('debater_a')} (A) "
          f"vs {get_llm_name('debater_b')} (B)")

    # ── Run 1 ─────────────────────────────────────────
    result1 = run_debate(
        topic      = topic,
        awareness  = awareness,
        run_number = 1,
    )

    run1_summary = {
        "winner":   result1.get("winner"),
        "total_a":  result1.get("total_a", 0),
        "total_b":  result1.get("total_b", 0),
        "model_a":  get_llm_name("debater_a"),
        "model_b":  get_llm_name("debater_b"),
        "topic":    topic,
    }

    # ── Swap models for Run 2 ─────────────────────────
    # Temporarily swap env vars
    original_a = settings.llm_debater_a
    original_b = settings.llm_debater_b

    os.environ["LLM_DEBATER_A"] = original_b
    os.environ["LLM_DEBATER_B"] = original_a

    # Reload settings with swapped models
    from importlib import reload
    import config as config_module
    reload(config_module)

    print(f"\n  Run 2 (swapped): "
          f"{get_llm_name('debater_a')} (A) "
          f"vs {get_llm_name('debater_b')} (B)")

    # ── Run 2 ─────────────────────────────────────────
    result2 = run_debate(
        topic       = topic,
        awareness   = awareness,
        run_number  = 2,
        run1_result = run1_summary,
    )

    # ── Restore original model assignments ────────────
    os.environ["LLM_DEBATER_A"] = original_a
    os.environ["LLM_DEBATER_B"] = original_b
    reload(config_module)

    # ── Bias analysis ─────────────────────────────────
    _print_bias_analysis(
        result1, result2,
        run1_summary["model_a"],
        run1_summary["model_b"],
    )


def _print_bias_analysis(
    result1:  dict,
    result2:  dict,
    model_a:  str,
    model_b:  str,
) -> None:
    """
    Compares two debate results and prints
    bias analysis to terminal.
    """
    winner1 = result1.get("winner")
    winner2 = result2.get("winner")

    # In Run 1: A=model_a, B=model_b
    # In Run 2: A=model_b, B=model_a (swapped)

    # Who won Run 1?
    run1_winning_model = (
        model_a if winner1 == "A" else model_b
    )
    run1_winning_pos = (
        "FOR" if winner1 == "A" else "AGAINST"
    )

    # Who won Run 2? (models swapped)
    run2_winning_model = (
        model_b if winner2 == "A" else model_a
    )
    run2_winning_pos = (
        "FOR" if winner2 == "A" else "AGAINST"
    )

    print(f"\n{'═'*52}")
    print(f"  BIAS ANALYSIS")
    print(f"{'═'*52}")
    print(f"\n  Run 1 winner: {run1_winning_model}")
    print(f"            Position: {run1_winning_pos}")
    print(f"            Score: "
          f"{result1.get('total_a')} vs "
          f"{result1.get('total_b')}")

    print(f"\n  Run 2 winner: {run2_winning_model}")
    print(f"            Position: {run2_winning_pos}")
    print(f"            Score: "
          f"{result2.get('total_a')} vs "
          f"{result2.get('total_b')}")

    print(f"\n  Analysis:")

    # ── Detect bias pattern ───────────────────────────
    same_model    = (
        run1_winning_model == run2_winning_model
    )
    same_position = (
        run1_winning_pos == run2_winning_pos
    )

    if same_position and same_model:
        print(f"  Both position AND model bias detected.")
        print(f"  {run1_winning_pos} position + "
              f"{run1_winning_model} both favoured.")

    elif same_model:
        print(f"  Model bias detected.")
        print(f"  {run1_winning_model} wins regardless"
              f" of position argued.")
        print(f"  This model appears stronger on "
              f"this topic.")

    elif same_position:
        print(f"  Position bias detected.")
        print(f"  {run1_winning_pos} position wins "
              f"regardless of model.")
        print(f"  This topic may structurally favour "
              f"one side.")

    else:
        score_diff1 = abs(
            result1.get("total_a", 0) -
            result1.get("total_b", 0)
        )
        score_diff2 = abs(
            result2.get("total_a", 0) -
            result2.get("total_b", 0)
        )

        if score_diff1 < 20 and score_diff2 < 20:
            print(f"  No significant bias detected.")
            print(f"  Both runs were closely contested.")
            print(f"  Debate appears balanced.")
        else:
            print(f"  Mixed results — no clear bias.")
            print(f"  Different winner in each run.")

    print(f"\n{'═'*52}\n")


# ════════════════════════════════════════════════════════
# TOPIC SELECTION
# Interactive topic picker for terminal.
# Three options: preset / generate / type
# ════════════════════════════════════════════════════════

def select_topic() -> str:
    """
    Interactive topic selection in terminal.
    Offers three methods:
        1. Choose from preset categories
        2. Type your own
        3. Use default from .env

    Returns:
        Validated topic string
    """
    print("\n  How would you like to choose a topic?\n")
    print("  1. Choose from presets")
    print("  2. Type your own topic")
    print(f"  3. Use default "
          f"({settings.debate_topic[:40]}...)")
    print()

    choice = input("  Enter 1, 2, or 3: ").strip()

    if choice == "1":
        return _select_preset_topic()
    elif choice == "2":
        return _type_custom_topic()
    else:
        return settings.debate_topic


def _select_preset_topic() -> str:
    """Shows categorised preset topics for selection."""
    categories = list(PRESET_TOPICS.keys())

    print("\n  Categories:\n")
    for i, cat in enumerate(categories, 1):
        print(f"  {i}. {cat}")

    print()
    cat_input = input(
        "  Choose category (1-"
        f"{len(categories)}): "
    ).strip()

    try:
        cat_idx  = int(cat_input) - 1
        category = categories[cat_idx]
        topics   = PRESET_TOPICS[category]
    except (ValueError, IndexError):
        print("  Invalid choice — using default topic")
        return settings.debate_topic

    print(f"\n  {category} topics:\n")
    for i, topic in enumerate(topics, 1):
        print(f"  {i}. {topic}")

    print()
    topic_input = input(
        f"  Choose topic (1-{len(topics)}): "
    ).strip()

    try:
        topic_idx = int(topic_input) - 1
        return topics[topic_idx]
    except (ValueError, IndexError):
        print("  Invalid choice — using default topic")
        return settings.debate_topic


def _type_custom_topic() -> str:
    """
    Prompts user to type a custom topic.
    Validates through Tier 1 check.
    Returns validated topic.
    """
    while True:
        topic = input(
            "\n  Enter debate topic: "
        ).strip()

        if not topic:
            print("  Topic cannot be empty. Try again.")
            continue

        # Tier 1 validation
        valid, reason = validate_topic_tier1(topic)
        if not valid:
            print(f"  Invalid topic: {reason}")
            print("  Please try a different topic.")
            continue

        return topic


# ════════════════════════════════════════════════════════
# ARGUMENT PARSER
# Handles command line arguments for advanced usage.
# All arguments are optional — prompts if not provided.
# ════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    """
    Parses command line arguments.

    Usage examples:
        python3 run.py
        python3 run.py --topic "AI will do more harm"
        python3 run.py --topic "..." --awareness scores
        python3 run.py --twice
        python3 run.py --list-topics
    """
    parser = argparse.ArgumentParser(
        prog="neurocolosseum",
        description="Two models enter. One argument wins.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--topic",
        type=str,
        help="Debate topic (prompts if not provided)",
    )
    parser.add_argument(
        "--awareness",
        type=str,
        choices=["none", "criteria", "scores", "trailing"],
        default=settings.debater_awareness,
        help="Debater score awareness (default: criteria)",
    )
    parser.add_argument(
        "--twice",
        action="store_true",
        help="Run twice with swapped models for bias analysis",
    )
    parser.add_argument(
        "--list-topics",
        action="store_true",
        help="List all preset topics and exit",
    )

    return parser.parse_args()


# ════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# Called when python3 run.py is executed.
# Orchestrates the full terminal experience.
# ════════════════════════════════════════════════════════

def main() -> None:
    """
    Main entry point for terminal interface.

    Flow:
        1. Print banner
        2. Parse arguments
        3. Select/validate topic
        4. Run debate (or run twice)
        5. Show results
    """
    # ── Print banner ──────────────────────────────────
    print_banner()

    # ── Parse arguments ───────────────────────────────
    args = parse_args()

    # ── List topics and exit ──────────────────────────
    if args.list_topics:
        print("  Preset debate topics:\n")
        for category, topics in PRESET_TOPICS.items():
            print(f"  {category}:")
            for topic in topics:
                print(f"    • {topic}")
            print()
        sys.exit(0)

    # ── Get topic ─────────────────────────────────────
    if args.topic:
        topic = args.topic.strip()
        # Validate provided topic
        valid, reason = validate_topic_tier1(topic)
        if not valid:
            print(f"\n  Invalid topic: {reason}\n")
            sys.exit(1)
    else:
        topic = select_topic()

    # ── Tier 2 validation ─────────────────────────────
    print(f"\n  Validating topic...")
    valid, reason = validate_topic_tier2(topic)
    if not valid:
        print(f"\n  Topic rejected: {reason}")
        print(f"  Please choose a genuinely "
              f"debatable topic.\n")
        sys.exit(1)

    print(f"  Topic validated ✅")

    # ── Print debate config ───────────────────────────
    print(f"\n  Configuration:")
    print(f"    Topic:     {topic}")
    print(f"    Awareness: {args.awareness}")
    print(f"    Run twice: {args.twice}")

    input(f"\n  Press Enter to begin...\n")

    # ── Run debate ────────────────────────────────────
    try:
        if args.twice:
            run_twice(
                topic     = topic,
                awareness = args.awareness,
            )
        else:
            run_debate(
                topic     = topic,
                awareness = args.awareness,
            )

    except TopicValidationError as e:
        print(f"\n  Topic error: {e}\n")
        sys.exit(1)

    except ConfigurationError as e:
        print(f"\n  Configuration error: {e}\n")
        sys.exit(1)

    except NeuroColosseumError as e:
        print(f"\n  Debate error: {e}\n")
        sys.exit(1)

    except KeyboardInterrupt:
        print(f"\n\n  Debate interrupted by user.\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
# edges.py
# ════════════════════════════════════════════════════════
# NeuroColosseum — Graph Routing Logic
# Two models enter. One argument wins.
#
# Key principle:
#   scored_phase = what judge just scored
#   current_phase = where debate goes next
#   These are set explicitly — never inferred
# ════════════════════════════════════════════════════════

from config import settings
from state import DebateState


def route_after_speech(state: DebateState) -> str:
    # print(f"\n  DEBUG route_after_speech:")
    # print(f"    phase:    {state['current_phase']}")
    # print(f"    speeches: {state['speeches_this_phase']}")
    # print(f"    turn:     {state['current_turn']}")
    if _should_abort(state):
        return "error"

    # Both spoke → advance phase
    # Do NOT score — judge only at end
    if state["speeches_this_phase"] >= 2:
        return "advance"      # ← was "judge", now just advance

    return "debater"


def route_after_scoring(state: DebateState) -> str:
    """Only called once — after closing."""
    if _should_abort(state):
        return "error"
    return _route_to_verdict_or_tiebreak(state)

def route_after_cross_exam(
    state: DebateState,
) -> str:
    if _should_abort(state):
        return "error"
    if state["cross_exam_turn"] >= 4:
        return "advance"
    return "cross_exam"


def route_after_phase_transition(
    state: DebateState,
) -> str:
    phase = state["current_phase"]

    if phase == "cross_exam":
        return "cross_exam"

    if phase == "verdict":
        return "judge"

    return "debater"

def route_after_tiebreak(state: DebateState) -> str:
    """Always goes to verdict."""
    return "verdict"


def get_next_phase(current: str) -> str:
    order = [
        "opening",
        "rebuttal_1",
        "rebuttal_2",
        "cross_exam",
        "closing",
        "verdict",   # ← triggers judge
    ]
    try:
        return order[order.index(current) + 1]
    except (ValueError, IndexError):
        return "verdict"


def get_rebuttal_first_turn(phase: str) -> str:
    if phase == "rebuttal_1":
        return "B"   # B challenges A's opening
    if phase == "rebuttal_2":
        return "A"   # A challenges B's opening
    return "A"


def _should_abort(state: DebateState) -> bool:
    """Abort if 2 or more errors accumulated."""
    return len(state.get("errors", [])) >= 2


def _route_to_verdict_or_tiebreak(
    state: DebateState,
) -> str:
    """Checks if tiebreak needed after closing."""
    difference = abs(
        state["total_a"] - state["total_b"]
    )
    if difference <= settings.tiebreak_threshold:
        print(f"\n  Tiebreak triggered! "
              f"Gap: {difference}")
        return "tiebreak"

    return "verdict"
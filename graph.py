# graph.py
# ════════════════════════════════════════════════════════
# NeuroColosseum — Graph Assembly
# Two models enter. One argument wins.
#
# This file assembles the complete LangGraph.
# It connects all nodes with edges.
# It defines the phase_transition_node inline
# because it is graph housekeeping not debate logic.
#
# Structure:
#   1. Import all nodes and edge functions
#   2. Define phase_transition_node inline
#   3. Build and return compiled graph
#
# Import hierarchy — graph.py is Layer 4:
#   Imports from config  (Layer 1)
#   Imports from state   (Layer 2)
#   Imports from nodes   (Layer 3)
#   Imports from edges   (Layer 3)
#   Never imports from run.py or streamlit_app.py
# ════════════════════════════════════════════════════════

from langgraph import graph
from langgraph.graph import StateGraph, END

from state import DebateState
from nodes import (
    setup_node,
    debater_node,
    cross_exam_node,
    judge_score_node,
    verdict_node,
    tiebreak_node,
    error_node
)
from edges import (
    route_after_speech,
    route_after_scoring,
    route_after_cross_exam,
    route_after_phase_transition,  # ← add this
    route_after_tiebreak,
    get_next_phase,
    get_rebuttal_first_turn,
)


# ════════════════════════════════════════════════════════
# PHASE TRANSITION NODE
# Defined inline here because it is graph housekeeping
# not debate logic. Too simple for its own file.
#
# Called after every judge_score_node completion.
# Updates three state fields:
#   current_phase       → next phase name
#   current_turn        → who goes first in next phase
#   speeches_this_phase → reset to 0
#
# Uses helpers from edges.py:
#   get_next_phase()         → what phase is next
#   get_rebuttal_first_turn() → who leads next phase
# ════════════════════════════════════════════════════════

def phase_transition_node(
    state: DebateState,
) -> dict:
    """
    Advances phase after speeches complete.
    closing → sets phase to "verdict"
              which triggers judge scoring.
    cross_exam handles its own transition
    so we skip it here.
    """
    # print(f"\n  DEBUG phase_transition:")
    # print(f"    current_phase:       {state['current_phase']}")
    # print(f"    scored_phase:        {state.get('scored_phase')}")
    # print(f"    speeches_this_phase: {state['speeches_this_phase']}")
    # print(f"    current_turn:        {state['current_turn']}")
    # print(f"    cross_exam_turn:     {state.get('cross_exam_turn')}")
    
    # Use current_phase — what we just finished
    # NOT scored_phase — that is for judge only
    current    = state["current_phase"]
    next_phase = get_next_phase(current)
    first_turn = get_rebuttal_first_turn(next_phase)

    print(f"\n  → {current.upper()} complete")
    print(f"  → Starting {next_phase.upper()}")

    return {
        "current_phase":       next_phase,
        "scored_phase":        current,   # record what just finished
        "current_turn":        first_turn,
        "speeches_this_phase": 0,
        "current_node":        "phase_transition_node",
    }

# ════════════════════════════════════════════════════════
# GRAPH BUILDER
# Assembles all nodes and edges into a compiled graph.
# Called once at startup.
# Returns a compiled graph ready for invoke().
# ════════════════════════════════════════════════════════

def build_graph():
    """
    Builds and compiles the NeuroColosseum debate graph.

    Creates a LangGraph StateGraph with:
        8 debate nodes  (from nodes.py)
        1 transition node (defined inline above)
        4 simple edges
        3 conditional edges

    Returns:
        Compiled LangGraph ready for invoke()

    Usage:
        graph = build_graph()
        result = graph.invoke(initial_state)

    The graph is immutable after compile().
    Call build_graph() once and reuse.
    """

    # ── Create graph with our state schema ───────────
    # StateGraph knows what fields to expect
    # Validates state structure at compile time
    graph = StateGraph(DebateState)


    # ════════════════════════════════════════════════
    # REGISTER ALL NODES
    # Every node must be registered before
    # it can be used in edges.
    # Format: graph.add_node("name", function)
    # "name" = string used in edge definitions
    # function = the actual node function
    # ════════════════════════════════════════════════

    graph.add_node("setup_node",            setup_node)
    graph.add_node("debater_node",          debater_node)
    graph.add_node("cross_exam_node",       cross_exam_node)
    graph.add_node("judge_score_node",      judge_score_node)
    graph.add_node("phase_transition_node", phase_transition_node)
    graph.add_node("verdict_node",          verdict_node)
    graph.add_node("tiebreak_node",         tiebreak_node)
    graph.add_node("error_node",            error_node)


    # ════════════════════════════════════════════════
    # SET ENTRY POINT
    # The first node that runs when graph.invoke()
    # is called. Always setup_node for us.
    # ════════════════════════════════════════════════

    graph.set_entry_point("setup_node")


    # ════════════════════════════════════════════════
    # SIMPLE EDGES
    # These always go to the same next node.
    # No decision needed.
    # Format: graph.add_edge("from", "to")
    # ════════════════════════════════════════════════

    # Setup always leads to first debater speech
    graph.add_edge(
        "setup_node",
        "debater_node"
    )

    graph.add_conditional_edges(
        "phase_transition_node",
        route_after_phase_transition,
        {
            "debater":    "debater_node",
            "cross_exam": "cross_exam_node",
            "judge":      "judge_score_node",
            "error":      "error_node",
        }
    )

    # After tiebreak always goes to verdict
    # tiebreak result stored in state for verdict to use
    graph.add_edge(
        "tiebreak_node",
        "verdict_node"
    )

    # Both verdict and error are terminal nodes
    graph.add_edge("verdict_node", END)
    graph.add_edge("error_node",   END)


    # ════════════════════════════════════════════════
    # CONDITIONAL EDGES
    # These call a routing function to decide
    # which node comes next.
    #
    # Format:
    #   graph.add_conditional_edges(
    #       "from_node",
    #       routing_function,
    #       {"return_value": "to_node", ...}
    #   )
    #
    # The routing function receives full state
    # and returns a string key from the dict.
    # LangGraph follows that key to the next node.
    # ════════════════════════════════════════════════

    # ── After debater speaks ──────────────────────
    # route_after_speech checks speeches_this_phase
    # "debater" → another speech needed (< 2 speeches)
    # "judge"   → both spoke, time to score
    # "error"   → too many failures, abort
    # route_after_speech now returns "advance" not "judge"
    graph.add_conditional_edges(
        "debater_node",
        route_after_speech,
        {
            "debater":  "debater_node",
            "advance":  "phase_transition_node",  # ← must exist
            "error":    "error_node",
        }
    )

    # ── After cross exam turn completes ───────────
    # route_after_cross_exam checks cross_exam_turn
    # "cross_exam" → more turns needed (turn < 4)
    # "judge"      → all 4 turns done, score it
    # "error"      → too many failures, abort
    graph.add_conditional_edges(
        "cross_exam_node",
        route_after_cross_exam,
        {
            "cross_exam": "cross_exam_node",
            "advance":    "phase_transition_node",  # ← add this
            "error":      "error_node",
        }
    )

    # judge → verdict or tiebreak
    graph.add_conditional_edges(
        "judge_score_node",
        route_after_scoring,
        {
            "verdict":   "verdict_node",
            "tiebreak":  "tiebreak_node",
            "error":     "error_node",
        }
    )

    # ════════════════════════════════════════════════
    # COMPILE
    # Validates the graph structure:
    #   All edge destinations are registered nodes
    #   Entry point exists
    #   All nodes have at least one path to END
    # Returns compiled graph ready for invoke()
    # ════════════════════════════════════════════════

    return graph.compile()


# ════════════════════════════════════════════════════════
# MODULE LEVEL INSTANCE
# Build the graph once when this module is imported.
# Reused across all debates — no rebuild needed.
#
# Why at module level?
#   Building the graph takes a small amount of time.
#   Doing it once at import = zero overhead per debate.
#   Both run.py and streamlit_app.py import this.
#   Both get the same pre-built graph instance. ✅
# ════════════════════════════════════════════════════════

debate_graph = build_graph()

"""
## The Complete Graph Visualised
```
START
  ↓
[setup_node]                    validates topic
  ↓ (simple edge)
[debater_node] ◄─────────────────────────────────┐
  ↓                                               │
route_after_speech                                │
  ├── "debater" ──────────────────────────────────┘
  │   (speeches < 2, need another)
  │
  └── "judge" ──────────────────────────────────┐
      (both spoke, score the round)             │
                                                ↓
                                    [judge_score_node]
                                                ↓
                                    route_after_scoring
                                      ├── "debater"
                                      │    ↓
                                      │   [phase_transition_node]
                                      │    updates phase/turn/count
                                      │    ↓ (simple edge)
                                      │   [debater_node] ← back to top
                                      │
                                      ├── "cross_exam"
                                      │    ↓
                                      │   [cross_exam_node] ◄──┐
                                      │    ↓                   │
                                      │   route_after_cross_exam│
                                      │    ├── "cross_exam" ───┘
                                      │    └── "judge" → judge_score_node
                                      │
                                      ├── "tiebreak"
                                      │    ↓
                                      │   [tiebreak_node]
                                      │    ↓ (simple edge)
                                      │   [verdict_node]
                                      │    ↓ (simple edge)
                                      │   END
                                      │
                                      └── "verdict"
                                           ↓
                                          [verdict_node]
                                           ↓ (simple edge)
                                          END

  Any node → "error" → [error_node] → END
"""
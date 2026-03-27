# state.py
# ════════════════════════════════════════════════════════
# NeuroColosseum — Debate State Definition
# Two models enter. One argument wins.
#
# DebateState is the single source of truth
# that travels through every node in the graph.
#
# Think of it as the debate's memory.
# Every node reads from it and adds to it.
# Nothing is lost between nodes.
#
# Import hierarchy — state.py is Layer 2:
#   Only imports from config (Layer 1)
#   Never imports from nodes, edges, graph
# ════════════════════════════════════════════════════════

from typing import TypedDict, Optional


# ════════════════════════════════════════════════════════
# DEBATE STATE
# The complete state that travels through every node.
#
# Field naming convention:
#   _a suffix  → belongs to Debater A
#   _b suffix  → belongs to Debater B
#   current_   → what is happening RIGHT NOW
#   cross_exam_ → specific to cross exam phase
#   run_        → specific to run twice bias test
# ════════════════════════════════════════════════════════

class DebateState(TypedDict):
    """
    Complete state for one NeuroColosseum debate.

    Travels through every node in the LangGraph.
    Every node receives the FULL state.
    Every node returns ONLY what it changed.
    LangGraph merges changes automatically.

    Initial state is created in run.py / streamlit_app.py
    and passed to graph.invoke(initial_state).

    Fields are grouped by purpose:
        Setup          → debate configuration
        Phase tracking → where we are in the debate
        Cross exam     → Q&A phase specific fields
        Debate content → speeches and transcript
        Scores         → all scoring data
        Bias test      → run twice comparison
        Output         → final results
        Meta           → errors and status
    """

    # ════════════════════════════════════════
    # SETUP
    # Set once in setup_node — never changes
    # ════════════════════════════════════════

    topic:          str
    # The debate topic
    # e.g. "AI will do more harm than good"

    position_a:     str
    # Always "FOR" — Debater A argues for the motion

    position_b:     str
    # Always "AGAINST" — Debater B argues against

    topic_valid:    bool
    # True after topic passes both guardrail tiers
    # Debate cannot start if False

    label_map:      dict
    # Anonymous labels for blind verdict scoring
    # Randomly assigned in setup_node
    # e.g. {"X": "A", "Y": "B"}
    # or   {"X": "B", "Y": "A"}
    # Revealed only after verdict declared

    awareness:      str
    # Debater awareness setting from .env
    # none / criteria / scores / trailing
    # Controls what debaters know about scores


    # ════════════════════════════════════════
    # PHASE TRACKING
    # Controls where we are in the debate.
    # These three fields are the "traffic lights"
    # that phase_router reads to make decisions.
    # ════════════════════════════════════════

    current_phase:        str
    # Which phase is active right now
    # Values: opening / rebuttal_1 / rebuttal_2 /
    #         cross_exam / closing / verdict

    current_turn:         str
    # Which debater speaks next
    # Values: "A" or "B"
    # Flipped by debater_node after each speech
    # Reset by phase_router when phase changes

    speeches_this_phase:  int
    # How many speeches completed in current phase
    # Starts at 0 for each new phase
    # When it reaches 2 → phase is complete
    # Reset to 0 by phase_router on phase change
    # NOT used for cross_exam (has own counter)


    # ════════════════════════════════════════
    # CROSS EXAMINATION
    # Fields specific to Phase 4.
    # cross_exam_turn is the counter here
    # (speeches_this_phase is not used in this phase)
    # ════════════════════════════════════════

    cross_exam_turn:       int
    # Tracks progress through cross exam
    # 0 → A generating questions
    # 1 → B answering A's questions
    # 2 → B generating questions
    # 3 → A answering B's questions
    # 4 → cross exam complete

    cross_exam_session:    str
    # Which session is active
    # Values: "A_asks" or "B_asks"
    # Used by phase_router for routing decisions

    cross_exam_mode:       str
    # What is happening within current session
    # Values: "asking" / "answering" / "done"

    cross_exam_questions:  list
    # Questions generated in current session
    # List of strings
    # Cleared at start of each session

    cross_exam_answers:    list
    # Answers given in current session
    # List of strings
    # Cleared at start of each session

    cross_exam_scores_a:   list
    # A's scores from both cross exam sessions
    # [session1_score, session2_score]
    # Averaged to get final cross exam score for A

    cross_exam_scores_b:   list
    # B's scores from both cross exam sessions
    # [session1_score, session2_score]
    # Averaged to get final cross exam score for B


    # ════════════════════════════════════════
    # DEBATE CONTENT
    # The actual words spoken in the debate.
    # ════════════════════════════════════════

    transcript:     list
    # Complete record of every speech in order
    # Each entry is a dict:
    # {
    #   "phase":    "opening",
    #   "debater":  "A",
    #   "position": "FOR",
    #   "round":    1,
    #   "argument": "AI has already saved...",
    #   "word_count": 387,
    #   "scores":   {coherence: 18, ...},
    #   "total":    71
    # }
    # Grows with every speech
    # Used by judge for context and final verdict

    last_argument:  str
    # The most recent speech text
    # Used by judge_score_node (scores this)
    # Used by next debater (responds to this)
    # Overwritten after every speech

    last_speaker:   str
    # Who gave the last argument
    # Values: "A" or "B"
    # Used by judge_score_node to know
    # which debater's scores to update


    # ════════════════════════════════════════
    # SCORES
    # All scoring data accumulated here.
    # Updated by judge_score_node after each round.
    # Never modified by debater_node.
    # ════════════════════════════════════════

    scores_a:           dict
    # Scores from the most recent round for A
    # e.g. {coherence: 18, evidence: 16, ...}
    # Overwritten each round
    # Individual round scores stored in transcript

    scores_b:           dict
    # Scores from the most recent round for B
    # Same structure as scores_a

    total_a:            int
    # Running total score for Debater A
    # Accumulates across all phases
    # Maximum possible: 500 (5 phases × 100)

    total_b:            int
    # Running total score for Debater B
    # Same structure as total_a

    criterion_totals_a: dict
    # Accumulated score per criterion for A
    # Across all phases
    # e.g. {coherence: 72, evidence: 68, ...}
    # Used by tiebreak_node (level 1)
    # Used by awareness "scores" mode

    criterion_totals_b: dict
    # Same structure as criterion_totals_a for B

    score_history:      list
    # Score after every completed round
    # Each entry:
    # {
    #   "phase":   "opening",
    #   "round":   1,
    #   "score_a": 71,
    #   "score_b": 74,
    #   "total_a": 71,
    #   "total_b": 74
    # }
    # Used by Streamlit to animate scoreboard
    # Used by bias_service for analysis


    # ════════════════════════════════════════
    # BIAS TEST (RUN TWICE)
    # Fields for the Run Twice feature.
    # Only populated on run 2.
    # ════════════════════════════════════════

    run_number:     int
    # Which run this is: 1 or 2
    # Run 1: models in original positions
    # Run 2: models swapped
    # Set in run.py / streamlit_app.py
    # Never changed by graph nodes

    run1_result:    Optional[dict]
    # Results from Run 1
    # Populated at start of Run 2
    # {winner, total_a, total_b, topic}
    # None during Run 1

    run2_result:    Optional[dict]
    # Results from Run 2
    # Populated after Run 2 verdict
    # None during Run 1

    bias_analysis:  Optional[str]
    # Human readable bias analysis
    # Generated after Run 2 completes
    # None during Run 1
    # e.g. "Position bias detected — FOR wins both runs"


    # ════════════════════════════════════════
    # TIEBREAK
    # Only populated if scores are tied.
    # ════════════════════════════════════════

    tiebreak_needed:      bool
    # True if score difference <= tiebreak_threshold
    # Checked by phase_router after closing scores

    tiebreak_level:       int
    # Which level resolved the tie (1-5)
    # 0 if tiebreak not needed

    tiebreak_winner:      Optional[str]
    # "A" or "B" — winner after tiebreak
    # None if tiebreak not needed or unresolved

    tiebreak_explanation: Optional[str]
    # Human readable explanation of how tie was broken
    # e.g. "A won 3 of 5 criteria (Level 1)"
    # None if tiebreak not needed


    # ════════════════════════════════════════
    # OUTPUT
    # Final results — populated by verdict_node
    # ════════════════════════════════════════

    winner:           Optional[str]
    # "A" or "B" — debate winner
    # None until verdict_node runs

    verdict:          Optional[str]
    # Full judge reasoning for the decision
    # None until verdict_node runs

    verdict_scores:   Optional[dict]
    # Final score breakdown from verdict
    # {
    #   "total_a": 350,
    #   "total_b": 363,
    #   "criteria_a": {...},
    #   "criteria_b": {...}
    # }
    # None until verdict_node runs

    transcript_path:  Optional[str]
    # File path where transcript was saved
    # e.g. "outputs/debate_2026-03-23_1432_a1b2c3.md"
    # None if save_transcript = False


    # ════════════════════════════════════════
    # META
    # Error tracking and debug info
    # ════════════════════════════════════════

    errors:           list
    # All errors encountered during debate
    # Each entry: string description
    # Empty list = clean run
    # 2+ entries = route to error_node

    current_node:     str
    # Name of the node currently executing
    # Updated by each node on entry
    # Used for debugging and LangSmith tracing

    scored_phase: str
    # What phase judge_score_node is currently scoring
    # Set before every judge_score_node call
    # Separate from current_phase to avoid confusion


# ════════════════════════════════════════════════════════
# INITIAL STATE FACTORY
# Creates a clean starting state for a new debate.
# Called by run.py and streamlit_app.py.
# Never called by graph nodes.
# ════════════════════════════════════════════════════════

def create_initial_state(
    topic:      str,
    awareness:  str  = "criteria",
    run_number: int  = 1,
    run1_result: dict | None = None,
) -> DebateState:
    """
    Creates a clean initial DebateState for a new debate.

    Sets all fields to their correct starting values.
    Nodes only receive this — they never create state.

    Args:
        topic:       The debate topic
        awareness:   Debater awareness setting
        run_number:  1 for first run, 2 for swap run
        run1_result: Results from run 1 (for run 2 only)

    Returns:
        Complete DebateState with all fields initialised

    Example:
        state = create_initial_state(
            topic="AI will do more harm than good",
            awareness="criteria"
        )
        result = graph.invoke(state)
    """
    import random

    # Randomly assign X/Y labels for blind scoring
    labels = ["A", "B"]
    random.shuffle(labels)
    label_map = {
        "X": labels[0],
        "Y": labels[1]
    }

    return DebateState(

        # ── Setup ────────────────────────────
        topic          = topic,
        position_a     = "FOR",
        position_b     = "AGAINST",
        topic_valid    = False,   # set True by setup_node
        label_map      = label_map,
        awareness      = awareness,

        # ── Phase tracking ───────────────────
        current_phase        = "opening",
        current_turn         = "A",
        speeches_this_phase  = 0,

        # ── Cross exam ───────────────────────
        cross_exam_turn      = 0,
        cross_exam_session   = "A_asks",
        cross_exam_mode      = "asking",
        cross_exam_questions = [],
        cross_exam_answers   = [],
        cross_exam_scores_a  = [],
        cross_exam_scores_b  = [],

        # ── Debate content ───────────────────
        transcript     = [],
        last_argument  = "",
        last_speaker   = "",

        # ── Scores ───────────────────────────
        scores_a            = {},
        scores_b            = {},
        total_a             = 0,
        total_b             = 0,
        criterion_totals_a  = {},
        criterion_totals_b  = {},
        score_history       = [],
        scored_phase = "opening",   # tracks what judge scores

        # ── Bias test ────────────────────────
        run_number    = run_number,
        run1_result   = run1_result,
        run2_result   = None,
        bias_analysis = None,

        # ── Tiebreak ─────────────────────────
        tiebreak_needed      = False,
        tiebreak_level       = 0,
        tiebreak_winner      = None,
        tiebreak_explanation = None,

        # ── Output ───────────────────────────
        winner          = None,
        verdict         = None,
        verdict_scores  = None,
        transcript_path = None,

        # ── Meta ─────────────────────────────
        errors       = [],
        current_node = "start",
    )

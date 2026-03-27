# nodes.py
# ════════════════════════════════════════════════════════
# NeuroColosseum — Graph Nodes
# Two models enter. One argument wins.
#
# Eight nodes — one per responsibility.
# Each node follows the same contract:
#   RECEIVES: full DebateState
#   DOES:     exactly one job
#   RETURNS:  dict of only changed fields
#   RAISES:   typed exception on failure
#
# Import hierarchy — nodes.py is Layer 3:
#   Imports from config (Layer 1)
#   Imports from state (Layer 2)
#   Never imports from edges or graph
# ════════════════════════════════════════════════════════

import os
import re
import random
import hashlib
from datetime import datetime
from langchain_core.messages import SystemMessage, HumanMessage

from config import (
    settings,
    get_llm,
    get_llm_name,
    get_awareness_prompt,
    validate_topic_tier1,
    ANTI_BIAS_PROMPT,
    SCORING_RUBRIC,
    PHASE_CRITERIA,
)
from state import DebateState
from exceptions import (
    LLMCallError,
    ScoringError,
    PhaseError,
    CrossExamError,
    TranscriptError,
    TopicValidationError,
    TiebreakError,
)


# ════════════════════════════════════════════════════════
# NODE 1 — SETUP
# First node in the graph.
# Validates topic, confirms positions,
# marks debate as ready to start.
# ════════════════════════════════════════════════════════

def setup_node(state: DebateState) -> dict:
    """
    Prepares the debate before first speech.

    Does three things:
        1. Validates topic (Tier 1 keyword check)
        2. Confirms positions A=FOR, B=AGAINST
        3. Sets topic_valid = True

    Note: label_map already set in create_initial_state
    Note: Tier 2 LLM validation happens in UI before
          graph.invoke() is called

    Returns:
        topic_valid, position_a, position_b, current_node

    Raises:
        TopicValidationError: if topic fails keyword check
    """
    print(f"\n{'═'*52}")
    print(f"  SETUP")
    print(f"{'═'*52}")

    topic = state["topic"].strip()
    print(f"  Topic:    {topic}")
    print(f"  Debater A: FOR  ({get_llm_name('debater_a')})")
    print(f"  Debater B: AGAINST  ({get_llm_name('debater_b')})")
    print(f"  Judge:    {get_llm_name('judge')}")

    # ── Tier 1 topic validation ───────────────────────
    valid, reason = validate_topic_tier1(topic)
    if not valid:
        raise TopicValidationError(reason)

    print(f"  Topic validated ✅")

    return {
        "topic_valid":    True,
        "position_a":     "FOR",
        "position_b":     "AGAINST",
        "current_node":   "setup_node",
    }


# ════════════════════════════════════════════════════════
# NODE 2 — DEBATER
# Handles ALL speeches for BOTH debaters.
# Adapts prompt based on current_phase and current_turn.
# One node handles opening, rebuttal, and closing.
# Cross examination uses its own dedicated node.
# ════════════════════════════════════════════════════════

def debater_node(state: DebateState) -> dict:
    """
    Generates one speech from one debater.
    Used for opening, rebuttal, and closing phases.

    Reads current_turn to know who speaks.
    Reads current_phase to build correct prompt.
    Injects awareness context from config.
    Streams argument live to terminal.
    Enforces word limit.

    Returns:
        last_argument, last_speaker, current_turn,
        speeches_this_phase, transcript, current_node

    Raises:
        LLMCallError: if LLM call fails
        PhaseError: if phase is unrecognised
    """
    turn      = state["current_turn"]     # "A" or "B"
    phase     = state["current_phase"]
    topic     = state["topic"]
    position  = (
        state["position_a"]
        if turn == "A"
        else state["position_b"]
    )
    model_name = get_llm_name(
        f"debater_{turn.lower()}"
    )

    # Safety guard — should never happen
    # but catches routing errors early
    if phase == "cross_exam":
        raise PhaseError(
            f"debater_node called with phase='cross_exam'.\n"
            f"Check routing — cross_exam_node should "
            f"handle this phase.\n"
            f"scored_phase={state.get('scored_phase')}\n"
            f"cross_exam_turn={state.get('cross_exam_turn')}"
        )

    print(f"\n{'═'*52}")
    print(f"  DEBATER {turn} — {phase.upper().replace('_', ' ')}")
    print(f"  Position: {position}")
    print(f"  Model:    {model_name}")
    print(f"{'═'*52}\n")

    # ── Build phase-specific instruction ─────────────
    instruction = _build_debater_instruction(
        turn, phase, topic, position, state
    )

    # ── Build awareness context ───────────────────────
    awareness = get_awareness_prompt(turn, state)

    # ── Build system prompt ───────────────────────────
    system_prompt = f"""You are Debater {turn} in a formal debate.
You are arguing {position} the motion: "{topic}"

Critical rules:
  → NEVER switch sides or acknowledge the opposing
    position as valid — you always argue {position}
  → Stay focused on the motion — no tangents
  → Be direct, persuasive, and use evidence
  → Maximum {settings.max_argument_words} words
  → Never exceed the word limit

{awareness}"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=instruction),
    ]

    # ── Call LLM with streaming ───────────────────────
    argument = _stream_llm(
        role=f"debater_{turn.lower()}",
        messages=messages,
        node="debater_node",
    )

    # ── Enforce word limit ────────────────────────────
    words    = argument.split()
    if len(words) > settings.max_argument_words:
        argument = " ".join(
            words[:settings.max_argument_words]
        )
        print(
            f"\n  [Word limit enforced: "
            f"{settings.max_argument_words} words]"
        )

    # ── Build transcript entry ────────────────────────
    entry = {
        "phase":      phase,
        "debater":    turn,
        "position":   position,
        "round":      state.get("speeches_this_phase", 0) + 1,
        "argument":   argument,
        "word_count": len(argument.split()),
        "scores":     {},    # filled by judge_score_node
        "total":      0,     # filled by judge_score_node
    }

    # ── Update state ──────────────────────────────────
    new_transcript = state["transcript"] + [entry]
    next_turn      = "B" if turn == "A" else "A"
    new_speeches   = state["speeches_this_phase"] + 1

    return {
        "last_argument":       argument,
        "last_speaker":        turn,
        "current_turn":        next_turn,
        "speeches_this_phase": new_speeches,
        "transcript":          new_transcript,
        "current_node":        "debater_node",
    }

def _get_phase_speech(
    state:   DebateState,
    phase:   str,
    debater: str,
) -> str:
    """Returns a specific debater's speech from a specific phase."""
    for entry in state["transcript"]:
        if entry["phase"] == phase and \
           entry["debater"] == debater:
            return entry["argument"]
    return ""


def _build_debater_instruction(
    turn:     str,
    phase:    str,
    topic:    str,
    position: str,
    state:    DebateState,
) -> str:
    """
    Builds the phase-specific instruction for a debater.
    Different phases require different approaches.
    """

    # ── Opening ───────────────────────────────────────
    if phase == "opening":

        independence_note = ""
        if turn == "B":
            independence_note = """
Important: This is your INDEPENDENT opening argument.
Do NOT directly reference or respond to your
opponent's opening statement.
Save rebuttals for the rebuttal phase.
Present your own case from first principles.
"""
        return f"""Present your opening argument {position} the motion:
"{topic}"

Structure your argument with:
  → A clear opening statement of your position
  → 2-3 strong supporting arguments with evidence
  → A compelling conclusion

{independence_note}
Maximum {settings.max_argument_words} words."""

    # ── Rebuttal ──────────────────────────────────────
    elif phase in ("rebuttal_1", "rebuttal_2"):

        # Who made the OPENING we are challenging?
        # rebuttal_1: B challenges A's opening
        # rebuttal_2: A challenges B's opening
        speeches_this = state["speeches_this_phase"]

        if speeches_this == 0:
            # First speaker — challenging opponent's opening
            # Find opponent's opening speech specifically
            if phase == "rebuttal_1":
                target_debater = "A"   # B challenges A
            else:
                target_debater = "B"   # A challenges B

            target_speech = _get_phase_speech(
                state, "opening", target_debater
            )

            return f"""Challenge your opponent's opening argument directly.

    Their opening statement:
    "{target_speech}"

    Identify the weakest claims and attack them with evidence.
    Maximum {settings.max_argument_words} words."""

        else:
            # Second speaker — replying to the challenge
            challenge = _get_last_speech(
                state,
                "B" if phase == "rebuttal_1" else "A"
            )

            return f"""Your opponent just challenged your opening argument:

    "{challenge}"

    Defend your position and counter their attack.
    Maximum {settings.max_argument_words} words."""
    # ── Closing ───────────────────────────────────────
    elif phase == "closing":

        # Reference cross exam if it happened
        cross_exam_note = ""
        if state.get("cross_exam_answers"):
            cross_exam_note = """
You may reference answers given during
cross examination to strengthen your case.
"""

        return f"""Give your closing statement {position} the motion:
"{topic}"

This is your final opportunity to persuade.
You must:
  → Summarise your strongest arguments
  → Address your opponent's best points
  → Make a compelling final case for your position
  → End with a memorable conclusion

{cross_exam_note}
Maximum {settings.max_argument_words} words."""

    else:
        raise PhaseError(
            f"debater_node called for unknown phase: "
            f"'{phase}'. "
            f"Cross exam uses cross_exam_node instead."
        )


def _get_last_speech(
    state:    DebateState,
    debater:  str,
) -> str:
    """
    Returns the most recent speech from a specific debater.
    Used to give context for rebuttals.
    Returns empty string if no speech found.
    """
    for entry in reversed(state["transcript"]):
        if entry["debater"] == debater:
            return entry["argument"]
    return ""


# ════════════════════════════════════════════════════════
# NODE 3 — CROSS EXAMINATION
# Manages the Q&A loop for Phase 4.
# Uses cross_exam_turn counter (0-4) not
# speeches_this_phase.
#
# Turn 0: A generates questions
# Turn 1: B answers A's questions
# Turn 2: B generates questions
# Turn 3: A answers B's questions
# Turn 4: complete → move to closing
# ════════════════════════════════════════════════════════

def cross_exam_node(state: DebateState) -> dict:
    """
    Manages one step of the cross examination phase.

    Called repeatedly by phase_router until
    cross_exam_turn reaches 4.

    Validates question quality and regenerates
    if questions are too vague (max 2 attempts).

    Returns:
        cross_exam_turn, cross_exam_questions,
        cross_exam_answers, last_argument,
        transcript, current_node

    Raises:
        CrossExamError: if question generation fails
        LLMCallError: if LLM call fails
    """
    turn = state["cross_exam_turn"]

    print(f"\n{'═'*52}")
    print(f"  CROSS EXAMINATION — Turn {turn}")
    print(f"{'═'*52}\n")

    # ── Turn 0: A generates questions ────────────────
    if turn == 0:
        return _generate_questions(state, asker="A")

    # ── Turn 1: B answers A's questions ──────────────
    elif turn == 1:
        return _answer_questions(
            state,
            answerer="B",
            questions=state["cross_exam_questions"],
        )

    # ── Turn 2: B generates questions ────────────────
    elif turn == 2:
        return _generate_questions(state, asker="B")

    # ── Turn 3: A answers B's questions ──────────────
    elif turn == 3:
        return _answer_questions(
            state,
            answerer="A",
            questions=state["cross_exam_questions"],
        )

    # ── Turn 4: complete ──────────────────────────────
    if turn == 4:
        print("  Cross examination complete ✅")
        return {
            "cross_exam_mode":     "done",
            "cross_exam_turn":     5,
            "scored_phase":        "cross_exam",  # score THIS
            "current_phase":       "cross_exam",  # keep as cross_exam
                                                # phase_transition will
                                                # advance to closing
            "current_node":        "cross_exam_node",
        }


def _generate_questions(
    state: DebateState,
    asker: str,
) -> dict:
    """
    Generates cross examination questions.
    Validates quality and regenerates if too vague.
    Maximum 2 regeneration attempts.
    """
    opponent   = "B" if asker == "A" else "A"
    model_name = get_llm_name(f"debater_{asker.lower()}")
    topic      = state["topic"]
    position   = (
        state["position_a"]
        if asker == "A"
        else state["position_b"]
    )

    print(f"  {asker} generating questions "
          f"(max {settings.cross_exam_questions})...")
    print(f"  Model: {model_name}\n")

    # Get opponent's full argument history
    opp_arguments = [
        entry["argument"]
        for entry in state["transcript"]
        if entry["debater"] == opponent
    ]
    opp_context = "\n\n".join(opp_arguments)

    system_prompt = f"""You are Debater {asker} conducting
cross examination of your opponent.
You argue {position} the motion: "{topic}"

Your goal: expose weaknesses in your opponent's
arguments through sharp, pointed questions."""

    instruction = f"""Your opponent has made these arguments:

{opp_context}

Generate exactly {settings.cross_exam_questions} sharp
cross examination questions.

Each question must:
  → Reference a SPECIFIC claim your opponent made
  → Expose a weakness, contradiction, or gap
  → Be a direct question (end with ?)
  → Maximum {settings.max_question_words} words each
  → Be impossible to dismiss without addressing

Format as a numbered list:
1. [question]
2. [question]
3. [question]"""

    # ── Generate with validation loop ────────────────
    questions  = []
    max_attempts = 2 + 1  # initial + 2 regenerations

    for attempt in range(max_attempts):
        response = _call_llm(
            role=f"debater_{asker.lower()}",
            messages=[
                SystemMessage(content=system_prompt),
                HumanMessage(content=instruction),
            ],
            node="cross_exam_node",
        )

        # Parse questions from numbered list
        questions = _parse_numbered_list(response)

        # Validate quality
        valid, reason = _validate_questions(
            questions,
            opp_context,
        )

        if valid:
            break

        if attempt < max_attempts - 1:
            print(f"  Questions too vague "
                  f"(attempt {attempt + 1}) — regenerating...")
            instruction += f"""

Your previous questions were rejected because:
{reason}

Generate STRONGER questions that directly target
specific claims your opponent made."""

    if not questions:
        raise CrossExamError(
            f"Failed to generate valid questions "
            f"after {max_attempts} attempts"
        )

    # ── Display questions ─────────────────────────────
    print(f"  Questions from Debater {asker}:")
    for i, q in enumerate(questions, 1):
        print(f"  Q{i}: {q}")

    # ── Build transcript entry ────────────────────────
    entry = {
        "phase":      "cross_exam",
        "debater":    asker,
        "position":   position,
        "round":      state["cross_exam_turn"],
        "argument":   "\n".join(
            [f"Q{i+1}: {q}"
             for i, q in enumerate(questions)]
        ),
        "word_count": sum(
            len(q.split()) for q in questions
        ),
        "scores":     {},
        "total":      0,
    }

    return {
        "cross_exam_turn":      state["cross_exam_turn"] + 1,
        "cross_exam_session":   "A_asks"
                                if asker == "A"
                                else "B_asks",
        "cross_exam_mode":      "answering",
        "cross_exam_questions": questions,
        "cross_exam_answers":   [],
        "last_argument":        entry["argument"],
        "last_speaker":         asker,
        "transcript":           state["transcript"] + [entry],
        "current_node":         "cross_exam_node",
    }


def _answer_questions(
    state:     DebateState,
    answerer:  str,
    questions: list,
) -> dict:
    """
    Generates answers to cross examination questions.
    Answerer must address each question directly.
    """
    model_name = get_llm_name(
        f"debater_{answerer.lower()}"
    )
    topic    = state["topic"]
    position = (
        state["position_a"]
        if answerer == "A"
        else state["position_b"]
    )

    print(f"  {answerer} answering questions...")
    print(f"  Model: {model_name}\n")

    questions_text = "\n".join(
        [f"{i+1}. {q}"
         for i, q in enumerate(questions)]
    )

    system_prompt = f"""You are Debater {answerer} answering
cross examination questions.
You argue {position} the motion: "{topic}"

Answer each question directly and confidently.
Do not evade — address each question head on
while maintaining your position."""

    instruction = f"""Answer these cross examination questions:

{questions_text}

For each question:
  → Address it directly — no evasion
  → Defend your position {position}
  → Keep each answer concise and clear
  → Number your answers to match the questions

Format:
1. [answer to Q1]
2. [answer to Q2]
3. [answer to Q3]"""

    response = _stream_llm(
        role=f"debater_{answerer.lower()}",
        messages=[
            SystemMessage(content=system_prompt),
            HumanMessage(content=instruction),
        ],
        node="cross_exam_node",
    )

    answers = _parse_numbered_list(response)

    # ── Build transcript entry ────────────────────────
    entry = {
        "phase":      "cross_exam",
        "debater":    answerer,
        "position":   position,
        "round":      state["cross_exam_turn"],
        "argument":   "\n".join(
            [f"A{i+1}: {a}"
             for i, a in enumerate(answers)]
        ),
        "word_count": sum(
            len(a.split()) for a in answers
        ),
        "scores":     {},
        "total":      0,
    }

    return {
        "cross_exam_turn":     state["cross_exam_turn"] + 1,
        "cross_exam_mode":     "done",
        "cross_exam_answers":  answers,
        "last_argument":       entry["argument"],
        "last_speaker":        answerer,
        "transcript":          state["transcript"] + [entry],
        "current_node":        "cross_exam_node",
    }


def _validate_questions(
    questions: list,
    opp_context: str,
) -> tuple[bool, str]:
    """
    Validates cross exam question quality.
    Returns (is_valid, reason_if_invalid).
    """
    if not questions:
        return False, "No questions generated"

    if len(questions) < settings.cross_exam_questions:
        return (
            False,
            f"Only {len(questions)} questions generated, "
            f"need {settings.cross_exam_questions}"
        )

    for i, q in enumerate(questions):
        # Must be a question
        if not q.strip().endswith("?"):
            return (
                False,
                f"Question {i+1} does not end with ?"
            )

        # Must not be too long
        if len(q.split()) > settings.max_question_words:
            return (
                False,
                f"Question {i+1} exceeds "
                f"{settings.max_question_words} word limit"
            )

        # Must not be too vague (too short)
        if len(q.split()) < 8:
            return (
                False,
                f"Question {i+1} is too vague "
                f"(less than 8 words)"
            )

    return True, ""


def _parse_numbered_list(text: str) -> list:
    """
    Parses a numbered list from LLM response.
    Handles formats: "1. item", "1) item", "1: item"
    Returns list of strings.
    """
    lines = text.strip().split("\n")
    items = []

    for line in lines:
        line = line.strip()
        # Match "1. ", "1) ", "1: "
        match = re.match(
            r'^\d+[\.\)\:]\s+(.+)$', line
        )
        if match:
            items.append(match.group(1).strip())

    return items


# ════════════════════════════════════════════════════════
# NODE 4 — JUDGE SCORING
# Scores both debaters after a complete round.
# Called ONCE per round after BOTH debaters finish.
# Completely blind — sees text only, no identities.
# ════════════════════════════════════════════════════════

def _score_cross_exam(
    state: DebateState,
) -> dict:
    """
    Scores cross examination specially.

    Two separate scoring calls:
        Session 1 — A asked, B answered
            A scored on questioner criteria
            B scored on answerer criteria

        Session 2 — B asked, A answered
            B scored on questioner criteria
            A scored on answerer criteria

    Final score = average of both sessions.

    Returns same structure as judge_score_node
    so rest of graph handles it identically.
    """
    print(f"  Scoring cross examination...")

    q_criteria = PHASE_CRITERIA["cross_exam_questioner"]
    a_criteria = PHASE_CRITERIA["cross_exam_answerer"]

    # ── Get cross exam speeches from transcript ───────
    cross_speeches = [
        entry for entry in state["transcript"]
        if entry["phase"] == "cross_exam"
    ]

    if len(cross_speeches) < 4:
        raise ScoringError(
            f"Expected 4 cross exam speeches "
            f"(Q1, A1, Q2, A2), "
            f"found {len(cross_speeches)}"
        )

    # Speeches in order:
    # [0] A questions  → A is questioner
    # [1] B answers    → B is answerer
    # [2] B questions  → B is questioner
    # [3] A answers    → A is answerer

    a_questions = cross_speeches[0]["argument"]
    b_answers   = cross_speeches[1]["argument"]
    b_questions = cross_speeches[2]["argument"]
    a_answers   = cross_speeches[3]["argument"]

    # ── Score Session 1 ───────────────────────────────
    # A as questioner
    print(f"  Scoring Session 1 (A questioned B)...")
    a_q_scores, a_q_total = _score_cross_exam_role(
        speech    = a_questions,
        criteria  = q_criteria,
        role      = "questioner",
        label     = "A",
    )

    # B as answerer
    b_a_scores, b_a_total = _score_cross_exam_role(
        speech    = b_answers,
        criteria  = a_criteria,
        role      = "answerer",
        label     = "B",
    )

    print(f"  Session 1 — A (questioner): {a_q_total}")
    print(f"  Session 1 — B (answerer):   {b_a_total}")

    # ── Score Session 2 ───────────────────────────────
    # B as questioner
    print(f"  Scoring Session 2 (B questioned A)...")
    b_q_scores, b_q_total = _score_cross_exam_role(
        speech    = b_questions,
        criteria  = q_criteria,
        role      = "questioner",
        label     = "B",
    )

    # A as answerer
    a_a_scores, a_a_total = _score_cross_exam_role(
        speech    = a_answers,
        criteria  = a_criteria,
        role      = "answerer",
        label     = "A",
    )

    print(f"  Session 2 — B (questioner): {b_q_total}")
    print(f"  Session 2 — A (answerer):   {a_a_total}")

    # ── Average scores per debater ────────────────────
    # A earned: questioner score (session 1)
    #         + answerer score  (session 2)
    # B earned: answerer score  (session 1)
    #         + questioner score(session 2)

    a_session1 = a_q_total   # A as questioner
    a_session2 = a_a_total   # A as answerer
    b_session1 = b_a_total   # B as answerer
    b_session2 = b_q_total   # B as questioner

    # Store both sessions for record
    new_cross_scores_a = (
        state.get("cross_exam_scores_a", []) +
        [a_session1, a_session2]
    )
    new_cross_scores_b = (
        state.get("cross_exam_scores_b", []) +
        [b_session1, b_session2]
    )

    # Average across both sessions
    total_a_round = (a_session1 + a_session2) // 2
    total_b_round = (b_session1 + b_session2) // 2

    print(f"\n  Cross exam final scores (averaged):")
    print(f"  A: ({a_session1} + {a_session2}) "
          f"/ 2 = {total_a_round}")
    print(f"  B: ({b_session1} + {b_session2}) "
          f"/ 2 = {total_b_round}")

    # ── Build combined scores dict ────────────────────
    # Merge questioner + answerer criteria for display
    scores_a = {
        **a_q_scores,  # question_sharpness, advancement
        **a_a_scores,  # answer_quality, composure
    }
    scores_b = {
        **b_a_scores,  # answer_quality, composure
        **b_q_scores,  # question_sharpness, advancement
    }

    # ── Update running totals ─────────────────────────
    new_total_a = state["total_a"] + total_a_round
    new_total_b = state["total_b"] + total_b_round

    new_crit_a  = _merge_criteria(
        state["criterion_totals_a"], scores_a
    )
    new_crit_b  = _merge_criteria(
        state["criterion_totals_b"], scores_b
    )

    history_entry = {
        "phase":   "cross_exam",
        "score_a": total_a_round,
        "score_b": total_b_round,
        "total_a": new_total_a,
        "total_b": new_total_b,
    }

    new_history = (
        state["score_history"] + [history_entry]
    )

    # ── Update transcript with scores ─────────────────
    new_transcript = _update_transcript_scores(
        state["transcript"],
        "cross_exam",
        scores_a, total_a_round,
        scores_b, total_b_round,
    )

    print(f"\n  Running total — "
          f"A: {new_total_a}  B: {new_total_b}")

    return {
        "scores_a":           scores_a,
        "scores_b":           scores_b,
        "total_a":            new_total_a,
        "total_b":            new_total_b,
        "criterion_totals_a": new_crit_a,
        "criterion_totals_b": new_crit_b,
        "score_history":      new_history,
        "transcript":         new_transcript,
        "cross_exam_scores_a": new_cross_scores_a,
        "cross_exam_scores_b": new_cross_scores_b,
        "current_node":       "judge_score_node",
    }


def _score_cross_exam_role(
    speech:   str,
    criteria: dict,
    role:     str,
    label:    str,
) -> tuple[dict, int]:
    """
    Scores one debater in one cross exam role.

    Args:
        speech:   The speech text to score
        criteria: Scoring criteria dict for this role
        role:     "questioner" or "answerer"
        label:    "A" or "B" for display

    Returns:
        (scores_dict, total)
    """
    criteria_list = "\n".join([
        f"  - {name.replace('_', ' ').title()}: "
        f"0-{points} points"
        for name, points in criteria.items()
    ])

    role_instruction = (
        "Score the quality of these questions. "
        "Did they expose weaknesses in the opponent's "
        "arguments? Were they sharp and specific?"
        if role == "questioner"
        else
        "Score the quality of these answers. "
        "Did they directly address the questions? "
        "Did the debater maintain composure and "
        "defend their position effectively?"
    )

    system_prompt = f"""{ANTI_BIAS_PROMPT}

You are scoring a cross examination {role}.
{role_instruction}

{SCORING_RUBRIC}

Criteria:
{criteria_list}"""

    instruction = f"""Score this cross examination {role}:

{speech}

Return in this EXACT format:
{chr(10).join([
    f'{name}: [score]'
    for name in criteria.keys()
])}
TOTAL: [total]"""

    response = _call_llm(
        role="judge",
        messages=[
            SystemMessage(content=system_prompt),
            HumanMessage(content=instruction),
        ],
        node="judge_score_node",
    )

    scores, total = _parse_scores(
        response, label, criteria
    )

    return scores, total

def judge_score_node(state: DebateState) -> dict:
    """
    Called ONCE after closing completes.
    Scores all phases from transcript.
    Returns accumulated totals.
    """
    print(f"\n{'─'*52}")
    print(f"  JUDGE SCORING — All phases")
    print(f"  Model: {get_llm_name('judge')}")
    print(f"{'─'*52}")

    total_a        = 0
    total_b        = 0
    crit_totals_a  = {}
    crit_totals_b  = {}
    score_history  = []
    new_transcript = list(state["transcript"])

    # Score each phase in order
    phases_to_score = [
        "opening",
        "rebuttal_1",
        "rebuttal_2",
        "cross_exam",
        "closing",
    ]

    for phase in phases_to_score:
        if phase == "cross_exam":
            result = _score_cross_exam(state)
        else:
            result = _score_phase(state, phase)

        if not result:
            continue

        sa, sb   = result["scores_a"], result["scores_b"]
        ta, tb   = result["total_a"],  result["total_b"]

        total_a += ta
        total_b += tb

        crit_totals_a = _merge_criteria(crit_totals_a, sa)
        crit_totals_b = _merge_criteria(crit_totals_b, sb)

        score_history.append({
            "phase":   phase,
            "score_a": ta,
            "score_b": tb,
            "total_a": total_a,
            "total_b": total_b,
        })

        print(f"  {phase:12} → A: {ta}  B: {tb}")

    print(f"\n  Final — A: {total_a}  B: {total_b}")

    return {
        "scores_a":           sa,
        "scores_b":           sb,
        "total_a":            total_a,
        "total_b":            total_b,
        "criterion_totals_a": crit_totals_a,
        "criterion_totals_b": crit_totals_b,
        "score_history":      score_history,
        "transcript":         new_transcript,
        "current_node":       "judge_score_node",
    }

def _score_phase(
    state: DebateState,
    phase: str,
) -> dict | None:
    """Scores one regular phase from transcript."""
    from langchain_core.messages import (
        SystemMessage, HumanMessage
    )

    criteria = PHASE_CRITERIA.get(phase)
    if not criteria:
        return None

    speeches = [
        e for e in state["transcript"]
        if e["phase"] == phase
    ]
    if len(speeches) < 2:
        return None

    label_map = state["label_map"]
    reverse   = {v: k for k, v in label_map.items()}

    speech_x = speeches[0]["argument"]
    speech_y = speeches[1]["argument"]
    debater_first = speeches[0]["debater"]

    criteria_list = "\n".join([
        f"  {k.replace('_',' ').title()}: 0-{v}"
        for k, v in criteria.items()
    ])

    response = _call_llm(
            role="judge",
            messages=[
                SystemMessage(content=f"{ANTI_BIAS_PROMPT}\n{SCORING_RUBRIC}"),
                HumanMessage(content=f"""Phase: {phase}

        Speech X:
        {speech_x}

        Speech Y:
        {speech_y}

        Criteria:
        {criteria_list}

        IMPORTANT: These are two different arguments.
        Score them independently. They will rarely
        deserve identical scores. Find real differences.

        Return EXACTLY this format with numbers only:
        X_SCORES:
        {chr(10).join([f'{k}: [number]' for k in criteria])}
        X_TOTAL: [number]
        Y_SCORES:
        {chr(10).join([f'{k}: [number]' for k in criteria])}
        Y_TOTAL: [number]"""),
            ],
            node="judge_score_node",
        )
    x_scores, x_total = _parse_scores(response, "X", criteria)
    y_scores, y_total = _parse_scores(response, "Y", criteria)

    if settings.length_penalty:
        if len(speech_x.split()) > settings.max_argument_words:
            x_total = min(x_total, 85)
        if len(speech_y.split()) > settings.max_argument_words:
            y_total = min(y_total, 85)

    if debater_first == "A":
        scores_a, ta = x_scores, x_total
        scores_b, tb = y_scores, y_total
    else:
        scores_a, ta = y_scores, y_total
        scores_b, tb = x_scores, x_total

    return {
        "scores_a": scores_a,
        "scores_b": scores_b,
        "total_a":  ta,
        "total_b":  tb,
    }

def _get_round_speeches(
    state: DebateState,
    phase: str,
) -> list:
    """
    Returns the two most recent speeches
    from the given phase.
    Used by judge_score_node to get what to score.
    """
    phase_speeches = [
        entry for entry in state["transcript"]
        if entry["phase"] == phase
    ]
    # Return the last 2 speeches in this phase
    return phase_speeches[-2:]


def _parse_scores(
    response:  str,
    label:     str,
    criteria:  dict,
) -> tuple[dict, int]:
    scores = {}

    # Find the section for this label (X or Y)
    # Look for X_SCORES: or Y_SCORES: block
    upper = response.upper()
    marker = f"{label}_SCORES:"
    start = upper.find(marker)

    # Use full response if marker not found
    section = (
        response[start:] if start != -1
        else response
    )

    for criterion in criteria.keys():
        # Flexible pattern — handles spaces,
        # dashes, different cases, colons
        pattern = (
            rf"{re.escape(criterion)}"
            rf"[\s\-_]*[:=\-]?\s*(\d+)"
        )
        match = re.search(
            pattern,
            section,
            re.IGNORECASE,
        )

        if match:
            score = int(match.group(1))
            max_s = criteria[criterion]
            scores[criterion] = min(
                max(score, 0), max_s
            )
        else:
            # Default to half max
            scores[criterion] = criteria[criterion] // 2
            print(
                f"  Warning: {criterion} score "
                f"not found for {label} — using default"
            )

    total = sum(scores.values())
    return scores, total


def _merge_criteria(
    existing: dict,
    new:      dict,
) -> dict:
    """
    Adds new criterion scores to existing totals.
    Creates new keys if they don't exist yet.
    """
    merged = dict(existing)
    for criterion, score in new.items():
        merged[criterion] = (
            merged.get(criterion, 0) + score
        )
    return merged


def _update_transcript_scores(
    transcript:     list,
    phase:          str,
    scores_a:       dict,
    total_a:        int,
    scores_b:       dict,
    total_b:        int,
) -> list:
    """
    Updates the last two speeches in the transcript
    with their scores.
    Returns updated transcript.
    """
    updated    = list(transcript)
    phase_idxs = [
        i for i, entry in enumerate(updated)
        if entry["phase"] == phase
    ]

    if len(phase_idxs) >= 2:
        # Second-to-last speech
        idx_a = phase_idxs[-2]
        # Last speech
        idx_b = phase_idxs[-1]

        debater_first  = updated[idx_a]["debater"]
        debater_second = updated[idx_b]["debater"]

        # Assign scores to correct debater
        if debater_first == "A":
            updated[idx_a]["scores"] = scores_a
            updated[idx_a]["total"]  = total_a
            updated[idx_b]["scores"] = scores_b
            updated[idx_b]["total"]  = total_b
        else:
            updated[idx_a]["scores"] = scores_b
            updated[idx_a]["total"]  = total_b
            updated[idx_b]["scores"] = scores_a
            updated[idx_b]["total"]  = total_a

    return updated


# ════════════════════════════════════════════════════════
# NODE 5 — VERDICT
# Final holistic judgment after all phases complete.
# Judge reads full transcript in chronological order.
# Debaters shown as anonymous X and Y labels.
# Saves transcript after declaring winner.
# ════════════════════════════════════════════════════════

def verdict_node(state: DebateState) -> dict:
    """
    Declares the debate winner.

    Judge reads entire transcript anonymously.
    Blind to debater identities and positions.
    Makes holistic qualitative judgment.
    Reveals X/Y labels after verdict declared.
    Saves transcript to markdown file.

    Returns:
        winner, verdict, verdict_scores,
        transcript_path, current_node

    Raises:
        LLMCallError: if judge call fails
        TranscriptError: if transcript save fails
    """
    label_map  = state["label_map"]
    model_name = get_llm_name("judge")

    print(f"\n{'═'*52}")
    print(f"  VERDICT")
    print(f"  Judge: {model_name}")
    print(f"{'═'*52}\n")

    # ── Build anonymous transcript ────────────────────
    anon_transcript = _build_anonymous_transcript(
        state["transcript"],
        label_map,
    )

    # ── Build verdict prompt ──────────────────────────
    system_prompt = f"""{ANTI_BIAS_PROMPT}

You are delivering the final verdict on a debate.
You have seen the complete debate anonymously.
Debaters are labelled X and Y — you do not know
their real identities or positions.

Make a holistic judgment based on:
  → Overall argument quality across all phases
  → Use of evidence throughout
  → Effectiveness in cross examination
  → Final closing statement impact
  → Who made the more compelling overall case"""

    instruction = f"""Here is the complete debate transcript:

{anon_transcript}

Final scores:
  Debater X: {_get_debater_total(state, label_map, 'X')} points
  Debater Y: {_get_debater_total(state, label_map, 'Y')} points

Declare the winner and explain your reasoning.

Return in this EXACT format:
WINNER: [X or Y]
REASONING: [3-5 sentences explaining why the winner
            was more persuasive overall]
KEY_MOMENT: [The single most decisive moment in the debate]"""

    # ── Call judge ────────────────────────────────────
    response = _call_llm(
        role="judge",
        messages=[
            SystemMessage(content=system_prompt),
            HumanMessage(content=instruction),
        ],
        node="verdict_node",
    )

    # ── Parse winner ──────────────────────────────────
    # Step 1 — replace X/Y with A/B first
    reverse_map  = {v: k for k, v in label_map.items()}
    verdict_text = response
    for debater, anon in reverse_map.items():
        verdict_text = verdict_text.replace(
            f"Debater {anon}", f"Debater {debater}"
        ).replace(
            f"debater {anon}", f"debater {debater}"
        ).replace(
            f"WINNER: {anon}", f"WINNER: {debater}"
        ).replace(
            f" {anon}\n", f" {debater}\n"
        )

    # Step 2 — parse winner from cleaned text
    winner_label   = _parse_winner(verdict_text)   # ← use verdict_text
    winner_debater = label_map.get(
        winner_label, winner_label               # ← fallback if already A/B
    )

    # ── Map anonymous label back to real debater ──────
    # label_map = {"X": "A", "Y": "B"} or {"X": "B", "Y": "A"}
    # winner_debater = label_map[winner_label]

    # ── Get winner model name ─────────────────────────
    winner_model = get_llm_name(
        f"debater_{winner_debater.lower()}"
    )
    winner_position = (
        state["position_a"]
        if winner_debater == "A"
        else state["position_b"]
    )

    # ── Display verdict ───────────────────────────────
    print(f"  WINNER: Debater {winner_debater}")
    print(f"  Model:  {winner_model}")
    print(f"  Position: {winner_position}")
    print(f"\n  Revealed: X = {label_map['X']}, "
          f"Y = {label_map['Y']}")
    print(f"\n  Final scores:")
    print(f"    A: {state['total_a']} / 500")
    print(f"    B: {state['total_b']} / 500")

    # ── Save transcript ───────────────────────────────
    transcript_path = None
    if settings.save_transcript:
        transcript_path = _save_transcript(
            state, winner_debater, verdict_text
        )

    return {
        "winner":          winner_debater,
        "verdict":         verdict_text,
        "verdict_scores":  {
            "total_a":      state["total_a"],
            "total_b":      state["total_b"],
            "criteria_a":   state["criterion_totals_a"],
            "criteria_b":   state["criterion_totals_b"],
        },
        "transcript_path": transcript_path,
        "current_node":    "verdict_node",
    }


def _build_anonymous_transcript(
    transcript: list,
    label_map:  dict,
) -> str:
    """
    Converts transcript to anonymous format.
    Replaces A/B with X/Y based on label_map.
    Keeps speeches in chronological order.
    """
    # Reverse map: A → X or Y
    reverse_map = {v: k for k, v in label_map.items()}

    lines = []
    for entry in transcript:
        debater      = entry["debater"]
        anon_label   = reverse_map.get(debater, debater)
        phase        = entry["phase"].replace("_", " ").title()
        argument     = entry["argument"]

        lines.append(
            f"[{phase}] Debater {anon_label}:\n"
            f"{argument}\n"
        )

    return "\n".join(lines)


def _get_debater_total(
    state:     DebateState,
    label_map: dict,
    label:     str,
) -> int:
    """Returns total score for an anonymous label."""
    debater = label_map[label]
    return state[f"total_{debater.lower()}"]


def _parse_winner(response: str) -> str:
    match = re.search(
        r"WINNER:\s*([XABY])",
        response,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).upper()

    # Fallback patterns
    for label in ["A", "B", "X", "Y"]:
        if f"debater {label} wins".lower() in response.lower():
            return label

    raise ScoringError(
        "Could not parse winner from verdict"
    )

# ════════════════════════════════════════════════════════
# NODE 6 — TIEBREAK
# Resolves tied or near-tied scores.
# Activated by phase_router when scores are within
# tiebreak_threshold of each other.
# Works through 5 levels until tie is resolved.
# ════════════════════════════════════════════════════════

def tiebreak_node(state: DebateState) -> dict:
    """
    Resolves a tied debate through 5 escalating levels.

    Level 1: Who won more individual criteria (3 of 5)
    Level 2: Adjusted speaks (drop high + low)
             Only if total speeches >= 6
    Level 3: Closing statement winner
    Level 4: Judge holistic call
    Level 5: Honourable draw

    Returns:
        tiebreak_needed, tiebreak_level,
        tiebreak_winner, tiebreak_explanation,
        current_node

    Raises:
        TiebreakError: if all levels exhausted
    """
    print(f"\n{'═'*52}")
    print(f"  TIEBREAK")
    print(f"  Score difference: "
          f"{abs(state['total_a'] - state['total_b'])}")
    print(f"{'═'*52}")

    # ── Level 1: Criteria wins ────────────────────────
    crit_a = state["criterion_totals_a"]
    crit_b = state["criterion_totals_b"]

    a_wins = sum(
        1 for c in crit_a
        if crit_a.get(c, 0) > crit_b.get(c, 0)
    )
    b_wins = sum(
        1 for c in crit_b
        if crit_b.get(c, 0) > crit_a.get(c, 0)
    )

    print(f"\n  Level 1 — Criteria wins:")
    print(f"    A won {a_wins} criteria")
    print(f"    B won {b_wins} criteria")

    if a_wins != b_wins:
        winner      = "A" if a_wins > b_wins else "B"
        explanation = (
            f"Level 1 — Debater {winner} won "
            f"{max(a_wins, b_wins)} of "
            f"{a_wins + b_wins} criteria"
        )
        print(f"  Winner: {winner} ✅")
        return _tiebreak_result(1, winner, explanation)

    # ── Level 2: Adjusted speaks ──────────────────────
    total_speeches = len(state["transcript"])

    if total_speeches >= 6:
        print(f"\n  Level 2 — Adjusted speaks:")
        adj_a = _adjusted_speaks(
            state["score_history"], "a"
        )
        adj_b = _adjusted_speaks(
            state["score_history"], "b"
        )
        print(f"    A adjusted: {adj_a}")
        print(f"    B adjusted: {adj_b}")

        if adj_a != adj_b:
            winner      = "A" if adj_a > adj_b else "B"
            explanation = (
                f"Level 2 — Adjusted speaks: "
                f"A={adj_a}, B={adj_b}"
            )
            print(f"  Winner: {winner} ✅")
            return _tiebreak_result(2, winner, explanation)

    # ── Level 3: Closing statement winner ────────────
    print(f"\n  Level 3 — Closing statement:")
    closing_a = _get_phase_score(
        state["score_history"], "closing", "a"
    )
    closing_b = _get_phase_score(
        state["score_history"], "closing", "b"
    )
    print(f"    A closing: {closing_a}")
    print(f"    B closing: {closing_b}")

    if closing_a != closing_b:
        winner      = "A" if closing_a > closing_b else "B"
        explanation = (
            f"Level 3 — Closing statement: "
            f"A={closing_a}, B={closing_b}"
        )
        print(f"  Winner: {winner} ✅")
        return _tiebreak_result(3, winner, explanation)

    # ── Level 4: Judge holistic call ──────────────────
    print(f"\n  Level 4 — Judge holistic call...")
    try:
        winner, reasoning = _judge_tiebreak_call(state)
        explanation = f"Level 4 — Judge holistic call: {reasoning}"
        print(f"  Winner: {winner} ✅")
        return _tiebreak_result(4, winner, explanation)
    except Exception as e:
        print(f"  Level 4 failed: {e}")

    # ── Level 5: Honourable draw ──────────────────────
    print(f"\n  Level 5 — Honourable draw")
    explanation = (
        "Level 5 — Both debaters argued equally well. "
        "This debate is declared an honourable draw."
    )
    return _tiebreak_result(5, None, explanation)


def _tiebreak_result(
    level:       int,
    winner:      str | None,
    explanation: str,
) -> dict:
    """Returns standardised tiebreak result dict."""
    return {
        "tiebreak_needed":      True,
        "tiebreak_level":       level,
        "tiebreak_winner":      winner,
        "tiebreak_explanation": explanation,
        "current_node":         "tiebreak_node",
    }


def _adjusted_speaks(
    score_history: list,
    debater_key:   str,
) -> int:
    """
    Calculates adjusted speaks for a debater.
    Drops their highest and lowest round score.
    Returns sum of remaining scores.
    """
    scores = [
        entry[f"score_{debater_key}"]
        for entry in score_history
    ]

    if len(scores) <= 2:
        return sum(scores)

    # Drop highest and lowest
    scores.sort()
    adjusted = scores[1:-1]
    return sum(adjusted)


def _get_phase_score(
    score_history: list,
    phase:         str,
    debater_key:   str,
) -> int:
    """Returns score for a specific phase."""
    for entry in score_history:
        if entry.get("phase") == phase:
            return entry.get(f"score_{debater_key}", 0)
    return 0


def _judge_tiebreak_call(
    state: DebateState,
) -> tuple[str, str]:
    """
    Level 4 tiebreak — asks judge to make
    a holistic qualitative decision.
    Returns (winner, reasoning).
    """
    anon_transcript = _build_anonymous_transcript(
        state["transcript"],
        state["label_map"],
    )

    response = _call_llm(
        role="judge",
        messages=[
            SystemMessage(content=f"""{ANTI_BIAS_PROMPT}
You must break a tie between two equally scored debaters.
Make a qualitative judgment on who made the more
compelling overall case — not based on scores."""),
            HumanMessage(content=f"""
The debate is tied on all metrics.
Read the transcript and decide the winner.

{anon_transcript}

Who made the more compelling overall case?

Return:
WINNER: [X or Y]
REASONING: [1-2 sentences]"""),
        ],
        node="tiebreak_node",
    )

    winner = _parse_winner(response)
    # Map X/Y to A/B
    real_winner = state["label_map"][winner]

    # Extract reasoning
    match = re.search(
        r"REASONING:\s*(.+)",
        response,
        re.DOTALL,
    )
    reasoning = (
        match.group(1).strip()
        if match
        else "Judge declared winner holistically"
    )

    return real_winner, reasoning


# ════════════════════════════════════════════════════════
# NODE 7 — ERROR
# Graceful failure handler.
# Activated after 2+ LLM failures in errors list.
# Saves whatever partial content exists.
# Never raises — always exits cleanly.
# ════════════════════════════════════════════════════════

def error_node(state: DebateState) -> dict:
    """
    Handles graceful failure after repeated errors.

    Saves whatever transcript content exists.
    Logs all errors clearly.
    Returns a partial result rather than crashing.

    Returns:
        winner=None, verdict=error summary,
        transcript_path (partial), current_node
    """
    errors = state.get("errors", [])

    print(f"\n{'═'*52}")
    print(f"  ERROR — Debate ended early")
    print(f"{'═'*52}")

    for i, error in enumerate(errors, 1):
        print(f"  Error {i}: {error}")

    # ── Save partial transcript ───────────────────────
    transcript_path = None
    try:
        if state.get("transcript"):
            transcript_path = _save_transcript(
                state,
                winner=None,
                verdict_response="[Debate ended due to errors]",
                partial=True,
            )
            print(f"\n  Partial transcript saved: "
                  f"{transcript_path}")
    except Exception as e:
        print(f"  Could not save partial transcript: {e}")

    verdict = (
        f"Debate ended early due to {len(errors)} error(s).\n"
        f"Completed phases: "
        f"{state.get('current_phase', 'unknown')}\n"
        f"Errors:\n" +
        "\n".join(f"  - {e}" for e in errors)
    )

    return {
        "winner":          None,
        "verdict":         verdict,
        "transcript_path": transcript_path,
        "current_node":    "error_node",
    }


# ════════════════════════════════════════════════════════
# PRIVATE HELPERS
# Internal functions used by multiple nodes.
# Not exported — prefix with _ convention.
# ════════════════════════════════════════════════════════

def _call_llm(
    role:     str,
    messages: list,
    node:     str,
    attempt:  int = 1,
) -> str:
    """
    Calls an LLM and returns response text.
    Does NOT stream — use _stream_llm for streaming.
    Wraps all errors in LLMCallError.

    Args:
        role:     "debater_a", "debater_b", or "judge"
        messages: list of SystemMessage + HumanMessage
        node:     calling node name for error context
        attempt:  which attempt this is

    Returns:
        Response content as string

    Raises:
        LLMCallError: with model and node context
    """
    model_name = get_llm_name(role)

    try:
        llm      = get_llm(role)
        response = llm.invoke(messages)
        return response.content

    except Exception as e:
        raise LLMCallError(
            message=str(e),
            model=model_name,
            node=node,
            attempt=attempt,
        )


def _stream_llm(
    role:     str,
    messages: list,
    node:     str,
) -> str:
    """
    Calls an LLM with streaming.
    Prints each chunk as it arrives.
    Returns complete response text.

    Args:
        role:     "debater_a", "debater_b", or "judge"
        messages: list of SystemMessage + HumanMessage
        node:     calling node name for error context

    Returns:
        Complete response content as string

    Raises:
        LLMCallError: with model and node context
    """
    model_name = get_llm_name(role)

    try:
        llm      = get_llm(role)
        content  = ""

        for chunk in llm.stream(messages):
            if chunk.content:
                print(chunk.content, end="", flush=True)
                content += chunk.content

        print()  # newline after streaming
        return content

    except Exception as e:
        raise LLMCallError(
            message=str(e),
            model=model_name,
            node=node,
        )


def _save_transcript(
    state:            DebateState,
    winner:           str | None,
    verdict_response: str,
    partial:          bool = False,
) -> str:
    """
    Saves complete debate transcript to markdown.
    Filename: debate_YYYY-MM-DD_HHMM_[hash].md

    Uses date + time + topic hash to ensure:
        Unique filenames even for same topic same day
        Short enough to avoid OS path limits
        Human readable date prefix

    Returns:
        File path of saved transcript

    Raises:
        TranscriptError: if save fails
    """
    try:
        # ── Generate filename ─────────────────────────
        now      = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H%M")

        # Short hash of topic (6 chars)
        topic_hash = hashlib.md5(
            state["topic"].encode()
        ).hexdigest()[:6]

        prefix   = "partial_" if partial else "debate_"
        filename = (
            f"{prefix}{date_str}_{time_str}"
            f"_{topic_hash}.md"
        )

        # ── Ensure output directory exists ────────────
        os.makedirs(settings.transcript_dir, exist_ok=True)
        filepath = os.path.join(
            settings.transcript_dir, filename
        )

        # ── Build markdown content ────────────────────
        content = _build_transcript_markdown(
            state, winner, verdict_response, partial
        )

        # ── Write file ────────────────────────────────
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"\n  Transcript saved: {filepath}")
        return filepath

    except Exception as e:
        raise TranscriptError(
            f"Failed to save transcript: {e}"
        )


def _build_transcript_markdown(
    state:            DebateState,
    winner:           str | None,
    verdict_response: str,
    partial:          bool = False,
) -> str:
    now         = datetime.now()
    topic       = state["topic"]
    model_a     = get_llm_name("debater_a")
    model_b     = get_llm_name("debater_b")
    model_judge = get_llm_name("judge")
    total_a     = state.get("total_a", 0)
    total_b     = state.get("total_b", 0)
    history     = state.get("score_history", [])

    # ── Winner line ───────────────────────────────────
    if winner:
        w_model = get_llm_name(f"debater_{winner.lower()}")
        w_pos   = (
            state.get("position_a")
            if winner == "A"
            else state.get("position_b")
        )
        winner_line = (
            f"Debater {winner} — "
            f"{w_model.split('/')[-1]} ({w_pos})"
        )
    else:
        winner_line = "No result"

    lines = []

    # ════════════════════════════════════════
    # HEADER
    # ════════════════════════════════════════
    lines += [
        f"# ⚔ NeuroColosseum",
        f"",
        f"## {topic}",
        f"",
        f"| | Model | Position |",
        f"|--|-------|----------|",
        f"| 🔵 A | `{model_a}` | FOR |",
        f"| 🔴 B | `{model_b}` | AGAINST |",
        f"| ⚖️ Judge | `{model_judge}` | Neutral |",
        f"",
        f"**Date:** {now.strftime('%Y-%m-%d %H:%M')}  ",
        f"**Awareness:** {state.get('awareness', 'criteria')}",
        f"",
        f"---",
        f"",
    ]

    # ════════════════════════════════════════
    # TL;DR — result at top so reader knows
    # outcome before reading full debate
    # ════════════════════════════════════════
    lines += [
        f"## 🏆 Result",
        f"",
        f"> **Winner: {winner_line}**  ",
        f"> Final score: **A {total_a} — B {total_b}**",
        f"",
    ]

    # Score table
    if history:
        lines += [
            f"| Phase | A | B | Winner |",
            f"|-------|---|---|--------|",
        ]
        for e in history:
            ph  = e["phase"].replace("_"," ").title()
            rw  = "🔵 A" if e["score_a"] > e["score_b"] else "🔴 B" if e["score_b"] > e["score_a"] else "—"
            lines.append(
                f"| {ph} | {e['score_a']} | {e['score_b']} | {rw} |"
            )
        lines += [
            f"| **Total** | **{total_a}** | **{total_b}** | **{'🔵 A' if total_a > total_b else '🔴 B'}** |",
            f"",
        ]

    lines += ["---", ""]

    # ════════════════════════════════════════
    # SPEECHES — grouped by phase
    # ════════════════════════════════════════
    phase_labels = {
        "opening":    "Phase 1 — Opening",
        "rebuttal_1": "Phase 2 — Rebuttal Round 1\n*B challenges A's opening*",
        "rebuttal_2": "Phase 3 — Rebuttal Round 2\n*A challenges B's opening*",
        "cross_exam": "Phase 4 — Cross Examination",
        "closing":    "Phase 5 — Closing",
    }

    current_phase = None

    for entry in state["transcript"]:
        phase   = entry["phase"]
        debater = entry["debater"]
        arg     = entry["argument"]
        wc      = entry.get("word_count", len(arg.split()))
        pos     = (
            state.get("position_a")
            if debater == "A"
            else state.get("position_b")
        )
        icon    = "🔵" if debater == "A" else "🔴"

        # Phase header
        if phase != current_phase:
            current_phase = phase
            label = phase_labels.get(
                phase,
                phase.replace("_"," ").title()
            )
            lines += [
                f"## {label}",
                f"",
            ]

        # Cross exam formatted differently
        if phase == "cross_exam":
            if "Q1:" in arg or "Q2:" in arg or arg.startswith("Q"):
                lines += [
                    f"**{icon} {debater} questions:**",
                    f"",
                ]
                for line in arg.split("\n"):
                    if line.strip():
                        lines.append(f"> {line.strip()}")
                lines.append("")
            else:
                lines += [
                    f"**{icon} {debater} answers:**",
                    f"",
                    arg,
                    f"",
                ]
        else:
            lines += [
                f"**{icon} Debater {debater} ({pos})** · {wc} words",
                f"",
                arg,
                f"",
                f"---",
                f"",
            ]

    # ════════════════════════════════════════
    # VERDICT
    # ════════════════════════════════════════
    lines += [
        f"## ⚖️ Judge's Verdict",
        f"",
        f"> **{winner_line}**",
        f"",
        verdict_response.strip(),
        f"",
        f"---",
        f"",
    ]

    # ════════════════════════════════════════
    # TIEBREAK
    # ════════════════════════════════════════
    if state.get("tiebreak_needed"):
        lines += [
            f"## Tiebreak",
            f"",
            f"**Level {state.get('tiebreak_level')}:** "
            f"{state.get('tiebreak_explanation','')}",
            f"",
            f"---",
            f"",
        ]

    # ════════════════════════════════════════
    # SOCIAL SHARE CARD
    # Ready to copy-paste for Twitter/LinkedIn
    # ════════════════════════════════════════
    score_line = " · ".join([
        f"{e['phase'].replace('rebuttal_','Reb').replace('cross_exam','XExam').replace('opening','Open').replace('closing','Close').title()} {e['score_a']}–{e['score_b']}"
        for e in history
    ])

    lines += [
        f"## 📱 Social Share",
        f"",
        f"```",
        f"⚔ NeuroColosseum",
        f"",
        f'"{topic}"',
        f"",
        f"🔵 {model_a.split('/')[-1]} (FOR)",
        f"🔴 {model_b.split('/')[-1]} (AGAINST)",
        f"",
        f"Scores: {score_line}",
        f"",
        f"🏆 {winner_line}",
        f"Final: A {total_a} — B {total_b}",
        f"",
        f"Judged blindly by {model_judge.split('/')[-1]}",
        f"Two models enter. One argument wins.",
        f"#NeuroColosseum #AI #LLM",
        f"```",
        f"",
        f"---",
        f"",
    ]

    # ════════════════════════════════════════
    # FOOTER
    # ════════════════════════════════════════
    lines += [
        f"*Generated by NeuroColosseum · "
        f"{now.strftime('%Y-%m-%d %H:%M:%S')}*  ",
        f"*Two models enter. One argument wins.*",
    ]

    return "\n".join(lines)
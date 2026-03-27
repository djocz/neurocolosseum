# exceptions.py
# ════════════════════════════════════════════
# NeuroColosseum — Custom Exceptions
# Two models enter. One argument wins.
# ════════════════════════════════════════════


class NeuroColosseumError(Exception):
    """
    Base exception for all NeuroColosseum errors.
    Catch this to handle any debate error.
    """
    pass


class TopicValidationError(NeuroColosseumError):
    """
    Topic failed the guardrail check.
    Tier 1: keyword blocklist
    Tier 2: LLM validation
    """
    pass


class LLMCallError(NeuroColosseumError):
    """
    An LLM API call failed.
    Model and node passed dynamically
    from config and calling function.
    Never hardcoded.
    """
    def __init__(self,
                 message: str,
                 model:   str = "unknown",
                 node:    str = "unknown",
                 attempt: int = 1):
        self.model   = model
        self.node    = node
        self.attempt = attempt
        super().__init__(
            f"{message}\n"
            f"  → model:   {model}\n"
            f"  → node:    {node}\n"
            f"  → attempt: {attempt}"
        )


class ScoringError(NeuroColosseumError):
    """
    Judge failed to score a round correctly.
    Malformed response or missing criteria.
    """
    pass


class PhaseError(NeuroColosseumError):
    """
    Phase transition encountered invalid state.
    Phase name unknown or counter out of sync.
    """
    pass


class CrossExamError(NeuroColosseumError):
    """
    Cross examination failed.
    Question generation or answer generation
    failed after max attempts.
    """
    pass


class TranscriptError(NeuroColosseumError):
    """
    Transcript save or load failed.
    File permission, missing folder, disk full.
    """
    pass


class TiebreakError(NeuroColosseumError):
    """
    All 5 tiebreak levels exhausted
    without resolving the tie.
    Extremely rare — should never happen
    if judge prompt is working correctly.
    """
    pass


class ConfigurationError(NeuroColosseumError):
    """
    Invalid or missing configuration.
    Missing API key, invalid model string,
    unsupported provider.
    """
    pass
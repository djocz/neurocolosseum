# config.py
# ════════════════════════════════════════════════════════
# NeuroColosseum — Central Configuration
# Two models enter. One argument wins.
#
# This is the control panel for the entire system.
# Every other file imports from here.
# Never hardcode settings anywhere else.
#
# Import hierarchy — config.py is Layer 1:
#   No imports from other app files
#   Only imports from standard library + third party
# ════════════════════════════════════════════════════════

import os
from typing import Optional
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from exceptions import ConfigurationError

load_dotenv()


# ════════════════════════════════════════════════════════
# PROVIDER REGISTRY
# Complete map of every supported LLM provider.
# Add new providers here — nowhere else.
#
# Each entry contains:
#   package  → pip package to install
#   env_var  → environment variable name for API key
#   setting  → field name in Settings class
#   example  → example model string for error messages
#   note     → optional human readable note
# ════════════════════════════════════════════════════════

PROVIDER_REGISTRY: dict[str, dict] = {

    # ── Major Cloud Providers ─────────────────────────

    "anthropic": {
        "package": "langchain-anthropic",
        "env_var": "ANTHROPIC_API_KEY",
        "setting": "anthropic_api_key",
        "example": "anthropic/claude-sonnet-4-6",
        "note":    "console.anthropic.com",
    },
    "openai": {
        "package": "langchain-openai",
        "env_var": "OPENAI_API_KEY",
        "setting": "openai_api_key",
        "example": "openai/gpt-4o",
        "note":    "platform.openai.com",
    },
    "groq": {
        "package": "langchain-groq",
        "env_var": "GROQ_API_KEY",
        "setting": "groq_api_key",
        "example": "groq/llama-3.3-70b-versatile",
        "note":    "console.groq.com — free tier available",
    },
    "google_genai": {
        "package": "langchain-google-genai",
        "env_var": "GOOGLE_API_KEY",
        "setting": "google_api_key",
        "example": "google_genai/gemini-2.5-flash",
        "note":    "aistudio.google.com",
    },
    "mistralai": {
        "package": "langchain-mistralai",
        "env_var": "MISTRAL_API_KEY",
        "setting": "mistral_api_key",
        "example": "mistralai/mistral-large-latest",
        "note":    "console.mistral.ai",
    },
    "cohere": {
        "package": "langchain-cohere",
        "env_var": "COHERE_API_KEY",
        "setting": "cohere_api_key",
        "example": "cohere/command-r-plus",
        "note":    "dashboard.cohere.com",
    },
    "fireworks": {
        "package": "langchain-fireworks",
        "env_var": "FIREWORKS_API_KEY",
        "setting": "fireworks_api_key",
        "example": "fireworks/accounts/fireworks/models/llama-v3p1-70b-instruct",
        "note":    "fireworks.ai",
    },
    "together": {
        "package": "langchain-together",
        "env_var": "TOGETHER_API_KEY",
        "setting": "together_api_key",
        "example": "together/meta-llama/Llama-3-70b-chat-hf",
        "note":    "api.together.xyz",
    },
    "huggingface": {
        "package": "langchain-huggingface",
        "env_var": "HUGGINGFACEHUB_API_TOKEN",
        "setting": "huggingface_api_key",
        "example": "huggingface/HuggingFaceH4/zephyr-7b-beta",
        "note":    "huggingface.co/settings/tokens",
    },
    "xai": {
        "package": "langchain-xai",
        "env_var": "XAI_API_KEY",
        "setting": "xai_api_key",
        "example": "xai/grok-beta",
        "note":    "x.ai — Grok models",
    },
    "perplexity": {
        "package": "langchain-perplexity",
        "env_var": "PPLX_API_KEY",
        "setting": "perplexity_api_key",
        "example": "perplexity/llama-3.1-sonar-large-128k-online",
        "note":    "perplexity.ai",
    },

    # ── Cloud Platform Providers ──────────────────────

    "azure_openai": {
        "package": "langchain-openai",
        "env_var": "AZURE_OPENAI_API_KEY",
        "setting": "azure_openai_api_key",
        "example": "azure_openai/gpt-4o",
        "note":    "Requires AZURE_OPENAI_ENDPOINT too",
    },
    "google_vertexai": {
        "package": "langchain-google-vertexai",
        "env_var": None,
        "setting": None,
        "example": "google_vertexai/gemini-2.5-flash",
        "note":    "Uses GOOGLE_APPLICATION_CREDENTIALS",
    },
    "bedrock_converse": {
        "package": "langchain-aws",
        "env_var": None,
        "setting": None,
        "example": "bedrock_converse/anthropic.claude-3-5-sonnet-20240620-v1:0",
        "note":    "Uses AWS CLI credentials — run: aws configure",
    },

    # ── Local / Free Providers ────────────────────────

    "ollama": {
        "package": "langchain-ollama",
        "env_var": None,
        "setting": None,
        "example": "ollama/llama3",
        "note":    "Local — no API key. Install: ollama.com",
    },
}

# Providers that require no API key
NO_KEY_PROVIDERS = {
    "ollama",
    "bedrock_converse",
    "google_vertexai",
}


# ════════════════════════════════════════════════════════
# SETTINGS
# All configuration loaded from .env at startup.
# Pydantic validates types and formats immediately.
# Missing required values raise ConfigurationError.
#
# All API keys are Optional — validated lazily
# only when that provider is first used.
# This means you only need keys for providers
# you actually configure in LLM_DEBATER_A/B/JUDGE.
# ════════════════════════════════════════════════════════

class Settings(BaseSettings):
    """
    Central settings for NeuroColosseum.

    Reads from .env automatically via pydantic-settings.
    All API keys are optional — only validated when used.
    Model strings validated for correct format at startup.

    Usage:
        from config import settings
        print(settings.debate_topic)
        print(settings.max_argument_words)
    """

    # ── Model Configuration ───────────────────────────
    # Which model each role uses
    # Format: provider/model-name
    # Change in .env — never touch this file

    llm_debater_a: str = Field(
        default="anthropic/claude-sonnet-4-6",
        description="Model for Debater A — always argues FOR"
    )
    llm_debater_b: str = Field(
        default="groq/llama-3.3-70b-versatile",
        description="Model for Debater B — always argues AGAINST"
    )
    llm_judge: str = Field(
        default="openai/gpt-4o",
        description="Model for judge — use different company for fairness"
    )

    # ── Major Cloud API Keys ──────────────────────────
    # All optional — only needed if using that provider

    anthropic_api_key:   Optional[str] = Field(default=None)
    openai_api_key:      Optional[str] = Field(default=None)
    groq_api_key:        Optional[str] = Field(default=None)
    google_api_key:      Optional[str] = Field(default=None)
    mistral_api_key:     Optional[str] = Field(default=None)
    cohere_api_key:      Optional[str] = Field(default=None)
    fireworks_api_key:   Optional[str] = Field(default=None)
    together_api_key:    Optional[str] = Field(default=None)
    huggingface_api_key: Optional[str] = Field(default=None)
    xai_api_key:         Optional[str] = Field(default=None)
    perplexity_api_key:  Optional[str] = Field(default=None)

    # ── Cloud Platform API Keys ───────────────────────
    # Azure OpenAI needs both key and endpoint

    azure_openai_api_key:  Optional[str] = Field(default=None)
    azure_openai_endpoint: Optional[str] = Field(default=None)
    openai_api_version:    Optional[str] = Field(
        default="2025-03-01-preview"
    )

    # Google Vertex AI — uses service account file
    # Set GOOGLE_APPLICATION_CREDENTIALS in environment
    # No field needed here

    # AWS Bedrock — uses AWS CLI credentials
    # Run: aws configure
    # No field needed here

    # ── Debate Settings ───────────────────────────────

    debate_topic: str = Field(
        default="AI will do more harm than good",
        description="Default topic for terminal interface"
    )
    debater_awareness: str = Field(
        default="criteria",
        description="What debaters know: none/criteria/scores/trailing"
    )
    cross_exam_questions: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Questions per cross exam session (1-5)"
    )

    # ── Scoring Settings ──────────────────────────────

    max_argument_words: int = Field(
        default=400,
        ge=100,
        le=1000,
        description="Maximum words per argument (100-1000)"
    )
    max_question_words: int = Field(
        default=80,
        ge=20,
        le=200,
        description="Maximum words per cross exam question"
    )
    tiebreak_threshold: int = Field(
        default=0,
        ge=0,
        le=20,
        description="Score gap to trigger tiebreak (0=exact tie only)"
    )
    length_penalty: bool = Field(
        default=True,
        description="Cap score at 85 for arguments over word limit"
    )

    # ── Output Settings ───────────────────────────────

    save_transcript: bool = Field(
        default=True,
        description="Save markdown transcript after each debate"
    )
    transcript_dir: str = Field(
        default="outputs",
        description="Directory for saved transcripts"
    )

    # ── Monitoring ────────────────────────────────────

    langsmith_api_key:  Optional[str] = Field(default=None)
    langsmith_project:  str           = Field(default="neurocolosseum")
    langsmith_tracing:  bool          = Field(default=False)

    # ── Validators ────────────────────────────────────

    @field_validator("debater_awareness")
    @classmethod
    def validate_awareness(cls, v: str) -> str:
        """
        Ensures awareness is one of four valid options.
        Validated at startup — fails immediately if wrong.
        """
        valid = ["none", "criteria", "scores", "trailing"]
        if v not in valid:
            raise ValueError(
                f"DEBATER_AWARENESS must be one of {valid}\n"
                f"Got: '{v}'"
            )
        return v

    @field_validator("llm_debater_a", "llm_debater_b", "llm_judge")
    @classmethod
    def validate_model_format(cls, v: str) -> str:
        """
        Ensures model strings follow provider/model-name format.
        Example: anthropic/claude-sonnet-4-6
        Validated at startup — fails immediately if wrong.
        """
        if "/" not in v:
            raise ValueError(
                f"Model string must be 'provider/model-name'\n"
                f"Got: '{v}'\n"
                f"Example: anthropic/claude-sonnet-4-6"
            )
        return v

    class Config:
        env_file          = ".env"
        env_file_encoding = "utf-8"
        case_sensitive    = False
        extra             = "ignore"


# ── Instantiate settings ──────────────────────────────
# Runs once at import time.
# All validators run here.
# Fails fast with clear error if .env is wrong.

try:
    settings = Settings()
except Exception as e:
    raise ConfigurationError(
        f"Failed to load configuration:\n\n"
        f"{e}\n\n"
        f"Fix your .env file. "
        f"Use .env.example as a template:\n"
        f"  cp .env.example .env"
    )


# ── Enable LangSmith tracing if configured ────────────

if settings.langsmith_tracing and settings.langsmith_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"]    = settings.langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"]    = settings.langsmith_project
    print(f"   LangSmith tracing → {settings.langsmith_project}")


# ════════════════════════════════════════════════════════
# LLM GETTERS
# Model-agnostic interface to all providers.
# Change provider by updating .env only.
# Never instantiate LLMs directly in nodes.
# ════════════════════════════════════════════════════════

# Maps role names to settings fields
_ROLE_TO_SETTING: dict[str, str] = {
    "debater_a": "llm_debater_a",
    "debater_b": "llm_debater_b",
    "judge":     "llm_judge",
}


def get_llm_name(role: str) -> str:
    """
    Returns the model string for a given role.
    Always reads from settings — never hardcoded.
    Used for error messages, logging, and LangSmith.

    Args:
        role: "debater_a", "debater_b", or "judge"

    Returns:
        Model string e.g. "anthropic/claude-sonnet-4-6"

    Raises:
        ConfigurationError: if role is unknown

    Example:
        get_llm_name("judge")
        → "openai/gpt-4o"
    """
    field = _ROLE_TO_SETTING.get(role)
    if not field:
        raise ConfigurationError(
            f"Unknown role: '{role}'\n"
            f"Valid roles: {list(_ROLE_TO_SETTING.keys())}"
        )
    return getattr(settings, field)


def get_llm(role: str):
    """
    Returns a configured LangChain LLM for a role.
    Model agnostic — works with any supported provider.
    Validates API key exists before attempting connection.

    Args:
        role: "debater_a", "debater_b", or "judge"

    Returns:
        LangChain BaseChatModel instance

    Raises:
        ConfigurationError: if API key missing or model invalid
        LLMCallError: if model initialisation fails

    Example:
        llm = get_llm("judge")
        response = llm.invoke(messages)
    """
    from exceptions import LLMCallError

    model_string = get_llm_name(role)
    provider, model = model_string.split("/", 1)

    # Validate and set API key for this provider
    _set_provider_key(provider, model_string)

    try:
        return init_chat_model(
            model=model,
            model_provider=provider
        )
    except Exception as e:
        raise LLMCallError(
            message=str(e),
            model=model_string,
            node=f"get_llm({role})"
        )


def _set_provider_key(provider: str,
                       model_string: str) -> None:
    """
    Sets the correct API key environment variable
    for a provider before LLM initialisation.

    Validates key exists when required.
    Providers in NO_KEY_PROVIDERS skip validation.
    Unknown providers are passed through to LangChain.

    Args:
        provider:     Provider name e.g. "anthropic"
        model_string: Full model string for error context

    Raises:
        ConfigurationError: if required API key is missing
    """
    # No key needed for local/platform providers
    if provider in NO_KEY_PROVIDERS:
        return

    # Look up provider in registry
    provider_info = PROVIDER_REGISTRY.get(provider)

    if not provider_info:
        # Unknown provider — pass through to LangChain
        # Might be a new provider added after this version
        return

    env_var = provider_info.get("env_var")
    setting = provider_info.get("setting")

    # Provider registered but needs no key
    if not env_var or not setting:
        return

    # Get key value from settings
    key_value = getattr(settings, setting, None)

    if not key_value:
        # Build helpful error message
        package = provider_info.get(
            "package", f"langchain-{provider}"
        )
        note    = provider_info.get("note", "")
        example = provider_info.get(
            "example", model_string
        )

        raise ConfigurationError(
            f"\nMissing API key for provider '{provider}'.\n"
            f"{'─' * 45}\n"
            f"Add to your .env file:\n"
            f"  {env_var}=your_key_here\n\n"
            f"Install the provider package:\n"
            f"  pip install {package}\n\n"
            f"Get your key at:\n"
            f"  {note}\n\n"
            f"Example model string:\n"
            f"  LLM_DEBATER_A={example}\n"
            f"{'─' * 45}"
        )

    # Set in environment for LangChain to read
    os.environ[env_var] = key_value

    # Azure OpenAI needs endpoint too
    if provider == "azure_openai":
        if not settings.azure_openai_endpoint:
            raise ConfigurationError(
                "Azure OpenAI requires AZURE_OPENAI_ENDPOINT in .env"
            )
        os.environ["AZURE_OPENAI_ENDPOINT"]  = \
            settings.azure_openai_endpoint
        os.environ["OPENAI_API_VERSION"]     = \
            settings.openai_api_version or \
            "2025-03-01-preview"


# ════════════════════════════════════════════════════════
# TOPIC GUARDRAIL
# Two-tier topic validation before debate starts.
# Tier 1: fast keyword check (free, instant)
# Tier 2: LLM check (called only on Start click)
# ════════════════════════════════════════════════════════

BLOCKED_KEYWORDS: list[str] = [
    # Violence
    "murder", "kill", "assassinate",
    "massacre", "genocide", "torture",
    # Weapons
    "bomb", "weapon", "explosive",
    "missile", "nuclear weapon",
    # Harmful instructions
    "drug synthesis", "hack into",
    "exploit", "malware", "virus",
    # Hate speech
    "racist", "sexist", "homophobic",
    "white supremacy", "ethnic cleansing",
    # Self harm
    "suicide", "self harm", "self-harm",
]


def validate_topic_tier1(topic: str) -> tuple[bool, str]:
    """
    Fast keyword check — no API call needed.
    Runs on every keystroke if desired.

    Args:
        topic: The debate topic string

    Returns:
        Tuple of (is_valid, reason)
        is_valid: True if topic passes
        reason:   Empty string if valid
                  Explanation if blocked

    Example:
        valid, reason = validate_topic_tier1(topic)
        if not valid:
            st.error(reason)
    """
    if not topic or not topic.strip():
        return False, "Topic cannot be empty"

    if len(topic.strip()) < 10:
        return False, "Topic is too short to debate meaningfully"

    if len(topic.strip()) > 200:
        return False, "Topic is too long — keep it under 200 characters"

    topic_lower = topic.lower()
    for keyword in BLOCKED_KEYWORDS:
        if keyword in topic_lower:
            return (
                False,
                f"Topic contains inappropriate content. "
                f"Please choose a respectful debate topic."
            )

    return True, ""


# ════════════════════════════════════════════════════════
# AWARENESS PROMPTS
# What debaters know about their scores.
# Injected into debater_node system prompt.
# Controlled by DEBATER_AWARENESS in .env.
# ════════════════════════════════════════════════════════

def get_awareness_prompt(role: str,
                          state: dict) -> str:
    """
    Returns awareness context for a debater.
    Content depends on DEBATER_AWARENESS setting.

    Args:
        role:  "A" or "B"
        state: current DebateState dictionary

    Returns:
        String injected into debater system prompt.
        Empty string if awareness = "none".

    Four modes:
        none     → pure debate, no score info
        criteria → knows scoring rubric only
        scores   → sees full scores + weakest area
        trailing → told scores only if losing by 10+
    """
    awareness = settings.debater_awareness

    # ── none ─────────────────────────────────────────
    if awareness == "none":
        return ""

    # ── criteria ──────────────────────────────────────
    if awareness == "criteria":
        return """
You are evaluated on five criteria.
Structure every argument to excel on all of them:
  1. Logical coherence   — are your arguments valid and well-structured?
  2. Use of evidence     — do you support claims with facts or examples?
  3. Rebuttal quality    — do you directly address your opponent's points?
  4. Persuasiveness      — would a neutral audience be convinced?
  5. Originality         — do you bring fresh angles not yet made?
"""

    # ── scores ────────────────────────────────────────
    if awareness == "scores":
        my_key    = f"total_{role.lower()}"
        opp_role  = "B" if role == "A" else "A"
        opp_key   = f"total_{opp_role.lower()}"
        crit_key  = f"criterion_totals_{role.lower()}"

        my_total  = state.get(my_key, 0)
        opp_total = state.get(opp_key, 0)
        gap       = my_total - opp_total

        # Find weakest criterion
        my_criteria = state.get(crit_key, {})
        weakest = (
            min(my_criteria, key=my_criteria.get)
            if my_criteria
            else "not yet scored"
        )

        gap_str = (
            f"+{gap}" if gap > 0
            else str(gap)
        )

        return f"""
Current debate scores:
  You:      {my_total} points
  Opponent: {opp_total} points
  Gap:      {gap_str} points {'(you are ahead)' if gap > 0 else '(you are behind)' if gap < 0 else '(tied)'}

Your weakest criterion so far: {weakest}
Focus on strengthening it in this argument.
"""

    # ── trailing ──────────────────────────────────────
    if awareness == "trailing":
        my_key   = f"total_{role.lower()}"
        opp_role = "B" if role == "A" else "A"
        opp_key  = f"total_{opp_role.lower()}"

        my_total  = state.get(my_key, 0)
        opp_total = state.get(opp_key, 0)
        gap       = my_total - opp_total

        if gap < -10:
            return f"""
You are currently losing by {abs(gap)} points.
You must argue more forcefully.
Directly challenge your opponent's weakest claims.
Bring stronger evidence and clearer reasoning.
"""
        return ""  # Not trailing enough — show nothing

    return ""  # Fallback for any unhandled value


# ════════════════════════════════════════════════════════
# SCORING CONSTANTS
# All scoring-related prompts and data structures.
# Injected into judge prompts — never hardcoded in nodes.
# ════════════════════════════════════════════════════════

ANTI_BIAS_PROMPT: str = """
You are a neutral debate judge.
You must actively resist these biases:

1. Verbosity bias
   Do NOT give higher scores for longer arguments.
   A concise 150-word argument scores the same as
   a 400-word argument if the reasoning quality is equal.

2. Position bias
   Do NOT favour arguments that match your training
   data consensus on this topic.
   Score logical merit only — not whether you agree.

3. Recency bias
   Do NOT give higher scores to the last argument read.
   Each argument is evaluated independently.

4. Style bias
   Do NOT favour formal academic language over clear
   conversational language.
   Score the argument — not the vocabulary.

5. Familiarity bias
   You may be evaluating arguments from models similar
   to yourself. Actively resist any sense of familiarity.
   Judge purely on logical merit.

Imagine you are a completely neutral human judge
with no prior opinion on this topic whatsoever.
"""


SCORING_RUBRIC: str = """
Score each criterion using this exact rubric:
  90-100 → Exceptional  — rare, truly outstanding argument
  75-89  → Strong       — well argued with clear evidence
  60-74  → Solid        — competent with minor gaps
  45-59  → Adequate     — makes points but lacks depth
  0-44   → Weak         — vague, off-topic, or unsupported

Critical rules:
  → Do NOT reward length — quality only
  → Arguments over word limit are capped at 85 maximum
  → Score each criterion SEPARATELY before totalling
  → When uncertain between two scores choose the LOWER
  → Reserve 90+ for truly exceptional arguments only
"""


PHASE_CRITERIA: dict[str, dict[str, int]] = {
    "opening": {
        "coherence":   25,
        "evidence":    25,
        "persuasion":  25,
        "originality": 25,
    },
    "rebuttal_1": {
        "rebuttal_quality": 25,
        "evidence":         25,
        "coherence":        25,
        "persuasion":       25,
    },
    "rebuttal_2": {
        "rebuttal_quality": 25,
        "evidence":         25,
        "coherence":        25,
        "persuasion":       25,
    },

    # ── Cross exam — two separate role criteria ───────
    # Used by judge_score_node cross exam special case
    "cross_exam_questioner": {
        "question_sharpness":   50,
        "argument_advancement": 50,
    },
    "cross_exam_answerer": {
        "answer_quality": 50,
        "composure":      50,
    },

    "closing": {
        "coherence":   25,
        "persuasion":  25,
        "originality": 25,
        "impact":      25,
    },
}


# ════════════════════════════════════════════════════════
# PRESET TOPICS
# Categorised list for the Streamlit topic picker.
# Add more topics freely — no code changes needed.
# ════════════════════════════════════════════════════════

PRESET_TOPICS: dict[str, list[str]] = {
    "Technology": [
        "AI will do more harm than good",
        "Social media has made society worse",
        "Remote work is better than office work",
        "Cryptocurrency is the future of money",
        "Open source AI is more dangerous than closed AI",
        "Self-driving cars should be legal on all roads",
        "Algorithmic content recommendation does more harm than good",
    ],
    "Society": [
        "Universal basic income should be implemented globally",
        "Cancel culture does more harm than good",
        "College education is no longer worth the cost",
        "Smartphones have made us less intelligent",
        "Social media influencers have too much power",
        "Meritocracy is a myth",
    ],
    "Environment": [
        "Nuclear energy is the solution to climate change",
        "Individual action cannot meaningfully solve climate change",
        "Electric vehicles will save the planet",
        "Veganism is the only ethical diet",
        "Carbon taxes are the most effective climate policy",
    ],
    "Economics": [
        "Capitalism does more harm than good",
        "A four-day work week should be the global standard",
        "Billionaires should not exist",
        "Globalisation has hurt more people than it helped",
        "Automation will create more jobs than it destroys",
    ],
    "Ethics": [
        "Genetic engineering of humans is ethically justified",
        "Mass surveillance makes society safer",
        "Violent video games cause real-world violence",
        "Animals should have the same rights as humans",
        "The ends justify the means",
    ],
    "Politics": [
        "Democracy is the best form of government",
        "Voting should be mandatory",
        "Term limits should apply to all elected officials",
        "Freedom of speech should have limits",
        "The United Nations has failed its mission",
    ],
}
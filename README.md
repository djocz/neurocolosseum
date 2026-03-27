# ⚔ NeuroColosseum

> Two models enter. One argument wins.

NeuroColosseum is an AI debate arena where two LLMs argue opposing sides of a topic, cross-examine each other, and are judged blindly by a third model.

Built with LangGraph and LangChain.

---

## How It Works

Six phases run sequentially:

1. **Opening** — each model presents their case independently
2. **Rebuttal 1** — B challenges A's opening, A replies
3. **Rebuttal 2** — A challenges B's opening, B replies
4. **Cross Examination** — models interrogate each other's weakest arguments
5. **Closing** — final statements
6. **Verdict** — judge scores all phases at once, declares winner

The judge never knows which model argued which side. Labels X/Y are revealed as A/B only after the verdict.

---

## Project Structure

```
neurocolosseum/
│
├── config.py          # settings, LLM getters, prompts
├── state.py           # DebateState TypedDict
├── nodes.py           # 8 node functions
├── edges.py           # routing logic
├── graph.py           # LangGraph assembly
├── exceptions.py      # typed custom exceptions
├── run.py             # terminal interface
│
├── .env               # API keys (never commit)
├── .env.example       # template
├── requirements.txt
│
└── outputs/           # saved transcripts
    └── archive/
```

---

## Setup

```bash
git clone https://github.com/yourname/neurocolosseum
cd neurocolosseum

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# Fill in your API keys
```

---

## Configuration

Edit `.env`:

```env
# Models
LLM_DEBATER_A=anthropic/claude-sonnet-4-6
LLM_DEBATER_B=groq/llama-3.3-70b-versatile
LLM_JUDGE=openai/gpt-4o-mini

# API keys — only fill what you use
ANTHROPIC_API_KEY=
GROQ_API_KEY=
OPENAI_API_KEY=

# Debate settings
DEBATE_TOPIC=AI will do more harm than good
DEBATER_AWARENESS=criteria
CROSS_EXAM_QUESTIONS=3
MAX_ARGUMENT_WORDS=400
MAX_QUESTION_WORDS=80
TIEBREAK_THRESHOLD=0
LENGTH_PENALTY=true
SAVE_TRANSCRIPT=true
TRANSCRIPT_DIR=outputs
```

### Supported Providers

| Provider | Package | Key |
|----------|---------|-----|
| Anthropic | `langchain-anthropic` | `ANTHROPIC_API_KEY` |
| OpenAI | `langchain-openai` | `OPENAI_API_KEY` |
| Groq | `langchain-groq` | `GROQ_API_KEY` (free tier) |
| Google | `langchain-google-genai` | `GOOGLE_API_KEY` |
| Mistral | `langchain-mistralai` | `MISTRAL_API_KEY` |
| Ollama | `langchain-ollama` | none — runs locally |

**Tip:** Use a different company for the judge to reduce familiarity bias.

### Debater Awareness

| Value | Effect |
|-------|--------|
| `none` | Pure debate — no score info |
| `criteria` | Knows scoring rubric only |
| `scores` | Sees full running scores |
| `trailing` | Told only if losing by 10+ points |

---

## Running

```bash
python3 run.py

# With flags:
python3 run.py --topic "Nuclear energy is the only climate solution"
python3 run.py --awareness scores
python3 run.py --twice          # swap models, detect bias
python3 run.py --list-topics    # show all preset topics
```

---

## Scoring

All phases scored in a single judge pass at the end.

| Phase | Criteria | Max |
|-------|----------|-----|
| Opening | Coherence, Evidence, Persuasion, Originality | 100 |
| Rebuttal 1 & 2 | Rebuttal Quality, Evidence, Coherence, Persuasion | 100 each |
| Cross Exam | Sharpness + Advancement (questioner) · Quality + Composure (answerer) — averaged | 100 |
| Closing | Coherence, Persuasion, Originality, Impact | 100 |
| **Total** | | **500** |

Arguments over the word limit are capped at 85/100.

---

## Bias Detection

```bash
python3 run.py --twice
```

Runs the same debate twice with models swapped. Detects:

- **Position bias** — same position wins both runs
- **Model bias** — same model wins both runs
- **Balanced** — different winner each run

---

## Transcripts

Saved to `outputs/` after every debate:

```
outputs/debate_2026-03-23_1432_a1b2c3.md
```

Each file includes the full debate, per-phase score table, judge verdict, and a copy-paste ready social media card. Debates are auto-archived after 30 days.

---

## Cost

| Setup | Cost per debate |
|-------|----------------|
| Claude + Groq + GPT-4o-mini | ~$0.005 |
| Groq only (free tier) | $0.00 |
| Ollama only (local) | $0.00 |

Groq free tier limit: 100k tokens/day.

---

## LangGraph Architecture

```
setup_node
  ↓
debater_node ◄─────────────────────────────┐
  ↓ speeches < 2 → loop                    │
  ↓ speeches = 2 → phase_transition_node   │
                        ↓                  │
              route_after_phase_transition │
                        ├── debater ────────┘
                        ├── cross_exam → cross_exam_node
                        └── judge     → judge_score_node
                                             ↓
                                        verdict_node
                                             ↓
                                            END
```

State travels as a single TypedDict through every node. Each node reads the full state and returns only changed fields.

---

## Monitoring

Optional LangSmith tracing:

```env
LANGSMITH_API_KEY=your_key
LANGSMITH_PROJECT=neurocolosseum
LANGSMITH_TRACING=true
```

Traces every LLM call with token usage, latency, and cost at smith.langchain.com.

---

## Known Limitations

- A=FOR always, B=AGAINST always — use `--twice` to control for model strength
- LLM judging LLM has inherent conflict of interest — mitigated via blind scoring but not eliminated
- Models under 30B parameters tend to score equally — use 70B+ for the judge

---

## Built With

- [LangGraph](https://github.com/langchain-ai/langgraph) — graph orchestration
- [LangChain](https://github.com/langchain-ai/langchain) — LLM abstraction
- [pyfiglet](https://github.com/pwaller/pyfiglet) — terminal banner

---

*Two models enter. One argument wins.*
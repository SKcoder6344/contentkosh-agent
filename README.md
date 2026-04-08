# ContentKosh AI Content Agent
### LangGraph + OpenAI | Stateful Multi-Step Educational Content Generation

> Built as a proof-of-work demo for the **AI Agent Developer** role at ContentKosh.

![CI](https://github.com/SKcoder6344/contentkosh-agent/actions/workflows/ci.yml/badge.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-stateful-orange)

---

## What It Does

A stateful, multi-step AI agent that automates educational content creation for coaching institutes. Given a topic and exam target, it executes a 4-node LangGraph pipeline:

```
[Topic Input]
     ↓
[Node 1] generate_notes    → Exam-ready study notes (200–250 words)
     ↓
[Node 2] generate_mcqs     → MCQ test set with answers + explanations
     ↓
[Node 3] review_quality    → Factual accuracy + MCQ quality check (JSON)
     ↓
[Node 4] assemble_package  → White-label-ready final content package
     ↓
[Output Package]
```

**Supported exams:** UPSC · SSC CGL · CUET · Class 10 Boards · State PCS · NEET · JEE

---

## Why This Architecture

| Feature | Simple LLM Call | This Agent |
|---|---|---|
| State shared across steps | ❌ | ✅ MCQs use notes as context |
| Quality gate before output | ❌ | ✅ Node 3 flags errors before delivery |
| Modular — swap any node | ❌ | ✅ Add translation, difficulty, plagiarism nodes |
| Scales to parallel branches | ❌ | ✅ LangGraph supports async + parallel nodes |
| Cost control | ❌ | ✅ Per-node token caps + session rate limiting |

---

## Production Features

- **Structured logging** (`logging` module) at every pipeline node
- **Input validation** with descriptive `ValueError` / `EnvironmentError` messages
- **Per-node `max_tokens` caps** — prevents runaway OpenAI spend
- **Session-level rate limiting** in the Streamlit UI (10 req/session default)
- **Graceful JSON fallback** — Node 4 handles malformed quality review responses
- **Non-root Docker user** — security-hardened container
- **GitHub Actions CI** — runs tests + basic secret scan on every push
- **MIT licensed**

---

## Quick Start

### Option 1: Local (Python)

```bash
git clone https://github.com/SKcoder6344/contentkosh-agent
cd contentkosh-agent

pip install -r requirements.txt

export OPENAI_API_KEY="your-key-here"

# Streamlit UI
make run

# CLI demo
make demo
```

### Option 2: Docker

```bash
docker build -t contentkosh-agent .
docker run -p 8501:8501 -e OPENAI_API_KEY="your-key-here" contentkosh-agent
# → Open http://localhost:8501
```

---

## Folder Structure

```
contentkosh-agent/
├── agent.py                  # Core pipeline (4 LangGraph nodes)
├── app.py                    # Streamlit UI
├── requirements.txt          # Pinned dependencies
├── Makefile                  # Dev shortcuts (run, test, docker-build…)
├── Dockerfile                # Multi-stage production container
├── LICENSE                   # MIT
├── metrics.csv               # Manual vs. agent performance comparison
├── tests/
│   └── test_agent.py         # pytest unit tests (mocked LLM)
└── .github/
    └── workflows/
        └── ci.yml            # GitHub Actions: test + secret scan
```

---

## API Usage

```python
from agent import run_agent

output = run_agent(
    topic="Photosynthesis",
    subject="Biology",
    exam_target="NEET",
    num_mcqs=5
)
print(output)
```

---

## Performance

| Metric | Manual Process | This Agent |
|---|---|---|
| Content package creation | 3–4 hours | ~45 seconds |
| MCQ generation per topic | 15–20 min | ~5 seconds |
| Factual accuracy check | Manual review | Automated (Node 3) |
| Scale (topics/day) | 10–15 | 100+ |

---

## Roadmap Alignment

| Current Node | Future Extension | Business Value |
|---|---|---|
| `generate_notes` | `generate_flashcards` | Spaced repetition feature |
| `review_quality` | `plagiarism_check` | Originality compliance |
| `assemble_package` | `translate_to_hindi` | Regional language expansion |
| New: `difficulty_adjuster` | Scale JEE vs Class 10 | Single agent, multiple exams |

---

## Running Tests

```bash
make test
# → pytest tests/ -v --cov=agent
```

---

## Author

**Sujal Kumar Nayak**  
GitHub: [github.com/SKcoder6344](https://github.com/SKcoder6344)  
Email: nayaksujalkumar@gmail.com

# ContentKosh AI Content Agent
### LangGraph + OpenAI | Multi-Step Educational Content Generation

Built as a proof-of-work demo for the **AI Agent Developer** role at ContentKosh.

---

## What This Does

A stateful, multi-step AI agent that automates educational content creation for coaching institutes. Given a topic and exam target, it runs a 4-node LangGraph pipeline:

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

---

## Why This Matters

ContentKosh needs agents that don't just generate—they **validate** before delivery. This architecture solves three production pain points:

**Stateful Quality Gates**  
Node 3 acts as a safety checkpoint. Unlike simple chains that output unchecked content, this agent reviews its own work (factual accuracy + relevance) before assembly. Critical for ed-tech where wrong MCQ answers damage credibility.

**White-Label Ready**  
Node 4 outputs structured packages with consistent formatting (headers, metadata, quality scores). Drops directly into ContentKosh's LMS without post-processing—unlike raw LLM text that needs manual cleanup.

**Modular Scaling**  
LangGraph nodes are swappable. Add a "Difficulty Adapter" node for JEE vs Class 10, or a "Regional Language" node for Hindi/Tamil—without rebuilding the pipeline. This mirrors ContentKosh's multi-exam, multi-language roadmap.

## Stack

| Component | Tech |
|---|---|
| Agent framework | LangGraph (stateful multi-step) |
| LLM | OpenAI GPT-4o-mini |
| Orchestration | LangChain `ChatOpenAI` |
| State management | `TypedDict` state across all nodes |
| Output | Structured, white-label-ready text package |

---


## Setup

```bash
git clone https://github.com/SKcoder6344/contentkosh-agent
cd contentkosh-agent

pip install -r requirements.txt

# Set your OpenAI API key
export OPENAI_API_KEY="your-key-here"

python agent.py
```

---

## Usage

```python
from agent import run_agent

output = run_agent(
    topic="Fundamental Rights in India",
    subject="Indian Polity",
    exam_target="UPSC",
    num_mcqs=5
)
print(output)
```

**Supported exam targets:** UPSC, SSC CGL, CUET, Class 10 Boards, State PCS, NEET, JEE

---

## Sample Output

```
╔══════════════════════════════════════════╗
   CONTENTKOSH — AI CONTENT PACKAGE
   Topic   : Fundamental Rights in India
   Subject : Indian Polity
   Exam    : UPSC
   Quality : Good
╚══════════════════════════════════════════╝

━━━━━━━━━━━━ STUDY NOTES ━━━━━━━━━━━━

Fundamental Rights are guaranteed under Part III (Articles 12–35)...
• Right to Equality (Art. 14–18)...
• Right to Freedom (Art. 19–22)...
...

━━━━━━━━━━━━ MCQ TEST SET ━━━━━━━━━━━

Q1. Which Article abolishes untouchability?
(A) Art. 14  (B) Art. 17  (C) Art. 19  (D) Art. 21
Answer: B
Explanation: Article 17 abolishes untouchability in any form...
...

━━━━━━━━━━━━ QUALITY REVIEW ━━━━━━━━━

Status  : Good
Summary : Content is accurate and exam-relevant for UPSC...
```

---

## Why LangGraph (not a simple chain)?

| Feature | Simple LLM Call | This Agent |
|---|---|---|
| State shared across steps | ❌ | ✅ MCQs use notes as context |
| Modular — swap any node | ❌ | ✅ Add translation node, difficulty selector |
| Quality gate before output | ❌ | ✅ Review node flags errors |
| Scale to parallel branches | ❌ | ✅ LangGraph supports async + parallel nodes |

This architecture scales directly to ContentKosh's LMS pipeline — each node can become a tool (n8n trigger, database write, content validator).

---

## Author

**Sujal Kumar Nayak**
GitHub: [github.com/SKcoder6344](https://github.com/SKcoder6344)
Email: nayaksujalkumar@gmail.com

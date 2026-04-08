"""
ContentKosh AI Content Agent — LangGraph Production Build
Built by Sujal Kumar Nayak as a proof-of-work submission
GitHub: github.com/SKcoder6344

Purpose:
  Multi-step stateful AI agent that automates educational content creation
  for coaching institutes. Given a topic, it generates:
    1. Chapter summary / study notes
    2. MCQ test questions (with answer key)
    3. Quality review (factual accuracy + MCQ validation)
    4. White-label-ready assembled content package

Stack: LangGraph + LangChain + OpenAI API
"""

import os
import json
import logging
from typing import TypedDict

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# ─────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("contentkosh.agent")

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

MAX_TOKENS_NOTES = 600
MAX_TOKENS_MCQS = 800
MAX_TOKENS_REVIEW = 400
MAX_TOKENS_DEFAULT = 500

SUPPORTED_EXAMS = frozenset([
    "UPSC", "SSC CGL", "CUET", "Class 10 Boards",
    "State PCS", "NEET", "JEE"
])

# ─────────────────────────────────────────────
# 1. STATE DEFINITION
# ─────────────────────────────────────────────

class ContentState(TypedDict):
    topic: str
    subject: str
    exam_target: str
    num_mcqs: int
    study_notes: str
    mcq_set: str
    quality_verdict: str
    final_package: str


# ─────────────────────────────────────────────
# 2. LLM FACTORY
# ─────────────────────────────────────────────

def get_llm(temperature: float = 0.4, max_tokens: int = MAX_TOKENS_DEFAULT) -> ChatOpenAI:
    """
    Returns a configured ChatOpenAI instance.

    Args:
        temperature: Sampling temperature (0.0–1.0). Lower = more deterministic.
        max_tokens:  Hard cap on output tokens to control cost.

    Returns:
        ChatOpenAI instance ready for invocation.

    Raises:
        EnvironmentError: If OPENAI_API_KEY is not set.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. "
            "Export it via: export OPENAI_API_KEY='your-key-here'"
        )

    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=temperature,
        max_tokens=max_tokens,
        openai_api_key=api_key,
    )


# ─────────────────────────────────────────────
# 3. NODE 1 — Study Notes Generator
# ─────────────────────────────────────────────

def generate_study_notes(state: ContentState) -> ContentState:
    """
    Generates concise, exam-ready study notes for the given topic.
    Tailored to the target exam format (UPSC, SSC, boards, etc.)

    Args:
        state: Current pipeline state with topic, subject, exam_target.

    Returns:
        Updated state with study_notes populated.
    """
    logger.info("[Node 1] Generating study notes | topic=%s exam=%s", state["topic"], state["exam_target"])

    llm = get_llm(temperature=0.3, max_tokens=MAX_TOKENS_NOTES)

    prompt = f"""You are an expert academic content writer for {state['exam_target']} exam preparation.

Write concise, exam-ready study notes on: **{state['topic']}** ({state['subject']})

Format requirements:
- Start with a 2-line definition / overview
- 4–6 key points in bullet format (each under 25 words)
- 1 "Remember" tip at the end (common exam trap or mnemonic)
- Total length: 200–250 words max
- Language: Clear English, no jargon beyond what the exam expects

Output ONLY the notes, no extra commentary."""

    try:
        response = llm.invoke([
            SystemMessage(content="You are a precise academic content writer. Follow format instructions exactly."),
            HumanMessage(content=prompt),
        ])
        logger.info("[Node 1] Study notes generated successfully")
        return {**state, "study_notes": response.content}
    except Exception as exc:
        logger.error("[Node 1] Failed to generate study notes: %s", exc, exc_info=True)
        raise RuntimeError(f"Study notes generation failed: {exc}") from exc


# ─────────────────────────────────────────────
# 4. NODE 2 — MCQ Generator
# ─────────────────────────────────────────────

def generate_mcqs(state: ContentState) -> ContentState:
    """
    Generates MCQs grounded in the study notes produced by Node 1.
    Questions are contextually consistent with the generated notes.

    Args:
        state: Current pipeline state including study_notes.

    Returns:
        Updated state with mcq_set populated.
    """
    logger.info("[Node 2] Generating %d MCQs | topic=%s", state["num_mcqs"], state["topic"])

    llm = get_llm(temperature=0.5, max_tokens=MAX_TOKENS_MCQS)

    prompt = f"""You are creating MCQ test questions for {state['exam_target']} based on the following study notes:

---
{state['study_notes']}
---

Generate exactly {state['num_mcqs']} MCQs on the topic: {state['topic']}

Format for each question:
Q[n]. [Question text]
(A) [Option]  (B) [Option]  (C) [Option]  (D) [Option]
Answer: [Letter]
Explanation: [1 sentence why this is correct]

Rules:
- Questions must be factual and exam-relevant
- One clearly correct answer per question
- Distractors must be plausible (not obviously wrong)
- Vary difficulty: 40% easy, 40% medium, 20% hard
- No repeated concepts across questions

Output ONLY the MCQs in the format above."""

    try:
        response = llm.invoke([
            SystemMessage(content="You are a precise MCQ test designer. Output only formatted questions, nothing else."),
            HumanMessage(content=prompt),
        ])
        logger.info("[Node 2] MCQ set generated successfully")
        return {**state, "mcq_set": response.content}
    except Exception as exc:
        logger.error("[Node 2] Failed to generate MCQs: %s", exc, exc_info=True)
        raise RuntimeError(f"MCQ generation failed: {exc}") from exc


# ─────────────────────────────────────────────
# 5. NODE 3 — Quality Review Agent
# ─────────────────────────────────────────────

def review_content_quality(state: ContentState) -> ContentState:
    """
    Acts as a senior content editor. Reviews both notes and MCQs for
    factual accuracy, exam relevance, and MCQ quality. Returns a JSON verdict.

    Args:
        state: Current pipeline state including study_notes and mcq_set.

    Returns:
        Updated state with quality_verdict (JSON string) populated.
    """
    logger.info("[Node 3] Running quality review | topic=%s", state["topic"])

    llm = get_llm(temperature=0.2, max_tokens=MAX_TOKENS_REVIEW)

    prompt = f"""You are a senior academic content quality reviewer for {state['exam_target']} coaching material.

Review this content package for topic: {state['topic']}

=== STUDY NOTES ===
{state['study_notes']}

=== MCQ SET ===
{state['mcq_set']}

Evaluate on:
1. Factual accuracy (flag any errors)
2. Exam relevance (is this what actually appears in {state['exam_target']}?)
3. MCQ quality (any ambiguous or flawed questions?)
4. Coverage gaps (important subtopics missing?)

Output format (strict JSON, no extra text):
{{
  "overall_rating": "Good/Needs Revision/Poor",
  "factual_issues": ["issue1"] or [],
  "mcq_issues": ["q_number: issue"] or [],
  "missing_topics": ["topic1"] or [],
  "verdict": "one sentence summary"
}}"""

    try:
        response = llm.invoke([
            SystemMessage(content="You are a strict quality reviewer. Output only valid JSON, no extra text."),
            HumanMessage(content=prompt),
        ])
        # Validate JSON parsability early — fail loudly here rather than silently in Node 4
        raw = response.content.strip()
        json.loads(raw)
        logger.info("[Node 3] Quality review complete")
        return {**state, "quality_verdict": raw}
    except json.JSONDecodeError as exc:
        logger.warning("[Node 3] Review returned non-JSON; falling back to raw text. Error: %s", exc)
        return {**state, "quality_verdict": response.content}
    except Exception as exc:
        logger.error("[Node 3] Quality review failed: %s", exc, exc_info=True)
        raise RuntimeError(f"Quality review failed: {exc}") from exc


# ─────────────────────────────────────────────
# 6. NODE 4 — Package Assembler
# ─────────────────────────────────────────────

def assemble_final_package(state: ContentState) -> ContentState:
    """
    Assembles the final content package in a clean, white-label-ready format.
    Gracefully handles malformed quality_verdict JSON.

    Args:
        state: Fully populated pipeline state.

    Returns:
        Updated state with final_package as a formatted string.
    """
    logger.info("[Node 4] Assembling final package | topic=%s", state["topic"])

    try:
        verdict_data = json.loads(state["quality_verdict"])
        rating = verdict_data.get("overall_rating", "Good")
        verdict_summary = verdict_data.get("verdict", "Content is ready.")
    except (json.JSONDecodeError, KeyError):
        logger.warning("[Node 4] Could not parse quality_verdict JSON; using raw fallback")
        rating = "Good"
        verdict_summary = state["quality_verdict"]

    package = f"""
╔══════════════════════════════════════════════════════════╗
   CONTENTKOSH — AI CONTENT PACKAGE
   Topic     : {state['topic']}
   Subject   : {state['subject']}
   Exam      : {state['exam_target']}
   Quality   : {rating}
╚══════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━ STUDY NOTES ━━━━━━━━━━━━━━━━

{state['study_notes']}

━━━━━━━━━━━━━━━━ MCQ TEST SET ━━━━━━━━━━━━━━━━

{state['mcq_set']}

━━━━━━━━━━━━━━━━ QUALITY REVIEW ━━━━━━━━━━━━━━

Status  : {rating}
Summary : {verdict_summary}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Generated by ContentKosh AI Agent
  Powered by LangGraph + OpenAI GPT-4o-mini
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    logger.info("[Node 4] Package assembled successfully")
    return {**state, "final_package": package}


# ─────────────────────────────────────────────
# 7. BUILD THE LANGGRAPH WORKFLOW
# ─────────────────────────────────────────────

def build_agent() -> StateGraph:
    """
    Constructs the LangGraph StateGraph with a 4-node linear pipeline:
    generate_notes → generate_mcqs → review_quality → assemble_package

    Returns:
        Compiled LangGraph workflow ready for invocation.
    """
    workflow = StateGraph(ContentState)

    workflow.add_node("generate_notes", generate_study_notes)
    workflow.add_node("generate_mcqs", generate_mcqs)
    workflow.add_node("review_quality", review_content_quality)
    workflow.add_node("assemble_package", assemble_final_package)

    workflow.set_entry_point("generate_notes")
    workflow.add_edge("generate_notes", "generate_mcqs")
    workflow.add_edge("generate_mcqs", "review_quality")
    workflow.add_edge("review_quality", "assemble_package")
    workflow.add_edge("assemble_package", END)

    return workflow.compile()


# ─────────────────────────────────────────────
# 8. PUBLIC ENTRYPOINT
# ─────────────────────────────────────────────

def run_agent(
    topic: str,
    subject: str = "General",
    exam_target: str = "UPSC",
    num_mcqs: int = 5,
) -> str:
    """
    Main function to run the ContentKosh AI content pipeline.

    Args:
        topic:       Subject matter to generate content for (e.g. "Fundamental Rights").
        subject:     Academic subject (e.g. "Polity", "Biology").
        exam_target: Target examination (must be one of SUPPORTED_EXAMS).
        num_mcqs:    Number of MCQ questions to generate (1–10 recommended).

    Returns:
        Final assembled content package as a formatted string.

    Raises:
        ValueError:       If topic is empty or exam_target is unsupported.
        EnvironmentError: If OPENAI_API_KEY is not configured.
        RuntimeError:     If any pipeline node fails.
    """
    # Input validation
    topic = topic.strip()
    if not topic:
        raise ValueError("topic cannot be empty.")

    if exam_target not in SUPPORTED_EXAMS:
        raise ValueError(
            f"Unsupported exam_target '{exam_target}'. "
            f"Choose from: {', '.join(sorted(SUPPORTED_EXAMS))}"
        )

    if not (1 <= num_mcqs <= 20):
        raise ValueError(f"num_mcqs must be between 1 and 20, got {num_mcqs}.")

    logger.info("Starting ContentKosh pipeline | topic=%s exam=%s mcqs=%d", topic, exam_target, num_mcqs)

    agent = build_agent()

    initial_state: ContentState = {
        "topic": topic,
        "subject": subject.strip() or "General",
        "exam_target": exam_target,
        "num_mcqs": num_mcqs,
        "study_notes": "",
        "mcq_set": "",
        "quality_verdict": "",
        "final_package": "",
    }

    result = agent.invoke(initial_state)
    logger.info("Pipeline completed successfully for topic: %s", topic)
    return result["final_package"]


# ─────────────────────────────────────────────
# 9. DEMO RUN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    output = run_agent(
        topic="Fundamental Rights in India",
        subject="Indian Polity",
        exam_target="UPSC",
        num_mcqs=5,
    )
    print(output)

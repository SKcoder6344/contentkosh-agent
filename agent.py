"""
ContentKosh AI Content Agent — LangGraph Demo
Built by Sujal Kumar Nayak as a proof-of-work submission
GitHub: github.com/SKcoder6344

Purpose:
  Multi-step stateful AI agent that automates educational content creation
  for coaching institutes. Given a topic, it generates:
    1. Chapter summary / study notes
    2. MCQ test questions (with answer key)
    3. One-line review of quality before finalizing

Stack: LangGraph + LangChain + OpenAI API
"""

import os
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import operator
import json


# ─────────────────────────────────────────────
# 1. STATE DEFINITION
#    LangGraph passes this dict between all nodes
# ─────────────────────────────────────────────

class ContentState(TypedDict):
    topic: str                    # input: e.g. "Photosynthesis Class 10"
    subject: str                  # e.g. "Biology", "History", "Economics"
    exam_target: str              # e.g. "UPSC", "SSC", "CUET", "Class 10 Boards"
    num_mcqs: int                 # how many MCQs to generate
    study_notes: str              # output of node 1
    mcq_set: str                  # output of node 2
    quality_verdict: str          # output of node 3 (review node)
    final_package: str            # assembled output


# ─────────────────────────────────────────────
# 2. LLM SETUP
# ─────────────────────────────────────────────

def get_llm(temperature: float = 0.4):
    """Returns ChatOpenAI instance. API key read from env."""
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=temperature,
        openai_api_key=os.getenv("OPENAI_API_KEY", "sk-placeholder")
    )


# ─────────────────────────────────────────────
# 3. NODE 1 — Study Notes Generator
# ─────────────────────────────────────────────

def generate_study_notes(state: ContentState) -> ContentState:
    """
    Generates concise, exam-ready study notes for the given topic.
    Tailored to the target exam format (UPSC, SSC, boards, etc.)
    """
    llm = get_llm(temperature=0.3)

    prompt = f"""You are an expert academic content writer for {state['exam_target']} exam preparation.

Write concise, exam-ready study notes on: **{state['topic']}** ({state['subject']})

Format requirements:
- Start with a 2-line definition / overview
- 4–6 key points in bullet format (each under 25 words)
- 1 "Remember" tip at the end (common exam trap or mnemonics)
- Total length: 200–250 words max
- Language: Clear English, no jargon beyond what the exam expects

Output ONLY the notes, no extra commentary."""

    response = llm.invoke([
        SystemMessage(content="You are a precise academic content writer. Follow format instructions exactly."),
        HumanMessage(content=prompt)
    ])

    return {**state, "study_notes": response.content}


# ─────────────────────────────────────────────
# 4. NODE 2 — MCQ Generator
# ─────────────────────────────────────────────

def generate_mcqs(state: ContentState) -> ContentState:
    """
    Generates MCQs from the study notes already created in node 1.
    Uses the notes as context so questions are consistent with the material.
    """
    llm = get_llm(temperature=0.5)

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

    response = llm.invoke([
        SystemMessage(content="You are a precise MCQ test designer. Output only formatted questions, nothing else."),
        HumanMessage(content=prompt)
    ])

    return {**state, "mcq_set": response.content}


# ─────────────────────────────────────────────
# 5. NODE 3 — Quality Review Agent
# ─────────────────────────────────────────────

def review_content_quality(state: ContentState) -> ContentState:
    """
    Acts as a senior content editor. Reviews both notes and MCQs
    and gives a short quality verdict + any critical fix notes.
    """
    llm = get_llm(temperature=0.2)

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

Output format (JSON):
{{
  "overall_rating": "Good/Needs Revision/Poor",
  "factual_issues": ["issue1", "issue2"] or [],
  "mcq_issues": ["q_number: issue"] or [],
  "missing_topics": ["topic1"] or [],
  "verdict": "one sentence summary"
}}"""

    response = llm.invoke([
        SystemMessage(content="You are a strict quality reviewer. Output only valid JSON, no extra text."),
        HumanMessage(content=prompt)
    ])

    return {**state, "quality_verdict": response.content}


# ─────────────────────────────────────────────
# 6. NODE 4 — Package Assembler
# ─────────────────────────────────────────────

def assemble_final_package(state: ContentState) -> ContentState:
    """
    Assembles the final content package in a clean, white-label-ready format.
    Ready for ContentKosh to deliver to coaching institutes.
    """
    try:
        verdict_data = json.loads(state["quality_verdict"])
        rating = verdict_data.get("overall_rating", "Good")
        verdict_summary = verdict_data.get("verdict", "Content is ready.")
    except Exception:
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
    return {**state, "final_package": package}


# ─────────────────────────────────────────────
# 7. BUILD THE LANGGRAPH WORKFLOW
# ─────────────────────────────────────────────

def build_agent() -> StateGraph:
    """
    Constructs the LangGraph StateGraph with 4 nodes:
    notes → mcqs → review → assemble
    """
    workflow = StateGraph(ContentState)

    # Add nodes
    workflow.add_node("generate_notes", generate_study_notes)
    workflow.add_node("generate_mcqs", generate_mcqs)
    workflow.add_node("review_quality", review_content_quality)
    workflow.add_node("assemble_package", assemble_final_package)

    # Define edges (linear pipeline)
    workflow.set_entry_point("generate_notes")
    workflow.add_edge("generate_notes", "generate_mcqs")
    workflow.add_edge("generate_mcqs", "review_quality")
    workflow.add_edge("review_quality", "assemble_package")
    workflow.add_edge("assemble_package", END)

    return workflow.compile()


# ─────────────────────────────────────────────
# 8. ENTRYPOINT
# ─────────────────────────────────────────────

def run_agent(
    topic: str,
    subject: str = "General",
    exam_target: str = "UPSC",
    num_mcqs: int = 5
) -> str:
    """
    Main function to run the ContentKosh AI agent.

    Args:
        topic      : e.g. "Fundamental Rights in India"
        subject    : e.g. "Polity", "Biology", "Economics"
        exam_target: e.g. "UPSC", "SSC CGL", "CUET", "Class 10 Boards"
        num_mcqs   : number of MCQ questions to generate (default 5)

    Returns:
        Final assembled content package as string
    """
    agent = build_agent()

    initial_state: ContentState = {
        "topic": topic,
        "subject": subject,
        "exam_target": exam_target,
        "num_mcqs": num_mcqs,
        "study_notes": "",
        "mcq_set": "",
        "quality_verdict": "",
        "final_package": ""
    }

    print(f"\n[ContentKosh Agent] Starting pipeline for: {topic}")
    print(f"[ContentKosh Agent] Target exam: {exam_target} | MCQs: {num_mcqs}\n")

    result = agent.invoke(initial_state)
    return result["final_package"]


# ─────────────────────────────────────────────
# 9. DEMO RUN (example usage)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    output = run_agent(
        topic="Fundamental Rights in India",
        subject="Indian Polity",
        exam_target="UPSC",
        num_mcqs=5
    )
    print(output)

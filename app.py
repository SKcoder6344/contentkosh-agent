"""
ContentKosh AI Agent — Streamlit Frontend
Production build with input sanitization, session rate limiting, and structured error handling.
"""

import os
import logging
import streamlit as st
from agent import run_agent, SUPPORTED_EXAMS

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("contentkosh.app")

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

MAX_REQUESTS_PER_SESSION = 10   # Prevents runaway API spend in a single session
MAX_TOPIC_LENGTH = 200          # Guard against oversized LLM prompts
MAX_SUBJECT_LENGTH = 100

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="ContentKosh AI Agent",
    page_icon="🎓",
    layout="centered",
)

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #FF4B4B; }
    .metric-card {
        padding: 12px 16px;
        border-radius: 8px;
        background-color: #f0f2f6;
        margin: 4px 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────

if "request_count" not in st.session_state:
    st.session_state.request_count = 0

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.markdown('<p class="main-header">🎓 ContentKosh AI Agent</p>', unsafe_allow_html=True)
st.markdown("**LangGraph + Groq** | Multi-step educational content generation")
st.markdown("---")

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Configuration")

    api_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Your free Groq key is used only for this session and never stored.",
    )
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key

    st.markdown("---")
    st.markdown("**Pipeline Nodes**")
    for step in ["1️⃣ Generate Notes", "2️⃣ Generate MCQs", "3️⃣ Quality Review", "4️⃣ Assemble Package"]:
        st.markdown(step)

    st.markdown("---")
    st.markdown(f"**Requests this session:** {st.session_state.request_count} / {MAX_REQUESTS_PER_SESSION}")
    st.markdown("**Built by:** Sujal Kumar Nayak")
    st.markdown("[GitHub](https://github.com/SKcoder6344/contentkosh-agent)")

# ─────────────────────────────────────────────
# MAIN FORM
# ─────────────────────────────────────────────

with st.form("content_form"):
    col1, col2 = st.columns(2)

    with col1:
        topic = st.text_input(
            "📚 Topic",
            placeholder="e.g., Fundamental Rights in India",
            max_chars=MAX_TOPIC_LENGTH,
        )
        subject = st.text_input(
            "📖 Subject",
            placeholder="e.g., Indian Polity",
            max_chars=MAX_SUBJECT_LENGTH,
        )

    with col2:
        exam_target = st.selectbox("🎯 Exam Target", sorted(SUPPORTED_EXAMS))
        num_mcqs = st.slider("📝 Number of MCQs", min_value=3, max_value=10, value=5)

    submitted = st.form_submit_button(
        "🚀 Generate Content Package",
        use_container_width=True,
        type="primary",
    )

# ─────────────────────────────────────────────
# FORM PROCESSING
# ─────────────────────────────────────────────

if submitted:
    # ── Guard: API key
    if not api_key:
        st.error("⚠️ Please enter your Groq API Key in the sidebar first.")
        st.info("Get your free key at: https://console.groq.com")
        st.stop()

    # ── Guard: Required fields
    topic_clean = topic.strip()
    subject_clean = subject.strip()
    if not topic_clean or not subject_clean:
        st.error("⚠️ Both Topic and Subject are required.")
        st.stop()

    # ── Guard: Session rate limit
    if st.session_state.request_count >= MAX_REQUESTS_PER_SESSION:
        st.error(
            f"⚠️ Session limit of {MAX_REQUESTS_PER_SESSION} requests reached. "
            "Refresh the page to start a new session."
        )
        st.stop()

    # ── Run pipeline
    logger.info("UI request | topic=%s exam=%s mcqs=%d", topic_clean, exam_target, num_mcqs)
    st.session_state.request_count += 1

    progress_bar = st.progress(0)
    status = st.empty()

    try:
        status.text("🔄 Node 1/4: Generating study notes…")
        progress_bar.progress(15)

        status.text("🔄 Nodes 1–4: Running full pipeline (notes → MCQs → review → assembly)…")
        progress_bar.progress(40)

        result = run_agent(
            topic=topic_clean,
            subject=subject_clean,
            exam_target=exam_target,
            num_mcqs=num_mcqs,
        )

        progress_bar.progress(100)
        status.text("✅ Done!")

        # ── Display results
        st.markdown("---")
        st.subheader("📦 Generated Content Package")

        tab_formatted, tab_raw = st.tabs(["🎨 Formatted View", "💻 Raw Text"])

        with tab_formatted:
            st.markdown(result)

        with tab_raw:
            st.code(result, language="text")

        col_dl, _ = st.columns([1, 3])
        with col_dl:
            st.download_button(
                label="📥 Download Package",
                data=result,
                file_name=f"{topic_clean.replace(' ', '_')}_{exam_target}_package.txt",
                mime="text/plain",
                use_container_width=True,
            )

        st.success(f"✅ Content generated for **{topic_clean}** ({exam_target})")
        st.balloons()
        logger.info("Pipeline success | topic=%s", topic_clean)

    except EnvironmentError as exc:
        st.error(f"🔑 API Key Error: {exc}")
        logger.error("EnvironmentError: %s", exc)

    except ValueError as exc:
        st.error(f"⚠️ Input Error: {exc}")
        logger.warning("ValueError: %s", exc)

    except RuntimeError as exc:
        st.error(f"❌ Pipeline Error: {exc}")
        st.info("💡 Check your API key validity and available credits.")
        logger.error("RuntimeError: %s", exc, exc_info=True)

    except Exception as exc:
        st.error("❌ An unexpected error occurred. Please try again.")
        st.info(f"Details: {exc}")
        logger.exception("Unexpected error during pipeline execution")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────

st.markdown("---")
st.caption(
    "🛡️ Built as a proof-of-work for ContentKosh | "
    "Stateless deployment — no data is stored | "
    "[GitHub Repo](https://github.com/SKcoder6344/contentkosh-agent)"
)

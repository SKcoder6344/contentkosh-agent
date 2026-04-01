import streamlit as st
import os
from agent import run_agent

# Page config
st.set_page_config(
    page_title="ContentKosh AI Agent",
    page_icon="🎓",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF4B4B;
    }
    .step-box {
        padding: 10px;
        border-radius: 5px;
        background-color: #f0f2f6;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🎓 ContentKosh AI Agent</p>', unsafe_allow_html=True)
st.markdown("**LangGraph + OpenAI** | Multi-step educational content generation")
st.markdown("---")

# Sidebar for API key
with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown("**Enter your OpenAI API Key**")
    api_key = st.text_input("API Key", type="password", help="Your key is secure and never stored")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    st.markdown("---")
    st.markdown("**Pipeline Nodes:**")
    st.markdown("1️⃣ Generate Notes")
    st.markdown("2️⃣ Generate MCQs")
    st.markdown("3️⃣ Quality Review")
    st.markdown("4️⃣ Assemble Package")

    st.markdown("---")
    st.markdown("**Built by:** Sujal Kumar Nayak")
    st.markdown("**Role:** AI Agent Developer Candidate")

# Main form
with st.form("content_form"):
    col1, col2 = st.columns(2)

    with col1:
        topic = st.text_input("📚 Topic", placeholder="e.g., Fundamental Rights in India", value="Fundamental Rights")
        subject = st.text_input("📖 Subject", placeholder="e.g., Indian Polity", value="Indian Polity")

    with col2:
        exam_target = st.selectbox(
            "🎯 Exam Target",
            ["UPSC", "SSC CGL", "CUET", "Class 10 Boards", "State PCS", "NEET", "JEE"]
        )
        num_mcqs = st.slider("📝 Number of MCQs", 3, 10, 5)

    submitted = st.form_submit_button("🚀 Generate Content Package", use_container_width=True, type="primary")

# Process
if submitted:
    if not api_key:
        st.error("⚠️ Please enter your OpenAI API Key in the sidebar first!")
        st.info("Get your key from: https://platform.openai.com/api-keys")
    elif not topic or not subject:
        st.error("⚠️ Please fill in both Topic and Subject!")
    else:
        # Progress tracking
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                # Node 1
                status_text.text("🔄 Node 1/4: Generating study notes...")
                progress_bar.progress(25)

                # Node 2-4 happen inside run_agent
                status_text.text("🔄 Nodes 2-4: Generating MCQs → Quality Check → Assembly...")
                progress_bar.progress(60)

                result = run_agent(
                    topic=topic,
                    subject=subject,
                    exam_target=exam_target,
                    num_mcqs=num_mcqs
                )

                progress_bar.progress(100)
                status_text.text("✅ Complete!")

                # Display results
                st.markdown("---")
                st.subheader("📦 Generated Content Package")

                tab1, tab2 = st.tabs(["🎨 Formatted View", "💻 Raw Text"])

                with tab1:
                    st.markdown(result)

                with tab2:
                    st.code(result, language="text")

                # Download button
                col_dl, col_space = st.columns([1, 3])
                with col_dl:
                    st.download_button(
                        label="📥 Download Package",
                        data=result,
                        file_name=f"{topic.replace(' ', '_')}_package.txt",
                        mime="text/plain",
                        use_container_width=True
                    )

                st.success(f"✅ Successfully generated content for **{topic}** ({exam_target})")
                st.balloons()

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("💡 Tip: Check if your OpenAI API key is valid and has available credits.")

# Footer
st.markdown("---")
st.caption("🛡️ Built as a proof-of-work for ContentKosh | Stateless deployment - no data is stored | [GitHub Repo](https://github.com/SKcoder6344/contentkosh-agent)")

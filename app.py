"""
app.py

Streamlit web UI for StudyMind.

FIXES APPLIED VS ORIGINAL:
1. display_name passed to ingest_pdf() — fixes the bug where temp file
   paths like /tmp/mynotes_abc123.pdf were stored as source metadata
   instead of the original filename the user uploaded.

2. Gap Finder tab added — the original app.py only had one tab ("Ask a
   Question"), leaving the find_gaps() feature completely inaccessible
   from the UI even though it was fully implemented in agent.py.

3. sys.path insert at top — ensures services/ and shared/ are importable
   regardless of which directory the user runs `streamlit run app.py` from.
   Without this, running from a parent folder caused ModuleNotFoundError.
"""

import os
import sys
import tempfile

# ─── Path fix ────────────────────────────────────────────────────────────────
# Ensures all local modules (services/, shared/, etc.) are importable
# regardless of which working directory `streamlit run app.py` is called from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st

from ingest import ingest_pdf
from services.rag_service import answer_question, search_notes

GAP_DISTANCE_THRESHOLD = 0.8

st.set_page_config(
    page_title="StudyMind",
    page_icon="📚",
    layout="centered"
)

st.title("📚 StudyMind")
st.caption("Ask questions. Get answers from YOUR notes only.")
st.divider()


# =============================================================================
# Sidebar — Upload PDFs
# =============================================================================
with st.sidebar:
    st.header("📂 Upload Notes")

    uploaded_files = st.file_uploader(
        "Upload your PDF notes",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("Upload Notes", use_container_width=True):
            for uploaded_file in uploaded_files:
                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".pdf",
                        # Use a clean prefix; the display_name fix below makes
                        # the actual prefix irrelevant to what gets stored in DB
                        prefix="studymind_upload_"
                    ) as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    with st.spinner(f"Uploading {uploaded_file.name}..."):
                        # FIX: pass the original filename as display_name so
                        # source metadata stores "myfile.pdf" not the temp path.
                        ingest_pdf(tmp_path, display_name=uploaded_file.name)

                except Exception as e:
                    st.error(f"Failed to upload {uploaded_file.name}: {e}")

                finally:
                    # Always clean up the temp file even if ingestion failed
                    if tmp_path and os.path.exists(tmp_path):
                        os.unlink(tmp_path)

            st.success("All notes uploaded successfully!")

    st.divider()
    st.caption("Made with 🩵 by Vaishnavv")


# =============================================================================
# Tabs — Ask a Question | Find Gaps
# FIX: original app.py only had 1 tab. Gap Finder tab added here so the
# find_gaps feature is accessible from the UI (it was already implemented
# in agent.py/rag_service.py but had no UI entry point).
# =============================================================================
tab1, tab2 = st.tabs(["💬 Ask a Question", "🔍 Find Gaps"])


# =============================================================================
# Tab 1 — Ask a Question
# =============================================================================
with tab1:
    question = st.text_input(
        "Ask a question from your notes",
        placeholder="e.g. What is photosynthesis?"
    )

    if st.button("Ask", use_container_width=True, type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching your notes..."):
                try:
                    result = answer_question(question)

                    if not result["success"]:
                        st.error(result["error"])
                    else:
                        st.markdown("### 🤖 Answer")
                        st.write(result["answer"])

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(
                                "LLM Judge",
                                "PASS ✅" if result["judge_result"] else "FAIL ❌"
                            )
                        with col2:
                            st.metric("Confidence", result["confidence"])

                        st.caption(f"📄 Sources: {', '.join(result['sources'])}")

                except Exception as e:
                    st.error(f"Something went wrong: {e}")


# =============================================================================
# Tab 2 — Find Gaps
# FIX: this entire tab was missing from the original app.py.
# =============================================================================
with tab2:
    st.markdown("### 🔍 Check Your Notes for Gaps")
    st.caption("Enter your syllabus topics to see which ones are covered in your notes.")

    topics_input = st.text_area(
        "Enter syllabus topics (one per line)",
        placeholder="e.g.\nPhotosynthesis\nMitosis\nDNA replication",
        height=150
    )

    if st.button("Check Gaps", use_container_width=True, type="primary"):
        topics = [t.strip() for t in topics_input.splitlines() if t.strip()]

        if not topics:
            st.warning("Please enter at least one topic.")
        else:
            covered = []
            missing = []

            with st.spinner("Checking your notes..."):
                for topic in topics:
                    try:
                        results = search_notes(topic, n_results=1)
                        is_covered = (
                            results["documents"][0]
                            and results["distances"][0][0] < GAP_DISTANCE_THRESHOLD
                        )
                        if is_covered:
                            covered.append(topic)
                        else:
                            missing.append(topic)
                    except Exception as e:
                        st.error(f"Error checking topic '{topic}': {e}")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### ✅ Covered")
                if covered:
                    for t in covered:
                        st.success(t)
                else:
                    st.info("None of the topics were found.")

            with col2:
                st.markdown("#### ❌ Missing")
                if missing:
                    for t in missing:
                        st.error(t)
                else:
                    st.info("All topics are covered! 🎉")
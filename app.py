import streamlit as st
import os
import tempfile
from agent import ask, find_gaps
from ingest import ingest_pdf


st.set_page_config(
    page_title="StudyMind",
    page_icon="📚",
    layout="centered"
)

st.title("📚 StudyMind")
st.caption("Ask questions. Get answers from YOUR notes only.")

st.divider()

with st.sidebar:
    st.header("📂 Upload Notes")
    uploaded_files = st.file_uploader(
        "Upload your PDF notes",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button(" Upload Notes", use_container_width=True):
            for uploaded_file in uploaded_files:
                
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf",
                    prefix=uploaded_file.name.replace(".pdf", "_")
                ) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                with st.spinner(f"Uploading {uploaded_file.name}..."):
                    ingest_pdf(tmp_path)
                os.unlink(tmp_path)  # clean up temp file

            st.success(" All notes Uploaded! Ready to query.")

    st.divider()
    st.caption("Made with ❤️ by Vaishnavv")

tab1= st.tabs([" Ask a Question"])

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
        
                from agent import search_notes
                from evals import keyword_score, llm_judge
                from google import genai
                from dotenv import load_dotenv

                load_dotenv()
                client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

                results = search_notes(question)

                if not results["documents"][0]:
                    st.error("No relevant content found in your notes.")
                else:
                    context_parts = []
                    sources = []

                    for i, (doc, metadata) in enumerate(zip(
                        results["documents"][0],
                        results["metadatas"][0]
                    )):
                        context_parts.append(
                            f"[Excerpt {i+1} from {metadata['source']}]\n{doc}"
                        )
                        if metadata["source"] not in sources:
                            sources.append(metadata["source"])

                    context = "\n\n".join(context_parts)

                    prompt = f"""You are a helpful study assistant. Answer the student's question 
ONLY using the provided excerpts from their notes. 

If the answer is not in the notes, say exactly: 
"I couldn't find this in your notes. You may need to check other sources."

Always mention which note/source your answer came from.

--- NOTES EXCERPTS ---
{context}
--- END OF EXCERPTS ---

Student's question: {question}

Answer:"""

                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=prompt
                    )
                    answer = response.text

                    # Eval
                    judge_result = llm_judge(client, question, answer, context)
                    confidence = "HIGH " if judge_result else "LOW "

                    # Display
                    st.markdown("###  Answer")
                    st.write(answer)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("LLM Judge", "PASS " if judge_result else "FAIL ")
                    with col2:
                        st.metric("Confidence", confidence)

                    st.caption(f"📄 Sources: {', '.join(sources)}")

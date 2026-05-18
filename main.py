"""
main.py — DocuMind AI

A RAG-powered Streamlit application that lets users upload PDF documents
and have a conversational Q&A session with them, grounded entirely in the
document's content, with source page citations.

Run:
    streamlit run main.py --server.port 5000
"""

import os
import shutil
import tempfile

import streamlit as st

from rag_engine import (
    load_and_split_pdf,
    build_vector_store,
    build_qa_chain,
    format_sources,
)

# ---------------------------------------------------------------------------
# Page configuration — must be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DocuMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
def init_session_state() -> None:
    defaults = {
        "chat_history": [],        # list of {"role", "content", "sources"}
        "qa_chain": None,          # active RunnableWithMessageHistory
        "vector_store": None,      # active Chroma vector store
        "chroma_dir": None,        # temp dir path for ChromaDB
        "processed_file": None,    # filename of the indexed PDF
        "doc_chunks_count": 0,     # number of text chunks indexed
        "session_id": "documind",  # static session id for message history
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


init_session_state()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_api_key() -> str | None:
    return os.environ.get("OPENAI_API_KEY") or None


def reset_pipeline() -> None:
    if st.session_state.chroma_dir and os.path.exists(st.session_state.chroma_dir):
        shutil.rmtree(st.session_state.chroma_dir, ignore_errors=True)

    st.session_state.chat_history = []
    st.session_state.qa_chain = None
    st.session_state.vector_store = None
    st.session_state.chroma_dir = None
    st.session_state.processed_file = None
    st.session_state.doc_chunks_count = 0


def process_uploaded_pdf(uploaded_file, api_key: str) -> None:
    with st.spinner("📚 Reading and indexing your document — this may take a moment…"):
        try:
            pdf_bytes = uploaded_file.read()
            chunks = load_and_split_pdf(pdf_bytes, uploaded_file.name)
            st.session_state.doc_chunks_count = len(chunks)

            chroma_dir = tempfile.mkdtemp(prefix="documind_chroma_")
            st.session_state.chroma_dir = chroma_dir

            vector_store = build_vector_store(chunks, api_key, chroma_dir)
            st.session_state.vector_store = vector_store

            st.session_state.qa_chain = build_qa_chain(vector_store, api_key)
            st.session_state.processed_file = uploaded_file.name

        except ValueError as exc:
            st.error(f"❌ {exc}")
        except Exception as exc:
            st.error(f"❌ An unexpected error occurred while processing the document: {exc}")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🧠 DocuMind AI")
    st.markdown("*Chat with your PDF documents using AI.*")
    st.divider()

    api_key = get_api_key()

    if not api_key:
        st.warning(
            "**OpenAI API key not found.**\n\n"
            "Add your key as `OPENAI_API_KEY` in your environment secrets "
            "to enable AI-powered document chat.",
        )
        st.divider()

    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"],
        help="Supports research papers, legal contracts, textbooks, manuals, and more.",
    )

    if uploaded_file is not None:
        if st.session_state.processed_file != uploaded_file.name:
            reset_pipeline()
            if not api_key:
                st.error(
                    "Please add your OpenAI API key before uploading a document."
                )
            else:
                process_uploaded_pdf(uploaded_file, api_key)

    if st.session_state.processed_file:
        st.success(f"✅ **{st.session_state.processed_file}**")
        st.caption(f"{st.session_state.doc_chunks_count} text chunks indexed")

        if st.button("🗑️ Remove document", use_container_width=True):
            reset_pipeline()
            st.rerun()

    st.divider()

    st.markdown(
        "**How it works**\n\n"
        "1. Upload any PDF document\n"
        "2. The text is extracted and split into chunks\n"
        "3. Chunks are embedded and stored in a local vector database\n"
        "4. Your questions are matched to the most relevant chunks\n"
        "5. GPT-4o-mini answers strictly from those chunks, with page citations"
    )


# ---------------------------------------------------------------------------
# Main area — header
# ---------------------------------------------------------------------------
st.markdown("## 🧠 DocuMind AI")
st.markdown(
    "Upload a PDF in the sidebar, then ask questions about it. "
    "Every answer is grounded in your document — no hallucinations, "
    "with source page citations so you can verify claims instantly."
)
st.divider()

if not st.session_state.processed_file:
    st.info(
        "👈 **Get started:** Upload a PDF in the sidebar to begin chatting with your document.",
    )

# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📎 Source pages", expanded=False):
                st.markdown(message["sources"])

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------
user_query = st.chat_input(
    placeholder="Ask a question about your document…",
    disabled=(st.session_state.qa_chain is None),
)

if user_query:
    if not user_query.strip():
        st.warning("Please enter a question before sending.")
    elif st.session_state.qa_chain is None:
        st.warning("Please upload a PDF document first.")
    else:
        st.session_state.chat_history.append(
            {"role": "user", "content": user_query, "sources": ""}
        )
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    result = st.session_state.qa_chain.invoke(
                        {"question": user_query},
                        config={"configurable": {"session_id": st.session_state.session_id}},
                    )
                    answer: str = result.get(
                        "answer",
                        "I was unable to generate a response. Please try again.",
                    )
                    source_docs = result.get("source_documents", [])
                    sources_text = format_sources(source_docs)

                    st.markdown(answer)

                    if sources_text:
                        with st.expander("📎 Source pages", expanded=False):
                            st.markdown(sources_text)

                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources_text,
                        }
                    )

                except Exception as exc:
                    error_msg = (
                        f"❌ An error occurred while generating a response: {exc}\n\n"
                        "Please check your OpenAI API key and try again."
                    )
                    st.error(error_msg)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": error_msg, "sources": ""}
                    )

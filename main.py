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
from langchain.memory import ConversationBufferMemory

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
# Session state — initialise all keys with defaults on first run
# ---------------------------------------------------------------------------
def init_session_state() -> None:
    """Initialise session-level state variables with safe defaults."""
    defaults = {
        # Chat messages: list of dicts with keys "role", "content", "sources"
        "chat_history": [],
        # Active ConversationalRetrievalChain (None until a PDF is indexed)
        "qa_chain": None,
        # Shared ConversationBufferMemory (None until a PDF is indexed)
        "memory": None,
        # Active Chroma vector store (None until a PDF is indexed)
        "vector_store": None,
        # Temporary directory path used for ChromaDB persistence this session
        "chroma_dir": None,
        # Filename of the currently indexed PDF (used for change detection)
        "processed_file": None,
        # Number of text chunks extracted from the current PDF
        "doc_chunks_count": 0,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


init_session_state()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_api_key() -> str | None:
    """Return the OpenAI API key from the environment, or None if not set."""
    return os.environ.get("OPENAI_API_KEY") or None


def reset_pipeline() -> None:
    """
    Tear down the active RAG pipeline and clear all session state.
    Called whenever a new document is uploaded or the user removes the current one.
    """
    # Remove the ChromaDB temp directory to free disk space
    if st.session_state.chroma_dir and os.path.exists(st.session_state.chroma_dir):
        shutil.rmtree(st.session_state.chroma_dir, ignore_errors=True)

    st.session_state.chat_history = []
    st.session_state.qa_chain = None
    st.session_state.memory = None
    st.session_state.vector_store = None
    st.session_state.chroma_dir = None
    st.session_state.processed_file = None
    st.session_state.doc_chunks_count = 0


def process_uploaded_pdf(uploaded_file, api_key: str) -> None:
    """
    Full pipeline: parse PDF → chunk text → embed → build vector store → build QA chain.
    Stores results in st.session_state. Shows spinners and user-friendly errors.

    Args:
        uploaded_file: Streamlit UploadedFile object.
        api_key:       Valid OpenAI API key string.
    """
    with st.spinner("📚 Reading and indexing your document — this may take a moment…"):
        try:
            # 1. Parse and split the PDF into overlapping chunks
            pdf_bytes = uploaded_file.read()
            chunks = load_and_split_pdf(pdf_bytes, uploaded_file.name)
            st.session_state.doc_chunks_count = len(chunks)

            # 2. Create a dedicated temp directory for ChromaDB persistence
            chroma_dir = tempfile.mkdtemp(prefix="documind_chroma_")
            st.session_state.chroma_dir = chroma_dir

            # 3. Generate embeddings and persist to ChromaDB
            vector_store = build_vector_store(chunks, api_key, chroma_dir)
            st.session_state.vector_store = vector_store

            # 4. Set up conversational memory and the QA chain
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer",
            )
            st.session_state.memory = memory
            st.session_state.qa_chain = build_qa_chain(vector_store, api_key, memory)
            st.session_state.processed_file = uploaded_file.name

        except ValueError as exc:
            # User-facing errors (empty PDF, no readable text, etc.)
            st.error(f"❌ {exc}")
        except Exception as exc:
            # Unexpected errors (network, API limits, etc.)
            st.error(f"❌ An unexpected error occurred while processing the document: {exc}")


# ---------------------------------------------------------------------------
# Sidebar — upload, status, and controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🧠 DocuMind AI")
    st.markdown("*Chat with your PDF documents using AI.*")
    st.divider()

    api_key = get_api_key()

    # Warn the user if no API key is configured
    if not api_key:
        st.warning(
            "**OpenAI API key not found.**\n\n"
            "Add your key as `OPENAI_API_KEY` in your environment secrets "
            "to enable AI-powered document chat.",
        )
        st.divider()

    # File uploader — accepts PDF only
    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"],
        help="Supports research papers, legal contracts, textbooks, manuals, and more.",
    )

    if uploaded_file is not None:
        # Only re-process if a new (or different) file is uploaded
        if st.session_state.processed_file != uploaded_file.name:
            reset_pipeline()

            if not api_key:
                st.error(
                    "Please add your OpenAI API key before uploading a document. "
                    "See the warning above for instructions."
                )
            else:
                process_uploaded_pdf(uploaded_file, api_key)

    # Show document status once a PDF has been successfully indexed
    if st.session_state.processed_file:
        st.success(f"✅ **{st.session_state.processed_file}**")
        st.caption(f"{st.session_state.doc_chunks_count} text chunks indexed")

        if st.button("🗑️ Remove document", use_container_width=True):
            reset_pipeline()
            st.rerun()

    st.divider()

    # How-it-works description
    st.markdown(
        "**How it works**\n\n"
        "1. Upload any PDF document\n"
        "2. The text is extracted and split into chunks\n"
        "3. Chunks are embedded and stored in a local vector database\n"
        "4. Your questions are matched to the most relevant chunks\n"
        "5. GPT-4o-mini answers strictly from those chunks, with page citations"
    )


# ---------------------------------------------------------------------------
# Main area — header and description
# ---------------------------------------------------------------------------
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("## 🧠 DocuMind AI")
    st.markdown(
        "Upload a PDF in the sidebar, then ask questions about it. "
        "Every answer is grounded in your document — no hallucinations, "
        "with source page citations so you can verify claims instantly."
    )

st.divider()

# Onboarding prompt when no document is loaded
if not st.session_state.processed_file:
    st.info(
        "👈 **Get started:** Upload a PDF in the sidebar to begin chatting with your document.",
    )

# ---------------------------------------------------------------------------
# Chat history — render all previous messages
# ---------------------------------------------------------------------------
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📎 Source pages", expanded=False):
                st.markdown(message["sources"])

# ---------------------------------------------------------------------------
# Chat input — disabled until a document is indexed
# ---------------------------------------------------------------------------
user_query = st.chat_input(
    placeholder="Ask a question about your document…",
    disabled=(st.session_state.qa_chain is None),
)

if user_query:
    # Guard: empty string after stripping whitespace
    if not user_query.strip():
        st.warning("Please enter a question before sending.")
    elif st.session_state.qa_chain is None:
        st.warning("Please upload a PDF document first.")
    else:
        # Append and immediately render the user's message
        st.session_state.chat_history.append(
            {"role": "user", "content": user_query, "sources": ""}
        )
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate and render the assistant's response
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    result = st.session_state.qa_chain.invoke(
                        {"question": user_query}
                    )
                    answer: str = result.get(
                        "answer", "I was unable to generate a response. Please try again."
                    )
                    source_docs = result.get("source_documents", [])
                    sources_text = format_sources(source_docs)

                    # Display the answer
                    st.markdown(answer)

                    # Display citations if available
                    if sources_text:
                        with st.expander("📎 Source pages", expanded=False):
                            st.markdown(sources_text)

                    # Persist to chat history
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

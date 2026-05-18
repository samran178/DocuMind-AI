"""
rag_engine.py — Core RAG (Retrieval-Augmented Generation) logic for DocuMind AI.

Handles PDF loading, text chunking, embedding generation, vector storage,
and the conversational QA chain setup.
"""

import os
import tempfile
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document


# ---------------------------------------------------------------------------
# Prompt: QA — strictly grounded in the retrieved document context
# ---------------------------------------------------------------------------
CUSTOM_QA_PROMPT = PromptTemplate(
    template="""You are DocuMind AI, a precise and reliable document analysis assistant.

Use ONLY the following context retrieved from the document to answer the question.

CONTEXT:
{context}

INSTRUCTIONS:
- Answer based ONLY on the provided context. Do not use any outside knowledge.
- If the answer is not in the context, respond exactly with: "I cannot find that in the document."
- Do NOT make up, infer, or guess information beyond what is explicitly stated.
- When the context includes page numbers or section metadata, reference them naturally
  (e.g., "According to page 3, ..." or "In Section 2.1, ...").
- Keep your answer concise and accurate.

Question: {question}

Answer:""",
    input_variables=["context", "question"],
)

# ---------------------------------------------------------------------------
# Prompt: Condense multi-turn follow-up questions into a standalone query
# ---------------------------------------------------------------------------
CONDENSE_QUESTION_PROMPT = PromptTemplate(
    template="""Given the conversation history below and a follow-up question,
rephrase the follow-up question as a complete, standalone question that
captures all necessary context from the history.

Chat History:
{chat_history}

Follow-up Question: {question}

Standalone Question:""",
    input_variables=["chat_history", "question"],
)


def load_and_split_pdf(pdf_bytes: bytes, filename: str) -> List[Document]:
    """
    Load a PDF from raw bytes, extract text page-by-page, and split into
    overlapping chunks suitable for embedding and retrieval.

    Args:
        pdf_bytes: Raw bytes of the uploaded PDF file.
        filename:  Original filename (stored in chunk metadata).

    Returns:
        A list of LangChain Document objects, each representing one chunk.

    Raises:
        ValueError: If the PDF is empty, unreadable, or contains no text.
    """
    # PyPDFLoader requires a file path, so write bytes to a temp file first
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
    finally:
        os.unlink(tmp_path)  # Always clean up the temp file

    # Guard: nothing was loaded
    if not pages:
        raise ValueError(
            "The PDF appears to be empty or could not be parsed. "
            "Please ensure the file is a valid, non-encrypted PDF."
        )

    # Guard: pages exist but contain no readable text (e.g. scanned image PDF)
    total_text = "".join(p.page_content for p in pages).strip()
    if not total_text:
        raise ValueError(
            "No readable text was found in this PDF. "
            "It may be a scanned image or an encrypted document. "
            "Please use a text-based PDF."
        )

    # Split into chunks with overlap to preserve context at boundaries
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)

    # Attach source filename to each chunk's metadata for citations
    for chunk in chunks:
        chunk.metadata["source_file"] = filename

    return chunks


def build_vector_store(
    chunks: List[Document],
    openai_api_key: str,
    persist_dir: str,
) -> Chroma:
    """
    Generate OpenAI embeddings for all chunks and persist them in a local
    ChromaDB vector store.

    Args:
        chunks:         Document chunks to embed.
        openai_api_key: OpenAI API key for the embedding model.
        persist_dir:    Local directory path for ChromaDB persistence.

    Returns:
        An initialised Chroma vector store ready for similarity search.
    """
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=openai_api_key,
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    return vector_store


def build_qa_chain(
    vector_store: Chroma,
    openai_api_key: str,
    memory: ConversationBufferMemory,
) -> ConversationalRetrievalChain:
    """
    Assemble a ConversationalRetrievalChain that retrieves the top-4
    most relevant chunks and passes them — along with conversation history —
    to GPT-4o-mini for answer generation.

    Args:
        vector_store:   Populated Chroma vector store.
        openai_api_key: OpenAI API key for the LLM.
        memory:         Shared ConversationBufferMemory instance.

    Returns:
        A ready-to-invoke ConversationalRetrievalChain.
    """
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0,          # Deterministic, fact-grounded responses
        openai_api_key=openai_api_key,
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},  # Retrieve top-4 most relevant chunks
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": CUSTOM_QA_PROMPT},
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        return_source_documents=True,
        verbose=False,
    )
    return chain


def format_sources(source_docs: List[Document]) -> str:
    """
    Build a deduplicated, human-readable citations block from source documents.

    Deduplicates by (filename, page) so that multiple chunks from the same
    page are listed only once.

    Args:
        source_docs: Source documents returned by the QA chain.

    Returns:
        A Markdown-formatted string of citations, or an empty string if none.
    """
    seen: set = set()
    citations: List[str] = []

    for doc in source_docs:
        meta = doc.metadata
        page = meta.get("page")  # 0-indexed page number from PyPDFLoader
        source = meta.get("source_file", meta.get("source", "Document"))

        # Convert 0-indexed page to 1-indexed for display
        page_label = f"Page {page + 1}" if page is not None else "Unknown page"
        key = (source, page)

        if key not in seen:
            seen.add(key)
            citations.append(f"- **{source}** — {page_label}")

    return "\n".join(citations) if citations else ""

"""
rag_engine.py — Core RAG logic for DocuMind AI.

Handles PDF loading, text chunking, embedding generation, vector storage,
and the conversational QA chain setup using LangChain 1.x APIs.
"""

import os
import tempfile
from typing import List, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# Step 1: Condense multi-turn follow-ups into a standalone search query
CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
    ("human",
     "Given the conversation above, rewrite my latest question as a concise, "
     "standalone question that captures all necessary context. "
     "Output only the rewritten question, nothing else."),
])

# Step 2: Answer strictly from the retrieved document context
QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     """You are DocuMind AI, a precise and reliable document analysis assistant.

Use ONLY the following context retrieved from the document to answer the question.

CONTEXT:
{context}

INSTRUCTIONS:
- Answer based ONLY on the provided context. Do not use any outside knowledge.
- If the answer is not in the context, respond exactly with: "I cannot find that in the document."
- Do NOT make up, infer, or guess information beyond what is explicitly stated.
- When the context includes page numbers or section metadata, reference them naturally
  (e.g., "According to page 3..." or "In Section 2.1...").
- Keep your answer concise and accurate."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])


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
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
    finally:
        os.unlink(tmp_path)

    if not pages:
        raise ValueError(
            "The PDF appears to be empty or could not be parsed. "
            "Please ensure the file is a valid, non-encrypted PDF."
        )

    total_text = "".join(p.page_content for p in pages).strip()
    if not total_text:
        raise ValueError(
            "No readable text was found in this PDF. "
            "It may be a scanned image or an encrypted document. "
            "Please use a text-based PDF."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)

    for chunk in chunks:
        chunk.metadata["source_file"] = filename

    return chunks


def build_vector_store(
    chunks: List[Document],
    openai_api_key: str,
    persist_dir: str,
) -> Chroma:
    """
    Generate OpenAI embeddings for all chunks and persist in ChromaDB.

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
) -> RunnableWithMessageHistory:
    """
    Assemble a conversational RAG chain using LangChain 1.x LCEL.

    The chain:
    1. Condenses the user's question using chat history into a standalone query.
    2. Retrieves the top-4 most relevant chunks from the vector store.
    3. Answers strictly from those chunks using GPT-4o-mini.

    Args:
        vector_store:   Populated Chroma vector store.
        openai_api_key: OpenAI API key for the LLM.

    Returns:
        A RunnableWithMessageHistory chain ready to invoke with session_id.
    """
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0,
        openai_api_key=openai_api_key,
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    def format_docs(docs: List[Document]) -> str:
        parts = []
        for doc in docs:
            page = doc.metadata.get("page")
            page_label = f"[Page {page + 1}]" if page is not None else ""
            parts.append(f"{page_label}\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)

    # Step 1: condense question with history → standalone search query
    condense_chain = CONDENSE_PROMPT | llm | StrOutputParser()

    # Step 2: retrieve relevant docs for the condensed query
    def retrieve_with_context(inputs: Dict[str, Any]) -> Dict[str, Any]:
        history = inputs.get("chat_history", [])
        question = inputs["question"]

        if history:
            standalone = condense_chain.invoke({
                "chat_history": history,
                "question": question,
            })
        else:
            standalone = question

        docs = retriever.invoke(standalone)
        return {
            "context": format_docs(docs),
            "question": question,
            "chat_history": history,
            "source_documents": docs,
        }

    # Full chain: retrieve → answer
    rag_chain = (
        RunnablePassthrough.assign(**{"retrieved": retrieve_with_context})
        | {
            "answer": (
                {
                    "context": lambda x: x["retrieved"]["context"],
                    "question": lambda x: x["retrieved"]["question"],
                    "chat_history": lambda x: x["retrieved"]["chat_history"],
                }
                | QA_PROMPT
                | llm
                | StrOutputParser()
            ),
            "source_documents": lambda x: x["retrieved"]["source_documents"],
        }
    )

    chain_with_history = RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: ChatMessageHistory(),  # per-session in-memory history
        input_messages_key="question",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return chain_with_history


def format_sources(source_docs: List[Document]) -> str:
    """
    Build a deduplicated, human-readable citations block from source documents.

    Args:
        source_docs: Source documents returned by the QA chain.

    Returns:
        A Markdown-formatted string of citations, or an empty string if none.
    """
    seen: set = set()
    citations: List[str] = []

    for doc in source_docs:
        meta = doc.metadata
        page = meta.get("page")
        source = meta.get("source_file", meta.get("source", "Document"))
        page_label = f"Page {page + 1}" if page is not None else "Unknown page"
        key = (source, page)

        if key not in seen:
            seen.add(key)
            citations.append(f"- **{source}** — {page_label}")

    return "\n".join(citations) if citations else ""

# DocuMind AI

DocuMind AI is a full-stack Retrieval-Augmented Generation (RAG) web application that solves the problem of digging through massive documents manually. Users can upload complex PDFs—like legal contracts, technical manuals, or textbooks—and chat with them in plain English. The application extracts the document's content, indexes it semantically, and returns direct, accurate answers backed up by precise page citations.

To eliminate common LLM hallucinations, the backend enforces strict prompt engineering guardrails, forcing the system to rely only on verified context from the uploaded file.

## 🚀 Key Features

* **Smart PDF Parsing & Dynamic Chunking:** Automatically processes multi-page documents and splits text using a sliding-window text splitter to preserve context across boundaries.
* **Vector Architecture:** Generates vector embeddings for high-speed semantic searches, mapping the structural meaning of user queries to the document data.
* **Persistent Session Chat:** A clean, minimal chat layout that saves conversational context across the session, allowing for continuous follow-up questions.
* **Strict Source Verifiability:** Every response highlights the specific pages or sections where the raw data lives, allowing the user to audit claims instantly.

## 🛠️ Tech Stack

* **Language & Core Logic:** Python 3.10+
* **Orchestration Framework:** LangChain (LangChain-OpenAI, LangChain-Community)
* **Frontend UI:** Streamlit
* **Vector Database:** ChromaDB (In-Memory Local Instance)
* **LLM & Embeddings:** OpenAI `gpt-4o-mini` & `text-embedding-3-small`

## 📂 Project Structure

```text
├── main.py              # Main Streamlit UI and RAG orchestration logic
├── requirements.txt     # Locked project dependencies
├── .env                 # Local environment keys (ignored by git)
└── README.md            # Project documentation

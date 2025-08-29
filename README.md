# ClauseWise — AI-Powered Contract Analyzer

ClauseWise lets you upload contract PDFs and ask natural-language questions. It indexes your documents locally, retrieves the most relevant snippets, and answers your questions with clear references to the source text.

**Current stack:**  
Streamlit (UI) · PyMuPDF/fitz (PDF parsing) · LangChain (orchestration) · FAISS (vector store) · OpenAIEmbeddings (embeddings) · Groq Llama 3 via ChatGroq (LLM)

---

## Features

- **Multi-PDF ingest**: drag & drop contracts (PDF) and process in one click  
- **Chunking & retrieval**: smart splitting for robust RAG performance  
- **Grounded answers**: responses cite source snippets for transparency  
- **Session memory**: keeps your vector index for subsequent queries  
- **Private by default**: vectors built locally; your PDFs stay on your machine  

---

## How it works (RAG in a nutshell)

1. **Ingest**: Parse PDFs → extract text with PyMuPDF  
2. **Split**: Chunk text with overlap to preserve context  
3. **Embed**: Convert chunks to vectors (OpenAIEmbeddings)  
4. **Index**: Store vectors in FAISS  
5. **Retrieve + Generate**: Pull top-k chunks → prompt Llama-3 via Groq for an answer  


PDFs → Text → Chunks → Embeddings → FAISS
↓
Retriever
↓
LLM (Groq Llama3)
↓
Answer

## Prerequisites

- Python **3.10+**  
- Accounts/keys as needed:
  - `GROQ_API_KEY` (for Llama-3 via Groq)  
  - `OPENAI_API_KEY` (for OpenAIEmbeddings)  
---

## 📦 Installation

```bash
git clone https://github.com/<your-username>/clausewise.git
cd clausewise
pip install -r requirements.txt

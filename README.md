
# RAG System: Intelligent Document Q&A with FastAPI and Large Language Models

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/) [![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Environment Configuration](#environment-configuration)
- [Running the API](#running-the-api)
- [Usage](#usage)
  - [Uploading Documents](#uploading-documents)
  - [Querying the System](#querying-the-system)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Project Overview
This project implements an enterprise-grade Retrieval-Augmented Generation (RAG) system for intelligent, context-aware question answering over large, heterogeneous collections of unstructured documents (e.g., insurance policies, contracts, emails).

Users submit natural language or shorthand queries, such as:
> "46-year-old male, knee surgery in Pune, 3-month-old insurance policy"

and receive precise, citation-backed answers extracted dynamically by combining semantic document retrieval with large language model generation.

The project is built with a modular FastAPI backend powering document ingestion, vector embeddings, hybrid retrieval, and generation pipelines—ready for scaling and production deployment.

---

## Features
- **Document Upload & Processing:** Support PDFs, scanned images, emails, and HTML regulatory documents with OCR and content chunking
- **Dense + Sparse Hybrid Retrieval:** Combines semantic vector search (via Pinecone) with BM25 lexical retrieval to boost accuracy
- **Retrieval-Augmented Generation (RAG):** Uses Open-Source LLMs (e.g., LLaMA 3 8B) to generate grounded, citation-rich answers
- **FastAPI REST API:** Interactive Swagger UI for uploading documents and querying with low latency
- **Security & Compliance Considerations:** Data encryption, role-based access controls, PII redaction (configurable)
- **Modular, Extensible Architecture:** Easily extendable for new document types, languages, and models

---

## Architecture

```
User Query
     ↓
FastAPI REST API
     ↓
Query Preprocessing & Expansion
     ↓
Hybrid Retriever (Pinecone vector DB + BM25)
     ↓
Retrieve Top-K Relevant Document Chunks
     ↓
RAG Prompt Construction
     ↓
LLM Answer Generation (with citations)
     ↓
Response Returned to User
```

Document Ingestion converts unstructured files into semantically chunked text embeddings, stored in Pinecone.
Query Pipeline processes user input, performs semantic + lexical search, then generates concise, transparent answers.
Entire pipeline is built for low latency (<1.5s), high accuracy, and scalable microservices deployment.

---

## Getting Started

### Prerequisites
- Python 3.9+
- Git
- Pinecone Vector Database account and API key
- Tesseract OCR installed (for scanned documents)
  - macOS: `brew install tesseract`
  - Ubuntu: `sudo apt-get install tesseract-ocr`
- Optional: Docker (for containerized deployment)

### Installation
Clone this repository:
```bash
git clone https://github.com/Avinash-ml07/hackrx_rag.git
cd hackrx_rag
```

Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows
```

Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## Environment Configuration
Create a `.env` file in the project root or export the following environment variables:
```
PINECONE_API_KEY=your_pinecone_api_key_here
```
Alternatively, export in your terminal session:
```bash
export PINECONE_API_KEY=your_pinecone_api_key_here
```

---

## Running the API
Start the FastAPI server using Uvicorn with hot reload:
```bash
uvicorn app.main:app --reload
```
Visit the interactive API docs at:
http://127.0.0.1:8000/docs

---

## Usage

### Uploading Documents
Use the `/upload-documents/` POST endpoint.
Upload one or more policy PDFs or related documents.
The backend processes, chunks, and indexes the documents asynchronously.

### Querying the System
Use the `/query/` POST endpoint.
Provide free-text or shorthand query strings describing the information you want.
Receive concise answers grounded in cited text chunks.

**Example query:**
```json
{
  "query": "Is knee surgery covered for a 46-year-old male in Pune with a 3-month-old policy?"
}
```

**Sample response:**
```json
{
  "answer": "According to Document 1, Clause 5.2, knee surgery is covered after a waiting period of 6 months.",
  "confidence": 0.92,
  "sources": ["insurance_policy_1.pdf"],
  "retrieved_chunks": 5
}
```

---

## Project Structure
```
hackrx_rag/
├── app/
│   ├── __init__.py
│   ├── main.py               # FastAPI app and endpoints
│   ├── rag_system.py         # Core RAG pipeline and logic
│   ├── document_processor.py # OCR, PDF parsing and preprocessing
│   ├── chunker.py            # Text chunking code
│   ├── embedding_manager.py  # Embedding generation and vector DB interface
│   ├── retriever.py          # Hybrid retrieval implementation
│   ├── response_generator.py # LLM prompting and answer synthesis
├── requirements.txt
├── Dockerfile
├── README.md
└── .env.example
```

---

## Technologies Used
- **FastAPI** — lightweight, async web framework for Python
- **Uvicorn** — lightning-fast ASGI server
- **Tesseract OCR / AWS Textract** — extract text from scanned documents
- **Sentence Transformers** — generate semantic embeddings
- **Pinecone Vector DB** — scalable vector similarity search
- **BM25 (rank_bm25)** — sparse lexical retrieval fallback
- **Transformers (Hugging Face)** — LLM integration for answer generation
- **spaCy** — NLP preprocessing, entity detection
- **Docker** — containerization
- **Prometheus + Grafana** — monitoring (optional)

---

## Future Enhancements
- Add multilingual support with IndicBERT embeddings (Hindi, Marathi)
- Implement active learning loop for embedding updates from user feedback
- Advanced structured retrieval—combine SQL-style queries and vectors
- Deploy mixture-of-experts LLMs for complex reasoning tasks
- Role-based access controls and enhanced compliance logging

---

## Contributing
Contributions are very welcome! If you:
- Find bugs
- Want to improve code or docs
- Add new features

please open issues or submit pull requests. Ensure tests pass and code style matches existing code.

---

## License
This project is licensed under the MIT License.

---

## Acknowledgements
Inspired by the HackRx 6.0 challenge
Thanks to the open-source community for FastAPI, Hugging Face Transformers, Pinecone, and related tools


Happy coding! 🚀 Feel free to raise issues or contact for help.



# LangChain Projects

This repository contains various LangChain experiments and projects.

## Projects

### code-interpreter
A multi-agent system that can analyze CSV data and execute Python code using LangChain agents.

Features:
- Python code execution via REPL
- CSV data analysis
- QR code generation
- LangSmith monitoring integration

### medium-rag
A Retrieval-Augmented Generation (RAG) system for querying legal technology content about AI agents.

Features:
- Document ingestion and chunking
- Vector search with Pinecone
- RAG pipeline with OpenAI
- LangSmith integration for monitoring
- Auto-creates Pinecone index
- Optimized for legal tech content

## Setup

### code-interpreter
1. Navigate to the project directory: `cd code-interpreter`
2. Install dependencies: `uv sync`
3. Configure environment variables in `.env`
4. Run: `uv run python main.py`

### medium-rag
1. Navigate to the project directory: `cd medium-rag`
2. Install dependencies: `uv sync`
3. Configure environment variables in `.env`
4. Ingest documents: `python ingestion.py`
5. Query the system: `python main.py`
# FinRAG
A RAG based Intelligent Assistant for Financial Reports (10K Filings)

This project implements an advanced Retrieval-Augmented Generation (RAG) system for querying financial reports such as 10k Filings using state-of-the-art natural language processing techniques.

## Features

- **Document Processing**: Efficiently loads and splits PDF documents into manageable chunks.
- **Advanced Retrieval Techniques**: Implements multiple retrieval methods including semantic search, hybrid search, query expansion, contextual compression, and multi-query retrieval.
- **Customizable Language Models**: Supports different LLM models (mistral and llama3.2) for generating responses.
- **Interactive User Interface**: Built with Streamlit for easy interaction and query submission.
- **Flexible Usage**: Allows querying of stored financial reports or uploading new documents for analysis.

## Components

### 1. Document Processing (`populate_database.py`)

- Loads PDF documents from a specified directory.
- Splits documents into chunks for efficient processing.
- Populates a Chroma vector database with document embeddings.

### 2. Embedding Generation (`get_embedding_function.py`)

- Utilizes the SentenceTransformer model 'BAAI/bge-large-en-v1.5' for generating embeddings.
- Implements a custom `FinanceEmbeddings` class for document and query embedding.

### 3. Advanced RAG Techniques (`advanced_rag_techniques.py`)

- Implements various retrieval methods:
  - Semantic Search
  - Hybrid Search
  - Query Expansion
  - Contextual Compression
  - Multi-Query Retrieval
- Includes re-ranking functionality using a CrossEncoder model.
- Supports query transformation for more precise information retrieval.

### 4. Main Application (`adv_rag_app.py`)

- Provides a Streamlit-based user interface for interacting with the system.
- Allows users to choose between using stored reports or uploading new ones.
- Offers options to select different retrieval methods and LLM models.
- Processes queries and displays responses with source information.

## Usage
1. Pull Ollama models using 'ollama pull mistral' and 'ollama pull llama3.2'
2. Ensure all dependencies are installed.
3. Run `populate_database.py` to process and store initial documents.
4. Launch the Streamlit app by running `streamlit run adv_rag_app.py`.
5. Use the interface to submit queries or upload new documents for analysis.

## Requirements

- Python 3.7+
- Langchain
- Streamlit
- Sentence Transformers
- PyPDF2
- Chroma
- Ollama

## Setup

1. Clone the repository.
2. Install required packages: `pip install -r requirements.txt`
3. Ensure you have the necessary LLM models available (mistral and llama3.2).
4. Run the Streamlit app to start querying financial reports.

## Contributing

Contributions to improve the system or add new features are welcome. Please submit a pull request or open an issue for discussion.


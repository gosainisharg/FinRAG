# RAG-Based-Financial-Reports-Assistant

This repository demonstrates a **Retrieval-Augmented Generation (RAG)** pipeline using **FAISS** and **state-of-the-art language models** like **LLaMA 3** and **OpenAI GPT** for querying financial reports. The system is optimized for analyzing financial documents, such as 10-K filings, to provide detailed, accurate, and user-friendly responses to natural language queries.

---

## Objective

Develop a robust RAG-based financial assistant that combines advanced vector search with fine-tuned generative models. The system enables users to interactively query structured financial documents, offering precise insights in a conversational format.

---

## Key Features

1. **Structured Financial Data**:
   - Utilizes 10-K filings, stored in a structured CSV format, with sections like "Risk Factors," "Financial Statements," and "Management Analysis."
   - Supports additional financial filings, such as 10-Q (quarterly reports) and 8-K (current events).

2. **FAISS and ChromaDB for Document Retrieval**:
   - **FAISS (Facebook AI Similarity Search)** provides efficient, similarity-based search over document embeddings and supports retrieval for models like GPT-4o and Meta-LLaMA 3-8B Instruct.
   - **ChromaDB** is used for advanced indexing and retrieval for LLaMA 3.2 and Mistral, ensuring accurate and scalable document processing.
   - Both systems store embeddings in indices for fast and context-aware retrieval of relevant sections.

3. **Pre-trained Financial Embeddings**:
   - Uses **FinLang/finance-embeddings-investopedia**, a specialized model for embedding financial text.
   - Embeds both user queries and document sections into a shared semantic space for accurate matching.

4. **Advanced Generative Models**:
   - **LLaMA 3.2**: A scalable model optimized for multi-turn dialogue and large-scale financial datasets, ensuring detailed contextual understanding.
   - **Mistral**: Efficiently designed for text generation and structured QA tasks, particularly on compact hardware setups.
   - **Meta-LLaMA 3-8B Instruct**: Fine-tuned for instruction-following tasks, enabling precise and context-sensitive responses.
   - **OpenAI GPT**: Combines retrieved context with user queries to generate fact-based, natural language responses.

5. **Interactive Querying**:
   - Users can query financial reports using natural language.
   - The system retrieves relevant sections and generates detailed, domain-specific answers.

---

## Workflow

#### **Step 1: Data Preparation and Embedding**

- **HTML Processing**: Extracted text using BeautifulSoup, cleaned and normalized, with sections like "Risk Factors" mapped to standardized fields. Output was structured as `structured_10k.csv`.  
- **PDF Processing**: Used PyPDFLoader and RecursiveCharacterTextSplitter to process files into 800-character chunks with 80-character overlap, assigning unique identifiers for traceability.  
- **Embedding Generation**: Text embeddings were generated using `all-mpnet-base-v2` for general-purpose tasks, `FinLang/finance-embeddings-investopedia` for financial contexts, and `BAAI/bge-large-en-v1.5` for balanced precision and generalizability. Indexed using **FAISS** for fast retrieval.

---

#### **Step 2: Query Processing and Retrieval**

- **Query Encoding**: User queries were embedded using the same models as document embeddings.  
- **Search and Retrieval**: **FAISS** retrieved relevant document chunks, while **ChromaDB** supported retrieval for **LLaMA 3.2** and **Mistral**.  
- **Context Aggregation**: Retrieved sections were combined into a coherent context for model input.

---

#### **Step 3: Response Generation**

- **Prompt Creation**: Combined user queries with retrieved context for structured input.  
- **Language Models**:  
  - **LLaMA 3.2** and **Mistral**: Fine-tuned for financial QA tasks.  
  - **GPT-4o**: Generated detailed, conversational responses.  
- **Output**: Delivered user-friendly responses via a **Streamlit interface**.

---

#### **Models Used**

- **GPT-4o**: General-purpose model for precise financial insights.  
- **Mistral**: Optimized for QA tasks on compact setups.  
- **LLaMA 3.2**: Handles multi-turn dialogue and large-scale datasets.  
- **Meta-LLaMA 3-8B Instruct**: Fine-tuned for instruction-following with domain-specific datasets.

---

## Example Query and Output

### Query:
*"What are the risk factors mentioned in the report?"*

### Pipeline Execution:
1. **Retrieval**: Retrieves relevant sections from the 10-K filing (e.g., "Risk Factors").
2. **Response Generation**: Combines the query and retrieved context to generate the following response:

### Output:
*"The risk factors include market volatility, regulatory changes, and supply chain disruptions, as mentioned in the report."*

---
## **Folder Structure**
```plaintext
RAG-Based-Financial-Reports-Assistant/
├── Evaluation/                     # Contains evaluation scripts and results for QA performance
│   ├── Evaluation.ipynb            # Notebook for evaluating retrieval and QA pair quality
│   ├── scored_qa.xlsx              # Excel file storing QA evaluation scores
├── FAISS/                          # Directory for FAISS-related files
│   ├── finance_10k_index.faiss     # FAISS index storing document embeddings for retrieval
├── data/                           # Directory for raw and processed financial data
├── RAG_Pipeline_with_GPT.ipynb     # Notebook implementing RAG pipeline using GPT for QA
├── RAG_pipeline_with_Llama.ipynb   # Notebook implementing RAG pipeline using LLaMA models
├── adv_rag_app.py                  # Streamlit app for interactive querying and response generation
├── advanced_rag_techniques.py      # Script for advanced RAG techniques like re-ranking and query expansion
├── get_embedding_function.py       # Script for generating text embeddings using various models
├── populate_database.py            # Script for populating ChromaDB with document embeddings
├── preprocess.py                   # Script for preprocessing HTML documents and normalizing text
├── section_mapping.json            # JSON file mapping document sections to indices for lookup
├── structured_10k.csv              # Structured dataset of 10-K filings for embeddings and retrieval
├── README.md                       # Project documentation with overview, setup, and usage instructions

```

## Key Components

### **FAISS Index**
- Efficiently stores document embeddings for fast similarity search.
- Ensures accurate retrieval of relevant sections from large datasets.

### **LLaMA 3**
- Fine-tuned using **LoRA** for parameter-efficient updates.
- Specialized for financial question answering.

### **OpenAI GPT-4o**
- Generates high-quality, natural language responses using retrieved context.
### **Mistral**
- Efficient model specialized for text generation and structured QA tasks, particularly on compact hardware setups.
### **LLaMA 3.2**
- A scalable language model optimized for multi-turn dialogue and large-scale financial datasets, ensuring detailed contextual understanding.

### **Embedding Models**
- Encodes financial text and user queries into a shared semantic space.

---

## Conclusion
This RAG-based financial assistant combines **FAISS** for precise retrieval and **state-of-the-art generative models** for response generation. It transforms complex financial reports into accessible insights, empowering users with easy-to-understand and actionable information.

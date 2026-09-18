import os
import shutil
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from populate_database import split_documents, add_to_chroma
from get_embedding_function import get_embedding_function
from advanced_rag_techniques import AdvancedRAGRetriever  # New import
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from typing import Literal

CHROMA_PATH = "chroma"
TEMP_CHROMA_PATH = "temp_chroma"
UPLOAD_PATH = "uploaded_files"
os.makedirs(UPLOAD_PATH, exist_ok=True)

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question precisely based on the above context: {question}
"""

def save_uploaded_file(uploaded_file):
    file_path = os.path.join(UPLOAD_PATH, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def clear_temp_chroma():
    if os.path.exists(TEMP_CHROMA_PATH):
        shutil.rmtree(TEMP_CHROMA_PATH)
    os.makedirs(TEMP_CHROMA_PATH, exist_ok=True)

def process_pdfs():
    try:
        documents = []
        for file in os.listdir(UPLOAD_PATH):
            if file.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(UPLOAD_PATH, file))  # Use UnstructuredPDFLoader if needed
                documents.extend(loader.load())

        chunks = split_documents(documents)

        db = Chroma(persist_directory=TEMP_CHROMA_PATH, embedding_function=get_embedding_function())
        db.add_documents(chunks)
        db.persist()

        return True, f"Successfully added {len(chunks)} document chunks to the temporary database."
    except Exception as e:
        return False, f"An error occurred: {e}"
    
def get_llm_model(model_name: Literal["mistral", "llama3.2"]):
    return Ollama(model=model_name)


def advanced_query_rag(query_text: str, db_path: str):

    # Add model selection in sidebar
    model_name = st.sidebar.selectbox(
        "Select LLM Model",
        ["mistral", "llama3.2"],
        index=0
    )

    # Initialize session state for retrieval method if not exists
    if 'retrieval_method' not in st.session_state:
        st.session_state.retrieval_method = "Semantic Search"
    
    embedding_function = get_embedding_function()
    rag_retriever = AdvancedRAGRetriever(db_path, embedding_function)

    # Use session state directly without callback
    retrieval_method = st.sidebar.radio(
        "Select Retrieval Method",
        ["Semantic Search", "Hybrid Search", "Query Expansion", 
         "Contextual Compression", "Multi-Query"],
        key="retrieval_method"  # This automatically updates session state
    )

    # Retrieve documents based on selected method
    if retrieval_method == "Semantic Search":
        results = rag_retriever.db.similarity_search_with_score(query_text, k=5)
    elif retrieval_method == "Hybrid Search":
        results = rag_retriever.hybrid_search(query_text)
    elif retrieval_method == "Query Expansion":
        expanded_queries = rag_retriever.query_expansion(query_text)
        results = []
        for query in expanded_queries:
            results.extend(rag_retriever.db.similarity_search_with_score(query, k=2))
        # Remove duplicates while preserving order
        results = list({result[0].metadata['id']: result for result in results}.values())
    elif retrieval_method == "Contextual Compression":
        compressed_docs = rag_retriever.contextual_compression(query_text)
        results = [(doc, 1.0) for doc in compressed_docs]
    elif retrieval_method == "Multi-Query":
        multi_query_docs = rag_retriever.multi_query_retrieval(query_text)
        results = [(doc, 1.0) for doc in multi_query_docs]

    # Optional Re-ranking
    if st.sidebar.checkbox("Enable Re-ranking"):
        docs = [doc for doc, _ in results]
        reranked_docs = rag_retriever.re_rank_results(query_text, docs)
        results = [(doc, 1.0) for doc in reranked_docs]

    # Optional Query Transformation
    if st.sidebar.checkbox("Use Query Transformation"):
        query_text = rag_retriever.query_transformer(query_text)

    # Prepare context
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    
    # Generate response
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = get_llm_model(model_name)
    response_text = model.invoke(prompt)

    # Extract sources
    sources = [doc.metadata.get("source", "Unknown") for doc, _score in results]
    
    return response_text, sources

# Streamlit App modifications
st.title("Advanced RAG-Based Financial Report QA")
st.write("Leverage advanced retrieval techniques for more precise financial report analysis.")

mode = st.radio("Choose an option:", ["Use Stored Financial Reports", "Upload New Financial Report"])

if mode == "Use Stored Financial Reports":
    st.write("Query data already stored in the database.")
    query_text = st.text_input("Enter your question:", key="query")

    # Execute query whenever input changes
    if query_text.strip():
        with st.spinner("Fetching results..."):
            try:
                response, sources = advanced_query_rag(query_text, CHROMA_PATH)
                st.success("Response:")
                st.write(response)
                st.write("Sources:")
                st.write(", ".join(sources))
            except Exception as e:
                st.error(f"An error occurred: {e}")

elif mode == "Upload New Financial Report":
    st.write("Upload PDFs to analyze and query.")
    uploaded_files = st.file_uploader("Upload PDF files:", accept_multiple_files=True, type=["pdf"])

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = save_uploaded_file(uploaded_file)
            st.write(f"✅ Saved: {uploaded_file.name}")

        if st.button("Process PDFs"):
            with st.spinner("Processing uploaded PDFs..."):
                clear_temp_chroma()
                success, message = process_pdfs()
                if success:
                    st.success(message)
                else:
                    st.error(message)

    query_text = st.text_input("Enter your question for the uploaded data:", "")
    if st.button("Submit Query for Uploaded Data"):
        if query_text.strip():
            with st.spinner("Fetching results..."):
                try:
                    response, sources = advanced_query_rag(query_text, TEMP_CHROMA_PATH)
                    st.success("Response:")
                    st.write(response)
                    st.write("Sources:")
                    st.write(", ".join(sources))
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a question!")

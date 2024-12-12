import json
from typing import List, Dict, Any
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from sentence_transformers import CrossEncoder
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms.ollama import Ollama

class AdvancedRAGRetriever:
    def __init__(self, chroma_path: str, embedding_function, model_name: str = "mistral"):
        self.db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)
        self.llm = Ollama(model=model_name)
        self.embedding_function = embedding_function
        
    def query_transformer(self, original_query: str) -> str:
        """Transform the original query to a more precise format."""
        prompt = PromptTemplate(
            input_variables=["question"],
            template="""You are a query transformation expert. Convert the given question 
            into a more precise, specific query that captures the core information need.
            
            Original Query: {question}
            Transformed Query:"""
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        transformed_query = chain.run(original_query).strip()
        return transformed_query

    def query_expansion(self, original_query: str) -> List[str]:
        """Generate multiple related queries to expand search coverage."""
        prompt = PromptTemplate(
            input_variables=["question"],
            template="""You are a query expansion expert. For the given query, 
            generate 3-4 semantically related alternative queries that might help 
            find more comprehensive information.
            
            Original Query: {question}
            Alternative Queries:"""
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        expanded_queries_str = chain.run(original_query).strip()
        
        try:
            expanded_queries = json.loads(expanded_queries_str)
        except:
            expanded_queries = expanded_queries_str.split('\n')
        
        return [original_query] + expanded_queries

    def hybrid_search(self, query: str, k: int = 5):
        """Perform hybrid semantic and keyword search."""
        # Semantic search
        semantic_results = self.db.similarity_search_with_score(query, k=k//2)
        
        # Keyword search (if supported by Chroma)
        try:
            keyword_results = self.db.max_marginal_relevance_search_with_score(query, k=k//2)
        except:
            keyword_results = self.db.similarity_search_with_score(query, k=k//2)
        
        # Combine and deduplicate results
        combined_results = semantic_results + keyword_results
        unique_results = {result[0].metadata['id']: result for result in combined_results}
        return list(unique_results.values())[:k]

    def contextual_compression(self, query: str, k: int = 5):
        """Apply contextual compression to retrieved documents."""
        compressor = LLMChainExtractor.from_llm(self.llm)
        compression_retriever = ContextualCompressionRetriever(
            base_retriever=self.db.as_retriever(search_kwargs={'k': k}),
            document_compressor=compressor
        )
        return compression_retriever.get_relevant_documents(query)

    def multi_query_retrieval(self, query: str, k: int = 5):
        """Retrieve documents using multiple query variations."""
        multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=self.db.as_retriever(search_kwargs={'k': k}),
            llm=self.llm
        )
        return multi_query_retriever.get_relevant_documents(query)

    def re_rank_results(self, query: str, documents: List[Dict[str, Any]], top_k: int = 3):
        """Re-rank retrieved documents using a cross-encoder."""
        cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
        
        # Prepare input for cross-encoder
        pairs = [(query, doc.page_content) for doc in documents]
        scores = cross_encoder.predict(pairs)
        
        # Sort documents by reranking scores
        ranked_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in ranked_docs[:top_k]]
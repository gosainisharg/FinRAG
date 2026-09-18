# from langchain_community.embeddings.ollama import OllamaEmbeddings
# from langchain_community.embeddings.bedrock import BedrockEmbeddings

# def get_embedding_function():
#     # embeddings = BedrockEmbeddings(
#     #     credentials_profile_name="default", region_name="us-east-1"
#     # )
#     embeddings = OllamaEmbeddings(model="nomic-embed-text")
#     return embeddings


from sentence_transformers import SentenceTransformer

def get_embedding_function():
    model = SentenceTransformer('BAAI/bge-large-en-v1.5')
    
    class FinanceEmbeddings:
        def __init__(self, model):
            self.model = model
        
        def embed_documents(self, texts):
            return self.model.encode(texts).tolist()
        
        def embed_query(self, text):
            return self.model.encode(text).tolist()
    
    return FinanceEmbeddings(model)
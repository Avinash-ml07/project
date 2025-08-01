from sentence_transformers import SentenceTransformer
import numpy as np
from pinecone import Pinecone, ServerlessSpec
import uuid

class EmbeddingManager:
    def __init__(self, model_name="all-MiniLM-L6-v2", pinecone_api_key=None):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        # Initialize Pinecone
        if pinecone_api_key:
            self.pc = Pinecone(api_key=pinecone_api_key)
        
    def generate_embeddings(self, texts):
        """Generate embeddings for list of texts"""
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings.tolist()
    
    def create_index(self, index_name):
        """Create Pinecone index"""
        if index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
        return self.pc.Index(index_name)
    
    def upsert_embeddings(self, index, chunks_with_embeddings):
        """Upsert embeddings to Pinecone in batches"""
        batch_size = 100
        
        for i in range(0, len(chunks_with_embeddings), batch_size):
            batch = chunks_with_embeddings[i:i + batch_size]
            vectors = []
            
            for chunk_data in batch:
                vector_id = str(uuid.uuid4())
                vectors.append({
                    'id': vector_id,
                    'values': chunk_data['embedding'],
                    'metadata': {
                        'text': chunk_data['text'],
                        'source': chunk_data.get('source', ''),
                        'chunk_index': chunk_data.get('chunk_index', 0),
                        'tokens': chunk_data.get('tokens', 0)
                    }
                })
            
            index.upsert(vectors=vectors)

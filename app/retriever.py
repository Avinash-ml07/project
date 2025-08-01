from rank_bm25 import BM25Okapi
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class HybridRetriever:
    def __init__(self, embedding_manager, pinecone_index, corpus_texts):
        self.embedding_manager = embedding_manager
        self.index = pinecone_index
        
        # Initialize BM25 for sparse retrieval
        tokenized_corpus = [text.lower().split() for text in corpus_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.corpus_texts = corpus_texts
        
    def dense_retrieval(self, query, top_k=20):
        """Perform dense vector retrieval"""
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]
        
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        return results.matches
    
    def sparse_retrieval(self, query, top_k=20):
        """Perform BM25 sparse retrieval"""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'text': self.corpus_texts[idx],
                'score': scores[idx],
                'index': idx
            })
        
        return results
    
    def hybrid_retrieval(self, query, top_k=10, dense_weight=0.7):
        """Combine dense and sparse retrieval with weighted scoring"""
        # Get results from both methods
        dense_results = self.dense_retrieval(query, top_k * 2)
        sparse_results = self.sparse_retrieval(query, top_k * 2)
        
        # Normalize scores and combine
        combined_results = {}
        
        # Process dense results
        for result in dense_results:
            text = result.metadata['text']
            combined_results[text] = {
                'dense_score': result.score * dense_weight,
                'sparse_score': 0,
                'metadata': result.metadata
            }
        
        # Process sparse results
        max_sparse_score = max([r['score'] for r in sparse_results]) if sparse_results else 1
        for result in sparse_results:
            text = result['text']
            normalized_score = result['score'] / max_sparse_score
            
            if text in combined_results:
                combined_results[text]['sparse_score'] = normalized_score * (1 - dense_weight)
            else:
                combined_results[text] = {
                    'dense_score': 0,
                    'sparse_score': normalized_score * (1 - dense_weight),
                    'metadata': {'text': text}
                }
        
        # Calculate final scores and rank
        final_results = []
        for text, scores in combined_results.items():
            final_score = scores['dense_score'] + scores['sparse_score']
            final_results.append({
                'text': text,
                'score': final_score,
                'metadata': scores['metadata']
            })
        
        # Sort by final score and return top k
        final_results.sort(key=lambda x: x['score'], reverse=True)
        return final_results[:top_k]

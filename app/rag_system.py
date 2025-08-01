from .document_processor import DocumentProcessor
from .pdf_processor import PDFProcessor
from .chunker import IntelligentChunker
from .embedding_manager import EmbeddingManager
from .retriever import HybridRetriever
from .query_processor import QueryProcessor
from .response_generator import ResponseGenerator


class HackRxRAGSystem:
    def __init__(self, pinecone_api_key, model_configs=None):
        # Initialize all components
        self.document_processor = DocumentProcessor()
        self.pdf_processor = PDFProcessor()
        self.chunker = IntelligentChunker()
        self.embedding_manager = EmbeddingManager(pinecone_api_key=pinecone_api_key)
        self.query_processor = QueryProcessor()
        self.response_generator = ResponseGenerator()
        
        # Create vector index
        self.index = self.embedding_manager.create_index("hackrx-documents")
        
        # Initialize retriever (will be set after document processing)
        self.retriever = None
        
    def process_documents(self, document_paths):
        """Process and index all documents"""
        all_chunks = []
        corpus_texts = []
        
        for doc_path in document_paths:
            print(f"Processing {doc_path}...")
            
            # Extract text based on file type
            if doc_path.lower().endswith('.pdf'):
                # Try multiple extraction methods
                text = self.pdf_processor.extract_with_pdfminer(doc_path)
                if not text or len(text.strip()) < 100:
                    structured_content = self.pdf_processor.extract_with_unstructured(doc_path)
                    text = " ".join(structured_content['text'])
            else:
                # For images, use OCR
                try:
                    text = self.document_processor.process_with_textract(doc_path)
                except:
                    text = self.document_processor.process_with_tesseract(doc_path)
            
            if text and len(text.strip()) > 50:
                # Chunk the document
                chunks = self.chunker.semantic_chunking(
                    text, 
                    metadata={'source': doc_path}
                )
                
                # Generate embeddings
                chunk_texts = [chunk['text'] for chunk in chunks]
                embeddings = self.embedding_manager.generate_embeddings(chunk_texts)
                
                # Prepare for indexing
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    chunk_data = {
                        'text': chunk['text'],
                        'embedding': embedding,
                        'source': doc_path,
                        'chunk_index': i,
                        'tokens': chunk['tokens']
                    }
                    all_chunks.append(chunk_data)
                    corpus_texts.append(chunk['text'])
        
        # Index all chunks
        print("Indexing documents...")
        self.embedding_manager.upsert_embeddings(self.index, all_chunks)
        
        # Initialize retriever
        self.retriever = HybridRetriever(
            self.embedding_manager, 
            self.index, 
            corpus_texts
        )
        
        print(f"Successfully processed and indexed {len(all_chunks)} chunks from {len(document_paths)} documents")
    
    def answer_query(self, user_query):
        """Process query and generate answer"""
        if not self.retriever:
            return "Error: No documents have been processed yet."
        
        # Preprocess query
        processed_query = self.query_processor.preprocess_query(user_query)
        expanded_query = self.query_processor.expand_query(processed_query)
        
        print(f"Original query: {user_query}")
        print(f"Processed query: {processed_query}")
        print(f"Expanded query: {expanded_query}")
        
        # Retrieve relevant contexts
        retrieved_contexts = self.retriever.hybrid_retrieval(expanded_query, top_k=5)
        
        if not retrieved_contexts:
            return "I couldn't find relevant information to answer your query."
        
        # Generate response
        raw_response = self.response_generator.generate_response(
            user_query, retrieved_contexts
        )
        
        # Validate response
        validated_response = self.response_generator.validate_response(
            raw_response, retrieved_contexts
        )
        
        # Prepare final response with metadata
        final_response = {
            'answer': validated_response['response'],
            'confidence': validated_response['confidence'],
            'sources': [ctx['metadata'].get('source', 'Unknown') for ctx in retrieved_contexts[:3]],
            'retrieved_chunks': len(retrieved_contexts)
        }
        
        return final_response

# Usage example
def main():
    # Initialize system
    rag_system = HackRxRAGSystem(pinecone_api_key="your-pinecone-api-key")
    
    # Process documents
    document_paths = [
        "insurance_policy_1.pdf",
        "medical_coverage_terms.pdf",
        "exclusions_document.pdf"
    ]
    
    rag_system.process_documents(document_paths)
    
    # Test queries
    test_queries = [
        "46-year-old male, knee surgery in Pune, 3-month-old insurance policy",
        "Is diabetes medication covered under the policy?",
        "What are the waiting periods for pre-existing conditions?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        response = rag_system.answer_query(query)
        print(f"Answer: {response['answer']}")
        print(f"Confidence: {response['confidence']:.2f}")
        print(f"Sources: {', '.join(response['sources'])}")
        print("-" * 80)

if __name__ == "__main__":
    main()

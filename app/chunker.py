import nltk
from nltk.tokenize import sent_tokenize
import tiktoken

class IntelligentChunker:
    def __init__(self, chunk_size=1024, overlap=128, model="text-embedding-3-small"):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        
    def count_tokens(self, text):
        """Count tokens in text"""
        return len(self.encoding.encode(text))
    
    def semantic_chunking(self, text, metadata=None):
        """Create semantically coherent chunks with overlap"""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            # If adding this sentence exceeds chunk size, finalize current chunk
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                chunks.append({
                    'text': current_chunk.strip(),
                    'tokens': current_tokens,
                    'metadata': metadata or {}
                })
                
                # Start new chunk with overlap
                overlap_text = self.create_overlap(current_chunk)
                current_chunk = overlap_text + " " + sentence
                current_tokens = self.count_tokens(current_chunk)
            else:
                current_chunk += " " + sentence
                current_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'tokens': current_tokens,
                'metadata': metadata or {}
            })
        
        return chunks
    
    def create_overlap(self, text):
        """Create overlap from end of previous chunk"""
        sentences = sent_tokenize(text)
        overlap_text = ""
        overlap_tokens = 0
        
        # Add sentences from the end until we reach overlap size
        for sentence in reversed(sentences):
            sentence_tokens = self.count_tokens(sentence)
            if overlap_tokens + sentence_tokens <= self.overlap:
                overlap_text = sentence + " " + overlap_text
                overlap_tokens += sentence_tokens
            else:
                break
        
        return overlap_text.strip()

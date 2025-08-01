from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class ResponseGenerator:
    def __init__(self, model_name="microsoft/DialoGPT-medium"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def create_rag_prompt(self, query, retrieved_contexts, max_context_length=2000):
        """Create a structured prompt for RAG generation"""
        
        # Truncate contexts if too long
        context_text = ""
        current_length = 0
        
        for i, context in enumerate(retrieved_contexts):
            context_snippet = f"Document {i+1}: {context['text'][:500]}...\n\n"
            if current_length + len(context_snippet) > max_context_length:
                break
            context_text += context_snippet
            current_length += len(context_snippet)
        
        prompt = f"""Based on the following policy documents, answer the user's question accurately and cite specific clauses where applicable.

POLICY DOCUMENTS:
{context_text}

USER QUESTION: {query}

INSTRUCTIONS:
1. Provide a clear, direct answer
2. Cite specific document sections that support your answer
3. If information is insufficient, state this clearly
4. Use the format: "According to Document X, [specific clause/information]"

ANSWER:"""
        
        return prompt
    
    def generate_response(self, query, retrieved_contexts, max_length=256):
        """Generate response using retrieved contexts"""
        prompt = self.create_rag_prompt(query, retrieved_contexts)
        
        # Tokenize input
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", 
                                     max_length=1024, truncation=True)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + max_length,
                num_return_sequences=1,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part
        generated_text = response[len(prompt):].strip()
        
        return generated_text
    
    def validate_response(self, response, retrieved_contexts):
        """Validate response against retrieved contexts"""
        # Simple validation - check if response mentions document citations
        has_citations = any(phrase in response.lower() 
                          for phrase in ["according to document", "document", "policy states"])
        
        # Check response length (avoid too short responses)
        adequate_length = len(response.split()) > 10
        
        confidence_score = 0.5
        if has_citations:
            confidence_score += 0.3
        if adequate_length:
            confidence_score += 0.2
        
        return {
            'response': response,
            'confidence': confidence_score,
            'has_citations': has_citations,
            'adequate_length': adequate_length
        }
        
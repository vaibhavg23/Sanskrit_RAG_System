"""
CPU-based LLM Generator for Sanskrit text generation
Using quantized models for efficient CPU inference
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
from typing import List, Dict

class SanskritLLMGenerator:
    def __init__(self, model_name: str = "google/flan-t5-base"):
        print(f"🤖 Initializing LLM Generator...")
        print(f"   Model: {model_name}")
        print(f"   Device: CPU")
        
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.generator = None
    
    def _load_model(self):
        """Load model lazily"""
        if self.model is None:
            print(f"📥 Loading LLM model (this may take 2-3 minutes)...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            
            self.generator = pipeline(
                "text2text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=-1  # Use CPU
            )
            
            print(f"✅ LLM model loaded!\n")
    
    def generate_response(self, query: str, context_docs: List[Dict], max_length: int = 512) -> Dict:
        """Generate response based on query and retrieved context"""
        self._load_model()
        
        # Combine context from retrieved documents
        context = "\n\n".join([doc['content'] for doc in context_docs])
        
        # Limit context length
        if len(context) > 2000:
            context = context[:2000] + "..."
        
        # Create prompt - simplified for better results
        prompt = f"""Context: {context}

Question: {query}

Based on the context above, provide a detailed answer in Sanskrit:"""
        
        # Generate response
        print("🤖 Generating response...")
        response = self.generator(
            prompt,
            max_length=max_length,
            min_length=50,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )
        
        generated_text = response[0]['generated_text']
        
        return {
            'answer': generated_text,
            'query': query,
            'context_used': len(context_docs),
            'model': self.model_name
        }
    
    def generate_simple_response(self, query: str, context_docs: List[Dict]) -> str:
        """Generate a simpler, rule-based response (fallback)"""
        
        # Extract key information from context
        context = "\n".join([doc['content'] for doc in context_docs])
        
        # Simple template-based response
        response = f"""प्रश्नस्य उत्तरम्:

{context[:800]}

[Based on retrieved Sanskrit documents]"""
        
        return response


# Test the generator
if __name__ == "__main__":
    # Simple test without full pipeline
    generator = SanskritLLMGenerator()
    
    test_context = [{
        'content': 'शंखनादः मूर्खः भृत्यः आसीत्। सः सर्वाणि कार्याणि विपरीतरूपेण करोति स्म।',
        'metadata': {'title': 'Test'}
    }]
    
    result = generator.generate_response(
        "मूर्खभृत्यस्य कथा किम्?",
        test_context
    )
    
    print("\n" + "="*60)
    print(f"Query: {result['query']}")
    print(f"Answer: {result['answer']}")
    print("="*60)
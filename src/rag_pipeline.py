"""
Main RAG Pipeline combining all components
"""

from src.logger import PerformanceLogger
from src.vector_store import SanskritVectorStore
from src.document_loader import SanskritDocumentLoader
from src.llm_generator import SanskritLLMGenerator

from src.config import DATA_DIR, MODEL_DIR, TOP_K_DOCS, CHUNK_SIZE
from typing import Dict
import time
import os

class SanskritRAGPipeline:
    def __init__(self, data_dir: str = DATA_DIR):
        print("\n" + "="*70)
        print("🕉️  Sanskrit RAG System - Initialization")
        print("="*70 + "\n")
        
        # Initialize components
        self.loader = SanskritDocumentLoader(data_dir)
        self.vector_store = SanskritVectorStore()
        self.llm = SanskritLLMGenerator()
        self.logger = PerformanceLogger()
        
        # Setup pipeline
        self._setup_pipeline()
    
    def _setup_pipeline(self):
        """Setup the complete RAG pipeline"""
        try:
            # Load documents
            documents = self.loader.load_documents()
            
            if not documents:
                print("❌ No documents found! Please add .txt files to the data/ directory.")
                return
            
            # Chunk documents
            chunks = self.loader.chunk_documents(chunk_size=CHUNK_SIZE)
            
            # Build vector index with caching
            cache_path = os.path.join(MODEL_DIR, "embeddings.pkl")
            self.vector_store.build_index(chunks, cache_path=cache_path)
            
            print("\n" + "="*70)
            print("✅ RAG Pipeline Ready!")
            print("="*70 + "\n")
            
        except Exception as e:
            print(f"\n❌ Error during setup: {e}")
            raise
    
    def query(self, question: str, k: int = TOP_K_DOCS, use_llm: bool = True) -> Dict:
        """Process a query through the RAG pipeline"""
        start_time = time.time()
        
        try:
            print("\n" + "─"*70)
            print(f"🔍 Query: {question}")
            print("─"*70)
            
            # Step 1: Retrieve relevant documents
            print(f"\n📚 Retrieving top {k} relevant documents...")
            retrieved_docs = self.vector_store.search(question, k=k)
            
            print(f"✅ Retrieved {len(retrieved_docs)} documents")
            for i, doc in enumerate(retrieved_docs):
                print(f"   {i+1}. {doc['metadata']['title'][:50]}... (score: {doc['similarity_score']:.3f})")
            
            # Step 2: Generate response
            if use_llm:
                print(f"\n🤖 Generating response with LLM...")
                result = self.llm.generate_response(question, retrieved_docs)
                response_text = result['answer']
            else:
                # Use simple context-based response
                print(f"\n📝 Creating context-based response...")
                response_text = self._create_simple_response(question, retrieved_docs)
            
            elapsed_time = time.time() - start_time
            
            # Log performance
            self.logger.log_query(question, elapsed_time, len(retrieved_docs), True)
            
            print(f"\n✅ Response generated in {elapsed_time:.2f}s")
            
            return {
                'query': question,
                'retrieved_docs': retrieved_docs,
                'response': response_text,
                'latency': elapsed_time,
                'num_docs_retrieved': len(retrieved_docs)
            }
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            self.logger.log_query(question, elapsed_time, 0, False)
            
            print(f"\n❌ Error processing query: {e}")
            
            return {
                'query': question,
                'error': str(e),
                'latency': elapsed_time
            }
    
    def _create_simple_response(self, query: str, retrieved_docs: list) -> str:
        """Create a simple response without LLM"""
        if not retrieved_docs:
            return "क्षम्यताम्, प्रश्नस्य उत्तरं दस्तावेजेषु न प्राप्तम्।"
        
        # Combine relevant context
        context_parts = []
        for doc in retrieved_docs:
            context_parts.append(f"【{doc['metadata']['title']}】\n{doc['content'][:300]}")
        
        response = "प्रश्नस्य आधारेण उत्तरम्:\n\n" + "\n\n".join(context_parts)
        return response
    
    def interactive_mode(self):
        """Run interactive query mode"""
        print("\n" + "="*70)
        print("🎯 Interactive Mode - Type your questions (or 'quit' to exit)")
        print("="*70)
        
        while True:
            print("\n")
            question = input("❓ Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                self.logger.print_statistics()
                break
            
            if not question:
                continue
            
            result = self.query(question, use_llm=False)  # Use simple mode for speed
            
            print("\n" + "─"*70)
            print("💡 Response:")
            print("─"*70)
            if 'error' in result:
                print(f"❌ Error: {result['error']}")
            else:
                print(result['response'])
            print("─"*70)


# Main execution
if __name__ == "__main__":
    try:
        # Initialize pipeline
        rag = SanskritRAGPipeline()
        
        # Example queries
        print("\n" + "="*70)
        print("📋 Running Example Queries")
        print("="*70)
        
        queries = [
            "मूर्खभृत्यस्य कथा किम्?",
            "कालीदासस्य चातुर्यं वर्णयतु",
            "वृद्धायाः कथायां किं घटितम्?"
        ]
        
        for query in queries:
            result = rag.query(query, use_llm=False)  # Use simple mode
            
            print("\n" + "="*70)
            print(f"Query: {result['query']}")
            print("="*70)
            if 'error' not in result:
                print(f"\n{result['response'][:500]}...")
                print(f"\n⏱️  Latency: {result['latency']:.2f}s")
            print("="*70)
        
        # Print statistics
        rag.logger.print_statistics()
        
        # Uncomment to run interactive mode
        # rag.interactive_mode()
        
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
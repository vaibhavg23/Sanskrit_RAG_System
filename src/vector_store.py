"""
Vector Store using FAISS for efficient similarity search
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from src.document_loader import SanskritDocumentLoader
from typing import List, Dict
import os
import pickle

class SanskritVectorStore:
    def __init__(self, model_name: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
        print(f"🔧 Initializing Vector Store...")
        print(f"   Model: {model_name}")
        
        self.model_name = model_name
        self.model = None
        self.index = None
        self.chunks = []
        self.dimension = 384  # MiniLM embedding dimension
    
    def _load_model(self):
        """Load embedding model"""
        if self.model is None:
            print(f"📥 Loading embedding model (this may take a minute)...")
            self.model = SentenceTransformer(self.model_name)
            print(f"✅ Model loaded!\n")
    
    def build_index(self, chunks: List[Dict], cache_path: str = None):
        """Build FAISS index from document chunks"""
        self._load_model()
        
        print("🔨 Building vector index...")
        self.chunks = chunks
        texts = [chunk['content'] for chunk in chunks]
        
        # Check if cached embeddings exist
        if cache_path and os.path.exists(cache_path):
            print(f"📂 Loading cached embeddings from {cache_path}...")
            with open(cache_path, 'rb') as f:
                embeddings = pickle.load(f)
            print("✅ Cached embeddings loaded!")
        else:
            # Generate embeddings
            print(f"🧮 Generating embeddings for {len(texts)} chunks...")
            embeddings = self.model.encode(
                texts, 
                show_progress_bar=True,
                batch_size=16
            )
            
            # Save embeddings to cache
            if cache_path:
                print(f"💾 Saving embeddings to {cache_path}...")
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, 'wb') as f:
                    pickle.dump(embeddings, f)
        
        embeddings = np.array(embeddings).astype('float32')
        
        # Create FAISS index
        print("🔍 Creating FAISS index...")
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings)
        
        print(f"✅ Index built with {self.index.ntotal} vectors\n")
    
    def search(self, query: str, k: int = 3) -> List[Dict]:
        """Search for similar documents"""
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        self._load_model()
        
        # Encode query
        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        
        # Search in FAISS
        distances, indices = self.index.search(query_embedding, k)
        
        # Retrieve results
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.chunks):
                results.append({
                    'content': self.chunks[idx]['content'],
                    'metadata': self.chunks[idx]['metadata'],
                    'distance': float(dist),
                    'similarity_score': 1 / (1 + float(dist))  # Convert distance to similarity
                })
        
        return results
    
    def save_index(self, index_path: str):
        """Save FAISS index to disk"""
        if self.index is not None:
            faiss.write_index(self.index, index_path)
            print(f"💾 Index saved to {index_path}")
    
    def load_index(self, index_path: str):
        """Load FAISS index from disk"""
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            print(f"📂 Index loaded from {index_path}")
        else:
            print(f"❌ Index file not found: {index_path}")


# Test the vector store
if __name__ == "__main__":
    from document_loader import SanskritDocumentLoader
    from config import DATA_DIR, MODEL_DIR
    
    # Load documents
    loader = SanskritDocumentLoader(DATA_DIR)
    documents = loader.load_documents()
    chunks = loader.chunk_documents()
    
    # Build vector store
    vector_store = SanskritVectorStore()
    cache_path = os.path.join(MODEL_DIR, "embeddings.pkl")
    vector_store.build_index(chunks, cache_path=cache_path)
    
    # Test search
    test_query = "मूर्खभृत्यस्य कथा"
    print(f"\n🔍 Test Search: {test_query}")
    results = vector_store.search(test_query, k=2)
    
    for i, result in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"Similarity: {result['similarity_score']:.4f}")
        print(f"Content: {result['content'][:150]}...")
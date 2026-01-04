"""
Document Loader for Sanskrit RAG System
Handles loading and preprocessing of Sanskrit text files
"""

import os
from typing import List, Dict
import re

class SanskritDocumentLoader:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.documents = []
    
    def load_documents(self) -> List[Dict]:
        """Load all text files from data directory"""
        print(f"📚 Loading Sanskrit documents from {self.data_dir}...")
        
        if not os.path.exists(self.data_dir):
            print(f"❌ Error: Data directory '{self.data_dir}' not found!")
            return []
        
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.txt'):
                filepath = os.path.join(self.data_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    self.documents.append({
                        'filename': filename,
                        'content': content,
                        'metadata': self._extract_metadata(filename, content)
                    })
                    print(f"  ✓ Loaded: {filename}")
                except Exception as e:
                    print(f"  ✗ Error loading {filename}: {e}")
        
        print(f"✅ Loaded {len(self.documents)} documents\n")
        return self.documents
    
    def _extract_metadata(self, filename: str, content: str) -> Dict:
        """Extract metadata from document"""
        lines = content.strip().split('\n')
        title = lines[0] if lines else filename
        
        # Remove excessive whitespace from title
        title = ' '.join(title.split())
        
        return {
            'title': title,
            'source': filename,
            'length': len(content),
            'num_lines': len(lines)
        }
    
    def chunk_documents(self, chunk_size: int = 500) -> List[Dict]:
        """Split documents into smaller chunks"""
        print(f"✂️  Chunking documents (size: {chunk_size} chars)...")
        chunks = []
        
        for doc in self.documents:
            content = doc['content']
            
            # Split by paragraphs first (double newline)
            paragraphs = content.split('\n\n')
            
            current_chunk = ""
            chunk_count = 0
            
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                
                # If adding this paragraph exceeds chunk size, save current chunk
                if len(current_chunk) + len(para) > chunk_size and current_chunk:
                    chunks.append({
                        'content': current_chunk.strip(),
                        'metadata': {
                            **doc['metadata'],
                            'chunk_id': chunk_count
                        }
                    })
                    chunk_count += 1
                    current_chunk = para + "\n\n"
                else:
                    current_chunk += para + "\n\n"
            
            # Add remaining content
            if current_chunk.strip():
                chunks.append({
                    'content': current_chunk.strip(),
                    'metadata': {
                        **doc['metadata'],
                        'chunk_id': chunk_count
                    }
                })
        
        print(f"✅ Created {len(chunks)} chunks\n")
        return chunks


# Test the document loader
if __name__ == "__main__":
    from config import DATA_DIR
    
    loader = SanskritDocumentLoader(DATA_DIR)
    documents = loader.load_documents()
    chunks = loader.chunk_documents(chunk_size=500)
    
    print("\n📊 Summary:")
    print(f"Total Documents: {len(documents)}")
    print(f"Total Chunks: {len(chunks)}")
    print(f"\nFirst chunk preview:")
    print(chunks[0]['content'][:200] + "...")
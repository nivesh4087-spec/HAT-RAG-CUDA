import os
from typing import List, Dict

class DocumentProcessor:
    """Handles parsing and chunking of cross-document text sources into fine-grained units."""
    
    def __init__(self, chunk_size: int = 250, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str, doc_id: str) -> List[Dict]:
        """Splits a single document text into overlapping chunks."""
        words = text.split()
        chunks = []
        
        start = 0
        chunk_idx = 0
        while start < len(words):
            end = start + self.chunk_size
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)
            
            chunks.append({
                "chunk_id": f"{doc_id}_chunk_{chunk_idx}",
                "doc_id": doc_id,
                "text": chunk_text,
                "token_count": len(chunk_words)
            })
            
            chunk_idx += 1
            start += (self.chunk_size - self.chunk_overlap)
            
        return chunks

    def process_documents(self, raw_documents: Dict[str, str]) -> List[Dict]:
        """Processes a collection of raw text documents into chunks."""
        all_chunks = []
        for doc_id, content in raw_documents.items():
            doc_chunks = self.chunk_text(content, doc_id)
            all_chunks.extend(doc_chunks)
        return all_chunks

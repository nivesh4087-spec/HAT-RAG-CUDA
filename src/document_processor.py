"""
Document Ingestion & Semantic Chunking Engine for HAT-RAG
Author: Nivesh Jain (Vishwakarma Institute of Technology, Pune)
"""

import os
import re
import json
import hashlib
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

try:
    import pypdf
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False


class DocumentProcessor:
    """
    Advanced Document Ingestion & Chunking Processor supporting multi-format parsing,
    sentence boundary alignment, section tracking, SHA-256 hashing, and provenance metadata.
    """

    def __init__(
        self,
        chunk_size: int = 250,
        chunk_overlap: int = 50,
        preserve_headings: bool = True
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.preserve_headings = preserve_headings

    def parse_pdf_file(self, file_path: str) -> Dict[str, Any]:
        """Parses PDF document and extracts full text along with page breakdown."""
        if not HAS_PYPDF:
            raise ImportError("pypdf is required to parse PDF files.")

        reader = pypdf.PdfReader(file_path)
        pages_text = []
        full_text_list = []

        for idx, page in enumerate(reader.pages):
            page_content = page.extract_text() or ""
            pages_text.append({"page": idx + 1, "text": page_content})
            full_text_list.append(page_content)

        return {
            "doc_id": os.path.basename(file_path),
            "file_type": "pdf",
            "page_count": len(reader.pages),
            "full_text": "\n\n".join(full_text_list),
            "pages": pages_text
        }

    def _split_into_sentences(self, text: str) -> List[str]:
        """Splits text into sentences using punctuation boundaries."""
        sentence_endings = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+')
        sentences = sentence_endings.split(text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def chunk_text(
        self,
        text: str,
        doc_id: str,
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Splits a single document text into sentence-aligned overlapping chunks.
        Extracts section headings and computes SHA-256 content hash for chunk tracking.
        """
        sentences = self._split_into_sentences(text)
        if not sentences:
            words = text.split()
            if not words:
                return []
            sentences = [text]

        chunks = []
        current_words: List[str] = []
        current_sentences: List[str] = []
        current_section = "General / Overview"
        chunk_idx = 0

        heading_pattern = re.compile(r'^(?:#+|\d+\.|\b(?:Section|Chapter|Abstract|Introduction|Methods|Results|Discussion|Conclusion)\b)', re.IGNORECASE)

        for sentence in sentences:
            if heading_pattern.match(sentence) and len(sentence.split()) < 15:
                current_section = sentence.strip("#").strip()

            sentence_words = sentence.split()
            
            if len(current_words) + len(sentence_words) > self.chunk_size and current_words:
                chunk_text_str = " ".join(current_sentences)
                chunk_hash = hashlib.sha256(chunk_text_str.encode("utf-8")).hexdigest()[:12]
                
                chunk_meta = {
                    "chunk_id": f"{doc_id}_c{chunk_idx}_{chunk_hash}",
                    "doc_id": doc_id,
                    "chunk_index": chunk_idx,
                    "section": current_section,
                    "text": chunk_text_str,
                    "token_count": len(current_words),
                    "hash": chunk_hash
                }
                if extra_metadata:
                    chunk_meta.update(extra_metadata)
                    
                chunks.append(chunk_meta)
                chunk_idx += 1

                overlap_words: List[str] = []
                overlap_sentences: List[str] = []
                for s in reversed(current_sentences):
                    s_w = s.split()
                    if len(overlap_words) + len(s_w) <= self.chunk_overlap:
                        overlap_words = s_w + overlap_words
                        overlap_sentences.insert(0, s)
                    else:
                        break

                current_words = overlap_words + sentence_words
                current_sentences = overlap_sentences + [sentence]
            else:
                current_words.extend(sentence_words)
                current_sentences.append(sentence)

        if current_sentences:
            chunk_text_str = " ".join(current_sentences)
            chunk_hash = hashlib.sha256(chunk_text_str.encode("utf-8")).hexdigest()[:12]
            
            chunk_meta = {
                "chunk_id": f"{doc_id}_c{chunk_idx}_{chunk_hash}",
                "doc_id": doc_id,
                "chunk_index": chunk_idx,
                "section": current_section,
                "text": chunk_text_str,
                "token_count": len(current_words),
                "hash": chunk_hash
            }
            if extra_metadata:
                chunk_meta.update(extra_metadata)
            chunks.append(chunk_meta)

        return chunks

    def process_documents(self, raw_documents: Dict[str, str]) -> List[Dict[str, Any]]:
        """Processes a dict of document_id -> text into chunks across all documents."""
        all_chunks = []
        for doc_id, content in raw_documents.items():
            doc_chunks = self.chunk_text(content, doc_id)
            all_chunks.extend(doc_chunks)
        logger.info(f"Processed {len(raw_documents)} documents into {len(all_chunks)} semantic chunks.")
        return all_chunks


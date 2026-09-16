"""
Abstract Summarization Engine for HAT-RAG
Author: Nivesh Jain (Vishwakarma Institute of Technology, Pune)
"""

import re
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class AbstractSummarizer:
    """
    Summarization Engine for generating higher-level abstract summaries at intermediate
    and root levels of the Hierarchical Abstract Tree (HAT).
    Supports HuggingFace BART/T5 models with TF-IDF/Extractive sentence ranking fallback.
    """

    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        self.model_name = model_name
        self.summarizer_pipeline = None

        if HAS_TRANSFORMERS:
            try:
                logger.info(f"Attempting to initialize summarization pipeline '{model_name}'...")
                self.summarizer_pipeline = pipeline("summarization", model=model_name)
            except Exception as e:
                logger.warning(f"Could not load HuggingFace pipeline '{model_name}': {e}. Using extractive summarizer.")

    def extract_key_entities(self, text: str) -> List[str]:
        """Extracts key technical terms and capitalized entities."""
        words = re.findall(r'\b[A-Z][a-zA-Z0-9_-]+\b', text)
        unique_entities = sorted(list(set(words)))
        return unique_entities[:10]

    def summarize_cluster(self, texts: List[str], max_length: int = 120) -> str:
        """Generates a cohesive abstract summary representing a cluster of child text passages."""
        if not texts:
            return "ABSTRACT SUMMARY: Empty Passage Cluster"

        combined_text = " ".join(texts)
        entities = self.extract_key_entities(combined_text)
        entity_str = f" [Entities: {', '.join(entities[:5])}]" if entities else ""

        if self.summarizer_pipeline is not None and len(combined_text.split()) > 30:
            try:
                summary = self.summarizer_pipeline(
                    combined_text[:1024],
                    max_length=max_length,
                    min_length=20,
                    do_sample=False
                )
                return f"ABSTRACT SUMMARY: {summary[0]['summary_text']}{entity_str}"
            except Exception as e:
                logger.warning(f"Summarizer pipeline inference failed: {e}. Using extractive fallback.")

        # Extractive Sentence Ranking Fallback
        sentences = [s.strip() for t in texts for s in t.split('.') if len(s.strip()) > 15]
        if not sentences:
            return f"ABSTRACT SUMMARY: {combined_text[:150]}...{entity_str}"

        # Frequency-based sentence ranking
        word_freq: Dict[str, int] = {}
        for s in sentences:
            for w in s.lower().split():
                if len(w) > 3:
                    word_freq[w] = word_freq.get(w, 0) + 1

        scored_sentences = []
        for s in sentences:
            score = sum(word_freq.get(w, 0) for w in s.lower().split() if len(w) > 3)
            scored_sentences.append((score, s))

        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        top_sentences = [s for _, s in scored_sentences[:min(3, len(scored_sentences))]]
        
        abstract_body = " | ".join(top_sentences)
        return f"ABSTRACT SUMMARY: {abstract_body}{entity_str}"


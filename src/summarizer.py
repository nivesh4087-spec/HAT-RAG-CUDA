import logging
from typing import List

logger = logging.getLogger(__name__)

# HuggingFace Transformers optional import
try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

class AbstractSummarizer:
    """
    Summarization Engine for abstract node generation in HAT-RAG.
    Supports Hugging Face pipelines with extractive fallback.
    """
    
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        self.model_name = model_name
        self.summarizer_pipeline = None
        
        if HAS_TRANSFORMERS:
            try:
                logger.info(f"Attempting to load summarization model: {model_name}")
                self.summarizer_pipeline = pipeline("summarization", model=model_name)
            except Exception as e:
                logger.warning(f"Could not load HuggingFace pipeline '{model_name}': {e}. Using extractive summarizer.")

    def summarize_cluster(self, texts: List[str], max_length: int = 100) -> str:
        """Generates an abstract summary for a cluster of child text passages."""
        if not texts:
            return "Empty Cluster Summary"
            
        combined_text = " ".join(texts)
        
        if self.summarizer_pipeline is not None and len(combined_text.split()) > 30:
            try:
                summary = self.summarizer_pipeline(
                    combined_text,
                    max_length=max_length,
                    min_length=20,
                    do_sample=False
                )
                return "ABSTRACT SUMMARY: " + summary[0]['summary_text']
            except Exception as e:
                logger.warning(f"Summarizer pipeline error: {e}")
                
        # Extractive Summary Fallback
        sentences = [s.strip() for t in texts for s in t.split('.') if len(s.strip()) > 10]
        if not sentences:
            return "ABSTRACT SUMMARY: " + combined_text[:120] + "..."
            
        top_sentences = sentences[:min(3, len(sentences))]
        return "ABSTRACT SUMMARY: " + " | ".join(top_sentences)

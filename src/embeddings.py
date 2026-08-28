import math
import random
import logging
from typing import List, Union

logger = logging.getLogger(__name__)

# Sentence Transformers optional import
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

class EmbeddingEngine:
    """
    Embedding Engine for Hierarchical Abstract Tree.
    Supports SentenceTransformers (e.g. all-MiniLM-L6-v2) on CUDA GPU
    with intelligent fallback to hash-vector math when model isn't downloaded.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dim: int = 64):
        self.model_name = model_name
        self.dim = dim
        self.st_model = None
        
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                logger.info(f"Loading SentenceTransformer model: {model_name}")
                self.st_model = SentenceTransformer(model_name)
            except Exception as e:
                logger.warning(f"Could not load SentenceTransformer '{model_name}': {e}. Using vector hash fallback.")

    def encode(self, texts: Union[str, List[str]]):
        """Encodes text or list of texts into embedding vectors."""
        is_single = isinstance(texts, str)
        text_list = [texts] if is_single else texts
        
        if self.st_model is not None:
            try:
                embeddings = self.st_model.encode(text_list, convert_to_numpy=True)
                return embeddings[0] if is_single else embeddings
            except Exception as e:
                logger.warning(f"Embedding inference failed: {e}. Falling back.")
                
        # Hash Vector Fallback
        results = []
        for text in text_list:
            seed_val = abs(hash(text)) % (2**31)
            rng = random.Random(seed_val)
            vec = [rng.uniform(-1.0, 1.0) for _ in range(self.dim)]
            norm_val = math.sqrt(sum(x*x for x in vec)) + 1e-9
            vec = [x / norm_val for x in vec]
            results.append(np.array(vec) if HAS_NUMPY else vec)
            
        return results[0] if is_single else results

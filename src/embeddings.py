"""
Dense Vector Embedding Engine for HAT-RAG
Author: Nivesh Jain (Vishwakarma Institute of Technology, Pune)
"""

import math
import random
import logging
from typing import List, Union, Dict, Any, Optional

logger = logging.getLogger(__name__)

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

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class EmbeddingEngine:
    """
    Production Dense Vector Embedding Engine with SentenceTransformers support,
    CUDA GPU acceleration, batch inference, L2 normalization, and caching.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dim: int = 64):
        self.model_name = model_name
        self.dim = dim
        self.st_model = None
        self.cache: Dict[str, Any] = {}

        if HAS_SENTENCE_TRANSFORMERS:
            try:
                device = "cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu"
                logger.info(f"Loading SentenceTransformer '{model_name}' on device '{device}'...")
                self.st_model = SentenceTransformer(model_name, device=device)
            except Exception as e:
                logger.warning(f"Could not initialize SentenceTransformer '{model_name}': {e}. Using hash fallback engine.")

    def _hash_vector_encode(self, text: str):
        """Generates deterministic pseudo-semantic vector representation for zero-dependency execution."""
        seed_val = abs(hash(text)) % (2**31)
        rng = random.Random(seed_val)
        
        words = text.lower().split()
        vec = [0.0] * self.dim
        
        for w in words:
            w_seed = abs(hash(w)) % (2**31)
            w_rng = random.Random(w_seed)
            for idx in range(self.dim):
                vec[idx] += w_rng.uniform(-1.0, 1.0)

        norm_val = math.sqrt(sum(x * x for x in vec)) + 1e-9
        norm_vec = [x / norm_val for x in vec]
        return np.array(norm_vec, dtype=np.float32) if HAS_NUMPY else norm_vec

    def encode(self, texts: Union[str, List[str]], use_cache: bool = True):
        """Encodes text or list of texts into L2-normalized vector embeddings."""
        is_single = isinstance(texts, str)
        text_list = [texts] if is_single else texts

        # Check cache
        if use_cache:
            uncached = [t for t in text_list if t not in self.cache]
        else:
            uncached = text_list

        if uncached:
            if self.st_model is not None:
                try:
                    embeddings = self.st_model.encode(
                        uncached,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                        normalize_embeddings=True
                    )
                    for t, emb in zip(uncached, embeddings):
                        self.cache[t] = emb
                except Exception as e:
                    logger.warning(f"SentenceTransformer encoding failed: {e}. Falling back to hash encoding.")
                    for t in uncached:
                        self.cache[t] = self._hash_vector_encode(t)
            else:
                for t in uncached:
                    self.cache[t] = self._hash_vector_encode(t)

        results = [self.cache[t] for t in text_list]
        return results[0] if is_single else results


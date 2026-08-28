import math
import logging

logger = logging.getLogger(__name__)

# Optional torch import
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Optional numpy import
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

def check_cuda_availability() -> dict:
    """Checks for CUDA GPU availability and returns hardware details."""
    if HAS_TORCH:
        cuda_available = torch.cuda.is_available()
        device_info = {
            "cuda_available": cuda_available,
            "device_count": torch.cuda.device_count() if cuda_available else 0,
            "device_name": torch.cuda.get_device_name(0) if cuda_available else "CPU (Fallback)",
            "cuda_version": torch.version.cuda if cuda_available else None
        }
    else:
        device_info = {
            "cuda_available": False,
            "device_count": 0,
            "device_name": "CPU Vector Engine (Native Pure-Python Math Engine)",
            "cuda_version": None
        }
    logger.info(f"GPU Hardware Status: {device_info}")
    return device_info

def gpu_batch_cosine_similarity(query_embed, doc_embeds):
    """
    Computes cosine similarity between a single query embedding and a batch of document embeddings
    using CUDA PyTorch tensor acceleration if available, otherwise pure math / NumPy vector matrix ops.
    """
    if HAS_TORCH and torch.cuda.is_available():
        device = torch.device("cuda")
        q_tensor = torch.tensor(query_embed, dtype=torch.float32, device=device)
        d_tensor = torch.tensor(doc_embeds, dtype=torch.float32, device=device)
        
        q_norm = torch.nn.functional.normalize(q_tensor, p=2, dim=-1)
        d_norm = torch.nn.functional.normalize(d_tensor, p=2, dim=-1)
        
        if q_norm.dim() == 1:
            q_norm = q_norm.unsqueeze(0)
            
        sims = torch.mm(q_norm, d_norm.transpose(0, 1)).squeeze(0)
        return sims.cpu().numpy()
    elif HAS_NUMPY:
        q_arr = np.array(query_embed)
        d_arr = np.array(doc_embeds)
        q_norm = q_arr / (np.linalg.norm(q_arr) + 1e-9)
        d_norm = d_arr / (np.linalg.norm(d_arr, axis=1, keepdims=True) + 1e-9)
        if q_norm.ndim == 1:
            q_norm = np.expand_dims(q_norm, axis=0)
        return np.dot(q_norm, d_norm.T).squeeze(0)
    else:
        # Pure Python implementation fallback
        def norm(vec):
            return math.sqrt(sum(x*x for x in vec)) + 1e-9
            
        def dot(v1, v2):
            return sum(x*y for x, y in zip(v1, v2))
            
        q_n = norm(query_embed)
        sims = []
        for d in doc_embeds:
            d_n = norm(d)
            sims.append(dot(query_embed, d) / (q_n * d_n))
        return sims



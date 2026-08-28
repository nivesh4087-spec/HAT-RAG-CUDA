import time
import math
import logging
from typing import Dict, List, Tuple, Any

logger = logging.getLogger(__name__)

# Optional PyTorch & CUDA imports
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Optional NumPy import
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

def check_cuda_availability() -> Dict[str, Any]:
    """Checks for CUDA GPU availability and returns complete hardware details."""
    if HAS_TORCH:
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            device_count = torch.cuda.device_count()
            total_memory_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
            allocated_gb = round(torch.cuda.memory_allocated(0) / (1024**3), 2)
            reserved_gb = round(torch.cuda.memory_reserved(0) / (1024**3), 2)
            cuda_version = torch.version.cuda
        else:
            device_name = "CPU (PyTorch Active)"
            device_count = 0
            total_memory_gb = 0.0
            allocated_gb = 0.0
            reserved_gb = 0.0
            cuda_version = None
            
        return {
            "cuda_available": cuda_available,
            "device_count": device_count,
            "device_name": device_name,
            "total_memory_gb": total_memory_gb,
            "allocated_gb": allocated_gb,
            "reserved_gb": reserved_gb,
            "cuda_version": cuda_version,
            "backend": "PyTorch CUDA" if cuda_available else "PyTorch CPU"
        }
    else:
        return {
            "cuda_available": False,
            "device_count": 0,
            "device_name": "CPU Vector Engine (Native Math Fallback)",
            "total_memory_gb": 0.0,
            "allocated_gb": 0.0,
            "reserved_gb": 0.0,
            "cuda_version": None,
            "backend": "Pure Math Fallback"
        }

def get_target_device():
    """Returns PyTorch CUDA device if available, otherwise CPU."""
    if HAS_TORCH and torch.cuda.is_available():
        return torch.device("cuda")
    elif HAS_TORCH:
        return torch.device("cpu")
    return "cpu"

def gpu_batch_cosine_similarity(query_embed, doc_embeds):
    """
    Computes cosine similarity between a query vector and doc embeddings batch.
    Uses CUDA PyTorch tensor acceleration if available, otherwise NumPy/Math.
    """
    start_time = time.perf_counter()
    
    if HAS_TORCH and torch.cuda.is_available():
        device = torch.device("cuda")
        q_tensor = torch.tensor(query_embed, dtype=torch.float32, device=device)
        d_tensor = torch.tensor(doc_embeds, dtype=torch.float32, device=device)
        
        q_norm = torch.nn.functional.normalize(q_tensor, p=2, dim=-1)
        d_norm = torch.nn.functional.normalize(d_tensor, p=2, dim=-1)
        
        if q_norm.dim() == 1:
            q_norm = q_norm.unsqueeze(0)
            
        sims = torch.mm(q_norm, d_norm.transpose(0, 1)).squeeze(0)
        result = sims.cpu().numpy()
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return result, elapsed_ms
    elif HAS_NUMPY:
        q_arr = np.array(query_embed)
        d_arr = np.array(doc_embeds)
        q_norm = q_arr / (np.linalg.norm(q_arr) + 1e-9)
        d_norm = d_arr / (np.linalg.norm(d_arr, axis=1, keepdims=True) + 1e-9)
        if q_norm.ndim == 1:
            q_norm = np.expand_dims(q_norm, axis=0)
        result = np.dot(q_norm, d_norm.T).squeeze(0)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return result, elapsed_ms
    else:
        def norm(vec):
            return math.sqrt(sum(x*x for x in vec)) + 1e-9
            
        def dot(v1, v2):
            return sum(x*y for x, y in zip(v1, v2))
            
        q_n = norm(query_embed)
        sims = []
        for d in doc_embeds:
            d_n = norm(d)
            sims.append(dot(query_embed, d) / (q_n * d_n))
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return sims, elapsed_ms

def benchmark_cuda_vs_cpu(vector_count: int = 1000, dim: int = 384) -> Dict[str, Any]:
    """Runs a benchmarking test comparing vector dot product execution speed."""
    if HAS_NUMPY:
        np.random.seed(42)
        q = np.random.randn(dim)
        docs = np.random.randn(vector_count, dim)
    else:
        import random
        rng = random.Random(42)
        q = [rng.uniform(-1, 1) for _ in range(dim)]
        docs = [[rng.uniform(-1, 1) for _ in range(dim)] for _ in range(vector_count)]
        
    _, time_taken_ms = gpu_batch_cosine_similarity(q, docs)
    hardware_status = check_cuda_availability()
    
    return {
        "vector_count": vector_count,
        "vector_dim": dim,
        "execution_time_ms": round(time_taken_ms, 3),
        "device_used": hardware_status["device_name"],
        "cuda_active": hardware_status["cuda_available"]
    }




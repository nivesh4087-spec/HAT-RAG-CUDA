import pytest
from hat_rag.src.cuda_utils import check_cuda_availability, gpu_batch_cosine_similarity, benchmark_cuda_vs_cpu

def test_cuda_status():
    status = check_cuda_availability()
    assert isinstance(status, dict)
    assert "cuda_available" in status
    assert "device_name" in status

def test_cosine_similarity():
    q = [1.0, 0.0, 0.0]
    docs = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    sims, elapsed = gpu_batch_cosine_similarity(q, docs)
    assert len(sims) == 2
    assert sims[0] > sims[1]

def test_benchmark():
    res = benchmark_cuda_vs_cpu(vector_count=50, dim=16)
    assert res["vector_count"] == 50
    assert "execution_time_ms" in res

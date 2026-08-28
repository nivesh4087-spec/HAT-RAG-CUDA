import os
import logging
from typing import Dict, List, Any, Optional

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from hat_rag.src.cuda_utils import check_cuda_availability, benchmark_cuda_vs_cpu
from hat_rag.src.document_processor import DocumentProcessor
from hat_rag.src.hierarchical_tree import HierarchicalAbstractTree
from hat_rag.src.retriever import HierarchicalRetriever
from hat_rag.src.generator import HATGenerator
from hat_rag.src.evaluator import RAGEvaluator

logger = logging.getLogger(__name__)

# Core Global Engine State
global_tree = None

def init_default_engine():
    global global_tree
    processor = DocumentProcessor()
    docs = {
        "cooling.txt": "Ceiling fans function by creating a wind chill factor. High speed rotation produces downward airflow, reducing temperature in halls by up to 4C.",
        "maintenance.txt": "Tool wear in stamping fan blades causes motor alignment errors. Daily torque inspections prevent excessive vibration, bearing wear, and failure.",
        "energy.txt": "Brushless DC motor (BLDC) ceiling fans consume up to 65% less power compared to standard induction motors with smart microcontroller adaptive speed control."
    }
    chunks = processor.process_documents(docs)
    global_tree = HierarchicalAbstractTree(max_levels=2, clusters_per_level=2)
    global_tree.build_tree(chunks)

init_default_engine()

if HAS_FASTAPI:
    app = FastAPI(
        title="HAT-RAG CUDA REST API Service",
        description="Hierarchical Abstract Tree for Cross-Document RAG Accelerated with NVIDIA CUDA",
        version="1.0.0"
    )

    class QueryRequest(BaseModel):
        query: str
        top_k: Optional[int] = 3

    class IngestRequest(BaseModel):
        documents: Dict[str, str]

    @app.get("/health")
    def health_check():
        return {
            "status": "online",
            "cuda_status": check_cuda_availability(),
            "nodes_indexed": len(global_tree.nodes) if global_tree else 0
        }

    @app.post("/ingest")
    def ingest_documents(req: IngestRequest):
        global global_tree
        processor = DocumentProcessor()
        chunks = processor.process_documents(req.documents)
        global_tree = HierarchicalAbstractTree()
        global_tree.build_tree(chunks)
        return {
            "message": "Documents indexed successfully into Hierarchical Abstract Tree",
            "documents_count": len(req.documents),
            "chunks_created": len(chunks),
            "total_nodes": len(global_tree.nodes)
        }

    @app.post("/query")
    def query_rag(req: QueryRequest):
        if not global_tree:
            raise HTTPException(status_code=400, detail="Tree not initialized")
            
        retriever = HierarchicalRetriever(global_tree)
        nodes, stats = retriever.top_down_search(req.query, top_k=req.top_k)
        
        generator = HATGenerator()
        res = generator.generate_response(req.query, nodes, search_stats=stats)
        return res

    @app.post("/benchmark")
    def run_benchmark(req: QueryRequest):
        if not global_tree:
            raise HTTPException(status_code=400, detail="Tree not initialized")
            
        evaluator = RAGEvaluator(global_tree)
        return evaluator.evaluate_query(req.query, top_k=req.top_k)

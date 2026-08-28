"""
Launcher for HAT-RAG System
Command Line Interface & Web Service Entry Point
"""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def main():
    parser = argparse.ArgumentParser(description="HAT-RAG: CUDA-Accelerated Hierarchical Abstract Tree Engine")
    parser.add_argument("--mode", choices=["demo", "app", "api", "test"], default="demo", help="Execution mode")
    parser.add_argument("--port", type=int, default=8000, help="Port for API service")

    args = parser.parse_args()

    if args.mode == "demo":
        from hat_rag.demo_hat_rag import main as run_demo
        run_demo()
    elif args.mode == "app":
        import os
        os.system("streamlit run hat_rag/app.py")
    elif args.mode == "api":
        import uvicorn
        uvicorn.run("hat_rag.src.api:app", host="0.0.0.0", port=args.port, reload=True)
    elif args.mode == "test":
        from hat_rag.run_tests import TestHATRAG
        import unittest
        unittest.main(module="hat_rag.run_tests", argv=["first-arg-is-ignored"])

if __name__ == "__main__":
    main()

import sys
import os

# Ensure project root is in sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src import cuda_utils
from src import document_processor
from src import embeddings
from src import summarizer
from src import hierarchical_tree
from src import retriever
from src import generator
from src import evaluator
from src import multi_approach
from src import api

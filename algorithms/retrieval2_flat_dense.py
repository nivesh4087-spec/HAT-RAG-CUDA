import argparse
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from retrieval1_hat_beam_traversal import (
    DEFAULT_TREE,
    DEMO_QUERIES,
    RetrievalResult,
    TreeIndex,
    print_result,
)


class FlatDenseRetriever:
    name = "Flat dense RAG"
    short = "Flat dense"

    def __init__(
        self,
        index: TreeIndex,
        mode: str = "dense",
        keyword_weight: float = 0.15,
        diversity_weight: float = 0.0,
        min_score: float = 0.0,
    ):
        self.index = index
        self.mode = mode
        self.keyword_weight = keyword_weight
        self.diversity_weight = diversity_weight
        self.min_score = min_score

        self.candidate_indices = index.leaf_idx
        self.leaf_embeddings = index.E[self.candidate_indices]
        self.total_candidates = len(self.candidate_indices)
        self.leaf_texts = [index.text(i) for i in self.candidate_indices]
        self.leaf_sources = [index.doc[i] for i in self.candidate_indices]

    def compute_dense_similarities(self, query_vector: np.ndarray) -> np.ndarray:
        dot_products = np.dot(self.leaf_embeddings, query_vector)
        embedding_norms = np.linalg.norm(self.leaf_embeddings, axis=1)
        query_norm = np.linalg.norm(query_vector)
        return dot_products / (embedding_norms * query_norm + 1e-9)

    def compute_keyword_overlaps(self, query_text: str) -> np.ndarray:
        cleaned_query = query_text.lower()
        query_words = set(re.findall(r"\b\w{3,}\b", cleaned_query))
        if not query_words:
            return np.zeros(self.total_candidates, dtype=np.float32)

        overlap_scores = np.zeros(self.total_candidates, dtype=np.float32)
        for idx, text in enumerate(self.leaf_texts):
            text_words = set(re.findall(r"\b\w{3,}\b", text.lower()))
            common_words = query_words.intersection(text_words)
            overlap_scores[idx] = len(common_words) / len(query_words)

        return overlap_scores

    def compute_diversity_rerank(
        self,
        base_scores: np.ndarray,
        top_k: int,
        lambda_param: float = 0.65,
    ) -> Tuple[List[int], List[float]]:
        unselected = list(range(self.total_candidates))
        selected_positions: List[int] = []
        selected_scores: List[float] = []

        first_pick = int(np.argmax(base_scores))
        selected_positions.append(first_pick)
        selected_scores.append(float(base_scores[first_pick]))
        unselected.remove(first_pick)

        while len(selected_positions) < min(top_k, self.total_candidates) and unselected:
            best_mmr_score = -float("inf")
            best_candidate = -1

            for candidate in unselected:
                relevance = base_scores[candidate]
                candidate_vector = self.leaf_embeddings[candidate]

                max_similarity_to_selected = max(
                    float(np.dot(candidate_vector, self.leaf_embeddings[sel]))
                    for sel in selected_positions
                )

                mmr_score = lambda_param * relevance - (1.0 - lambda_param) * max_similarity_to_selected
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_candidate = candidate

            selected_positions.append(best_candidate)
            selected_scores.append(float(base_scores[best_candidate]))
            unselected.remove(best_candidate)

        return selected_positions, selected_scores

    def compute_score_statistics(self, scores: np.ndarray) -> Dict[str, float]:
        sorted_scores = np.sort(scores)[::-1]
        score_margin = float(sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) > 1 else 0.0
        return {
            "max": float(np.max(scores)),
            "min": float(np.min(scores)),
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "margin": score_margin,
        }

    def retrieve(self, query: str, k: int = 5) -> RetrievalResult:
        query_vector = self.index.encode_query(query)
        return self.search(query_vector, k=k, query=query)

    def search(self, query_vector: np.ndarray, k: int = 5, query: str = "") -> RetrievalResult:
        start_time = time.perf_counter()

        dense_scores = self.compute_dense_similarities(query_vector)
        final_scores = dense_scores.copy()
        if self.mode == "hybrid" and query:
            keyword_scores = self.compute_keyword_overlaps(query)
            final_scores = (1.0 - self.keyword_weight) * dense_scores + self.keyword_weight * keyword_scores

        if self.mode == "mmr" or self.diversity_weight > 0.0:
            diversity_lambda = 1.0 - self.diversity_weight if self.diversity_weight > 0.0 else 0.7
            ranked_local_indices, ranked_scores = self.compute_diversity_rerank(
                final_scores, top_k=k, lambda_param=diversity_lambda
            )
        else:
            sorted_order = np.argsort(-final_scores)
            ranked_local_indices = [int(i) for i in sorted_order[:k]]
            ranked_scores = [float(final_scores[i]) for i in ranked_local_indices]

        if self.min_score > 0.0:
            filtered_indices: List[int] = []
            filtered_scores: List[float] = []
            for local_idx, score in zip(ranked_local_indices, ranked_scores):
                if score >= self.min_score:
                    filtered_indices.append(local_idx)
                    filtered_scores.append(score)
            ranked_local_indices = filtered_indices
            ranked_scores = filtered_scores

        chosen_node_ids = [self.index.ids[self.candidate_indices[i]] for i in ranked_local_indices]
        elapsed_time = time.perf_counter() - start_time

        return RetrievalResult(
            method=self.short,
            query=query,
            node_ids=chosen_node_ids,
            scores=ranked_scores,
            nodes_scored=self.total_candidates,
            seconds=elapsed_time,
            trace=[],
        )


def _cli():
    parser = argparse.ArgumentParser(description="Retrieval 2: Flat Dense Retrieval Engine")
    parser.add_argument("--tree", default=str(DEFAULT_TREE))
    parser.add_argument("--query", action="append")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mode", choices=["dense", "hybrid", "mmr"], default="dense")
    parser.add_argument("--keyword-weight", type=float, default=0.2)
    parser.add_argument("--diversity-weight", type=float, default=0.15)
    parser.add_argument("--min-score", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = _cli()
    print("=" * 100)
    print(f" RETRIEVAL 2: FLAT DENSE RETRIEVAL ENGINE [Mode: {args.mode.upper()}]")
    print("=" * 100)
    index = TreeIndex.from_json(args.tree, device=args.device)
    print(f"[Index] {index.n_leaves} leaf chunks | {index.embedder.banner()}")
    retriever = FlatDenseRetriever(
        index,
        mode=args.mode,
        keyword_weight=args.keyword_weight,
        diversity_weight=args.diversity_weight,
        min_score=args.min_score,
    )
    for q in args.query or DEMO_QUERIES:
        print_result(index, retriever.retrieve(q, k=args.k))
    print("=" * 100)

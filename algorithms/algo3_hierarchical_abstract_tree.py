"""
ALGORITHM 3: HIERARCHICAL ABSTRACT TREE (HAT) CONSTRUCTION

Offline knowledge construction. Consumes the chunk collection C produced by
Algorithm 1 and builds the tree H:

    L0     leaf chunk nodes        (one per chunk, dense transformer embedding)
    L1..Lk abstract nodes          (k-means clusters, LLM-written abstracts)
    Root   global abstract         (cross-document synthesis)
    +      explicit alpha cross-child edges linking semantically related nodes
           that live under different parents / different source documents.

Everything runs on plain CPU by default. Every tensor op is written so that a
single device switch (device="cuda") moves it onto the GPU, which is where a
CUDA port would start -- but no GPU is required to build the tree.

This module covers knowledge construction only -- no query-time retrieval.
"""

import os


os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import hashlib
import json
import math
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ----------------------------------------------------------------------------
# Device resolution
# ----------------------------------------------------------------------------
def resolve_device(requested: str = "auto") -> str:
    """Return the torch device string actually usable on this machine."""
    if not HAS_TORCH:
        return "cpu"
    requested = (requested or "auto").lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested


def describe_device(device: str) -> str:
    """Human readable device banner, e.g. 'CUDA:0 (NVIDIA RTX 4090, 24.0 GB)'."""
    if device.startswith("cuda") and HAS_TORCH and torch.cuda.is_available():
        idx = 0 if ":" not in device else int(device.split(":")[1])
        props = torch.cuda.get_device_properties(idx)
        return f"CUDA:{idx} ({props.name}, {props.total_memory / 1024 ** 3:.1f} GB VRAM)"
    threads = torch.get_num_threads() if HAS_TORCH else 1
    return f"CPU ({threads} threads)"


# ----------------------------------------------------------------------------
# Deterministic fallback encoder (only used when the transformer is unavailable)
# ----------------------------------------------------------------------------
class HashingFallbackEncoder:
    """
    Deterministic feature-hashing encoder over word unigrams/bigrams and
    character 4-grams. Unlike a random projection of the whole string, shared
    vocabulary between two passages produces shared dimensions, so cosine
    similarity still carries lexical signal.
    """

    name = "feature-hashing-fallback"

    def __init__(self, dim: int = 384):
        self.dim = dim
        self._token_re = re.compile(r"[a-z0-9$%.]+")

    def _hash(self, token: str) -> int:
        return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "big")

    def _features(self, text: str) -> List[str]:
        low = text.lower()
        words = self._token_re.findall(low)
        feats = list(words)
        feats += [f"{a}_{b}" for a, b in zip(words, words[1:])]
        squashed = " ".join(words)
        feats += [squashed[i:i + 4] for i in range(0, max(0, len(squashed) - 3), 2)]
        return feats

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feat in self._features(text):
                h = self._hash(feat)
                out[row, h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        return out


# ----------------------------------------------------------------------------
# Embedding statistics
# ----------------------------------------------------------------------------
@dataclass
class EmbeddingStats:
    backend: str = "uninitialised"
    device: str = "cpu"
    dim: int = 0
    texts_encoded: int = 0
    cache_hits: int = 0
    batches: int = 0
    encode_seconds: float = 0.0
    load_seconds: float = 0.0
    peak_batch: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def throughput(self) -> float:
        return self.texts_encoded / self.encode_seconds if self.encode_seconds > 0 else 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "device": self.device,
            "embedding_dim": self.dim,
            "texts_encoded": self.texts_encoded,
            "cache_hits": self.cache_hits,
            "batches": self.batches,
            "encode_seconds": round(self.encode_seconds, 4),
            "model_load_seconds": round(self.load_seconds, 4),
            "throughput_texts_per_sec": round(self.throughput, 2),
        }


# ----------------------------------------------------------------------------
# Dense text encoder
# ----------------------------------------------------------------------------
class DenseEmbeddingModel:
    """
    Batched dense sentence encoder. Runs on CPU unless a CUDA device is
    available and requested, in which case the same code path uses it.

    Parameters
    ----------
    model_name : HuggingFace sentence-transformers id.
    device     : 'auto' | 'cuda' | 'cuda:1' | 'cpu'.
    batch_size : texts per forward pass (raised automatically on CUDA).
    normalize  : L2-normalise outputs so that dot product == cosine similarity.
    use_fp16   : half precision inference on CUDA (ignored on CPU).
    allow_fallback : degrade to HashingFallbackEncoder instead of raising.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "auto",
        batch_size: int = 32,
        normalize: bool = True,
        use_fp16: bool = True,
        allow_fallback: bool = True,
        cache_embeddings: bool = True,
        verbose: bool = True,
    ):
        self.model_name = model_name
        self.device = resolve_device(device)
        self.normalize = normalize
        self.use_fp16 = use_fp16 and self.device.startswith("cuda")
        self.allow_fallback = allow_fallback
        self.cache_embeddings = cache_embeddings
        self.verbose = verbose
        self.batch_size = batch_size * 4 if self.device.startswith("cuda") else batch_size

        self._cache: Dict[str, np.ndarray] = {}
        self._backend = None
        self._fallback: Optional[HashingFallbackEncoder] = None
        self.stats = EmbeddingStats(device=self.device)
        self.load_error: Optional[str] = None

        self._load_backend()

    # -- loading -------------------------------------------------------------
    def _load_backend(self):
        t0 = time.time()
        try:
            from sentence_transformers import SentenceTransformer
            self._backend = SentenceTransformer(self.model_name, device=self.device)
            if self.use_fp16:
                self._backend = self._backend.half()
            self.stats.backend = f"sentence-transformers::{self.model_name}"
            self.stats.dim = int(self._backend.get_sentence_embedding_dimension())
        except Exception as exc:  # noqa: BLE001 - any load failure degrades gracefully
            if not self.allow_fallback:
                raise
            self.load_error = f"{type(exc).__name__}: {exc}"
            self._fallback = HashingFallbackEncoder(dim=384)
            self.stats.backend = f"fallback::{self._fallback.name}"
            self.stats.dim = self._fallback.dim
            self.device = "cpu"
            self.stats.device = "cpu"
            if self.verbose:
                print(f"  [!] Transformer encoder unavailable ({self.load_error})")
                print(f"  [!] Falling back to deterministic {self._fallback.name} encoder.")
        self.stats.load_seconds = time.time() - t0

    # -- properties ----------------------------------------------------------
    @property
    def dim(self) -> int:
        return self.stats.dim

    @property
    def is_transformer(self) -> bool:
        return self._backend is not None

    def banner(self) -> str:
        return (
            f"backend={self.stats.backend} | device={describe_device(self.device)} | "
            f"dim={self.dim} | batch={self.batch_size} | fp16={self.use_fp16}"
        )

    # -- encoding ------------------------------------------------------------
    def _encode_raw(self, texts: List[str]) -> np.ndarray:
        if self._backend is not None:
            vecs = self._backend.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
            return np.asarray(vecs, dtype=np.float32)
        return self._fallback.encode(texts)

    @staticmethod
    def _l2_normalize(mat: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0.0] = 1e-9
        return mat / norms

    def encode(
        self,
        texts: Union[str, Sequence[str]],
        label: str = "",
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode texts into an (n, d) float32 matrix of L2-normalised rows."""
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        if not items:
            return np.zeros((0, self.dim), dtype=np.float32)

        out = np.zeros((len(items), self.dim), dtype=np.float32)
        pending_idx: List[int] = []
        pending_txt: List[str] = []

        for i, text in enumerate(items):
            key = hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()
            if self.cache_embeddings and key in self._cache:
                out[i] = self._cache[key]
                self.stats.cache_hits += 1
            else:
                pending_idx.append(i)
                pending_txt.append(text)

        if pending_txt:
            t0 = time.time()
            n_batches = 0
            for start in range(0, len(pending_txt), self.batch_size):
                batch = pending_txt[start:start + self.batch_size]
                vecs = self._encode_raw(batch)
                if self.normalize:
                    vecs = self._l2_normalize(vecs)
                for offset, vec in enumerate(vecs):
                    out[pending_idx[start + offset]] = vec
                n_batches += 1
                self.stats.peak_batch = max(self.stats.peak_batch, len(batch))
                if show_progress and len(pending_txt) > self.batch_size:
                    done = min(start + self.batch_size, len(pending_txt))
                    print(f"    embedding batch {n_batches}: {done}/{len(pending_txt)} texts", end="\r")
            if show_progress and len(pending_txt) > self.batch_size:
                print(" " * 70, end="\r")

            elapsed = time.time() - t0
            self.stats.texts_encoded += len(pending_txt)
            self.stats.batches += n_batches
            self.stats.encode_seconds += elapsed
            self.stats.history.append({
                "label": label or f"call_{len(self.stats.history)}",
                "texts": len(pending_txt),
                "batches": n_batches,
                "seconds": round(elapsed, 4),
            })

            if self.cache_embeddings:
                for i, text in zip(pending_idx, pending_txt):
                    key = hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()
                    self._cache[key] = out[i]

        return out[0] if single else out

    def encode_chunks(self, chunks: Sequence[Dict[str, Any]], show_progress: bool = True) -> np.ndarray:
        """Encode Algorithm-1 chunk dicts (uses 'section' as a light context prefix)."""
        texts = [
            f"{c.get('section', '')}: {c['text']}".strip(": ") if c.get("section") else c["text"]
            for c in chunks
        ]
        return self.encode(texts, label="leaf_chunks", show_progress=show_progress)

    # -- similarity ----------------------------------------------------------
    def similarity(self, a: np.ndarray, b: Optional[np.ndarray] = None) -> np.ndarray:
        """Cosine similarity matrix; runs on GPU when available."""
        b = a if b is None else b
        a2 = a.reshape(1, -1) if a.ndim == 1 else a
        b2 = b.reshape(1, -1) if b.ndim == 1 else b
        if HAS_TORCH and self.device.startswith("cuda"):
            ta = torch.as_tensor(a2, device=self.device, dtype=torch.float32)
            tb = torch.as_tensor(b2, device=self.device, dtype=torch.float32)
            return (ta @ tb.T).cpu().numpy()
        return a2 @ b2.T


DEFAULT_SUMMARIZER_MODEL = "facebook/bart-large-cnn"


# ============================================================================
# Tree node
# ============================================================================
class TreeNode:
    def __init__(
        self,
        node_id: str,
        level: int,
        text: str,
        embedding: Optional[np.ndarray] = None,
        doc_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.node_id = node_id
        self.level = level
        self.text = text
        self.embedding = None if embedding is None else np.asarray(embedding, dtype=np.float32)
        self.doc_id = doc_id
        self.metadata = metadata or {}
        self.children: List["TreeNode"] = []
        self.parent: Optional["TreeNode"] = None
        # Explicit alpha cross-child relationships (dashed edges of the HAT):
        # [{"target_id", "score", "relation"}]
        self.cross_links: List[Dict[str, Any]] = []

    def add_child(self, child_node: "TreeNode"):
        child_node.parent = self
        self.children.append(child_node)

    @property
    def source_docs(self) -> List[str]:
        """Documents covered by this subtree (leaves carry a single doc_id)."""
        if self.doc_id:
            return [self.doc_id]
        docs: List[str] = []
        for child in self.children:
            for d in child.source_docs:
                if d not in docs:
                    docs.append(d)
        return docs

    def to_dict(self, include_embedding: bool = True) -> Dict[str, Any]:
        emb = None
        if include_embedding and self.embedding is not None:
            emb = [round(float(x), 6) for x in self.embedding]
        return {
            "node_id": self.node_id,
            "level": self.level,
            "text": self.text,
            "doc_id": self.doc_id,
            "source_docs": self.source_docs,
            "metadata": self.metadata,
            "parent_id": self.parent.node_id if self.parent else None,
            "children_ids": [c.node_id for c in self.children],
            "cross_links": self.cross_links,
            "embedding": emb,
        }


# ============================================================================
# Spherical k-means (cosine)
# ============================================================================
@dataclass
class ClusterResult:
    labels: np.ndarray
    centroids: np.ndarray
    inertia: float
    iterations: int
    cohesion: float          # mean cosine(point, own centroid)
    device: str
    seconds: float


class SphericalKMeans:
    """
    Spherical k-means over L2-normalised embeddings.

    Cosine similarity reduces to a dot product on normalised vectors, so the
    assignment step is a single (n x d) @ (d x k) matmul. That runs fine on CPU
    at this corpus size and is also the one operation worth moving to CUDA later
    -- pass device="cuda" and the same code runs there. k-means++ seeding keeps
    the clustering stable and empty clusters are re-seeded from the worst fit.
    """

    def __init__(self, n_clusters: int, max_iter: int = 50, tol: float = 1e-5,
                 seed: int = 42, device: str = "auto"):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.device = resolve_device(device)
        self.backend = "torch" if HAS_TORCH else "numpy"

    # -- numpy path ----------------------------------------------------------
    @staticmethod
    def _normalize_np(x: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(x, axis=1, keepdims=True)
        n[n == 0] = 1e-9
        return x / n

    def _kmeanspp_np(self, x: np.ndarray, k: int, rng: np.random.RandomState) -> np.ndarray:
        n = x.shape[0]
        centers = [int(rng.randint(n))]
        for _ in range(1, k):
            sims = x @ x[centers].T                 # (n, |centers|)
            d2 = np.clip(1.0 - sims.max(axis=1), 0.0, None) ** 2
            if d2.sum() <= 1e-12:
                remaining = [i for i in range(n) if i not in centers]
                centers.append(int(rng.choice(remaining)) if remaining else centers[-1])
            else:
                centers.append(int(rng.choice(n, p=d2 / d2.sum())))
        return x[centers].copy()

    def fit(self, embeddings: np.ndarray) -> ClusterResult:
        t0 = time.time()
        x = self._normalize_np(np.asarray(embeddings, dtype=np.float32))
        n = x.shape[0]
        k = max(1, min(self.n_clusters, n))
        rng = np.random.RandomState(self.seed)
        centroids = self._kmeanspp_np(x, k, rng)

        if HAS_TORCH:
            labels, centroids, inertia, iters = self._fit_torch(x, centroids, k)
        else:
            labels, centroids, inertia, iters = self._fit_numpy(x, centroids, k)

        cohesion = float(np.mean([float(x[i] @ centroids[labels[i]]) for i in range(n)])) if n else 0.0
        return ClusterResult(
            labels=labels,
            centroids=centroids,
            inertia=float(inertia),
            iterations=iters,
            cohesion=cohesion,
            device=self.device if HAS_TORCH else "cpu/numpy",
            seconds=time.time() - t0,
        )

    def _fit_numpy(self, x, centroids, k):
        labels = np.zeros(x.shape[0], dtype=np.int64)
        prev = -np.inf
        iters = 0
        for iters in range(1, self.max_iter + 1):
            sims = x @ centroids.T
            labels = sims.argmax(axis=1)
            score = float(sims.max(axis=1).sum())
            for c in range(k):
                mask = labels == c
                if not mask.any():
                    worst = int(sims.max(axis=1).argmin())
                    centroids[c] = x[worst]
                    labels[worst] = c
                    continue
                centroids[c] = x[mask].mean(axis=0)
            centroids = self._normalize_np(centroids)
            if abs(score - prev) < self.tol:
                break
            prev = score
        inertia = float((1.0 - (x @ centroids.T).max(axis=1)).sum())
        return labels, centroids, inertia, iters

    def _fit_torch(self, x_np, centroids_np, k):
        dev = self.device
        x = torch.as_tensor(x_np, device=dev)
        centroids = torch.as_tensor(centroids_np, device=dev)
        prev = float("-inf")
        iters = 0
        labels = torch.zeros(x.shape[0], dtype=torch.long, device=dev)

        for iters in range(1, self.max_iter + 1):
            sims = x @ centroids.T                          # GPU matmul: assignment step
            best_sim, labels = sims.max(dim=1)
            score = float(best_sim.sum())

            # update step: mean of members, re-normalised back onto the sphere
            new_centroids = torch.zeros_like(centroids)
            counts = torch.zeros(k, device=dev)
            new_centroids.index_add_(0, labels, x)
            counts.index_add_(0, labels, torch.ones_like(best_sim))

            empty = (counts == 0).nonzero(as_tuple=True)[0]
            for c in empty.tolist():                        # re-seed from worst fit
                worst = int(best_sim.argmin())
                new_centroids[c] = x[worst]
                counts[c] = 1.0
                labels[worst] = c
                best_sim[worst] = 1.0

            new_centroids = new_centroids / counts.unsqueeze(1).clamp(min=1.0)
            centroids = new_centroids / new_centroids.norm(dim=1, keepdim=True).clamp(min=1e-9)

            if abs(score - prev) < self.tol:
                break
            prev = score

        inertia = float((1.0 - (x @ centroids.T).max(dim=1).values).sum())
        return labels.cpu().numpy(), centroids.cpu().numpy().astype(np.float32), inertia, iters


# ============================================================================
# Abstract summarisation (LLM with extractive fallback)
# ============================================================================
class AbstractSummarizer:
    """
    Writes the abstract text carried by an internal HAT node.

    mode='llm'        : abstractive seq2seq summariser (default bart-large-cnn)
    mode='extractive' : centroid-based sentence selection, no model download
    mode='auto'       : try the LLM, silently fall back to extractive
    """

    def __init__(
        self,
        mode: str = "auto",
        model_name: str = DEFAULT_SUMMARIZER_MODEL,
        device: str = "auto",
        max_input_chars: int = 3500,
        verbose: bool = True,
    ):
        self.requested_mode = mode
        self.model_name = model_name
        self.device = resolve_device(device)
        self.max_input_chars = max_input_chars
        self.verbose = verbose
        self._pipe = None
        self.mode = "extractive"
        self.load_seconds = 0.0
        self.calls = 0
        self.llm_seconds = 0.0
        self.load_error: Optional[str] = None

        if mode in ("llm", "auto"):
            self._load_llm(strict=(mode == "llm"))

    def _load_llm(self, strict: bool):
        t0 = time.time()
        try:
            from transformers import pipeline
            self._pipe = pipeline(
                "summarization",
                model=self.model_name,
                device=0 if self.device.startswith("cuda") else -1,
            )
            self.mode = "llm"
        except Exception as exc:  # noqa: BLE001
            self.load_error = f"{type(exc).__name__}: {exc}"
            if strict:
                raise
            self.mode = "extractive"
            if self.verbose:
                print(f"  [!] Summarizer LLM unavailable ({self.load_error}); using extractive abstracts.")
        self.load_seconds = time.time() - t0

    def banner(self) -> str:
        if self.mode == "llm":
            return f"abstractive::{self.model_name} on {describe_device(self.device)}"
        return "extractive::centroid-sentence-selection"

    # -- helpers -------------------------------------------------------------
    STOPWORDS = {
        "this", "that", "with", "from", "were", "have", "been", "which", "under",
        "across", "during", "while", "their", "these", "those", "there", "into",
        "total", "also", "than", "such", "will", "other", "over", "each", "both",
    }

    def extract_keywords(self, text: str, top_k: int = 6) -> List[str]:
        words = re.findall(r"\b[A-Za-z][A-Za-z0-9&/-]{3,}\b", text)
        freq: Dict[str, int] = {}
        for w in words:
            key = w if w.isupper() else w.lower()
            if key.lower() in self.STOPWORDS:
                continue
            freq[key] = freq.get(key, 0) + 1
        ranked = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0].lower()))
        return [w for w, _ in ranked[:top_k]]

    @staticmethod
    def _sentences(text: str) -> List[str]:
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [p.strip() for p in parts if len(p.strip()) > 25]

    def _extractive(
        self,
        child_texts: Sequence[str],
        centroid: Optional[np.ndarray],
        embedder: Optional[DenseEmbeddingModel],
        n_sentences: int = 3,
    ) -> str:
        combined = " ".join(child_texts)
        sentences = self._sentences(combined)
        if not sentences:
            return combined[:400]
        if centroid is None or embedder is None:
            return " ".join(sentences[:n_sentences])

        vecs = embedder.encode(sentences, label="extractive_summary")
        scores = vecs @ np.asarray(centroid, dtype=np.float32)
        order = np.argsort(-scores)[: min(n_sentences, len(sentences))]
        # keep original reading order for the selected sentences
        chosen = [sentences[i] for i in sorted(order.tolist())]
        return " ".join(chosen)

    # -- public --------------------------------------------------------------
    def summarize(
        self,
        child_texts: Sequence[str],
        level: int,
        is_root: bool = False,
        centroid: Optional[np.ndarray] = None,
        embedder: Optional[DenseEmbeddingModel] = None,
        source_docs: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Return {'text', 'method', 'keywords', 'seconds'} for one abstract node."""
        t0 = time.time()
        combined = " ".join(child_texts)
        keywords = self.extract_keywords(combined)
        method = self.mode

        if self.mode == "llm":
            source = combined[: self.max_input_chars]
            n_words = len(source.split())
            max_len = max(45, min(140, int(n_words * 0.45)))
            min_len = max(20, int(max_len * 0.45))
            try:
                out = self._pipe(
                    source,
                    max_length=max_len,
                    min_length=min_len,
                    do_sample=False,
                    num_beams=4 if self.device.startswith("cuda") else 1,
                    truncation=True,
                )
                body = out[0]["summary_text"].strip()
                self.calls += 1
            except Exception as exc:  # noqa: BLE001 - never break the build
                if self.verbose:
                    print(f"  [!] LLM summarisation failed ({type(exc).__name__}); extractive fallback used.")
                body = self._extractive(child_texts, centroid, embedder)
                method = "extractive-fallback"
        else:
            body = self._extractive(child_texts, centroid, embedder)

        elapsed = time.time() - t0
        if method.startswith("llm") or method == "llm":
            self.llm_seconds += elapsed

        return {
            "text": body,
            "method": method,
            "keywords": keywords,
            "seconds": elapsed,
        }


# ============================================================================
# HAT builder
# ============================================================================
@dataclass
class BuildMetrics:
    embed_seconds: float = 0.0
    cluster_seconds: float = 0.0
    summarize_seconds: float = 0.0
    crosslink_seconds: float = 0.0
    total_seconds: float = 0.0
    level_stats: List[Dict[str, Any]] = field(default_factory=list)
    cross_links: int = 0
    cross_doc_links: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "embed_seconds": round(self.embed_seconds, 3),
            "cluster_seconds": round(self.cluster_seconds, 3),
            "summarize_seconds": round(self.summarize_seconds, 3),
            "crosslink_seconds": round(self.crosslink_seconds, 3),
            "total_seconds": round(self.total_seconds, 3),
            "cross_links": self.cross_links,
            "cross_doc_links": self.cross_doc_links,
            "levels": self.level_stats,
        }


class HierarchicalAbstractTreeBuilder:
    """
    Builds the Hierarchical Abstract Tree H.

    Parameters
    ----------
    embedder        : DenseEmbeddingModel used for leaves and abstracts.
    max_levels      : maximum number of abstract levels above the leaves.
    target_children : desired children per abstract node -> k = ceil(n / target).
    alpha           : cosine threshold for explicit cross-child edges. Either a
                      fixed float or "auto" (mean + 1 std of the observed
                      similarity distribution at that level, clamped).
    max_cross_links : cap on alpha edges kept per node.
    force_single_root : collapse remaining top nodes into one global root.
    summarizer_mode : 'auto' | 'llm' | 'extractive'.
    """

    def __init__(
        self,
        embedder: Optional[DenseEmbeddingModel] = None,
        max_levels: int = 3,
        target_children: int = 4,
        alpha: Union[float, str] = "auto",
        max_cross_links: int = 3,
        force_single_root: bool = True,
        collapse_singletons: bool = True,
        summarizer_mode: str = "auto",
        summarizer_model: str = DEFAULT_SUMMARIZER_MODEL,
        embedding_model: str = DEFAULT_MODEL,
        device: str = "auto",
        seed: int = 42,
        verbose: bool = True,
        **legacy,
    ):
        # backwards compatibility with the old constructor signature
        if "branching_factor" in legacy:
            target_children = max(2, int(legacy.pop("branching_factor")))

        self.max_levels = max_levels
        self.target_children = max(2, target_children)
        self.alpha = alpha if isinstance(alpha, str) else float(alpha)
        self.collapse_singletons = collapse_singletons
        self.resolved_alpha: Dict[int, float] = {}
        self.max_cross_links = max_cross_links
        self.force_single_root = force_single_root
        self.seed = seed
        self.verbose = verbose
        self.device = resolve_device(device)

        self.embedder = embedder or DenseEmbeddingModel(
            model_name=embedding_model, device=device, verbose=verbose
        )
        self.summarizer = AbstractSummarizer(
            mode=summarizer_mode, model_name=summarizer_model, device=device, verbose=verbose
        )

        self.nodes: Dict[str, TreeNode] = {}
        self.root_nodes: List[TreeNode] = []
        self.metrics = BuildMetrics()

    # -- logging -------------------------------------------------------------
    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    # -- construction --------------------------------------------------------
    def _make_leaves(self, leaf_chunks: Sequence[Dict[str, Any]]) -> List[TreeNode]:
        t0 = time.time()
        self._log(f"  [L0] Embedding {len(leaf_chunks)} leaf chunks "
                  f"({self.embedder.stats.backend.split('::')[0]} on {describe_device(self.embedder.device)}) ...")
        matrix = self.embedder.encode_chunks(leaf_chunks)
        self.metrics.embed_seconds += time.time() - t0

        leaves: List[TreeNode] = []
        for chunk, vec in zip(leaf_chunks, matrix):
            node = TreeNode(
                node_id=chunk["chunk_id"],
                level=0,
                text=chunk["text"],
                embedding=vec,
                doc_id=chunk.get("doc_id"),
                metadata={
                    "token_count": chunk.get("token_count", len(chunk["text"].split())),
                    "section": chunk.get("section", "General"),
                    "chunk_index": chunk.get("chunk_index"),
                    "hash": chunk.get("hash"),
                },
            )
            self.nodes[node.node_id] = node
            leaves.append(node)
        self._log(f"  [L0] Embedding matrix E: {matrix.shape} | "
                  f"{self.embedder.stats.throughput:.1f} texts/sec")
        return leaves

    def _k_for_level(self, n_nodes: int, level: int) -> int:
        """Adaptive cluster count: enough parents so each holds ~target_children."""
        remaining_levels = self.max_levels - level
        k = math.ceil(n_nodes / self.target_children)
        if remaining_levels <= 1 and self.force_single_root:
            k = max(1, min(k, self.target_children))
        return max(1, min(k, n_nodes))

    def _build_level(self, children: List[TreeNode], level: int) -> List[TreeNode]:
        k = self._k_for_level(len(children), level - 1)
        matrix = np.stack([c.embedding for c in children])

        t0 = time.time()
        kmeans = SphericalKMeans(n_clusters=k, seed=self.seed, device=self.device)
        result = kmeans.fit(matrix)
        self.metrics.cluster_seconds += time.time() - t0
        self._log(f"  [L{level}] k-means: n={len(children)} -> k={k} | "
                  f"backend={kmeans.backend}/{result.device} | iters={result.iterations} | "
                  f"cohesion={result.cohesion:.3f} | {result.seconds * 1000:.1f} ms")

        buckets: Dict[int, List[TreeNode]] = {}
        for node, lab in zip(children, result.labels):
            buckets.setdefault(int(lab), []).append(node)

        parents: List[TreeNode] = []
        promoted = 0
        for c_idx in sorted(buckets):
            cluster = buckets[c_idx]
            centroid = result.centroids[c_idx]

            # A one-member cluster would only paraphrase its single child into a
            # chain node: promote the child to the next level untouched instead.
            if self.collapse_singletons and len(cluster) == 1:
                parents.append(cluster[0])
                promoted += 1
                continue
            docs: List[str] = []
            for child in cluster:
                for d in child.source_docs:
                    if d not in docs:
                        docs.append(d)

            t1 = time.time()
            summary = self.summarizer.summarize(
                [c.text for c in cluster],
                level=level,
                is_root=False,
                centroid=centroid,
                embedder=self.embedder,
                source_docs=docs,
            )
            self.metrics.summarize_seconds += time.time() - t1

            t2 = time.time()
            parent_vec = self.embedder.encode(summary["text"], label=f"abstract_L{level}")
            self.metrics.embed_seconds += time.time() - t2

            parent = TreeNode(
                node_id=f"L{level}_A{c_idx}",
                level=level,
                text=summary["text"],
                embedding=parent_vec,
                metadata={
                    "child_count": len(cluster),
                    "cluster_id": c_idx,
                    "keywords": summary["keywords"],
                    "summary_method": summary["method"],
                    "cluster_cohesion": round(
                        float(np.mean([float(np.asarray(c.embedding) @ centroid) for c in cluster])), 4
                    ),
                    "is_cross_document": len(docs) > 1,
                },
            )
            for child in cluster:
                parent.add_child(child)
            self.nodes[parent.node_id] = parent
            parents.append(parent)
            self._log(f"        L{level}_A{c_idx}: {len(cluster)} children | {len(docs)} doc(s) | "
                      f"{summary['method']} | {summary['seconds']:.1f}s")

        self.metrics.level_stats.append({
            "level": level,
            "input_nodes": len(children),
            "clusters": len(parents),
            "abstract_nodes_created": len(parents) - promoted,
            "promoted_singletons": promoted,
            "kmeans_iterations": result.iterations,
            "mean_cohesion": round(result.cohesion, 4),
            "inertia": round(result.inertia, 4),
            "cross_document_clusters": sum(1 for p in parents if p.metadata.get("is_cross_document")),
        })
        return parents

    def _make_root(self, tops: List[TreeNode], level: int) -> TreeNode:
        docs: List[str] = []
        for t in tops:
            for d in t.source_docs:
                if d not in docs:
                    docs.append(d)
        t1 = time.time()
        summary = self.summarizer.summarize(
            [t.text for t in tops], level=level, is_root=True,
            centroid=None, embedder=self.embedder, source_docs=docs,
        )
        self.metrics.summarize_seconds += time.time() - t1

        t2 = time.time()
        vec = self.embedder.encode(summary["text"], label="root_abstract")
        self.metrics.embed_seconds += time.time() - t2

        root = TreeNode(
            node_id="ROOT",
            level=level,
            text=summary["text"],
            embedding=vec,
            metadata={
                "child_count": len(tops),
                "keywords": summary["keywords"],
                "summary_method": summary["method"],
                "is_cross_document": len(docs) > 1,
                "corpus_documents": len(docs),
            },
        )
        for t in tops:
            root.add_child(t)
        self.nodes[root.node_id] = root
        self._log(f"  [ROOT] Global abstract over {len(tops)} nodes / {len(docs)} documents "
                  f"({summary['method']}, {summary['seconds']:.1f}s)")
        return root

    def _resolve_alpha(self, sim: np.ndarray, level: int) -> float:
        """Fixed alpha, or a data-driven one at mean + 1 std of this level's
        off-diagonal similarities (clamped so it never becomes degenerate)."""
        if not isinstance(self.alpha, str):
            self.resolved_alpha[level] = float(self.alpha)
            return float(self.alpha)
        vals = sim[sim > -1.0]
        thr = 0.6 if vals.size == 0 else float(vals.mean() + vals.std())
        thr = float(min(0.90, max(0.35, thr)))
        self.resolved_alpha[level] = round(thr, 4)
        return thr

    # -- explicit alpha cross-child relationships ----------------------------
    def build_cross_links(self, nodes_by_level: Dict[int, List[TreeNode]]):
        """
        Dashed edges of the HAT: for every node, attach up to `max_cross_links`
        peers at the same level whose cosine similarity >= alpha but which sit
        under a *different* parent. These are the explicit relationships that
        let related material be reached without walking back up through the root.
        """
        t0 = time.time()
        total = 0
        cross_doc = 0

        for level, nodes in sorted(nodes_by_level.items()):
            if len(nodes) < 2:
                continue
            matrix = np.stack([n.embedding for n in nodes])
            sim = self.embedder.similarity(matrix)
            np.fill_diagonal(sim, -1.0)
            threshold = self._resolve_alpha(sim, level)

            for i, node in enumerate(nodes):
                order = np.argsort(-sim[i])
                kept = 0
                for j in order:
                    score = float(sim[i][j])
                    if score < threshold or kept >= self.max_cross_links:
                        break
                    peer = nodes[int(j)]
                    same_parent = (
                        node.parent is not None
                        and peer.parent is not None
                        and node.parent.node_id == peer.parent.node_id
                    )
                    if same_parent:
                        continue
                    docs_a, docs_b = set(node.source_docs), set(peer.source_docs)
                    is_cross_doc = bool(docs_a) and bool(docs_b) and not (docs_a & docs_b)
                    node.cross_links.append({
                        "target_id": peer.node_id,
                        "score": round(score, 4),
                        "relation": "cross_document" if is_cross_doc else "cross_cluster",
                    })
                    kept += 1
                    total += 1
                    cross_doc += int(is_cross_doc)

            n_level = sum(len(n.cross_links) for n in nodes)
            self._log(f"  [alpha>={threshold:.3f}] level {level}: {n_level} explicit cross-child edges "
                      f"over {len(nodes)} nodes")

        self.metrics.crosslink_seconds += time.time() - t0
        self.metrics.cross_links = total
        self.metrics.cross_doc_links = cross_doc

    # -- entry point ---------------------------------------------------------
    def build_tree(self, leaf_chunks: Sequence[Dict[str, Any]]) -> List[TreeNode]:
        t_start = time.time()
        self.nodes.clear()
        self.root_nodes = []
        self.metrics = BuildMetrics()

        if not leaf_chunks:
            return []

        current = self._make_leaves(leaf_chunks)

        level = 0
        while level < self.max_levels and len(current) > 1:
            level += 1
            current = self._build_level(current, level)
            if len(current) == 1:
                break

        if self.force_single_root and len(current) > 1:
            top_level = max(n.level for n in current) + 1
            current = [self._make_root(current, top_level)]
        elif len(current) == 1 and current[0].level > 0:
            self._rename_node(current[0], "ROOT")

        self.root_nodes = current
        nodes_by_level: Dict[int, List[TreeNode]] = {}
        for node in self.nodes.values():
            nodes_by_level.setdefault(node.level, []).append(node)
        self.build_cross_links(nodes_by_level)
        self.metrics.total_seconds = time.time() - t_start
        return self.root_nodes

    def _rename_node(self, node: TreeNode, new_id: str):
        """Give the single top abstract the canonical ROOT id."""
        if node.node_id == new_id or new_id in self.nodes:
            return
        old_id = node.node_id
        self.nodes.pop(old_id, None)
        node.node_id = new_id
        self.nodes[new_id] = node
        for other in self.nodes.values():
            for link in other.cross_links:
                if link["target_id"] == old_id:
                    link["target_id"] = new_id

    # -- persistence ---------------------------------------------------------
    def save_tree_json(self, filepath: str, include_embeddings: bool = True):
        data = {
            "schema": "HAT-v2",
            "config": {
                "max_levels": self.max_levels,
                "target_children": self.target_children,
                "alpha": self.alpha,
                "resolved_alpha_per_level": self.resolved_alpha,
                "max_cross_links": self.max_cross_links,
                "force_single_root": self.force_single_root,
                "seed": self.seed,
            },
            "embedding_model": self.embedder.stats.as_dict(),
            "summarizer": {
                "mode": self.summarizer.mode,
                "model": self.summarizer.model_name if self.summarizer.mode == "llm" else None,
                "llm_calls": self.summarizer.calls,
            },
            "metrics": self.metrics.as_dict(),
            "total_nodes": len(self.nodes),
            "root_ids": [r.node_id for r in self.root_nodes],
            "nodes": {nid: n.to_dict(include_embeddings) for nid, n in self.nodes.items()},
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_tree_json(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.nodes = {}
        for nid, d in data["nodes"].items():
            emb = d.get("embedding")
            node = TreeNode(
                node_id=d["node_id"],
                level=d["level"],
                text=d["text"],
                embedding=np.asarray(emb, dtype=np.float32) if emb else None,
                doc_id=d.get("doc_id"),
                metadata=d.get("metadata", {}),
            )
            node.cross_links = d.get("cross_links", [])
            self.nodes[nid] = node

        for nid, d in data["nodes"].items():
            parent = self.nodes[nid]
            for cid in d.get("children_ids", []):
                if cid in self.nodes:
                    parent.add_child(self.nodes[cid])

        self.root_nodes = [self.nodes[r] for r in data["root_ids"] if r in self.nodes]
        cfg = data.get("config", {})
        self.max_levels = cfg.get("max_levels", self.max_levels)
        self.alpha = cfg.get("alpha", self.alpha)
        return data

    # -- reporting -----------------------------------------------------------
    def level_counts(self) -> Dict[int, int]:
        counts: Dict[int, int] = {}
        for n in self.nodes.values():
            counts[n.level] = counts.get(n.level, 0) + 1
        return dict(sorted(counts.items()))

    def display_tree(self, max_leaves_per_parent: int = 3, text_width: int = 96):
        print("\n" + "=" * 100)
        print(" HIERARCHICAL ABSTRACT TREE (H) -- TOPOLOGY")
        print("=" * 100)

        def _print(node: TreeNode, indent: str = "", is_last: bool = True):
            marker = "`-- " if is_last else "|-- "
            docs = node.source_docs
            if node.level == 0:
                tag = f"(doc: {node.doc_id} | {node.metadata.get('section', '')[:40]})"
            else:
                kw = ", ".join(node.metadata.get("keywords", [])[:4])
                tag = (f"({node.metadata.get('child_count', 0)} children | {len(docs)} docs | "
                       f"cohesion {node.metadata.get('cluster_cohesion', '-')}) topics: {kw}")
            print(f"{indent}{marker}[L{node.level}] {node.node_id} {tag}")

            child_indent = indent + ("    " if is_last else "|   ")
            text = node.text if len(node.text) <= text_width else node.text[:text_width] + "..."
            print(f'{child_indent}  "{text}"')
            for link in node.cross_links:
                print(f"{child_indent}  ~~> alpha-link {link['target_id']} "
                      f"(cos={link['score']}, {link['relation']})")

            shown = node.children
            hidden = 0
            if node.children and node.children[0].level == 0 and len(node.children) > max_leaves_per_parent:
                shown = node.children[:max_leaves_per_parent]
                hidden = len(node.children) - len(shown)
            for idx, child in enumerate(shown):
                _print(child, child_indent, idx == len(shown) - 1 and hidden == 0)
            if hidden:
                print(f"{child_indent}`-- ... {hidden} more leaf chunk(s)")

        for i, root in enumerate(self.root_nodes):
            _print(root, is_last=(i == len(self.root_nodes) - 1))
        print("=" * 100)

    def print_metrics(self):
        counts = self.level_counts()
        leaves = counts.get(0, 0)
        print("\n" + "=" * 100)
        print(" HAT CONSTRUCTION METRICS")
        print("=" * 100)
        print(f"  Embedding backend        : {self.embedder.stats.backend}")
        print(f"  Embedding device / dim   : {describe_device(self.embedder.device)} / {self.embedder.dim}")
        print(f"  Summarizer               : {self.summarizer.banner()}")
        print(f"  Total nodes              : {len(self.nodes)}")
        for lvl, cnt in counts.items():
            label = "leaf chunks" if lvl == 0 else ("root abstract" if lvl == max(counts) else "abstract nodes")
            print(f"    level {lvl:<2}               : {cnt:>4} {label}")
        print(f"  Compression (L0 -> root) : {leaves / max(1, counts.get(max(counts), 1)):.1f}x")
        alpha_txt = (f"auto -> {self.resolved_alpha}" if isinstance(self.alpha, str)
                     else f"{self.alpha:.2f}")
        print(f"  Explicit alpha edges     : {self.metrics.cross_links} "
              f"({self.metrics.cross_doc_links} cross-document, alpha={alpha_txt})")
        print("  Phase timings (s)        : "
              f"embed={self.metrics.embed_seconds:.2f} | cluster={self.metrics.cluster_seconds:.2f} | "
              f"summarize={self.metrics.summarize_seconds:.2f} | crosslink={self.metrics.crosslink_seconds:.2f}")
        print(f"  Total build time         : {self.metrics.total_seconds:.2f}s")
        for ls in self.metrics.level_stats:
            print(f"    L{ls['level']}: {ls['input_nodes']} -> {ls['clusters']} nodes "
                  f"({ls['abstract_nodes_created']} new abstracts, "
                  f"{ls['promoted_singletons']} promoted) | cohesion={ls['mean_cohesion']} | "
                  f"iters={ls['kmeans_iterations']} | cross-doc clusters={ls['cross_document_clusters']}")
        print("=" * 100)


# ============================================================================
# Demonstration
# ============================================================================
def run_hat_tree_demo(summarizer_mode: str = "auto", device: str = "auto",
                      alpha: Union[float, str] = "auto"):
    print("=" * 100)
    print(" ALGORITHM 3: HIERARCHICAL ABSTRACT TREE (HAT) CONSTRUCTION")
    print("=" * 100)

    sample_leaf_chunks = [
        {"chunk_id": "Doc1_c0", "doc_id": "Paper_01_ThermalDynamics", "section": "Blade Design",
         "text": "High-speed ceiling fan blade profiles are engineered with optimal pitch angles between 12 to 15 degrees to maximize downward volumetric air displacement while reducing turbulent boundary layer drag.",
         "token_count": 28},
        {"chunk_id": "Doc1_c1", "doc_id": "Paper_01_ThermalDynamics", "section": "Thermal Analysis",
         "text": "Empirical thermal imaging demonstrates a uniform 3.8 degree Celsius decrease in surface temperature across a 40 square meter test chamber, eliminating thermal stratification zones near the ceiling.",
         "token_count": 27},
        {"chunk_id": "Doc2_c0", "doc_id": "Paper_02_MotorDiagnostics", "section": "Vibration Telemetry",
         "text": "Progressive tool wear during sheet metal stamping induces structural micro-asymmetries in rotor brackets that manifest as high-frequency harmonic vibrations exceeding 120 Hz during continuous motor operation.",
         "token_count": 26},
        {"chunk_id": "Doc2_c1", "doc_id": "Paper_02_MotorDiagnostics", "section": "Sensor Ingestion",
         "text": "Triaxial accelerometer telemetry combined with edge microcontroller inference enables real-time bearing fault classification and reduces unplanned motor downtime by 41 percent in industrial facilities.",
         "token_count": 25},
        {"chunk_id": "Doc3_c0", "doc_id": "Paper_03_EnergyOptimization", "section": "Inverter Topology",
         "text": "Permanent magnet Brushless DC motor architectures achieve electrical efficiency ratings exceeding 88 percent with sensorless field-oriented control that minimizes harmonic acoustic noise.",
         "token_count": 24},
        {"chunk_id": "Doc3_c1", "doc_id": "Paper_03_EnergyOptimization", "section": "Speed Scheduling",
         "text": "Ambient temperature-adaptive pulse-width modulation dynamically modulates rotor RPM based on room occupancy, cutting annual electrical consumption by 62 percent versus AC induction motors.",
         "token_count": 24},
    ]

    print(f"\n[Input] {len(sample_leaf_chunks)} leaf chunks from 3 documents (Algorithm 1 output)")
    builder = HierarchicalAbstractTreeBuilder(
        max_levels=2, target_children=3, alpha=alpha,
        summarizer_mode=summarizer_mode, device=device,
    )
    print(f"[Config] embedder: {builder.embedder.banner()}")
    print(f"[Config] summarizer: {builder.summarizer.banner()}")

    print("\n[Build] Constructing H ...")
    builder.build_tree(sample_leaf_chunks)
    builder.display_tree()

    json_path = Path(__file__).resolve().parent / "hat_tree_structure.json"
    builder.save_tree_json(str(json_path))
    print(f"\n[Persistence] Tree exported to '{json_path.name}'")

    reload_test = HierarchicalAbstractTreeBuilder(
        embedder=builder.embedder, summarizer_mode="extractive", verbose=False
    )
    reload_test.load_tree_json(str(json_path))
    links = sum(len(n.cross_links) for n in reload_test.nodes.values())
    print(f"[Persistence] Reloaded {len(reload_test.nodes)} nodes, "
          f"{len(reload_test.root_nodes)} root(s), {links} alpha edges -- round trip OK")

    builder.print_metrics()
    return builder


def _cli():
    p = argparse.ArgumentParser(description="Algorithm 3: HAT construction demo")
    p.add_argument("--summarizer", choices=["auto", "llm", "extractive"], default="auto")
    p.add_argument("--device", default="auto", help="auto | cuda | cuda:0 | cpu")
    p.add_argument("--alpha", default="auto",
                   help="cross-child similarity threshold: a float, or 'auto'")
    return p.parse_args()


if __name__ == "__main__":
    args = _cli()
    alpha_arg = args.alpha if args.alpha == "auto" else float(args.alpha)
    run_hat_tree_demo(summarizer_mode=args.summarizer, device=args.device, alpha=alpha_arg)

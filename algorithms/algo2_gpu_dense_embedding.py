"""
ALGORITHM 2: GPU-ACCELERATED DENSE EMBEDDING

Maps the chunk collection C produced by Algorithm 1 into a dense matrix E of
L2-normalised vectors using a real transformer encoder, executed on CUDA when a
GPU is present and transparently on CPU otherwise.

    C = {c_1 ... c_n}   ->   E in R^{n x d},  ||e_i||_2 = 1

The encoder is a genuine sentence-embedding model (default:
sentence-transformers/all-MiniLM-L6-v2, d = 384). If the transformer stack or
the model weights are unavailable, the module degrades to a deterministic
feature-hashing encoder so that the indexing pipeline never hard-fails.
"""

import os

# Keep the transformer stack quiet and free of the TensorFlow import path.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import hashlib
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
# Algorithm 2 : dense embedding model
# ----------------------------------------------------------------------------
class DenseEmbeddingModel:
    """
    Batched, GPU-accelerated dense encoder.

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


# ----------------------------------------------------------------------------
# Demonstration
# ----------------------------------------------------------------------------
def _load_finance_chunks(max_docs: int = 5) -> List[Dict[str, Any]]:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from algo1_document_chunking import DocumentChunker, load_documents_corpus

    corpus = load_documents_corpus()
    corpus = dict(list(corpus.items())[:max_docs])
    chunker = DocumentChunker(chunk_size=60, chunk_overlap=15)
    chunks: List[Dict[str, Any]] = []
    for doc_id, text in corpus.items():
        chunks.extend(chunker.chunk_document(text, doc_id))
    return chunks


def run_embedding_demo(device: str = "auto", model_name: str = DEFAULT_MODEL):
    print("=" * 90)
    print(" ALGORITHM 2: GPU-ACCELERATED DENSE EMBEDDING")
    print("=" * 90)

    print("\n[Model Init] Loading dense encoder ...")
    embedder = DenseEmbeddingModel(model_name=model_name, device=device)
    print(f"  {embedder.banner()}")
    print(f"  Model load time: {embedder.stats.load_seconds:.2f}s")

    chunks = _load_finance_chunks()
    print(f"\n[Input] {len(chunks)} chunks from {len(set(c['doc_id'] for c in chunks))} documents (Algorithm 1 output)")

    print("\n[Forward Pass] Encoding chunk collection C -> embedding matrix E ...")
    E = embedder.encode_chunks(chunks)
    print(f"  Embedding matrix E: shape={E.shape}, dtype={E.dtype}")
    print(f"  Row norm check (should be 1.0): {float(np.linalg.norm(E[0])):.6f}")

    s = embedder.stats
    print("\n[Throughput]")
    print(f"  Texts encoded : {s.texts_encoded}")
    print(f"  Batches       : {s.batches} (max batch {s.peak_batch})")
    print(f"  Encode time   : {s.encode_seconds:.3f}s")
    print(f"  Throughput    : {s.throughput:.1f} texts/sec on {describe_device(s.device)}")

    # cache verification
    before = s.texts_encoded
    embedder.encode_chunks(chunks, show_progress=False)
    print(f"  Cache check   : re-encode added {s.texts_encoded - before} forward passes "
          f"({s.cache_hits} cache hits)")

    print("\n[Semantic Sanity Check] Nearest cross-document neighbours by cosine similarity:")
    sim = embedder.similarity(E)
    np.fill_diagonal(sim, -1.0)
    for i in range(min(4, len(chunks))):
        j = int(np.argmax(sim[i]))
        print(f"  {chunks[i]['doc_id'][:28]:<28} | {chunks[i]['section'][:34]:<34}")
        print(f"    -> closest: {chunks[j]['doc_id'][:28]:<28} | {chunks[j]['section'][:34]:<34} "
              f"| cos={sim[i][j]:.3f}")

    print("\n" + "=" * 90)
    print(" ALGORITHM 2 SUMMARY")
    print("=" * 90)
    for k, v in embedder.stats.as_dict().items():
        print(f"  {k:<26}: {v}")
    print("=" * 90)
    return embedder, E


if __name__ == "__main__":
    run_embedding_demo()

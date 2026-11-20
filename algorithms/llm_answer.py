"""
GENERATION: grounded answers from retrieved evidence (Groq / Gemini)

The last stage of HAT-RAG: the evidence set R chosen by any of the four
retrievers is numbered, packed into a prompt and sent to an LLM, which must
answer from those sources only and cite them as [1], [2], ...

Providers are reached over their OpenAI-compatible REST endpoints with plain
`requests`, so no SDK is needed:

    groq    https://api.groq.com/openai/v1             GROQ_API_KEY   (default)
    gemini  https://generativelanguage.googleapis.com/v1beta/openai
                                                       GEMINI_API_KEY / GOOGLE_API_KEY

Keys are read from the environment or a .env file at the project root (see
.env.example). Provider "auto" uses Groq and falls back to Gemini when Groq is
missing or rate-limited. The model is GROQ_MODEL / GEMINI_MODEL if set,
otherwise the first model in the preference list that the account's /models
endpoint actually offers.

    python algorithms/llm_answer.py --query "How does Tesla set its warranty reserves?"
    python algorithms/llm_answer.py --tree data/multihop_rag/trees/hat_multihop_full.json \
        --query "Who is facing a fraud trial according to The Verge and TechCrunch?"
"""

import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

import requests

ROOT = Path(__file__).resolve().parents[1]

PROVIDERS: Dict[str, Dict[str, Any]] = {
    "groq": {
        "label": "Groq",
        "base": "https://api.groq.com/openai/v1",
        "keys": ["GROQ_API_KEY"],
        "model_env": "GROQ_MODEL",
        "prefer": ["llama-3.3-70b-versatile", "openai/gpt-oss-120b", "meta-llama/llama-4-maverick-17b-128e-instruct",
                   "qwen/qwen3-32b", "llama-3.1-8b-instant", "openai/gpt-oss-20b"],
    },
    "gemini": {
        "label": "Gemini",
        "base": "https://generativelanguage.googleapis.com/v1beta/openai",
        "keys": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        "model_env": "GEMINI_MODEL",
        "prefer": ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.5-flash-lite", "gemini-1.5-flash"],
    },
}
# substrings that mark a non-chat model in a /models listing
NON_CHAT = ("whisper", "tts", "guard", "embed", "vision-preview", "playai", "orpheus", "imagen", "aqa")


def load_env(path: Optional[Path] = None) -> None:
    """Read KEY=VALUE lines from .env into os.environ (existing values win)."""
    path = path or ROOT / ".env"
    if not path.exists():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(path, override=False)
        return
    except ImportError:
        pass
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def api_key(provider: str) -> Optional[str]:
    load_env()
    for name in PROVIDERS[provider]["keys"]:
        val = os.environ.get(name, "").strip()
        if val:
            return val
    return None


def available_providers() -> List[str]:
    return [p for p in PROVIDERS if api_key(p)]


class LLMError(RuntimeError):
    def __init__(self, message: str, status: Optional[int] = None, retry_after: Optional[float] = None):
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after


# ----------------------------------------------------------------------------
# Client
# ----------------------------------------------------------------------------
_MODEL_CACHE: Dict[str, List[str]] = {}


class LLMClient:
    """One provider + model. chat() returns text; stream() yields text deltas."""

    def __init__(self, provider: str = "groq", model: Optional[str] = None, timeout: float = 60.0):
        if provider not in PROVIDERS:
            raise ValueError(f"unknown provider {provider!r}")
        self.provider = provider
        self.cfg = PROVIDERS[provider]
        self.key = api_key(provider)
        if not self.key:
            raise LLMError(f"No API key for {self.cfg['label']}: set {' or '.join(self.cfg['keys'])} in .env")
        self.timeout = timeout
        self.model = model or os.environ.get(self.cfg["model_env"], "").strip() or self.pick_model()

    @property
    def label(self) -> str:
        return f"{self.cfg['label']} · {self.model}"

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}

    def list_models(self) -> List[str]:
        if self.provider not in _MODEL_CACHE:
            try:
                r = requests.get(f"{self.cfg['base']}/models", headers=self._headers(), timeout=15)
                r.raise_for_status()
                ids = [m["id"].removeprefix("models/") for m in r.json().get("data", [])]
                _MODEL_CACHE[self.provider] = sorted(i for i in ids if not any(x in i.lower() for x in NON_CHAT))
            except (requests.RequestException, ValueError, KeyError):
                _MODEL_CACHE[self.provider] = []
        return _MODEL_CACHE[self.provider]

    def pick_model(self) -> str:
        offered = self.list_models()
        for m in self.cfg["prefer"]:
            if not offered or m in offered:
                return m
        return offered[0]

    @property
    def reasoning(self) -> bool:
        m = self.model.lower()
        return any(x in m for x in ("gpt-oss", "qwen3", "deepseek-r1", "-thinking", "gemini-2.5"))

    def _payload(self, messages, temperature, max_tokens, stream) -> Dict[str, Any]:
        body = {"model": self.model, "messages": messages, "temperature": temperature,
                "max_tokens": max_tokens, "stream": stream}
        if self.reasoning:
            # hidden reasoning tokens count against max_tokens: keep the effort low and
            # leave headroom, or a short answer budget is spent before any text appears
            body["max_tokens"] = max_tokens + 1024
            m = self.model.lower()
            if self.provider == "groq" and "gpt-oss" in m:
                body["reasoning_effort"] = "low"
            elif self.provider == "groq" and "qwen3" in m:
                body["reasoning_format"] = "hidden"
            elif self.provider == "gemini":
                body["reasoning_effort"] = "low"
        return body

    def _raise_for(self, r: requests.Response):
        if r.status_code < 400:
            return
        try:
            body = r.json()
            body = body[0] if isinstance(body, list) and body else body     # Gemini wraps errors in a list
            msg = (body.get("error") or {}).get("message") or r.text
        except (ValueError, AttributeError):
            msg = r.text
        retry = r.headers.get("retry-after")
        raise LLMError(f"{self.cfg['label']} HTTP {r.status_code}: {str(msg)[:300]}", r.status_code,
                       float(retry) if retry and retry.replace(".", "", 1).isdigit() else None)

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.1, max_tokens: int = 800,
             retries: int = 3) -> Dict[str, Any]:
        for attempt in range(retries + 1):
            t0 = time.perf_counter()
            r = requests.post(f"{self.cfg['base']}/chat/completions", headers=self._headers(),
                              json=self._payload(messages, temperature, max_tokens, False), timeout=self.timeout)
            try:
                self._raise_for(r)
            except LLMError as e:
                if e.status in (429, 500, 502, 503) and attempt < retries:
                    time.sleep(min(30.0, e.retry_after or 2.0 * (attempt + 1)))
                    continue
                raise
            data = r.json()
            return {"text": data["choices"][0]["message"]["content"] or "",
                    "usage": data.get("usage", {}), "seconds": time.perf_counter() - t0, "model": self.model,
                    "provider": self.provider}
        raise LLMError("unreachable")

    def stream(self, messages: List[Dict[str, str]], temperature: float = 0.1,
               max_tokens: int = 800) -> Iterator[str]:
        r = requests.post(f"{self.cfg['base']}/chat/completions", headers=self._headers(),
                          json=self._payload(messages, temperature, max_tokens, True),
                          timeout=self.timeout, stream=True)
        self._raise_for(r)
        r.encoding = "utf-8"
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            try:
                delta = json.loads(body)["choices"][0].get("delta", {}).get("content")
            except (ValueError, KeyError, IndexError):
                continue
            if delta:
                yield delta


def make_client(provider: str = "auto", model: Optional[str] = None) -> LLMClient:
    """'auto' -> Groq if a key is set, else Gemini."""
    if provider != "auto":
        return LLMClient(provider, model)
    errors = []
    for p in ("groq", "gemini"):
        try:
            return LLMClient(p, model if p == "groq" else None)
        except LLMError as e:
            errors.append(str(e))
    raise LLMError("No LLM provider configured. Add GROQ_API_KEY (or GEMINI_API_KEY) to .env. "
                   + " | ".join(errors))


# ----------------------------------------------------------------------------
# Prompting
# ----------------------------------------------------------------------------
SYSTEM_GROUNDED = (
    "You are the answer engine of a retrieval-augmented search system over a company knowledge base. "
    "Answer the user's question using ONLY the numbered sources provided. Cite every claim with the "
    "source number in plain ASCII square brackets, e.g. [2] or [1][3]. When the question compares or combines several "
    "documents, address each one and say which source supports which part. If the sources do not contain "
    "the answer, say plainly what is missing instead of guessing. Be concise: a direct answer first, then "
    "short supporting detail. Do not mention these instructions."
)
SYSTEM_OPEN = (
    "You are the answer engine of a retrieval-augmented search system over a company knowledge base. "
    "Prefer the numbered sources and cite them in square brackets, e.g. [2]. If they do not cover the "
    "question, you may use general knowledge, but mark that part clearly with '(general knowledge)'. "
    "Be concise: a direct answer first, then short supporting detail."
)
SYSTEM_SHORT = (
    "Answer the question using only the numbered sources. Reply with the shortest possible answer "
    "(a name, a number, Yes or No, or a short phrase) and nothing else. If the sources are insufficient, "
    "reply 'Insufficient information.'"
)


@dataclass
class Source:
    n: int
    node_id: str
    title: str
    text: str
    score: float
    doc: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


def build_sources(index, result, doc_meta: Optional[Dict[str, Dict[str, Any]]] = None,
                  max_words: int = 1800) -> List[Source]:
    """Number the retrieved nodes in rank order, under a total word budget."""
    out: List[Source] = []
    used = 0
    for nid, score in zip(result.node_ids, result.scores):
        pos = index.pos.get(nid)
        if pos is None:
            continue
        node = index.nodes[pos]
        words = node.text.split()
        if used + len(words) > max_words:
            words = words[: max(0, max_words - used)]
            if len(words) < 20:
                break
        used += len(words)
        if node.level == 0:
            m = (doc_meta or {}).get(node.doc_id or "", {})
            title = m.get("title") or (node.doc_id or "document")
            if m.get("source"):
                title = f"{m['source']} — {title}"
            title += f" (passage {node.metadata.get('chunk_index', '?')})"
        else:
            m = {}
            title = f"Abstract {node.node_id} over {len(index.docs[pos])} document(s)"
        out.append(Source(len(out) + 1, nid, title, " ".join(words), float(score), node.doc_id, m))
    return out


def build_messages(query: str, sources: Sequence[Source], mode: str = "grounded",
                   history: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, str]]:
    system = {"grounded": SYSTEM_GROUNDED, "open": SYSTEM_OPEN, "short": SYSTEM_SHORT}[mode]
    blocks = "\n\n".join(f"[{s.n}] {s.title}\n{s.text}" for s in sources) or "(no sources retrieved)"
    msgs = [{"role": "system", "content": system}]
    for turn in (history or [])[-4:]:                 # a little conversational memory
        msgs.append({"role": turn["role"], "content": turn["content"][:1500]})
    msgs.append({"role": "user", "content": f"Sources:\n\n{blocks}\n\nQuestion: {query}"})
    return msgs


_CITE_RE = re.compile(r"[\[【]\s*(\d{1,2}(?:\s*[,;†]\s*(?:L\d+-L\d+|\d{1,2}))*)\s*[\]】]")


def normalize_citations(text: str) -> str:
    """'【4】', '[1, 3]' and '【4†L1-L3】' -> '[4]', '[1][3]', '[4]'."""
    def fix(m):
        nums = [n for n in re.split(r"\s*[,;†]\s*", m.group(1)) if n.isdigit()]
        return "".join(f"[{n}]" for n in nums)
    return _CITE_RE.sub(fix, text)


def cited(text: str, n_sources: int) -> List[int]:
    """Source numbers the answer actually cites."""
    nums = {int(x) for x in re.findall(r"\[(\d{1,2})\]", normalize_citations(text))}
    return sorted(n for n in nums if 1 <= n <= n_sources)


def answer(index, retriever, query: str, k: int = 8, client: Optional[LLMClient] = None,
           doc_meta: Optional[Dict[str, Dict[str, Any]]] = None, mode: str = "grounded",
           history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    """Retrieve with `retriever`, then generate. Returns answer, sources, timings."""
    client = client or make_client()
    t0 = time.perf_counter()
    result = retriever.retrieve(query, k=k)
    t_retrieve = time.perf_counter() - t0
    sources = build_sources(index, result, doc_meta)
    reply = client.chat(build_messages(query, sources, mode, history))
    text = normalize_citations(reply["text"])
    return {"answer": text, "sources": sources, "cited": cited(text, len(sources)),
            "result": result, "retrieve_seconds": t_retrieve, "generate_seconds": reply["seconds"],
            "usage": reply["usage"], "model": client.label}


def _cli():
    import argparse
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from retrieval1_hat_beam_traversal import DEFAULT_TREE, HATBeamRetriever, TreeIndex

    p = argparse.ArgumentParser(description="Answer a question with HAT-RAG retrieval + an LLM")
    p.add_argument("--tree", default=str(DEFAULT_TREE))
    p.add_argument("--query", required=True)
    p.add_argument("-k", type=int, default=8)
    p.add_argument("--provider", default="auto", choices=["auto", "groq", "gemini"])
    p.add_argument("--model", default=None)
    a = p.parse_args()

    client = make_client(a.provider, a.model)
    index = TreeIndex.from_json(a.tree)
    meta = None
    if "multihop" in Path(a.tree).name:                # publisher + title instead of raw doc ids
        sys.path.insert(0, str(Path(__file__).resolve().parent / "compare-algos"))
        from multihop_corpus import article_meta, load_articles
        meta = article_meta(load_articles())
    out = answer(index, HATBeamRetriever(index), a.query, k=a.k, client=client, doc_meta=meta)
    print(f"[{out['model']}] retrieval {out['retrieve_seconds'] * 1000:.1f} ms, "
          f"generation {out['generate_seconds']:.2f} s\n")
    print(out["answer"], "\n")
    for s in out["sources"]:
        print(f"  [{s.n}] {s.title}  ({s.score:.3f})")


if __name__ == "__main__":
    _cli()

"""
BENCHMARK D: END-TO-END ANSWER ACCURACY ON MULTIHOP-RAG

Retrieval recall says whether the evidence reached the context window; this
says whether the answer came out right. For a stratified sample of held-out
MultiHop-RAG test questions, each of the four retrievers supplies its top-k
passages, the same LLM answers from them with a short-answer prompt, and the
answer is scored the way the MultiHop-RAG paper scores it: correct when the
normalised gold answer appears in the normalised response (Yes/No answers
must match exactly).

Every LLM response is cached in results/answers_cache.jsonl keyed by
(model, method, question), so a run interrupted by a rate limit or a daily
token cap resumes where it stopped.

    python algorithms/compare-algos/run_answer_eval.py --per-type 25
    python algorithms/compare-algos/run_answer_eval.py --provider gemini --model gemini-2.5-flash
"""

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from llm_answer import LLMError, build_messages, build_sources, make_client
from multihop_corpus import article_meta, load_articles, load_queries
from run_comparison import RESULTS_DIR, make_retrievers
from run_multihop import FULL_TREE, answerable
from retrieval1_hat_beam_traversal import TreeIndex

CACHE = RESULTS_DIR / "answers_cache.jsonl"
TYPES = ["inference", "comparison", "temporal"]


def normalise(text: str) -> str:
    text = text.lower().replace("’", "'")
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(re.sub(r"[^a-z0-9' ]+", " ", text).split())


def correct(response: str, gold: str) -> bool:
    r, g = normalise(response), normalise(gold)
    if g in ("yes", "no"):
        first = r.split()[:3]
        return g in first and ({"yes", "no"} - {g}).isdisjoint(first)
    return bool(g) and g in r


def load_cache() -> Dict[str, Dict[str, Any]]:
    out = {}
    if CACHE.exists():
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                out[rec["key"]] = rec
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--per-type", type=int, default=25, help="questions per question type")
    p.add_argument("-k", type=int, default=6, help="passages given to the LLM")
    p.add_argument("--provider", default="auto", choices=["auto", "groq", "gemini"])
    p.add_argument("--model", default=None)
    p.add_argument("--rpm", type=float, default=12.0, help="request rate cap (requests per minute)")
    a = p.parse_args()

    client = make_client(a.provider, a.model)
    print(f"[llm] {client.label}")
    articles = load_articles()
    queries, _ = load_queries(articles)
    meta = article_meta(articles)
    index = TreeIndex.from_json(FULL_TREE)
    retrievers = make_retrievers(index, reference=False)

    rng = random.Random(7)
    test = [q for q in answerable(index, queries) if q["split"] == "test"]
    sample: List[Dict[str, Any]] = []
    for t in TYPES:
        pool = [q for q in test if q["type"] == t]
        sample += rng.sample(pool, min(a.per_type, len(pool)))
    print(f"[sample] {len(sample)} test questions ({a.per_type} per type), k={a.k}")

    cache = load_cache()
    gap = 60.0 / a.rpm
    last = 0.0
    records = []
    with CACHE.open("a", encoding="utf-8") as fh:
        for n, q in enumerate(sample, 1):
            qvec = index.encode_query(q["query"])
            for m, r in retrievers.items():
                key = f"{client.model}|{m}|{q['id']}|k{a.k}"
                if key not in cache:
                    res = r.search(qvec, k=a.k, query=q["query"])
                    msgs = build_messages(q["query"], build_sources(index, res, meta), "short")
                    wait = gap - (time.time() - last)
                    if wait > 0:
                        time.sleep(wait)
                    try:
                        out = client.chat(msgs, temperature=0.0, max_tokens=60, retries=6)
                    except LLMError as e:
                        print(f"\n[stop] {e}\n       {len(cache)} answers cached; re-run to resume.")
                        return summarise(records, client, a, partial=True)
                    last = time.time()
                    cache[key] = {"key": key, "model": client.model, "method": m, "id": q["id"],
                                  "type": q["type"], "response": out["text"].strip(), "gold": q["answer"],
                                  "tokens": out["usage"].get("total_tokens")}
                    fh.write(json.dumps(cache[key]) + "\n")
                    fh.flush()
                rec = dict(cache[key])
                rec["correct"] = correct(rec["response"], q["answer"])
                records.append(rec)
            print(f"  {n}/{len(sample)} answered", end="\r")
    summarise(records, client, a)


def summarise(records, client, a, partial: bool = False):
    if not records:
        return
    methods = list(dict.fromkeys(r["method"] for r in records))
    ids = {r["id"] for r in records}
    complete = [i for i in ids if sum(r["id"] == i for r in records) == len(methods)]
    rows = [r for r in records if r["id"] in complete]
    summary = {}
    for m in methods:
        mine = [r for r in rows if r["method"] == m]
        summary[m] = {"accuracy": sum(r["correct"] for r in mine) / max(1, len(mine)), "n": len(mine),
                      "by_type": {t: sum(r["correct"] for r in mine if r["type"] == t) /
                                  max(1, sum(r["type"] == t for r in mine)) for t in TYPES}}
    out = {"model": client.model, "provider": client.provider, "k": a.k, "questions": len(complete),
           "partial": partial, "metric": "gold answer contained in the normalised response (MultiHop-RAG)",
           "summary": summary}
    (RESULTS_DIR / "answers.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[D] answer accuracy, {len(complete)} questions, {client.label}, k={a.k}")
    for m, s in summary.items():
        print(f"    {m:<18} {s['accuracy'] * 100:5.1f}%   " +
              "  ".join(f"{t} {v * 100:4.1f}%" for t, v in s["by_type"].items()))


if __name__ == "__main__":
    main()

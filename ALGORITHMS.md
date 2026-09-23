# The Two Algorithms — What They Do, Why They Differ, and How We Know They Work

This document covers the two algorithms that make up the completed half of HAT-RAG:

- **Algorithm 1** — document chunking (`algorithms/algo1_document_chunking.py`)
- **Algorithm 3** — hierarchical abstract tree construction (`algorithms/algo3_hierarchical_abstract_tree.py`)

Algorithm 3 also contains the dense encoder it depends on. Keeping it there is a
deliberate choice: the encoder is a component of tree construction, and folding it into
a separate module would fragment the code before the CUDA port, which is a later step.

---

## Part 1 — Algorithm 1: Document Chunking

### What it does

Turns a raw document into a list of overlapping passages, each carrying enough metadata
to be a useful node later:

```
{ chunk_id, doc_id, chunk_index, section, text, token_count, hash }
```

### The design decisions that matter

**Sentence boundaries, not character counts.** The splitter (`split_sentences`) uses a
regex with negative lookbehinds for abbreviations and initials, so it does not break on
`$383,285 million.` or `Inc.`. A fixed-width character splitter would routinely cut a
financial sentence in half, leaving a fragment like *"...reducing annual depreciation
expense by approximately"* — a chunk that embeds to nothing useful and pollutes whatever
cluster it lands in.

**Overlap carried as whole sentences.** When a chunk fills up, the tail sentences that
fit within `chunk_overlap` words are carried into the next chunk intact
(`algo1_document_chunking.py:86`). Overlap exists so that a fact spanning a chunk
boundary survives in at least one chunk; carrying half-sentences would defeat that.

**Section detection.** Headings (`# Section 2: Balance Sheet...`) are recognised and
consumed as labels rather than emitted as content, and each chunk records the section it
belongs to. The section label is prepended to the text at embedding time, which gives
the encoder context that a bare passage lacks.

**Content-addressed identity.** Each chunk gets a SHA-256 prefix of its own text, and
the chunk id embeds it. Identical text always produces the same id; changed text always
produces a different one. That makes the index diffable and makes duplicate detection
free.

### A real bug we found and fixed here

Section labels were originally stamped at *flush* time, using whichever heading had most
recently been seen. Because chunks are emitted lazily — a chunk is written only when the
next sentence would overflow it — a chunk could be written *after* the parser had already
walked into the following section. The result: chunks were labelled with a section they
did not contain. `Paper_01..._c0` held Section 1 text but was labelled "Section 2".

This was not cosmetic. The section label is prepended to the chunk text before
embedding, so a wrong label actively pushes a chunk toward the wrong cluster. The fix
(`_dominant_section`, `algo1_document_chunking.py:29`) tracks the section of every
sentence in a chunk and labels the chunk with the section that most of its sentences
came from, ties broken by document order.

### Complexity

Linear in corpus length, single pass, no model calls. On the finance corpus: 5 documents
→ 24 chunks, averaging 46.5 words, in well under a second.

---

## Part 2 — The dense encoder

Before Algorithm 3 can cluster anything, chunks must become vectors.

**What it is:** `sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions, batched, with
L2-normalised output so that every downstream dot product *is* cosine similarity.

**What it replaced:** the original code used a placeholder that hashed each word to seed
a random number generator and summed the resulting random vectors. That construction has
no semantic content whatsoever — two passages about balance sheets were no more similar
than a balance sheet and a blade-pitch specification. Every clustering and similarity
number produced under it was meaningless. This was the single most important change in
the project, because it is the difference between a tree that *looks* hierarchical and
one that *is* semantically hierarchical.

**Device handling:** `resolve_device()` returns CUDA when a GPU is present and CPU
otherwise; on GPU the batch size is quadrupled and the weights switch to fp16. Nothing
requires a GPU — all results here are CPU results — but the device argument is threaded
through every tensor operation so the later CUDA port is a switch, not a rewrite.

**Graceful degradation:** if the transformer stack or the weights are unavailable, the
encoder falls back to a deterministic feature-hashing scheme over word uni/bigrams and
character 4-grams rather than crashing. This is genuinely weaker than the transformer,
but it is not noise: shared vocabulary produces shared dimensions, so lexically similar
passages still score higher. In testing, the fallback correctly matched the two
"total assets" passages to each other.

**Caching:** encodings are cached by content hash, so the tree builder re-encoding the
leaf set costs nothing (24 cache hits in the recorded finance build).

---

## Part 3 — Algorithm 3: Hierarchical Abstract Tree Construction

### The loop

```
leaves   <- embed every chunk                         (L0)
level    <- 0
while level < max_levels and |current| > 1:
    level <- level + 1
    k     <- ceil(|current| / target_children)
    labels, centroids <- SphericalKMeans(k).fit(embeddings(current))
    for each cluster:
        if |cluster| == 1: promote the child unchanged
        else:
            abstract <- LLM summary of the cluster's texts
            parent   <- node(text=abstract, embedding=encode(abstract))
            attach every cluster member as a child of parent
    current <- the new parents
root <- the single remaining node, or a synthesised global abstract
add explicit alpha cross-child edges at every level
```

### Component by component

**Clustering — spherical k-means.** Embeddings are unit vectors, so cosine similarity is
a dot product and the assignment step is one `(n×d)@(d×k)` matmul. Seeding is k-means++
(farthest-point-weighted), iteration is capped at 50 with a 1e-5 tolerance, and empty
clusters are re-seeded from the worst-fitting point so `k` clusters always come back
populated. The finance build converges in 3–4 iterations per level, in 3–9 ms.

*What this replaced:* the original code assigned nodes to clusters with `index % k` —
pure round-robin. It was not clustering at all; it produced a tree whose shape was
determined by chunk ordering. Two passages landed together because of where they sat in
a list, not because they were about the same thing.

**Adaptive cluster count.** `k = ceil(n / target_children)` per level, instead of a fixed
branching factor. A fixed `k=2` forces 24 chunks into two buckets regardless of how many
distinct topics exist; the adaptive rule lets the corpus decide the shape and keeps
parents at a readable fan-out.

**Abstracts — real abstractive summarisation.** Each internal node's text is written by
`facebook/bart-large-cnn` over its children's text. The previous implementation
concatenated keyword lists into a template string (`"CLUSTER ABSTRACT (L1): Focus on
[...]"`), which is a label, not an abstract — it could not be read, and embedding it
produced a vector describing the template rather than the content. A centroid-based
extractive summariser is retained as a fallback (`--summarizer extractive`) for runs
where the LLM is unavailable or speed matters; it selects the sentences closest to the
cluster centroid.

**Parents live in the leaf vector space.** The generated abstract is embedded with the
same encoder, so an abstract node and a chunk node are directly comparable. This is what
makes top-down traversal possible later: a query vector can be scored against a level-2
abstract and a level-0 chunk with the same operation.

**Singleton collapse.** A cluster of one would otherwise produce a parent that merely
paraphrases its only child — a chain node that adds depth, costs an LLM call, and carries
no new information. Such children are promoted to the next level unchanged. One node was
promoted this way in the recorded finance build.

**Explicit α cross-child edges — the distinguishing feature.** After the hierarchy is
built, each node is linked to up to `max_cross_links` peers at the same level whose
cosine similarity is ≥ α *and which sit under a different parent*. Each edge is tagged
`cross_document` or `cross_cluster`.

These are the dashed edges in the architecture diagram, and they are the reason this is a
*cross-document* index rather than a hierarchy of per-document summaries. A strict tree
forces any relationship between two chunks under different parents to travel up to their
common ancestor and back down — through abstracts that have already discarded the detail
that made them related. The α edges preserve those relationships directly.

α can be a fixed float or `auto`, which sets the threshold at mean + 1σ of the observed
similarity distribution at that level, clamped to [0.35, 0.90]. The auto mode exists
because the right threshold is corpus-dependent: a fixed 0.60 that produces sensible
edges on the finance corpus produces zero edges on a heterogeneous corpus, and a
threshold that produces edges everywhere is not a threshold.

---

## Part 4 — How this differs from other approaches

| Approach | How it indexes | What it cannot do | What HAT does differently |
|---|---|---|---|
| **Flat vector RAG** | One embedding per chunk, one flat index | No notion of scale — a query needing a corpus-level overview gets k isolated passages and no synthesis | Abstracts at every level are first-class, retrievable nodes, so "what is the overall picture" has something to match against |
| **RAPTOR-style tree RAG** | Recursive clustering + summarisation into a tree | Relationships that cross the tree's branches are only reachable through a common ancestor | Explicit α cross-child edges, stored in the index, link related nodes under different parents directly |
| **GraphRAG** | Entity/relation graph extracted by an LLM | Extraction quality drives everything; no notion of abstraction level; expensive to build | Structure comes from the embedding geometry, not from LLM-extracted triples; abstraction level is explicit |
| **Per-document summarisation** | One summary tree per document | Cross-document synthesis never happens at index time | Clustering is run over the *pooled* chunk set, so a single abstract can and does span five companies |

The practical difference shows up in the recorded build: `L1_A5` is one abstract node
whose 8 children are balance-sheet passages from **all five companies**, and `L1_A0` is
one abstract over income-statement passages from four. Under per-document trees, those
nodes cannot exist. Under a strict tree with no α edges, the 17 cross-company
relationships recorded in this index would be unreachable except through the root.

---

## Part 5 — What we actually did to make it work

The original implementation had the right *shape* — chunk, cluster, summarise, recurse —
but every step that was supposed to carry meaning was a placeholder. The work was
replacing each placeholder with something real, and then fixing what that exposed:

| # | Change | Why it was necessary |
|---|---|---|
| 1 | Random-projection encoder → `all-MiniLM-L6-v2` | Similarities were meaningless; nothing downstream could be correct |
| 2 | `index % k` round-robin → spherical k-means with k-means++ | Grouping was determined by list order, not content |
| 3 | Keyword template strings → abstractive LLM summaries | Node text was a label, not readable content, and embedded as such |
| 4 | Fixed branching factor → adaptive `k = ceil(n/target)` | Tree shape was imposed rather than discovered |
| 5 | Added explicit α cross-child edges | The cross-document claim had no implementation at all |
| 6 | Fixed section mis-attribution in Algorithm 1 | Wrong section labels were feeding the encoder and skewing clusters |
| 7 | Stripped the decorative header from abstract text before summarising the level above | The `[ABSTRACT L1 | topics: ...]` prefix leaked into parent summaries, so the root's "topics" included the literal word *ABSTRACT* |
| 8 | Added singleton collapse | Chain nodes that paraphrase a single child add depth without information |
| 9 | Added `auto` α, per level | A single fixed threshold cannot be right across corpora |
| 10 | Added fallbacks (numpy k-means, hashing encoder, extractive summariser) | The pipeline must not hard-fail when torch, weights or the LLM are unavailable |
| 11 | Added build metrics, richer JSON schema, round-trip load | Without measurements there is no way to tell a good tree from a bad one |

---

## Part 6 — How we know it is working properly

Evidence from the committed artifact `algorithms/hat_finance_tree_structure.json`, built
on CPU with `seed=42`:

**Structure.** 24 leaf chunks → 5 L1 abstracts → 2 L2 abstracts → 1 root; 32 nodes total,
24× compression from leaves to root.

**Cluster quality.** Mean cohesion (cosine of each member to its own centroid) is 0.829 at
L1, 0.726 at L2, 0.853 at L3. Convergence in 3–4 iterations. Values near 0.8 on 384-d
unit vectors indicate clusters that are genuinely tight rather than arbitrary partitions.

**Semantic correctness, checked by reading it.** The clusters correspond to recognisable
accounting themes, not to file order:
- `L1_A5` (8 children, 5 companies, cohesion 0.802) — balance sheet and working capital
- `L1_A0` (8 children, 4 companies, cohesion 0.804) — income statement and margins
- `L1_A4` (2 children, 2 companies, cohesion 0.877) — revenue recognition under ASC 606
- `L1_A1` (2 children, 1 company, cohesion 0.922) — Tesla automotive segment revenue

**Cross-document linkage.** 29 α edges, **17 of them between different companies**, scores
0.587–0.685. The strongest links pair passages of comparable financial content across
companies: NVIDIA's results section with Microsoft's balance-sheet section (0.657),
Tesla's revenue section with Apple's executive summary (0.638), Microsoft's revenue
recognition policy with Amazon's revenue streams (0.587).

**Persistence integrity.** Every build reloads its own JSON and verifies node count, root
count and α edge count. The recorded run round-trips 32 nodes, 1 root, 29 edges.

**Degraded paths tested.** Forcing `HAS_TORCH=False` exercises the numpy k-means path and
still returns populated clusters; pointing the encoder at a non-existent model triggers
the hashing fallback and still completes an end-to-end build; `k > n`, single-chunk and
empty corpora all return sane trees rather than raising.

**Determinism.** Seeded k-means: repeated runs of the same command produce the same
topology.

**Cost.** Embedding 0.33 s, clustering 0.06 s, α edges 0.009 s, LLM summarisation 43.8 s —
44.2 s total, on CPU. The summariser is 99% of the cost, which is worth stating plainly:
the parts people assume need a GPU are the parts that are already effectively free at
this corpus size.

---

## Part 7 — Honest limitations

- **The corpus is small.** 5 documents, 24 chunks. The structure is correct and the
  clusters are meaningful, but scaling behaviour is untested; adaptive `k` and the α
  threshold will need re-examination at thousands of chunks.
- **`bart-large-cnn` is extraction-leaning.** Its abstracts are fluent and faithful but
  often lead with one child's content rather than synthesising across all of them. A
  summariser better suited to multi-document input would improve the upper levels.
- **α edges concentrate at the leaf level.** In the recorded build all 29 edges are L0.
  Higher-level abstracts are few and semantically spread out, so cross-parent pairs do
  not clear the auto threshold. This is honest behaviour rather than a bug, but it means
  the value of α edges above the leaves is currently unproven.
- **No retrieval, so no end-task metric.** Tree quality is evidenced by cohesion,
  cross-document composition and inspection. There is no recall@k or answer-accuracy
  number yet, because there is no retriever — that is the next half of the work.
- **CUDA is untested on real hardware.** The device paths are written and guarded, but
  every measurement here is CPU.

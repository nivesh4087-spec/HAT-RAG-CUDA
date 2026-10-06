### Verdict

| Rank | Method | Finance recall@5 | Synthetic recall@5 (mean, 5 sizes) | Combined score | Corpus scored / query |
|---:|---|---:|---:|---:|---:|
| 1 | **HAT-RAG** | 85.0% | 96.7% | 90.9 | 39% |
| 2 | Graph PPR | 83.4% | 93.8% | 88.6 | 100% |
| 3 | Flat dense | 83.9% | 88.6% | 86.3 | 100% |
| 4 | RAPTOR collapsed | 74.6% | 85.1% | 79.9 | 118% |

### Finance corpus: all metrics (52 queries, k = 5)

| Method | Recall@5 | Hit@5 | All gold found | MRR | Filing coverage | Precision | Nodes scored | Latency (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HAT-RAG | 85.0% | 90.4% | 78.8% | 0.660 | 96.8% | 67.7% | 23.9 | 0.295 |
| Flat dense | 83.9% | 90.4% | 75.0% | 0.677 | 98.1% | 62.3% | 24.0 | 0.085 |
| RAPTOR collapsed | 74.6% | 76.9% | 69.2% | 0.600 | 97.7% | 80.4% | 32.0 | 0.086 |
| Graph PPR | 83.4% | 90.4% | 75.0% | 0.678 | 97.1% | 64.6% | 24.0 | 0.204 |
| HAT-RAG (paper spec) | 62.6% | 67.3% | 55.8% | 0.568 | 100.0% | 39.6% | 23.3 | 0.047 |

### Finance corpus: recall@5 by query type

| Method | Single-hop (31) | Cross-document (12) | Thematic (9) |
|---|---:|---:|---:|
| HAT-RAG | 100.0% | 54.2% | 74.6% |
| Flat dense | 96.8% | 58.3% | 73.5% |
| RAPTOR collapsed | 90.3% | 33.3% | 75.7% |
| Graph PPR | 96.8% | 54.2% | 76.5% |
| HAT-RAG (paper spec) | 71.0% | 33.3% | 72.6% |

### Synthetic scaling: recall@5 · nodes scored · latency

| Method | 8 docs, 99 chunks | 16 docs, 196 chunks | 32 docs, 393 chunks | 64 docs, 786 chunks | 128 docs, 1,582 chunks |
|---|---:|---:|---:|---:|---:|
| HAT-RAG | 100.0% · 54 · 0.64 ms | 92.7% · 65 · 0.82 ms | 97.3% · 84 · 0.85 ms | 97.2% · 104 · 0.70 ms | 96.2% · 154 · 1.39 ms |
| Flat dense | 100.0% · 99 · 0.18 ms | 94.8% · 196 · 0.14 ms | 88.4% · 393 · 0.16 ms | 82.6% · 786 · 0.51 ms | 77.4% · 1,582 · 1.67 ms |
| RAPTOR collapsed | 97.7% · 114 · 0.19 ms | 86.5% · 226 · 0.14 ms | 86.6% · 451 · 0.18 ms | 79.9% · 901 · 0.58 ms | 75.0% · 1,807 · 1.91 ms |
| Graph PPR | 98.9% · 99 · 0.47 ms | 93.8% · 196 · 0.49 ms | 93.8% · 393 · 0.63 ms | 93.8% · 786 · 1.04 ms | 88.9% · 1,582 · 3.83 ms |
| HAT-RAG (paper spec) | 93.2% · 51 · 0.20 ms | 81.2% · 61 · 0.18 ms | 80.4% · 75 · 0.19 ms | 72.9% · 90 · 0.20 ms | 49.0% · 135 · 0.78 ms |

### Index sizes per synthetic corpus

| Documents | Chunks N | Tree nodes | Level sizes (leaf to root) | Tree build (s) | Graph build (s) |
|---:|---:|---:|---|---:|---:|
| 8 | 99 | 114 | 99 / 13 / 1 / 1 | 1.4 | 0.00 |
| 16 | 196 | 226 | 196 / 25 / 4 / 1 | 3.1 | 0.01 |
| 32 | 393 | 451 | 393 / 50 / 7 / 1 | 6.5 | 0.03 |
| 64 | 786 | 901 | 786 / 99 / 13 / 2 / 1 | 13.3 | 0.07 |
| 128 | 1,582 | 1,807 | 1582 / 196 / 25 / 3 / 1 | 28.6 | 0.22 |

### HAT-RAG ablations: recall@5

| Variant | Finance | Synthetic, 64 docs | Nodes scored (synthetic) |
|---|---:|---:|---:|
| full HAT-RAG | 85.0% | 97.2% | 104 |
| - entity linking | 80.2% | 72.9% | 96 |
| - PageRank (cosine ranking) | 78.4% | 86.8% | 104 |
| - reading-order edges | 82.6% | 97.2% | 101 |
| - alpha edges | 83.4% | 97.2% | 75 |
| - diversity | 85.0% | 93.8% | 104 |
| ungated diversity | 85.0% | 100.0% | 104 |
| paper spec | 62.6% | 72.9% | 90 |

### Beam width (synthetic, 64 docs)

| β | HAT-RAG recall@5 | nodes scored | paper spec recall@5 | nodes scored |
|---:|---:|---:|---:|---:|
| 1 | 97.9% | 69 | 58.3% | 54 |
| 2 | 97.9% | 90 | 70.8% | 76 |
| 3 | 97.2% | 104 | 72.9% | 90 |
| 5 | 97.9% | 131 | 81.2% | 118 |
| 8 | 97.9% | 160 | 79.2% | 148 |
| 12 | 97.9% | 201 | 82.6% | 190 |
| 16 | 97.9% | 219 | 82.6% | 209 |
| 24 | 97.9% | 271 | 83.3% | 262 |
| 32 | 97.9% | 323 | 84.0% | 314 |

### Beam width and diversity weight (finance)

| β | recall@5 | nodes scored |
|---:|---:|---:|
| 1 | 69.5% | 18.1 |
| 2 | 83.1% | 21.2 |
| 3 | 85.0% | 23.9 |
| 4 | 85.4% | 26.5 |
| 5 | 87.3% | 24.0 |
| 6 | 87.3% | 24.0 |

| λ | recall@5 | filing coverage | precision |
|---:|---:|---:|---:|
| 0.0 | 85.0% | 96.8% | 68.1% |
| 0.05 | 85.0% | 96.8% | 68.1% |
| 0.1 | 85.0% | 96.8% | 68.1% |
| 0.15 | 85.0% | 96.8% | 67.7% |
| 0.2 | 85.0% | 96.8% | 67.7% |
| 0.3 | 85.0% | 96.8% | 67.3% |
| 0.5 | 85.0% | 96.8% | 66.5% |
| 1.0 | 83.1% | 96.8% | 64.2% |

### Per-query head-to-head (all corpora)

| HAT-RAG vs | better | same | worse |
|---|---:|---:|---:|
| Flat dense | 69 | 300 | 7 |
| RAPTOR collapsed | 84 | 289 | 3 |
| Graph PPR | 31 | 341 | 4 |

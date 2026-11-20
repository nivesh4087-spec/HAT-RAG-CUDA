### Verdict

| Rank | Method | Finance recall@5 | Synthetic recall@5 (mean, 5 sizes) | Combined score | Corpus scored / query |
|---:|---|---:|---:|---:|---:|
| 1 | **HAT-RAG** | 85.0% | 96.7% | 90.9 | 30% |
| 2 | Graph PPR | 83.4% | 92.4% | 87.9 | 100% |
| 3 | Flat dense | 83.9% | 79.5% | 81.7 | 100% |
| 4 | RAPTOR collapsed | 74.6% | 76.2% | 75.4 | 117% |

### Finance corpus: all metrics (52 queries, k = 5)

| Method | Recall@5 | Hit@5 | All gold found | MRR | Filing coverage | Precision | Nodes scored | Latency (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HAT-RAG | 85.0% | 90.4% | 78.8% | 0.660 | 96.8% | 67.7% | 23.9 | 0.790 |
| Flat dense | 83.9% | 90.4% | 75.0% | 0.677 | 98.1% | 62.3% | 24.0 | 0.194 |
| RAPTOR collapsed | 74.6% | 76.9% | 69.2% | 0.600 | 97.7% | 80.4% | 32.0 | 0.197 |
| Graph PPR | 83.4% | 90.4% | 75.0% | 0.678 | 97.1% | 64.6% | 24.0 | 1.454 |
| HAT-RAG (paper spec) | 62.6% | 67.3% | 55.8% | 0.568 | 100.0% | 39.6% | 23.3 | 0.204 |

### Finance corpus: recall@5 by query type

| Method | Single-hop (31) | Cross-document (12) | Thematic (9) |
|---|---:|---:|---:|
| HAT-RAG | 100.0% | 54.2% | 74.6% |
| Flat dense | 96.8% | 58.3% | 73.5% |
| RAPTOR collapsed | 90.3% | 33.3% | 75.7% |
| Graph PPR | 96.8% | 54.2% | 76.5% |
| HAT-RAG (paper spec) | 71.0% | 33.3% | 72.6% |

### Synthetic scaling: recall@5 · nodes scored · latency

| Method | 8 docs, 99 chunks | 16 docs, 196 chunks | 32 docs, 393 chunks | 64 docs, 786 chunks | 128 docs, 1,582 chunks | 256 docs, 3,154 chunks | 512 docs, 6,332 chunks |
|---|---:|---:|---:|---:|---:|---:|---:|
| HAT-RAG | 100.0% · 54 · 1.14 ms | 92.7% · 65 · 1.28 ms | 97.3% · 84 · 1.28 ms | 97.2% · 104 · 1.67 ms | 96.2% · 154 · 1.86 ms | 96.1% · 174 · 3.14 ms | 97.6% · 199 · 4.86 ms |
| Flat dense | 100.0% · 99 · 0.36 ms | 94.8% · 196 · 0.36 ms | 88.4% · 393 · 0.36 ms | 82.6% · 786 · 0.40 ms | 77.4% · 1,582 · 0.41 ms | 62.5% · 3,154 · 0.50 ms | 50.8% · 6,332 · 0.68 ms |
| RAPTOR collapsed | 97.7% · 114 · 0.29 ms | 86.5% · 226 · 0.33 ms | 86.6% · 451 · 0.31 ms | 79.9% · 901 · 0.37 ms | 75.0% · 1,807 · 0.36 ms | 57.7% · 3,599 · 0.47 ms | 49.7% · 7,218 · 0.68 ms |
| Graph PPR | 98.9% · 99 · 1.70 ms | 93.8% · 196 · 1.97 ms | 93.8% · 393 · 2.17 ms | 93.8% · 786 · 2.72 ms | 88.9% · 1,582 · 3.67 ms | 89.6% · 3,154 · 5.81 ms | 88.3% · 6,332 · 10.27 ms |
| HAT-RAG (paper spec) | 93.2% · 51 · 0.35 ms | 81.2% · 61 · 0.50 ms | 80.4% · 75 · 0.57 ms | 72.9% · 90 · 0.64 ms | 49.0% · 135 · 0.89 ms | 42.0% · 144 · 1.10 ms | 29.6% · 155 · 1.01 ms |

### Index sizes per synthetic corpus

| Documents | Chunks N | Tree nodes | Level sizes (leaf to root) | Tree build (s) | Graph build (s) |
|---:|---:|---:|---|---:|---:|
| 8 | 99 | 114 | 99 / 13 / 1 / 1 | 3.4 | 0.01 |
| 16 | 196 | 226 | 196 / 25 / 4 / 1 | 5.7 | 0.02 |
| 32 | 393 | 451 | 393 / 50 / 7 / 1 | 10.9 | 0.03 |
| 64 | 786 | 901 | 786 / 99 / 13 / 2 / 1 | 22.4 | 0.06 |
| 128 | 1,582 | 1,807 | 1582 / 196 / 25 / 3 / 1 | 44.7 | 0.16 |
| 256 | 3,154 | 3,599 | 3154 / 389 / 49 / 6 / 1 | 89.9 | 0.44 |
| 512 | 6,332 | 7,218 | 6332 / 775 / 96 / 12 / 2 / 1 | 183.7 | 1.43 |

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
| Flat dense | 361 | 471 | 8 |
| RAPTOR collapsed | 389 | 447 | 4 |
| Graph PPR | 108 | 728 | 4 |

### MultiHop-RAG: full corpus (22,838 passages, 25,673 nodes; test split, 1,789 questions)

| Method | Recall@5 | Recall@10 | Hit@5 | All evidence@10 | MRR@5 | Nodes scored | Share of index | Latency (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HAT-RAG | 20.5% | 27.8% | 45.3% | 5.9% | 0.304 | 936 | 3.6% | 67.68 |
| Flat dense | 20.9% | 28.7% | 45.5% | 5.3% | 0.297 | 22,838 | 89.0% | 2.64 |
| RAPTOR collapsed | 19.9% | 27.1% | 43.2% | 4.5% | 0.284 | 25,673 | 100.0% | 2.38 |
| Graph PPR | 23.7% | 29.9% | 49.4% | 8.3% | 0.320 | 22,838 | 89.0% | 68.54 |
| HAT-RAG (paper spec) | 10.7% | 15.8% | 25.4% | 0.7% | 0.173 | 203 | 0.8% | 1.41 |

| Method (recall@5) | Inference | Comparison | Temporal |
|---|---:|---:|---:|
| HAT-RAG | 13.3% | 28.5% | 18.6% |
| Flat dense | 14.1% | 30.0% | 16.8% |
| RAPTOR collapsed | 13.2% | 28.5% | 16.2% |
| Graph PPR | 16.5% | 32.8% | 20.2% |
| HAT-RAG (paper spec) | 8.1% | 13.8% | 9.4% |

| HAT-RAG vs (recall@5, per question) | better | same | worse |
|---|---:|---:|---:|
| Flat dense | 139 | 1495 | 155 |
| RAPTOR collapsed | 176 | 1470 | 143 |
| Graph PPR | 135 | 1392 | 262 |
| HAT-RAG (paper spec) | 464 | 1246 | 79 |

| Variant (MultiHop-RAG, test, k = 5) | Recall@5 | Nodes scored |
|---|---:|---:|
| full HAT-RAG | 20.5% | 936 |
| - entity linking | 14.7% | 293 |
| - PageRank (cosine ranking) | 20.2% | 936 |
| - reading-order edges | 20.5% | 935 |
| - alpha edges | 19.6% | 845 |
| - diversity | 20.5% | 936 |
| ungated diversity | 20.6% | 936 |
| paper spec | 10.7% | 203 |
| capped links (beta=3, max_link=200) | 19.8% | 262 |

| β | Recall@5 | Nodes scored |
|---:|---:|---:|
| 1 | 18.5% | 765 |
| 2 | 19.8% | 816 |
| 3 | 20.0% | 858 |
| 5 | 20.5% | 936 |
| 8 | 20.9% | 1,031 |
| 16 | 20.9% | 1,271 |

### MultiHop-RAG growing haystack (761 fixed test questions): recall@5 · nodes scored · latency

| Method | 100 articles, 4,207 passages | 200 articles, 7,861 passages | 400 articles, 14,934 passages | 609 articles, 22,838 passages |
|---|---:|---:|---:|---:|
| HAT-RAG | 18.2% · 318 · 12.7 ms | 17.8% · 369 · 14.1 ms | 16.6% · 563 · 18.3 ms | 15.7% · 761 · 55.0 ms |
| Flat dense | 18.6% · 4,207 · 0.9 ms | 17.9% · 7,861 · 1.3 ms | 16.6% · 14,934 · 1.8 ms | 16.2% · 22,838 · 2.2 ms |
| RAPTOR collapsed | 17.5% · 4,718 · 0.8 ms | 17.1% · 8,839 · 1.2 ms | 15.2% · 16,776 · 1.5 ms | 15.4% · 25,673 · 2.3 ms |
| Graph PPR | 19.8% · 4,207 · 18.6 ms | 19.2% · 7,861 · 31.8 ms | 18.2% · 14,934 · 26.1 ms | 17.7% · 22,838 · 82.8 ms |
| HAT-RAG (paper spec) | 12.1% · 149 · 2.0 ms | 11.5% · 157 · 1.8 ms | 10.9% · 193 · 2.8 ms | 9.1% · 202 · 2.7 ms |

### End-to-end answer accuracy (45 test questions, openai/gpt-oss-120b, k = 5)

| Method | Accuracy | Inference | Comparison | Temporal |
|---|---:|---:|---:|---:|
| HAT-RAG | 35.6% | 73.3% | 6.7% | 26.7% |
| Flat dense | 44.4% | 80.0% | 26.7% | 26.7% |
| RAPTOR collapsed | 35.6% | 73.3% | 20.0% | 13.3% |
| Graph PPR | 46.7% | 86.7% | 33.3% | 20.0% |

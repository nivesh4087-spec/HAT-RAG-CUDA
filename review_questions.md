# Review Questions: HAT-RAG

| Review Question | Answer |
|---|---|
| **Has resource/environmental impact been considered?** | Yes. The tree is built once offline and cached, and top-down traversal scores only ~20% of nodes, not 100% as in flat RAG. <br> It runs on a CPU; a GPU is optional, and small local models (MiniLM, BART) keep power use low. |
| **Is the system maintainable and scalable?** | Yes. The pipeline is modular (chunk → embed → tree → JSON), and each stage caches its output so it can be rerun on its own. <br> Search cost is O(k log N), and moving to CUDA is a device switch rather than a rewrite. |
| **Have potential safety risks been identified?** | Yes. LLM summaries can hallucinate or drop facts, so every answer links back to the leaf passages it came from. <br> It is a research and decision-support tool, and financial outputs are not presented as advice. |
| **Have security threats and controls been identified?** | Yes. Threats: prompt injection from source documents, data leakage, and tampered model weights. <br> Controls: runs fully locally with no external API calls, models come from trusted hubs, and the planned API will validate input with Pydantic. |
| **Are privacy, consent, fairness and IP considered?** | Yes. Documents never leave the local machine, and the test data is public finance filings with no personal data. <br> All models and libraries are openly licensed (MIT/Apache). Abstracts can carry embedding bias, so answers cite their sources. |
| **Has feasibility/cost been estimated?** | Yes. It needs no paid APIs or licences, only a standard laptop (8 GB RAM, ~3 GB disk) with an optional NVIDIA GPU. <br> It is already working on the finance dataset, so the remaining cost is developer time for the API/UI and the CUDA port. |

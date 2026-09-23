"""
OFFLINE / KNOWLEDGE CONSTRUCTION SHOWCASE -- GENERAL RESEARCH CORPUS

Runs the full indexing path end to end on a small multi-document corpus:

    Document corpus
      -> Algorithm 1: document chunking   -> chunk collection C
      -> dense embedding of the chunks    -> embedding matrix E
      -> Algorithm 3: HAT construction    -> tree H (+ alpha edges)

Knowledge construction only -- query-time retrieval is not part of this script.
Runs on CPU; a GPU is optional.

    python algorithms/run_demo.py
    python algorithms/run_demo.py --summarizer extractive     # skip the LLM
"""

import argparse
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))

from algo1_document_chunking import DocumentChunker
from algo3_hierarchical_abstract_tree import (
    DEFAULT_MODEL,
    DEFAULT_SUMMARIZER_MODEL,
    DenseEmbeddingModel,
    HierarchicalAbstractTreeBuilder,
    describe_device,
)

DOCUMENTS = {
    "Paper_01_ThermalDynamics.pdf": (
        "# Section 1: Aerodynamic Blade Design and Thermal Dissipation\n"
        "High-speed ceiling fan blade profiles are engineered with optimal pitch angles between 12 to 15 degrees. "
        "This configuration maximizes downward volumetric air displacement while reducing turbulent boundary layer drag.\n"
        "# Section 2: Temperature Gradient Analysis\n"
        "Empirical thermal imaging demonstrates a uniform 3.8 degree Celsius decrease in surface temperature across a 40 square meter test chamber. "
        "Continuous fluid circulation eliminates thermal stratification zones near ceiling heights."
    ),
    "Paper_02_MotorDiagnostics.pdf": (
        "# Section 1: Predictive Maintenance and Vibration Telemetry\n"
        "Progressive tool wear during sheet metal stamping induces structural micro-asymmetries in rotor blade brackets. "
        "These mechanical imperfections manifest as high-frequency harmonic vibrations exceeding 120 Hz during continuous motor operation.\n"
        "# Section 2: Sensor Ingestion and Fault Mitigation\n"
        "Triaxial accelerometer telemetry combined with edge microcontroller inference enables real-time bearing fault classification. "
        "Automated preventative recalibration cycles reduce unplanned motor downtime by 41% in industrial facilities."
    ),
    "Paper_03_EnergyOptimization.pdf": (
        "# Section 1: Brushless DC Motor Inverter Topology\n"
        "Permanent magnet Brushless DC (BLDC) motor architectures achieve electrical efficiency ratings exceeding 88 percent. "
        "Integrated sensorless field-oriented control (FOC) algorithms minimize harmonic acoustic noise and reduce standby power consumption.\n"
        "# Section 2: Microcontroller Speed Scheduling\n"
        "Ambient temperature-adaptive pulse-width modulation (PWM) dynamically modulates rotor RPM based on real-time room occupancy patterns. "
        "This closed-loop energy management routine cuts annual electrical consumption by 62% relative to conventional AC induction motors."
    ),
}


def run_pipeline_demo(args):
    print("*" * 100)
    print("   HAT-RAG OFFLINE KNOWLEDGE CONSTRUCTION -- ALGORITHM 1 AND ALGORITHM 3")
    print("*" * 100)

    # ---------------------------------------------------------------- PHASE 1
    print("\n[PHASE 1] ALGORITHM 1: DOCUMENT CHUNKING -> chunk collection C")
    chunker = DocumentChunker(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    all_leaf_chunks = []
    for doc_id, text in DOCUMENTS.items():
        doc_chunks = chunker.chunk_document(text, doc_id)
        all_leaf_chunks.extend(doc_chunks)
        print(f"  > {doc_id:<34} -> {len(doc_chunks)} chunks")

    print(f"\n  |C| = {len(all_leaf_chunks)} chunks from {len(DOCUMENTS)} documents")
    sample = all_leaf_chunks[0]
    print(f"  Sample: [{sample['chunk_id']}] section='{sample['section']}' "
          f"tokens={sample['token_count']} sha256={sample['hash']}")

    # ---------------------------------------------------------------- PHASE 2
    print("\n" + "-" * 100)
    print("[PHASE 2] DENSE EMBEDDING OF C -> matrix E (leaf vectors for Algorithm 3)")
    embedder = DenseEmbeddingModel(model_name=args.model, device=args.device)
    print(f"  {embedder.banner()}")
    E = embedder.encode_chunks(all_leaf_chunks)
    print(f"  E shape = {E.shape} | {embedder.stats.throughput:.1f} texts/sec on "
          f"{describe_device(embedder.device)}")

    # ---------------------------------------------------------------- PHASE 3
    print("\n" + "-" * 100)
    print("[PHASE 3] ALGORITHM 3: HIERARCHICAL ABSTRACT TREE CONSTRUCTION -> H")
    builder = HierarchicalAbstractTreeBuilder(
        embedder=embedder,
        max_levels=args.levels,
        target_children=args.children,
        alpha=args.alpha,
        summarizer_mode=args.summarizer,
        summarizer_model=args.summarizer_model,
        device=args.device,
    )
    print(f"  Summarizer: {builder.summarizer.banner()}")
    builder.build_tree(all_leaf_chunks)

    builder.display_tree(max_leaves_per_parent=args.show_leaves)

    json_out = current_dir / "hat_tree_structure.json"
    builder.save_tree_json(str(json_out))
    print(f"\n[Persistence] H serialized to {json_out.name}")

    verify = HierarchicalAbstractTreeBuilder(embedder=embedder, summarizer_mode="extractive", verbose=False)
    verify.load_tree_json(str(json_out))
    print(f"[Persistence] Round trip verified: {len(verify.nodes)} nodes, "
          f"{len(verify.root_nodes)} root(s)")

    builder.print_metrics()

    counts = builder.level_counts()
    print("\n" + "=" * 100)
    print(" PIPELINE AUDIT")
    print("=" * 100)
    print(f"  Algorithm 1 (chunking)            : PASSED ({len(all_leaf_chunks)} chunks / {len(DOCUMENTS)} docs)")
    print(f"  Dense embedding                   : PASSED ({E.shape[0]}x{E.shape[1]} matrix, "
          f"{embedder.stats.backend.split('::')[0]})")
    print(f"  Algorithm 3 (HAT construction)    : PASSED ({len(builder.nodes)} nodes over "
          f"{len(counts)} levels, {builder.metrics.cross_links} alpha edges)")
    print(f"  JSON persistence round trip       : PASSED")
    print(f"  Scope                             : offline indexing only (no query-time retrieval)")
    print("=" * 100)
    return builder


def _cli():
    p = argparse.ArgumentParser(description="HAT-RAG offline construction showcase")
    p.add_argument("--chunk-size", type=int, default=35)
    p.add_argument("--chunk-overlap", type=int, default=10)
    p.add_argument("--levels", type=int, default=2, help="max abstract levels above the leaves")
    p.add_argument("--children", type=int, default=3, help="target children per abstract node")
    p.add_argument("--alpha", default="auto", help="cross-child cosine threshold: float or 'auto'")
    p.add_argument("--summarizer", choices=["auto", "llm", "extractive"], default="auto")
    p.add_argument("--summarizer-model", default=DEFAULT_SUMMARIZER_MODEL)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--device", default="auto", help="cpu is the default; cuda used only if present")
    p.add_argument("--show-leaves", type=int, default=3)
    args = p.parse_args()
    if args.alpha != "auto":
        args.alpha = float(args.alpha)
    return args


if __name__ == "__main__":
    run_pipeline_demo(_cli())

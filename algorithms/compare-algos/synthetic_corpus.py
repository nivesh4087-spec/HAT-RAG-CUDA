"""
Synthetic cross-document corpus for the scaling study.

Adapted from the paper's benchmark generator (IBERAMIA/experiments): M
technical dossiers, eight sections each, one planted fact per section buried
in distractor sentences drawn from a shared pool, plus bridge sentences that
tie a component in document A to a programme in document B. Gold evidence is
known by construction, so the corpus can be grown from ~100 to ~1,600 leaf
chunks while every query stays exactly scorable.

Gold items use the same format as finance_queries.json:
    {"doc": doc_id, "terms": [...]}
"""

import random
from typing import Any, Dict, List, Tuple

TOPICS = [
    ("thermal", "thermal management", ["heat sink", "coolant loop", "thermal interface", "fan curve", "duct geometry"]),
    ("power", "power electronics", ["inverter stage", "gate driver", "DC link", "rectifier bank", "current sensor"]),
    ("storage", "energy storage", ["cell stack", "electrolyte blend", "state estimator", "balancing board", "pack housing"]),
    ("control", "control software", ["scheduler core", "watchdog task", "state machine", "telemetry buffer", "fault handler"]),
    ("network", "network fabric", ["edge router", "switch plane", "link aggregator", "queue manager", "time sync unit"]),
    ("sensing", "sensor instrumentation", ["strain array", "optical probe", "pressure cell", "drift monitor", "sampling head"]),
    ("materials", "structural materials", ["alloy sheet", "composite ply", "weld seam", "coating layer", "fastener set"]),
    ("safety", "operational safety", ["interlock chain", "isolation valve", "audit trail", "shutdown path", "clearance log"]),
    ("logistics", "supply logistics", ["staging depot", "transit lane", "packaging spec", "yard sequencer", "customs record"]),
    ("quality", "quality assurance", ["sampling plan", "gauge study", "defect taxonomy", "review board", "release gate"]),
    ("maintenance", "maintenance programme", ["inspection cycle", "spares kit", "torque schedule", "wear model", "service log"]),
    ("compliance", "regulatory compliance", ["emission clause", "reporting window", "conformity file", "notified body", "label rule"]),
]

FILLER = [
    "Operating procedures are reviewed each quarter by the responsible engineering group.",
    "Deviation from the nominal configuration must be recorded in the change register.",
    "Historical records indicate stable behaviour across the last four reporting periods.",
    "The subsystem interacts with adjacent modules through a documented interface contract.",
    "Field data is aggregated weekly and archived for longitudinal analysis.",
    "Commissioning tests are repeated after any firmware or hardware revision.",
    "Threshold values are derived from the internal design standard and vendor datasheets.",
    "Anomalies are escalated to the review board when two consecutive samples fall out of band.",
    "Capacity planning assumes the reference duty cycle described in the design dossier.",
    "The measurement chain is calibrated against a traceable laboratory reference.",
    "Redundant paths are exercised during scheduled availability windows.",
    "Documentation is versioned together with the corresponding configuration baseline.",
]

METRICS = ["nominal efficiency", "mean latency", "duty margin", "failure rate", "settling time",
           "throughput index", "leakage current", "recovery interval"]


def build_corpus(n_docs: int = 16, paras_per_doc: int = 8, seed: int = 17,
                 max_single: int = 40) -> Tuple[Dict[str, str], List[Dict[str, Any]], Dict[str, List[str]]]:
    """Return (documents, queries, doc_aliases)."""
    rng = random.Random(seed)
    documents: Dict[str, str] = {}
    facts: List[Dict[str, Any]] = []
    # A..Z, then AA, BA, ... so project names stay unique past 26 documents
    projects = [f"Project {chr(65 + i % 26)}{'' if i < 26 else chr(64 + i // 26)}{rng.randint(10, 99)}"
                for i in range(n_docs)]
    aliases: Dict[str, List[str]] = {}

    for d in range(n_docs):
        tkey, tname, components = TOPICS[d % len(TOPICS)]
        doc_id = f"doc{d:03d}_{tkey}.txt"
        project = projects[d]
        aliases[doc_id] = [project]
        paragraphs = []
        for p in range(paras_per_doc):
            comp = components[p % len(components)]
            metric = METRICS[(d + p) % len(METRICS)]
            value = round(rng.uniform(2.0, 99.0), 2)
            lines = [
                f"This section of the {tname} dossier for {project} describes the {comp}.",
                f"The {comp} of {project} records a {metric} of {value} units under the reference duty cycle.",
            ]
            lines += rng.sample(FILLER, 4)
            paragraphs.append(" ".join(lines))
            facts.append({"doc_id": doc_id, "project": project, "component": comp,
                          "metric": metric, "value": value})
        documents[doc_id] = "\n".join(paragraphs)

    # bridge sentences: a component in doc A is qualified by a programme in doc B
    bridges = []
    doc_ids = list(documents)
    for i in range(0, n_docs - 1, 2):
        a, b = doc_ids[i], doc_ids[i + 1]
        fa = [f for f in facts if f["doc_id"] == a][0]
        fb = [f for f in facts if f["doc_id"] == b][2]
        documents[a] += (f"\nThe {fa['component']} used in {fa['project']} is supplied and qualified by the "
                         f"{fb['component']} programme described for {fb['project']}.")
        bridges.append((fa, fb))

    queries: List[Dict[str, Any]] = []
    for n, f in enumerate(facts[:: max(1, len(facts) // max_single)][:max_single]):
        queries.append({
            "id": f"S{n + 1:02d}",
            "type": "single",
            "query": f"What {f['metric']} does the {f['component']} of {f['project']} record?",
            "gold": [{"doc": f["doc_id"],
                      "terms": [f"{f['component']} of {f['project']}", f"{f['metric']} of {f['value']}"]}],
        })
    for n, (fa, fb) in enumerate(bridges):
        queries.append({
            "id": f"C{n + 1:02d}",
            "type": "cross",
            "query": (f"How does the {fa['component']} of {fa['project']} relate to the "
                      f"{fb['component']} of {fb['project']}?"),
            "gold": [{"doc": fa["doc_id"], "terms": [f"{fa['component']} used in {fa['project']}"]},
                     {"doc": fb["doc_id"], "terms": [f"{fb['component']} of {fb['project']}"]}],
        })
    return documents, queries, aliases

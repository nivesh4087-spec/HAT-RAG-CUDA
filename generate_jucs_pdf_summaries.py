import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#4A5568"))
        if self._pageNumber > 1:
            self.drawString(54, 750, "J.UCS Research Summary Report — HAT-RAG-CUDA Framework Alignment")
            self.setStrokeColor(colors.HexColor("#CBD5E0"))
            self.setLineWidth(0.5)
            self.line(54, 742, 558, 742)
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 36, page_str)
        self.drawString(54, 36, "Confidential & Research Documentation | J.UCS & HAT-RAG CUDA")
        self.setStrokeColor(colors.HexColor("#CBD5E0"))
        self.setLineWidth(0.5)
        self.line(54, 48, 558, 48)
        self.restoreState()

def get_styles():
    styles = getSampleStyleSheet()
    title = ParagraphStyle('DocTitle', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=13, leading=16, textColor=colors.HexColor("#1A365D"), spaceAfter=4)
    subtitle = ParagraphStyle('DocSubtitle', parent=styles['Normal'], fontName='Helvetica-Oblique', fontSize=9.5, leading=13, textColor=colors.HexColor("#2B6CB0"), spaceAfter=8)
    h1 = ParagraphStyle('Heading1_Custom', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=10.5, leading=13, textColor=colors.HexColor("#1A365D"), spaceBefore=6, spaceAfter=3)
    body = ParagraphStyle('Body_Custom', parent=styles['BodyText'], fontName='Helvetica', fontSize=8.5, leading=11.5, textColor=colors.HexColor("#2D3748"), spaceAfter=4)
    bullet = ParagraphStyle('Bullet_Custom', parent=body, leftIndent=10, spaceAfter=2)
    cell = ParagraphStyle('TableCell', parent=styles['Normal'], fontName='Helvetica', fontSize=8, leading=10, textColor=colors.HexColor("#1A202C"))
    cell_b = ParagraphStyle('TableCellBold', parent=cell, fontName='Helvetica-Bold', textColor=colors.HexColor("#1A365D"))
    return title, subtitle, h1, body, bullet, cell, cell_b


def create_pdf_paper1(filename):
    doc = SimpleDocTemplate(filename, pagesize=letter, leftMargin=54, rightMargin=54, topMargin=54, bottomMargin=54)
    title, subtitle, h1, body, bullet, cell, cell_b = get_styles()
    story = []
    story.append(Paragraph("J.UCS Research Paper Summary Report (Paper 1 of 2)", subtitle))
    story.append(Paragraph("Optimization of Parallel Number Theoretic Transform Algorithms for Multi-Core Digital Signal Processors", title))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1A365D"), spaceAfter=6))
    
    meta_data = [
        [Paragraph("Journal:", cell_b), Paragraph("Journal of Universal Computer Science (J.UCS), Vol. 32, No. 9 (2026), pp. 1354–1374", cell)],
        [Paragraph("Authors:", cell_b), Paragraph("Dingxing Xie, Xinchao Hu, Lian Peng, Guokai Liu, Ning Cheng, Jing Liu, Yang Zhang", cell)],
        [Paragraph("PDF Link:", cell_b), Paragraph("https://lib.jucs.org/article/207845/download/pdf/", cell)],
        [Paragraph("Core Domain:", cell_b), Paragraph("Parallel Computing, Multi-Core Acceleration, CUDA Vector Optimization, Memory Pipeline", cell)],
        [Paragraph("HAT-RAG Mapping:", cell_b), Paragraph("CUDA Similarity Acceleration Engine (cuda_utils.py) & Batch Cosine SIMD Vectorization", cell)]
    ]
    t = Table(meta_data, colWidths=[100, 404])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F7FAFC")),
        ('BOX', (0,0), (-1,-1), 0.75, colors.HexColor("#E2E8F0")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#EDF2F7")),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t)
    story.append(Spacer(1, 6))

    story.append(Paragraph("1. Executive Summary & Core Research Problem", h1))
    story.append(Paragraph(
        "This research paper from J.UCS introduces a high-performance parallel optimization framework addressing memory access bottlenecks and hardware SIMD stalls in vector transformations. "
        "By proposing a four-step matrix decomposition, VLIW/SIMD microkernels, vectorized modular arithmetic across 16 parallel streams, and DMA double-buffering, "
        "the paper achieves a <b>2.84x execution speedup</b> and a <b>42.5% reduction in memory bandwidth consumption</b>.",
        body
    ))

    story.append(Paragraph("2. Key Technical Innovations", h1))
    story.append(Paragraph("• <b>Four-Step 2D Matrix Decomposition:</b> Replaces standard six-step algorithms by restructuring 1D vector arrays into 2D matrices.", bullet))
    story.append(Paragraph("• <b>VLIW / SIMD Microkernel:</b> Applies aggressive loop unrolling and explicit register reuse to saturate functional ALUs.", bullet))
    story.append(Paragraph("• <b>16-Stream Parallel Vector Execution:</b> Enables multi-threaded SIMD data-paths for modular vector mathematics.", bullet))
    story.append(Paragraph("• <b>Double-Buffering DMA Pipelining:</b> Overlaps memory transfers with active computation.", bullet))

    story.append(Paragraph("3. Architectural Mapping to HAT-RAG CUDA Repository", h1))

    mapping_data = [
        [Paragraph("J.UCS Concept", cell_b), Paragraph("HAT-RAG-CUDA Module", cell_b), Paragraph("Architectural Synergy & Impact", cell_b)],
        [Paragraph("Four-Step 2D Decomposition", cell), Paragraph("<code>src/cuda_utils.py</code><br/>Batch Cosine Similarity", cell), Paragraph("Transforms 1D chunk comparisons into 2D GPU tensor batch matrix multiplications Q * D^T.", cell)],
        [Paragraph("16-Way SIMD Stream Parallelism", cell), Paragraph("<code>src/cuda_utils.py</code><br/>PyTorch CUDA Streams", cell), Paragraph("Executes multi-threaded tensor similarity operations across GPU CUDA warps with FP16/BF16 math.", cell)],
        [Paragraph("Double-Buffering DMA Pipelining", cell), Paragraph("<code>src/retriever.py</code><br/>Top-Down Logarithmic Traversal", cell), Paragraph("Asynchronously fetches child cluster metadata while scoring parent node embeddings.", cell)],
        [Paragraph("Index Locality & Memory Placement", cell), Paragraph("<code>src/hierarchical_tree.py</code><br/>Node Metadata Index", cell), Paragraph("Maintains structured node ID indexing to optimize memory access during top-down branch traversal.", cell)]
    ]

    t_map = Table(mapping_data, colWidths=[125, 155, 224])
    t_map.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#EBF8FF")),
        ('BOX', (0,0), (-1,-1), 0.75, colors.HexColor("#CBD5E0")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_map)
    story.append(Spacer(1, 6))

    story.append(Paragraph("4. Benchmark Results & Research Synergy", h1))
    story.append(Paragraph(
        "<b>Quantitative Benchmark:</b> Latency dropped from 4.82ms down to 1.70ms (2.84x speedup) and bus contention decreased by 42.5%.<br/>"
        "<b>Synergy with HAT-RAG:</b> Demonstrates that hardware-aligned parallel algorithms combined with structured memory pipelining deliver scalable throughput. "
        "These empirical findings substantiate the CUDA GPU similarity acceleration and logarithmic search engine in HAT-RAG.",
        body
    ))
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated {filename}")

def create_pdf_paper2(filename):
    doc = SimpleDocTemplate(filename, pagesize=letter, leftMargin=54, rightMargin=54, topMargin=54, bottomMargin=54)
    title, subtitle, h1, body, bullet, cell, cell_b = get_styles()
    story = []
    story.append(Paragraph("J.UCS Research Paper Summary Report (Paper 2 of 2)", subtitle))
    story.append(Paragraph("Using Generative Artificial Intelligence to Improve User Engagement in Content Marketing", title))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1A365D"), spaceAfter=6))
    
    meta_data = [
        [Paragraph("Journal:", cell_b), Paragraph("Journal of Universal Computer Science (J.UCS), Vol. 31, No. 12 (2025), pp. 1274–1296", cell)],
        [Paragraph("Authors:", cell_b), Paragraph("Irene Ruiz-Pozo, Juan Morales-García, Claudia Ximena Aguirre-Mejía, Antonio Serrano", cell)],
        [Paragraph("PDF Link:", cell_b), Paragraph("https://lib.jucs.org/article/151691/download/pdf/", cell)],
        [Paragraph("Core Domain:", cell_b), Paragraph("Generative Artificial Intelligence (GenAI), LLM Synthesis, Grounded Prompting, Content Retrieval", cell)],
        [Paragraph("HAT-RAG Mapping:", cell_b), Paragraph("Grounded Evidence Generator (generator.py) & Automated Faithfulness Verification Engine", cell)]
    ]
    t = Table(meta_data, colWidths=[100, 404])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F7FAFC")),
        ('BOX', (0,0), (-1,-1), 0.75, colors.HexColor("#E2E8F0")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#EDF2F7")),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t)
    story.append(Spacer(1, 6))

    story.append(Paragraph("1. Executive Summary & Core Research Problem", h1))
    story.append(Paragraph(
        "This J.UCS research article investigates the impact of Generative AI (GenAI) and Large Language Models (LLMs) on text synthesis, semantic relevance, and grounded document generation. "
        "The authors compare ungrounded standalone LLM generation with context-conditioned semantic retrieval. "
        "The empirical findings demonstrate that conditioning generative synthesis on domain-specific retrieved context yields a <b>34.2% increase in user engagement</b> and reduces hallucinations by <b>58.6%</b>.",
        body
    ))

    story.append(Paragraph("2. Key Technical Innovations", h1))
    story.append(Paragraph("• <b>Context-Conditioned Evidence Generation:</b> Combines structured document contexts with generative LLMs to produce accurate responses.", bullet))
    story.append(Paragraph("• <b>Multi-Tier Content Abstraction:</b> Structures text output into multi-tier semantic summaries tailored to diverse user abstraction levels.", bullet))
    story.append(Paragraph("• <b>Automated Grounding Metrics:</b> Establishes quantitative criteria for checking whether generated claims match source text facts.", bullet))
    story.append(Paragraph("• <b>Hallucination Control:</b> Proves that structured evidence injection eliminates factual drifts in long-form generation.", bullet))

    story.append(Paragraph("3. Architectural Mapping to HAT-RAG CUDA Repository", h1))

    mapping_data = [
        [Paragraph("J.UCS Concept", cell_b), Paragraph("HAT-RAG-CUDA Module", cell_b), Paragraph("Architectural Synergy & Impact", cell_b)],
        [Paragraph("Evidence-Grounded Generation", cell), Paragraph("<code>src/generator.py</code><br/>Multi-Document Synthesis", cell), Paragraph("Synthesizes final LLM responses using retrieved cross-document hierarchical contexts with explicit citation tags.", cell)],
        [Paragraph("Multi-Tier Content Abstraction", cell), Paragraph("<code>src/hierarchical_tree.py</code><br/>Abstract Summary Hierarchy", cell), Paragraph("Structures document corpora into Root Abstracts (Level 2), Local Abstracts (Level 1), and Leaf Passages (Level 0).", cell)],
        [Paragraph("Hallucination Control", cell), Paragraph("<code>src/generator.py</code><br/>Faithfulness Score Engine", cell), Paragraph("Calculates n-gram and term coverage between generated claims and retrieved source contexts to verify factual correctness.", cell)],
        [Paragraph("User Engagement & Relevance", cell), Paragraph("<code>src/retriever.py</code><br/>MMR Reranking & Diversity", cell), Paragraph("Filters redundant passages and selects diverse cross-document evidence to maximize answer completeness.", cell)]
    ]

    t_map = Table(mapping_data, colWidths=[125, 155, 224])
    t_map.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#EBF8FF")),
        ('BOX', (0,0), (-1,-1), 0.75, colors.HexColor("#CBD5E0")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_map)
    story.append(Spacer(1, 6))

    story.append(Paragraph("4. Benchmark Results & Research Synergy", h1))
    story.append(Paragraph(
        "<b>Quantitative Benchmark:</b> Grounded AI content achieved 34.2% higher engagement and 58.6% lower hallucination rates.<br/>"
        "<b>Synergy with HAT-RAG:</b> Validates the primary objective of HAT-RAG-CUDA: combining multi-tiered hierarchical text retrieval with evidence-grounded LLM synthesis "
        "delivers vastly superior context quality, lower token consumption, and verifiable factual accuracy.",
        body
    ))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated {filename}")


if __name__ == '__main__':
    os.makedirs('docs', exist_ok=True)
    create_pdf_paper1("docs/JUCS_Paper1_Parallel_Optimization_Summary.pdf")
    create_pdf_paper2("docs/JUCS_Paper2_Generative_AI_RAG_Summary.pdf")



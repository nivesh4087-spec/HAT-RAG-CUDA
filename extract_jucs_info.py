import pypdf
import sys

sys.stdout.reconfigure(encoding='utf-8')

def inspect_pdf(pdf_path):
    print(f"\n==========================================")
    print(f"Reading {pdf_path}")
    print(f"==========================================")
    reader = pypdf.PdfReader(pdf_path)
    print(f"Num pages: {len(reader.pages)}")
    text_sample = ""
    for i in range(min(5, len(reader.pages))):
        text_sample += f"--- Page {i+1} ---\n" + reader.pages[i].extract_text()[:600] + "\n"
    print(text_sample[:2000])

inspect_pdf("docs/JUCS_2026_Parallel_Optimization_NTT.pdf")
inspect_pdf("docs/JUCS_2026_Generative_AI_RAG.pdf")


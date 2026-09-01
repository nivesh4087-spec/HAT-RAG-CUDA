# -*- coding: utf-8 -*-
import os
import markdown2
import subprocess
from pathlib import Path

def convert_md_to_html_and_pdf(md_path, pdf_path):
    print(f"Reading Markdown from {md_path}...")
    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    html_body = markdown2.markdown(
        md_content,
        extras=[
            "tables",
            "fenced-code-blocks",
            "header-ids",
            "code-friendly",
            "break-on-newline"
        ]
    )

    css_style = """<style>
        @page {
            size: A4;
            margin: 15mm;
        }
        body {
            font-family: Arial, sans-serif;
            color: #1e293b;
            line-height: 1.5;
            font-size: 10pt;
        }
        h1 {
            color: #0f172a;
            font-size: 20pt;
            border-bottom: 3px solid #2563eb;
            padding-bottom: 6px;
        }
        h2 {
            color: #1e3a8a;
            font-size: 14pt;
            border-bottom: 2px solid #cbd5e1;
            padding-bottom: 4px;
            margin-top: 20px;
        }
        h3 {
            color: #0284c7;
            font-size: 12pt;
            margin-top: 16px;
        }
        h4 {
            color: #334155;
            font-size: 10pt;
            margin-top: 10px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 12px 0;
            font-size: 8.5pt;
        }
        th {
            background-color: #1e293b;
            color: #ffffff;
            font-weight: bold;
            padding: 6px 8px;
            border: 1px solid #334155;
        }
        td {
            padding: 5px 8px;
            border: 1px solid #cbd5e1;
            vertical-align: top;
        }
        tr:nth-child(even) td {
            background-color: #f8fafc;
        }
        code {
            background-color: #f1f5f9;
            color: #0f172a;
            padding: 2px 4px;
            border-radius: 3px;
            font-size: 8.5pt;
        }
        pre {
            background-color: #0f172a;
            color: #f8fafc;
            padding: 10px;
            border-radius: 4px;
            font-size: 8.5pt;
        }
        hr {
            border: none;
            height: 1px;
            background-color: #cbd5e1;
            margin: 16px 0;
        }
    </style>"""

    full_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>29 Research Papers Synthesis Documentation</title>
    {css_style}
</head>
<body>
    {html_body}
</body>
</html>"""

    html_path = md_path.with_suffix(".html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"Generated intermediate HTML: {html_path}")

    edge_executable = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    chrome_executable = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

    executable = None
    if os.path.exists(edge_executable):
        executable = edge_executable
    elif os.path.exists(chrome_executable):
        executable = chrome_executable

    if executable:
        print(f"Converting HTML to PDF using Headless Browser ({executable})...")
        cmd = [
            executable,
            "--headless",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf_path}",
            str(html_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0:
            print(f"[SUCCESS] PDF generated at: {pdf_path} ({os.path.getsize(pdf_path)} bytes)")
            return True
        else:
            print(f"Browser printing warning: {result.stderr}")
    
    try:
        import weasyprint
        print("Using WeasyPrint fallback to generate PDF...")
        weasyprint.HTML(filename=str(html_path)).write_pdf(str(pdf_path))
        print(f"[SUCCESS] PDF generated via WeasyPrint at: {pdf_path}")
        return True
    except Exception as e:
        print(f"WeasyPrint error: {e}")

    return False

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    targets = [
        base_dir / "docs" / "RESEARCH_PAPERS_SUMMARY_DOCUMENTATION.md",
        base_dir / "RESEARCH_PAPERS_SUMMARY_DOCUMENTATION.md"
    ]
    
    for md_path in targets:
        if md_path.exists():
            pdf_path = md_path.with_suffix(".pdf")
            convert_md_to_html_and_pdf(md_path, pdf_path)

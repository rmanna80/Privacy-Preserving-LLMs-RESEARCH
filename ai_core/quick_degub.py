# quick_debug.py
from pathlib import Path
from pypdf import PdfReader

pdf_path = Path("data/raw_pdfs/2024_John_and_Sally_Smith.pdf")
reader = PdfReader(str(pdf_path))

for i, page in enumerate(reader.pages):
    text = page.extract_text() or ""
    if "SSN" in text.upper() or "SOCIAL" in text.upper() or "-" in text:
        print("\n===== PAGE", i, "=====")
        print(text[:2000])

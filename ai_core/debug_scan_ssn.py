# debug_scan_ssn.py
from pathlib import Path
from ai_core.pdf_loader import load_pdfs
import re

SSN = re.compile(r"\b\d{3}[- ]?\d{2}[- ]?\d{4}\b")

docs = load_pdfs(Path("data/raw_pdfs"))
all_text = "\n".join(d.page_content for d in docs)

print("Found SSNs:", sorted(set(SSN.findall(all_text))))
print("Contains 222-22-2222?", "222-22-2222" in all_text)

"""
ai_core/extractor.py — AI-assisted structured extraction (sub-phase 4d.2).

Given a Document row, load its PDF text, ask the local Ollama model to
fill in the canonical field schema for that doc_type, and create
Extraction rows (extracted_by='llm') that an advisor can then verify or
reject in the UI.

Honest design notes (v1 limitations, deliberate):
  - Single-prompt extraction over truncated text. We concatenate the
    document's pages and truncate to MAX_DOC_CHARS so the prompt fits
    qwen2.5:7b's context comfortably. Long trust documents may lose
    later pages. v2 will do per-field retrieval over chunks.
  - Page attribution is best-effort. We ask the model for the page it
    found each fact on, but a 7B model's page numbers are unreliable.
    Confidence is set to 0.6 for all LLM extractions; the advisor's
    verification is the real quality gate.
  - PII fields (is_pii=True in the schema) are encrypted before storage
    via db.crypto, same as Person.ssn.

The extraction is LOCAL ONLY — routed through Purpose.DOCUMENT_EXTRACTION
in ai_routing, which always resolves to Ollama. Client documents never
leave the machine.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Optional

from ai_core.extraction_schema import fields_for_doc_type

MAX_DOC_CHARS = 15000  # keep the prompt well inside qwen2.5:7b context


# ─────────────────────────────────────────────────────────────────────
# Document text loading
# ─────────────────────────────────────────────────────────────────────

def _load_document_text(file_path: str) -> list[tuple[int, str]]:
    """Load a single PDF and return [(page_number, page_text), ...].

    Reuses the existing hybrid loader (PyMuPDF + OCR fallback) by staging
    the single file in a temp directory, since load_pdfs_hybrid scans a
    folder.
    """
    from ai_core.pdf_loader import load_pdfs_hybrid

    src = Path(file_path)
    if not src.exists():
        raise FileNotFoundError(f"Document file not found: {file_path}")

    staging = Path("data/chroma/_extract_staging")
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, staging / src.name)

    pages = load_pdfs_hybrid(staging)
    shutil.rmtree(staging, ignore_errors=True)

    out: list[tuple[int, str]] = []
    for pg in pages:
        page_no = pg.metadata.get("page", 0)
        out.append((int(page_no) if page_no is not None else 0, pg.page_content))
    out.sort(key=lambda t: t[0])
    return out


def _build_doc_text(pages: list[tuple[int, str]]) -> str:
    """Concatenate pages with page markers, truncated to MAX_DOC_CHARS."""
    parts: list[str] = []
    total = 0
    for page_no, text in pages:
        block = f"\n--- PAGE {page_no + 1} ---\n{text.strip()}\n"
        if total + len(block) > MAX_DOC_CHARS:
            remaining = MAX_DOC_CHARS - total
            if remaining > 200:
                parts.append(block[:remaining])
            break
        parts.append(block)
        total += len(block)
    return "".join(parts)


# ─────────────────────────────────────────────────────────────────────
# Prompt + parsing
# ─────────────────────────────────────────────────────────────────────

def _build_prompt(doc_type: str, fields: list[dict], doc_text: str) -> str:
    field_lines = "\n".join(
        f'  "{f["key"]}": {f["hint"]} (type: {f["value_type"]})'
        for f in fields
    )
    return f"""You are a meticulous estate-planning paralegal. Extract structured facts from the document below.

DOCUMENT TYPE: {doc_type}

FIELDS TO EXTRACT:
{field_lines}

RULES:
- Respond with ONLY a JSON object. No prose, no markdown fences.
- One key per field listed above.
- Each value is an object: {{"value": "<extracted value>", "page": <page number or null>, "quote": "<short supporting quote from the document>"}}
- If a field is not present in the document, use null for that key (not an empty string).
- For yes_no fields, the value must be exactly "yes" or "no".
- For list fields, join multiple items with "; ".
- Do not invent values. Only extract what the document actually states.

DOCUMENT TEXT:
{doc_text}

JSON:"""


def _parse_llm_json(raw: str) -> dict:
    """Parse the model's JSON, tolerating markdown fences and stray text."""
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    # If the model added prose, grab the first {...} block
    if not cleaned.startswith("{"):
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(0)
    return json.loads(cleaned)


# ─────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────

def run_extraction(document_id: int, *, verbose: bool = False) -> dict:
    """Run AI extraction on a document. Returns a summary dict:
        {"proposed": int, "skipped": int, "errors": [str]}

    Creates Extraction rows with extracted_by='llm', confidence=0.6.
    Skips fields that already have an extraction row (so re-running
    doesn't duplicate; advisor should reject old rows first to re-extract
    a field).
    """
    from db.repositories import (
        get_document,
        list_extractions_for_document,
        create_extraction,
    )
    from ai_core.llm_ollama import build_ollama_llm

    doc = get_document(document_id)
    if doc is None:
        return {"proposed": 0, "skipped": 0, "errors": ["Document not found."]}

    fields = fields_for_doc_type(doc.doc_type)
    existing_keys = {e.field_key for e in list_extractions_for_document(document_id)}
    target_fields = [f for f in fields if f["key"] not in existing_keys]

    if not target_fields:
        return {"proposed": 0, "skipped": len(fields),
                "errors": ["All fields already have extractions. Reject old rows to re-extract."]}

    # Load + truncate text
    try:
        pages = _load_document_text(doc.file_path)
    except Exception as e:
        return {"proposed": 0, "skipped": 0, "errors": [f"Could not load PDF: {e}"]}
    doc_text = _build_doc_text(pages)
    if not doc_text.strip():
        return {"proposed": 0, "skipped": 0, "errors": ["No text extracted from PDF."]}

    # Ask the model — local Ollama only (Purpose.DOCUMENT_EXTRACTION policy)
    llm = build_ollama_llm()
    prompt = _build_prompt(doc.doc_type, target_fields, doc_text)
    if verbose:
        print(f"[Extractor] Prompting for {len(target_fields)} fields, "
              f"{len(doc_text)} chars of document text")
    try:
        response = llm.invoke(prompt)
        raw = response.content if hasattr(response, "content") else str(response)
        parsed = _parse_llm_json(raw)
    except json.JSONDecodeError as e:
        return {"proposed": 0, "skipped": 0,
                "errors": [f"Model returned invalid JSON: {e}"]}
    except Exception as e:
        return {"proposed": 0, "skipped": 0, "errors": [f"LLM call failed: {e}"]}

    # Create extraction rows
    field_by_key = {f["key"]: f for f in target_fields}
    proposed, errors = 0, []
    for key, payload in parsed.items():
        fdef = field_by_key.get(key)
        if fdef is None or payload is None:
            continue
        value = payload.get("value") if isinstance(payload, dict) else payload
        if value is None or str(value).strip() in ("", "null", "None", "N/A"):
            continue
        page = payload.get("page") if isinstance(payload, dict) else None
        quote = payload.get("quote") if isinstance(payload, dict) else None
        try:
            create_extraction(
                document_id=document_id,
                field_key=key,
                field_value=str(value).strip(),
                extraction_type=fdef["value_type"],
                is_pii=fdef["is_pii"],
                page_number=int(page) if isinstance(page, (int, float)) else None,
                text_snippet=(str(quote)[:500] if quote else None),
                extracted_by="llm",
                confidence=0.6,
            )
            proposed += 1
        except Exception as e:
            errors.append(f"{key}: {e}")

    if verbose:
        print(f"[Extractor] Proposed {proposed} extractions")
    return {"proposed": proposed,
            "skipped": len(fields) - len(target_fields),
            "errors": errors}

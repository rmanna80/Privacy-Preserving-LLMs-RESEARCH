from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple
from langchain_core.documents import Document

SSN_PATTERN = re.compile(r"\b\d{3}[- ]?\d{2}[- ]?\d{4}\b")

# Simple "likely person name" heuristic (ALL CAPS names in many IRS forms)
# You can expand later (Title Case, commas, etc.)
NAME_PATTERN = re.compile(r"\b[A-Z]{2,}(?:\s+[A-Z]{2,})+\b")


def looks_like_ssn_question(q: str) -> bool:
    q = q.lower()
    return ("ssn" in q) or ("social security" in q)


def normalize_name(s: str) -> str:
    s = re.sub(r"[^A-Za-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.upper()


def extract_ssns(text: str) -> List[str]:
    return list(dict.fromkeys(SSN_PATTERN.findall(text)))


def extract_candidate_names(text: str) -> List[str]:
    """
    Extract candidate names. This is heuristic. Works well for IRS-style ALL CAPS.
    """
    raw = NAME_PATTERN.findall(text.upper())
    # Filter out obvious non-names you might see in forms
    blacklist = {
        "UNITED STATES", "INTERNAL REVENUE SERVICE", "DEPARTMENT OF THE TREASURY",
        "PREVIEW COPY", "DO NOT FILE"
    }
    out = []
    for n in raw:
        n = normalize_name(n)
        if len(n) < 5:
            continue
        if n in blacklist:
            continue
        # Avoid capturing long paragraphs as "names"
        if len(n.split()) > 4:
            continue
        out.append(n)
    return list(dict.fromkeys(out))


def build_name_ssn_pairs_from_docs(docs: List[Document]) -> Dict[str, str]:
    """
    Build a best-effort mapping name -> SSN from retrieved text.
    We use "same line" pairing first, then proximity-window fallback.
    """
    text = "\n".join(d.page_content for d in docs)
    upper = text.upper()

    pairs: Dict[str, str] = {}

    # 1) Same-line pairing: strongest signal
    for line in upper.splitlines():
        ssns = SSN_PATTERN.findall(line)
        if not ssns:
            continue
        names = extract_candidate_names(line)
        if not names:
            continue
        # If multiple names and one SSN, assign to each (common on joint lines)
        # You can adjust this policy later.
        for name in names:
            pairs.setdefault(name, ssns[0])

    # 2) Proximity pairing: find name then SSN within a small window after it
    window = 200
    for name in extract_candidate_names(upper):
        if name in pairs:
            continue
        idx = upper.find(name)
        while idx != -1:
            segment = upper[idx : idx + window]
            m = SSN_PATTERN.search(segment)
            if m:
                pairs[name] = m.group(0)
                break
            idx = upper.find(name, idx + 1)

    return pairs


def extract_requested_name(question: str) -> Optional[str]:
    """
    Try to extract the person name the user asked about.
    Examples:
      "What is Garrett McGovern's SSN?"
      "SSN for John Smith"
    """
    q = question.strip()

    # Possessive: "<name>'s ssn"
    m = re.search(r"([A-Za-z]+(?:\s+[A-Za-z]+){0,3})'s\s+(?:ssn|social security)", q, re.IGNORECASE)
    if m:
        return normalize_name(m.group(1))

    # "ssn for <name>"
    m = re.search(r"(?:ssn|social security)\s+for\s+([A-Za-z]+(?:\s+[A-Za-z]+){0,3})", q, re.IGNORECASE)
    if m:
        return normalize_name(m.group(1))

    return None


def best_name_match(target: str, candidates: List[str]) -> Optional[str]:
    """
    Simple match strategy:
    - exact match
    - contains match (handles middle initials, etc.)
    """
    if not target:
        return None
    target = normalize_name(target)

    if target in candidates:
        return target

    # contains match
    for c in candidates:
        if target in c or c in target:
            return c

    return None

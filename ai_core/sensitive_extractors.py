# ai_core/sensitive_extractors.py
from __future__ import annotations

import re
from typing import Dict, List, Optional
from langchain_core.documents import Document

SSN_PATTERN = re.compile(r"\b\d{3}[- ]?\d{2}[- ]?\d{4}\b")
NAME_PATTERN = re.compile(r"\b[A-Z]{2,}(?:\s+[A-Z]{2,}){1,3}\b")

BLACKLIST = {
    "UNITED STATES",
    "INTERNAL REVENUE SERVICE",
    "DEPARTMENT OF THE TREASURY",
    "PREVIEW COPY",
    "DO NOT FILE",
    "SEE INSTRUCTIONS",
    "GO TO WWW",
    "ADD LINES",
    "NAME PHONE NO",
    "PHONE NO",
    "SOCIAL SECURITY",
    "SOCIAL SECURITY NUMBER",
}

FORM_WORDS = {
    "SEE", "USE", "FOR", "THE", "AND", "OR", "IF", "NOT",
    "FROM", "LINE", "FORM", "ADD", "ENTER", "CHECK", "FILE",
    "TAX", "YOUR", "YOU", "DID", "ARE", "WAS", "WILL", "THIS",
    "THAT", "WITH", "HAVE", "ONLY", "ALSO", "EACH", "ANY",
    "TOTAL", "AMOUNT", "NUMBER", "ATTACH", "SCHEDULE", "COPY",
    "PREVIEW", "POLICY", "DATE", "TYPE", "RATE", "PLAN",
    "DIGITAL", "ASSET", "EXEMPT", "INCOME", "CREDIT", "TRUST",
    "ESTATE", "SOCIAL", "SECURITY", "QUALIFIED", "ORDINARY",
    "SUBTRACT", "BLINDNESS", "INTEREST", "DIVIDENDS", "INST",
    "JOINT", "RETURN", "SPOUSE", "NAME", "PREPARER", "EMPLOYED",
    "FIRM", "ADDRESS", "HERE", "DISTRIBUTION", "DISTRIBUTIONS",
    "FOREIGN", "THEIR", "IRA", "PRIVACY", "ACT", "PENSIONS",
    "ANNUITIES", "BENEFITS", "SUM", "ELECTION", "METHOD",
    "DISCLOSURE", "SEPARATE", "INSTRUCTIONS", "EARNED", "RESERVED",
    "FUTURE", "LAST", "WILL", "REAL", "STATE", "ZIP", "APPROXIMATE",
    "VALUE", "FANTASTIC", "AVE", "CORPORATE", "WAY", "FINANCIAL",
    "SINGLE", "MARRIED", "FILING", "HEAD", "HOUSEHOLD", "SURVIVING",
    "QUALIFYING", "DEPENDENT", "DEDUCTION", "STANDARD", "ITEMIZED",
    "CAPITAL", "GAIN", "LOSS", "SHORT", "LONG", "TERM", "FEDERAL",
    "WITHHOLD", "WITHHELD", "PAYMENT", "REFUND", "OWE", "PENALTY",
    "OFFICE", "SERVICE", "STREET", "ROAD", "BLVD", "DRIVE",
}


def looks_like_ssn_question(q: str) -> bool:
    q = q.lower()
    return ("ssn" in q) or ("social security" in q)


def normalize_name(s: str) -> str:
    s = re.sub(r"[^A-Za-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.upper()


def extract_ssns(text: str) -> List[str]:
    return list(dict.fromkeys(SSN_PATTERN.findall(text)))


def extract_requested_name(question: str) -> Optional[str]:
    q = question.strip()

    patterns = [
        r"([A-Za-z]+(?:\s+[A-Za-z]+){1,3})'s\s+(?:ssn|social security)",
        r"(?:ssn|social security)\s+(?:number\s+)?for\s+([A-Za-z]+(?:\s+[A-Za-z]+){1,3})",
        r"(?:what\s+is\s+)?([A-Za-z]+(?:\s+[A-Za-z]+){1,3})'s\s+(?:ssn|social security)",
    ]

    for pattern in patterns:
        m = re.search(pattern, q, re.IGNORECASE)
        if m:
            return normalize_name(m.group(1))

    return None


def best_name_match(target: str, candidates: List[str]) -> Optional[str]:
    if not target:
        return None

    target = normalize_name(target)
    target_parts = set(target.split())

    if target in candidates:
        return target

    best_candidate = None
    best_score = 0

    for candidate in candidates:
        candidate_norm = normalize_name(candidate)
        candidate_parts = set(candidate_norm.split())

        if target in candidate_norm or candidate_norm in target:
            return candidate

        overlap = len(target_parts.intersection(candidate_parts))
        if overlap > best_score:
            best_score = overlap
            best_candidate = candidate

    return best_candidate if best_score >= 2 else None


def _is_valid_name_word(word: str) -> bool:
    return (
        word.isalpha()
        and len(word) >= 2
        and word not in FORM_WORDS
        and word not in BLACKLIST
    )


def _is_person_name(name: str) -> bool:
    """
    Accept likely person names with 2 to 4 words.
    Example:
    - JOHN SMITH
    - JOHN A SMITH
    - MARY ANN JONES
    """
    words = name.split()
    if not (2 <= len(words) <= 4):
        return False

    valid_words = [w for w in words if _is_valid_name_word(w)]
    return len(valid_words) >= 2


def _is_last_name_only(word: str) -> bool:
    word = word.strip()
    return _is_valid_name_word(word) and len(word) >= 3


def _extract_candidate_names_from_line(line: str) -> List[str]:
    raw_names = NAME_PATTERN.findall(line)
    names: List[str] = []

    for raw in raw_names:
        name = normalize_name(raw)
        if name in BLACKLIST:
            continue
        if _is_person_name(name):
            names.append(name)

    return list(dict.fromkeys(names))


def _is_ssn_only_line(line: str) -> bool:
    return bool(re.match(r"^\s*\d{3}[- ]?\d{2}[- ]?\d{4}\s*$", line.strip()))


def _extract_name_from_previous_lines(lines: List[str], ssn_index: int) -> Optional[str]:
    """
    Look back up to 4 lines to reconstruct a likely name.
    Handles patterns like:
    - JOHN SMITH
    - JOHN / SMITH
    - JOHN A / SMITH
    """
    window_start = max(0, ssn_index - 4)
    previous_lines = [lines[i].strip() for i in range(window_start, ssn_index)]

    # First try: exact multi-word name on any prior line
    for line in reversed(previous_lines):
        candidates = _extract_candidate_names_from_line(line)
        if candidates:
            return candidates[0]

    # Second try: combine adjacent lines, e.g. JOHN + SMITH
    for i in range(len(previous_lines) - 1):
        left = normalize_name(previous_lines[i])
        right = normalize_name(previous_lines[i + 1])

        if not left or not right:
            continue

        if " " not in left and " " not in right:
            combined = f"{left} {right}"
            if _is_person_name(combined):
                return combined

        # Example: "JOHN A" + "SMITH"
        if " " in left and " " not in right:
            combined = f"{left} {right}"
            if _is_person_name(combined):
                return combined

    return None


def build_name_ssn_pairs_from_docs(docs: List[Document]) -> Dict[str, str]:
    """
    Build name -> SSN pairs from retrieved documents.

    Strategies:
    1) Same line:      JOHN SMITH 111-11-1111
    2) SSN-only line:  look back across prior lines for a likely name
    3) Nearby line:    name line followed by SSN within the next few lines
    """
    pairs: Dict[str, str] = {}

    for doc in docs:
        upper = doc.page_content.upper()
        lines = upper.splitlines()

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or "&" in line:
                continue

            ssns = SSN_PATTERN.findall(line)

            # Strategy 1: same line name + SSN
            if ssns:
                names = _extract_candidate_names_from_line(line)
                if len(names) == 1 and len(ssns) >= 1:
                    if names[0] not in pairs:
                        pairs[names[0]] = ssns[0]
                    continue

            # Strategy 2: SSN-only line, reconstruct name from previous lines
            if _is_ssn_only_line(line):
                ssn_match = SSN_PATTERN.search(line)
                if ssn_match:
                    name = _extract_name_from_previous_lines(lines, i)
                    if name and name not in pairs:
                        pairs[name] = ssn_match.group(0)
                continue

            # Strategy 3: name on current line, SSN in next 1-4 lines
            names = _extract_candidate_names_from_line(line)
            if names:
                for k in range(i + 1, min(i + 5, len(lines))):
                    future_line = lines[k]
                    if "&" in future_line:
                        continue
                    ahead_ssns = SSN_PATTERN.findall(future_line)
                    if ahead_ssns:
                        for name in names:
                            if name not in pairs:
                                pairs[name] = ahead_ssns[0]
                        break

    return pairs
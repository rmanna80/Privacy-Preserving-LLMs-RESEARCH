# ai_core/sensitive_extractors.py
from __future__ import annotations
import re
from typing import Dict, List, Optional
from langchain_core.documents import Document

SSN_PATTERN = re.compile(r"\b\d{3}[- ]?\d{2}[- ]?\d{4}\b")
NAME_PATTERN = re.compile(r"\b[A-Z]{2,}(?:\s+[A-Z]{2,})+\b")

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
}

FORM_WORDS = {
    'SEE', 'USE', 'FOR', 'THE', 'AND', 'OR', 'IF', 'NOT',
    'FROM', 'LINE', 'FORM', 'ADD', 'ENTER', 'CHECK', 'FILE',
    'TAX', 'YOUR', 'YOU', 'DID', 'ARE', 'WAS', 'WILL', 'THIS',
    'THAT', 'WITH', 'HAVE', 'ONLY', 'ALSO', 'EACH', 'ANY',
    'TOTAL', 'AMOUNT', 'NUMBER', 'ATTACH', 'SCHEDULE', 'COPY',
    'PREVIEW', 'POLICY', 'DATE', 'TYPE', 'RATE', 'PLAN',
    'DIGITAL', 'ASSET', 'EXEMPT', 'INCOME', 'CREDIT', 'TRUST',
    'ESTATE', 'SOCIAL', 'SECURITY', 'QUALIFIED', 'ORDINARY',
    'SUBTRACT', 'BLINDNESS', 'INTEREST', 'DIVIDENDS', 'INST',
    'JOINT', 'RETURN', 'SPOUSE', 'NAME', 'PREPARER', 'EMPLOYED',
    'FIRM', 'ADDRESS', 'HERE', 'DISTRIBUTION', 'DISTRIBUTIONS',
    'FOREIGN', 'THEIR', 'IRA', 'PRIVACY', 'ACT', 'PENSIONS',
    'ANNUITIES', 'BENEFITS', 'SUM', 'ELECTION', 'METHOD',
    'DISCLOSURE', 'SEPARATE', 'INSTRUCTIONS', 'EARNED', 'RESERVED',
    'FUTURE', 'LAST', 'WILL', 'REAL', 'STATE', 'ZIP', 'APPROXIMATE',
    'VALUE', 'FANTASTIC', 'AVE', 'CORPORATE', 'WAY', 'FINANCIAL',
    'SINGLE', 'MARRIED', 'FILING', 'HEAD', 'HOUSEHOLD', 'SURVIVING',
    'QUALIFYING', 'DEPENDENT', 'DEDUCTION', 'STANDARD', 'ITEMIZED',
    'CAPITAL', 'GAIN', 'LOSS', 'SHORT', 'LONG', 'TERM', 'FEDERAL',
    'WITHHOLD', 'WITHHELD', 'PAYMENT', 'REFUND', 'OWE', 'PENALTY',
    'OFFICE', 'SERVICE', 'STREET', 'ROAD', 'BLVD', 'DRIVE',
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

    m = re.search(r"([A-Za-z]+(?:\s+[A-Za-z]+){1,3})'s\s+(?:ssn|social security)", q, re.IGNORECASE)
    if m:
        return normalize_name(m.group(1))

    m = re.search(r"(?:ssn|social security)\s+(?:number\s+)?for\s+([A-Za-z]+(?:\s+[A-Za-z]+){1,3})", q, re.I)
    if m:
        return normalize_name(m.group(1))

    m = re.search(r"(?:what\s+is\s+)?([A-Za-z]+(?:\s+[A-Za-z]+){1,3})'s\s+(?:ssn|social security)", q, re.I)
    if m:
        return normalize_name(m.group(1))

    return None


def best_name_match(target: str, candidates: List[str]) -> Optional[str]:
    if not target:
        return None
    target = normalize_name(target)

    if target in candidates:
        return target

    for c in candidates:
        if target in c or c in target:
            return c
    return None


def _is_person_name(name: str) -> bool:
    """Return True if name looks like a real 2-word person name."""
    words = name.split()
    if len(words) != 2:
        return False
    for word in words:
        if len(word) < 2:
            return False
        if not word.isalpha():
            return False
        if word in FORM_WORDS:
            return False
    return True


def _is_last_name_only(word: str) -> bool:
    """Return True if a single word could be a last name."""
    word = word.strip()
    return (
        word.isalpha()
        and len(word) >= 3
        and word not in FORM_WORDS
        and word not in BLACKLIST
    )


def build_name_ssn_pairs_from_docs(docs: List[Document]) -> Dict[str, str]:
    """
    Build name->SSN pairs processing each document page individually.

    Handles three IRS form layouts:
    1) Same line:    "JOHN SMITH 111 11 1111"
    2) Split lines:  line N-1="JOHN", line N="SMITH", line N+1="111 11 1111"
    3) Proximity:    name line followed by SSN within 3 lines
    """
    pairs: Dict[str, str] = {}

    for doc in docs:
        upper = doc.page_content.upper()
        lines = upper.splitlines()

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue

            ssns = SSN_PATTERN.findall(line)

            # ── Strategy 1: Same-line pairing ──────────────────────────────
            if ssns and "&" not in line:
                raw_names = NAME_PATTERN.findall(line)
                names = []
                for n in raw_names:
                    n = normalize_name(n)
                    if n in BLACKLIST:
                        continue
                    if _is_person_name(n):
                        names.append(n)

                if len(names) == 1 and len(ssns) == 1:
                    name, ssn = names[0], ssns[0]
                    if name not in pairs:
                        pairs[name] = ssn
                        print(f"[DEBUG] Same-line paired: {name} -> {ssn}")
                    continue

            # ── Strategy 2: SSN-only line → reconstruct name from prior lines
            is_ssn_only = bool(re.match(r'^\s*\d{3}[- ]?\d{2}[- ]?\d{4}\s*$', line))
            if is_ssn_only and "&" not in line:
                ssn = SSN_PATTERN.search(line).group(0)

                # Look back: expect pattern like:
                # i-2: "JOHN"  (first name)
                # i-1: "SMITH" (last name)
                # i:   "111 11 1111" (SSN)
                last_name = None
                first_name = None

                if i >= 1:
                    prev1 = lines[i - 1].strip()
                    if re.match(r'^[A-Z]{2,}$', prev1) and _is_last_name_only(prev1):
                        last_name = prev1

                if last_name and i >= 2:
                    prev2 = lines[i - 2].strip()
                    # Could be "JOHN" or "JOHN MIDDLENAME" — take first word
                    parts = prev2.split()
                    if parts and re.match(r'^[A-Z]{2,}$', parts[0]) and parts[0] not in FORM_WORDS:
                        first_name = parts[0]

                if first_name and last_name:
                    full_name = f"{first_name} {last_name}"
                    if _is_person_name(full_name) and full_name not in pairs:
                        pairs[full_name] = ssn
                        print(f"[DEBUG] Split-line paired: {full_name} -> {ssn}")
                    continue

            # ── Strategy 3: Proximity pairing ──────────────────────────────
            if not ssns and "&" not in line:
                raw_names = NAME_PATTERN.findall(line)
                names = []
                for n in raw_names:
                    n = normalize_name(n)
                    if n in BLACKLIST:
                        continue
                    if _is_person_name(n):
                        names.append(n)

                if names:
                    for k in range(i + 1, min(i + 4, len(lines))):
                        ahead_ssns = SSN_PATTERN.findall(lines[k])
                        if ahead_ssns and "&" not in lines[k]:
                            for name in names:
                                if name not in pairs:
                                    pairs[name] = ahead_ssns[0]
                                    print(f"[DEBUG] Proximity paired: {name} -> {ahead_ssns[0]}")
                            break

    print(f"[DEBUG] Final pairs: {pairs}")
    return pairs
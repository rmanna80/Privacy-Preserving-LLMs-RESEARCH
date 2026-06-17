"""
ai_core/ssn_resolver.py — accurate, family-aware SSN retrieval (Option 2).

Replaces the blacklist-guessing approach in sensitive_extractors with a
match-against-known-people approach. We know who the family members are
(from the people table), so instead of guessing which capitalized phrases
in a dense insurance contract are names, we only ever match SSNs to actual
people in the family. False positives like "PAPERWORK REDUCTION ACT"
become impossible because that's not a person in the family.

Resolution order for "what is <name>'s SSN?":
  1. DB FIRST — if the Person record has an encrypted SSN, return it.
     Authoritative; no scanning, no false positives.
  2. DOCUMENT SCAN — if not in the DB, scan the indexed chunks for an
     SSN near that known person's name.

Access policy (Option 2 interim — graduated per-relationship rules come
in the next phase once User<->Person identity exists):
  - Advisor (authorized=True)  -> full SSN
  - Client / unauthorized      -> masked  •••-••-1234

The returned value is a plain string built deterministically here. It is
NEVER passed through the LLM, so no model guardrail can refuse it and no
hallucination can corrupt it.
"""

from __future__ import annotations

import re
from typing import Optional

from langchain_core.documents import Document

SSN_PATTERN = re.compile(r"\b\d{3}[- ]?\d{2}[- ]?\d{4}\b")


# ─────────────────────────────────────────────────────────────────────
# Name helpers
# ─────────────────────────────────────────────────────────────────────

def _normalize(s: str) -> str:
    s = re.sub(r"[^A-Za-z\s]", " ", s or "")
    return re.sub(r"\s+", " ", s).strip().upper()


def extract_requested_name(question: str) -> Optional[str]:
    """Pull the target name out of an SSN question. Returns normalized
    upper-case name, or None if the question is generic ('list all SSNs')."""
    # Strip common leading question phrasing so it can't be captured as
    # part of the name (e.g. "what is John Smith's ssn" -> "John Smith").
    q = question.strip()
    q = re.sub(r"^\s*(what\s+is|what's|whats|tell\s+me|give\s+me|show\s+me|"
               r"please|can\s+you|could\s+you|find|get)\s+", "", q,
               flags=re.IGNORECASE)

    patterns = [
        r"([A-Za-z]+(?:\s+[A-Za-z]+){0,3})'s\s+(?:ssn|social security)",
        r"(?:ssn|social security)\s+(?:number\s+)?(?:of|for)\s+([A-Za-z]+(?:\s+[A-Za-z]+){0,3})",
        r"(?:what\s+is\s+)?([A-Za-z]+(?:\s+[A-Za-z]+){0,3})'s\s+(?:ssn|social security)",
    ]
    for pat in patterns:
        m = re.search(pat, q, re.IGNORECASE)
        if m:
            name = _normalize(m.group(1))
            # Guard: drop any leftover leading filler tokens
            filler = {"WHAT", "IS", "WHATS", "TELL", "ME", "GIVE", "SHOW",
                      "PLEASE", "CAN", "YOU", "COULD", "FIND", "GET", "THE"}
            tokens = [t for t in name.split() if t not in filler]
            if tokens:
                return " ".join(tokens)
    return None


def _name_matches(requested: str, person_full: str) -> bool:
    """Does the requested name refer to this person? Token-overlap match
    so 'JOHN SMITH' matches 'JOHN A SMITH' and 'SMITH' matches if unique."""
    req = set(_normalize(requested).split())
    person = set(_normalize(person_full).split())
    if not req or not person:
        return False
    # Full subset match (requested is contained in the person's name)
    if req.issubset(person):
        return True
    # At least 2 shared tokens (first + last)
    return len(req & person) >= 2


def mask_ssn(ssn: str) -> str:
    """Return •••-••-1234 form, preserving the last 4 digits."""
    digits = re.sub(r"\D", "", ssn or "")
    if len(digits) < 4:
        return "•••-••-••••"
    return f"•••-••-{digits[-4:]}"


# ─────────────────────────────────────────────────────────────────────
# Document scan (fallback) — matched ONLY against known people
# ─────────────────────────────────────────────────────────────────────

def _scan_documents_for_person_ssn(
    chunks: list[Document],
    person_full_name: str,
) -> Optional[str]:
    """Find an SSN near this specific known person's name in the chunks.

    Because we only ever look for ONE known name, we can't produce the
    garbage matches the old blacklist approach did.
    """
    target_tokens = set(_normalize(person_full_name).split())
    if not target_tokens:
        return None

    for doc in chunks:
        text = doc.page_content
        lines = text.splitlines()
        for i, line in enumerate(lines):
            line_tokens = set(_normalize(line).split())
            # Is the person's name on this line (first + last present)?
            if len(target_tokens & line_tokens) >= 2:
                # SSN on the same line?
                m = SSN_PATTERN.search(line)
                if m:
                    return m.group(0)
                # SSN within the next 4 lines?
                for k in range(i + 1, min(i + 5, len(lines))):
                    m2 = SSN_PATTERN.search(lines[k])
                    if m2:
                        return m2.group(0)
    return None


# ─────────────────────────────────────────────────────────────────────
# Public resolver
# ─────────────────────────────────────────────────────────────────────

def resolve_ssn_question(
    question: str,
    family_id: int,
    chunks: list[Document],
    *,
    authorized: bool,
) -> str:
    """Answer an SSN question deterministically (no LLM involved).

    authorized=True  -> full SSN  (advisor / AUTHORIZED disclosure mode)
    authorized=False -> masked    (client / REDACTED disclosure mode)
    """
    from db.repositories import list_people_in_family

    people = list_people_in_family(family_id)
    if not people:
        return "I don't have any people on file for this family yet."

    requested = extract_requested_name(question)

    # ── Generic "list all SSNs" question ──────────────────────────────
    if not requested:
        names_with_ssn = []
        for p in people:
            has_ssn = bool(getattr(p, "ssn", None))
            if not has_ssn:
                # check documents
                doc_ssn = _scan_documents_for_person_ssn(chunks, p.full_name)
                has_ssn = bool(doc_ssn)
            if has_ssn:
                names_with_ssn.append(p.display_name)
        if not names_with_ssn:
            return "I couldn't find a social security number for anyone in this family."
        listed = "\n".join(f"- {n}" for n in names_with_ssn)
        return (
            "I have a social security number on file for:\n"
            f"{listed}\n\nWhose SSN would you like?"
        )

    # ── Specific person ───────────────────────────────────────────────
    match = None
    for p in people:
        if _name_matches(requested, p.full_name):
            match = p
            break

    if match is None:
        return (
            f"I couldn't find anyone named {requested.title()} in this "
            f"family's records."
        )

    # 1) DB first — the authoritative encrypted record
    ssn = getattr(match, "ssn", None)

    # 2) Fall back to document scan
    if not ssn:
        ssn = _scan_documents_for_person_ssn(chunks, match.full_name)

    if not ssn:
        return (
            f"I couldn't find a social security number for "
            f"{match.display_name} in the records or documents."
        )

    # ── Apply access policy ───────────────────────────────────────────
    if authorized:
        return f"{match.display_name}'s SSN is **{ssn}**."
    else:
        return (
            f"{match.display_name}'s SSN ends in **{mask_ssn(ssn)}**. "
            f"Full SSN access for clients is configured per-person by your "
            f"advisor."
        )

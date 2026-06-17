"""
ai_core/promotion.py — promote verified extractions into structured data.

Model B: a verified extraction is a trusted FACT. Promotion turns that
fact into actual graph data (Person / Entity / Role / attribute updates)
that the Family Tree, Org Ownership, and Entities pages render natively —
the thing that actually competes with Ester.

Design principle: NEVER auto-mutate. Promotion always goes through an
advisor confirm step in the UI. This module provides:
  - classify_field(): what KIND of structured target does a field map to?
  - suggest_person_match(): best existing Person for a name string
  - the actual create/link operations (called only after advisor confirms)

Field → target mapping
----------------------
  ROLE fields      (trustee, successor_trustee, beneficiary, executor…)
      → create a Role linking a Person to an Entity
  PERSON fields    (trustor, spouse, guardian, testator, insured…)
      → ensure a Person exists (match or create)
  ENTITY_ATTR      (trust_name, jurisdiction, trust_type, trust_ein…)
      → update fields on an Entity
  PLAIN            (AGI, total_tax, dates, yes/no clauses…)
      → no structured target; stays display-only (not promotable)
"""

from __future__ import annotations

import re
from typing import Optional


# ─────────────────────────────────────────────────────────────────────
# Field classification
# ─────────────────────────────────────────────────────────────────────

# field_key -> role_type for ROLE fields
ROLE_FIELD_MAP: dict[str, str] = {
    "current_trustee":        "trustee",
    "successor_trustee":      "successor_trustee",
    "second_successor_trustee": "successor_trustee",
    "beneficiaries_current":  "beneficiary",
    "beneficiaries_remainder": "remainder_beneficiary",
    "executor_primary":       "executor",
    "executor_alternate":     "executor",
    "beneficiary_primary":    "beneficiary",
    "insured":                "insured",
    "owner":                  "owner",
}

# field_key -> nothing structured; these define/locate a PERSON
PERSON_FIELDS = {
    "trustor", "testator", "guardian_minors", "spouse_name",
    "primary_taxpayer",
}

# field_key -> entity attribute to set
ENTITY_ATTR_MAP: dict[str, str] = {
    "trust_name":    "name",
    "trust_type":    "sub_type",
    "jurisdiction":  "jurisdiction",
    "creation_date": "formation_date",
    "trust_ein":     "tax_id",          # encrypted setter on Entity
    "carrier":       "name",
    "policy_type":   "sub_type",
}


def classify_field(field_key: str) -> str:
    """Return 'role' | 'person' | 'entity_attr' | 'plain'."""
    if field_key in ROLE_FIELD_MAP:
        return "role"
    if field_key in PERSON_FIELDS:
        return "person"
    if field_key in ENTITY_ATTR_MAP:
        return "entity_attr"
    return "plain"


def is_promotable(field_key: str) -> bool:
    return classify_field(field_key) != "plain"


# ─────────────────────────────────────────────────────────────────────
# Name parsing & matching
# ─────────────────────────────────────────────────────────────────────

def split_names(value: str) -> list[str]:
    """A list-typed extraction value may hold several names joined by
    '; ' or ' and '. Return individual name strings."""
    parts = re.split(r";|\band\b|&|,", value)
    return [p.strip() for p in parts if p.strip()]


def _normalize(s: str) -> str:
    s = re.sub(r"[^A-Za-z\s]", " ", s or "")
    return re.sub(r"\s+", " ", s).strip().upper()


def suggest_person_match(name: str, people: list) -> Optional[int]:
    """Return the person_id of the best existing match, or None.

    Token-overlap heuristic: prefer full containment, then 2+ shared
    tokens (first + last). Conservative — when unsure, returns None so
    the advisor decides."""
    target = set(_normalize(name).split())
    if not target:
        return None

    best_id, best_score = None, 0
    for p in people:
        cand = set(_normalize(p.full_name).split())
        if not cand:
            continue
        if target.issubset(cand) or cand.issubset(target):
            return p.id
        score = len(target & cand)
        if score > best_score:
            best_score, best_id = score, p.id
    return best_id if best_score >= 2 else None


def split_first_last(name: str) -> tuple[str, str]:
    """Best-effort split of a display name into (first, last)."""
    parts = name.strip().split()
    if not parts:
        return ("", "")
    if len(parts) == 1:
        return (parts[0], "")
    return (parts[0], parts[-1])


# ─────────────────────────────────────────────────────────────────────
# Promotion operations (called AFTER advisor confirms in the UI)
# ─────────────────────────────────────────────────────────────────────

def promote_role(
    *,
    family_id: int,
    person_id: int,
    entity_id: int,
    role_type: str,
):
    """Create a Role linking a Person to an Entity. De-dupes: if an
    identical active role exists, returns it instead of duplicating."""
    from db.database import get_session
    from db.models import Role
    from sqlmodel import select

    with get_session() as s:
        existing = s.exec(
            select(Role).where(
                (Role.person_id == person_id)
                & (Role.entity_id == entity_id)
                & (Role.role_type == role_type)
            )
        ).first()
        if existing is not None:
            return existing
        r = Role(
            person_id=person_id,
            entity_id=entity_id,
            role_type=role_type,
            is_active=True,
        )
        s.add(r)
        s.flush()
        s.refresh(r)
        return r


def ensure_person(
    *,
    family_id: int,
    first_name: str,
    last_name: str,
):
    """Create a Person if no clear match exists. Returns the Person."""
    from db.repositories import create_person
    return create_person(
        family_id=family_id,
        first_name=first_name or "(unknown)",
        last_name=last_name or "",
    )


def update_entity_attr(
    *,
    entity_id: int,
    attr: str,
    value: str,
):
    """Set one attribute on an Entity. Handles the encrypted tax_id."""
    from db.database import get_session
    from db.models import Entity

    with get_session() as s:
        e = s.get(Entity, entity_id)
        if e is None:
            return None
        if attr == "tax_id":
            e.tax_id = value           # encrypted setter
        else:
            setattr(e, attr, value)
        s.add(e)
        s.flush()
        s.refresh(e)
        return e


def create_entity_for_family(
    *,
    family_id: int,
    name: str,
    entity_type: str = "trust",
):
    """Create a new Entity (used when a trust extraction has no entity to
    attach to yet)."""
    from db.repositories import create_entity
    return create_entity(
        family_id=family_id,
        name=name,
        entity_type=entity_type,
    )


def mark_extraction_promoted(extraction_id: int) -> None:
    """Record that an extraction has been promoted, so the UI can show it
    as done. We reuse text_snippet-adjacent state by setting a flag in
    notes-like field; simplest is a dedicated check via Role/Person existence,
    but to keep it explicit we stamp the extraction's extracted_by tag."""
    from db.database import get_session
    from db.models import Extraction

    with get_session() as s:
        e = s.get(Extraction, extraction_id)
        if e is None:
            return
        # Append a marker to extraction_type so we can detect promotion
        # without a schema change. e.g. 'text' -> 'text|promoted'
        if e.extraction_type and "|promoted" not in e.extraction_type:
            e.extraction_type = f"{e.extraction_type}|promoted"
        s.add(e)
        s.flush()


def is_promoted(extraction) -> bool:
    return bool(extraction.extraction_type) and "|promoted" in extraction.extraction_type

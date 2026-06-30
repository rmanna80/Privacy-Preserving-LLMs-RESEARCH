"""
api/schemas.py — request/response shapes for the API.

Keeping these explicit (rather than returning ORM objects directly)
means the API contract is stable and we never accidentally leak a field
(like an encrypted SSN/EIN blob) just because it's on the model.

PII NOTE: PersonOut and EntityOut deliberately DO NOT include ssn /
tax_id. Those are encrypted at rest and have no business in a list/read
view. If a single authorized "reveal" endpoint is ever needed, it gets
its own explicit route with its own audit — never bundled into a list.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    name: str


class MeResponse(BaseModel):
    username: str
    role: str
    name: str


class FamilyOut(BaseModel):
    id: int
    name: str
    notes: Optional[str] = None
    people_count: int
    entities_count: int
    documents_count: int


class PersonOut(BaseModel):
    id: int
    display_name: str
    full_name: str
    first_name: str
    last_name: str
    middle_name: Optional[str] = None
    preferred_name: Optional[str] = None
    dob: Optional[date] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    is_deceased: bool = False
    date_of_death: Optional[date] = None
    has_ssn: bool = False   # whether one is on file — NOT the value


class EntityOut(BaseModel):
    id: int
    name: str
    entity_type: str
    sub_type: Optional[str] = None
    jurisdiction: Optional[str] = None
    formation_date: Optional[date] = None
    termination_date: Optional[date] = None
    has_tax_id: bool = False   # whether one is on file — NOT the value


class RoleOut(BaseModel):
    id: int
    person_id: int
    person_name: str
    entity_id: int
    role_type: str
    interest_percentage: Optional[float] = None
    is_active: bool = True
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    notes: Optional[str] = None


class RelationshipOut(BaseModel):
    id: int
    person_a_id: int
    person_a_name: str
    person_b_id: int
    person_b_name: str
    relationship_type: str
    start_date: Optional[date] = None
    notes: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────
# Write request bodies
#
# PII (ssn / tax_id) is ACCEPTED on write — an advisor entering a
# client's SSN is legitimate — but is never returned in any response.
# The response models (PersonOut / EntityOut) have no such field.
# ─────────────────────────────────────────────────────────────────────

class PersonCreate(BaseModel):
    first_name: str
    last_name: str
    middle_name: Optional[str] = None
    preferred_name: Optional[str] = None
    dob: Optional[date] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    ssn: Optional[str] = None
    is_deceased: bool = False
    date_of_death: Optional[date] = None
    notes: Optional[str] = None


class PersonUpdate(BaseModel):
    # All optional — only provided fields are changed.
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    middle_name: Optional[str] = None
    preferred_name: Optional[str] = None
    dob: Optional[date] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    ssn: Optional[str] = None
    is_deceased: Optional[bool] = None
    date_of_death: Optional[date] = None
    notes: Optional[str] = None


class EntityCreate(BaseModel):
    name: str
    entity_type: str
    sub_type: Optional[str] = None
    jurisdiction: Optional[str] = None
    formation_date: Optional[date] = None
    termination_date: Optional[date] = None
    tax_id: Optional[str] = None
    notes: Optional[str] = None


class EntityUpdate(BaseModel):
    name: Optional[str] = None
    entity_type: Optional[str] = None
    sub_type: Optional[str] = None
    jurisdiction: Optional[str] = None
    formation_date: Optional[date] = None
    termination_date: Optional[date] = None
    tax_id: Optional[str] = None
    notes: Optional[str] = None


class RoleCreate(BaseModel):
    person_id: int
    entity_id: int
    role_type: str
    interest_percentage: Optional[float] = None
    is_active: bool = True
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    notes: Optional[str] = None


class DocumentOut(BaseModel):
    """PII/internal-safe document view.

    Deliberately omits file_path (internal disk location) and file_hash
    (internal dedup detail) — neither belongs over the API. Exposes only
    what a frontend needs to display and act on a document.
    """
    id: int
    original_filename: str
    category: Optional[str] = None
    category_label: str
    doc_type: str
    doc_year: Optional[int] = None
    file_size_bytes: Optional[int] = None
    mime_type: Optional[str] = None
    notes: Optional[str] = None
    person_id: Optional[int] = None
    entity_id: Optional[int] = None
    indexed_in_vectorstore: bool = False
    extraction_status: Optional[str] = None
    uploaded_at: Optional[datetime] = None


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    documents_indexed: bool   # whether the family had an index to answer from
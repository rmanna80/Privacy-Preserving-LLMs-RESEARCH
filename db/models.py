"""
SQLModel schemas for the wealth intelligence platform.

Design philosophy
-----------------
- Family is the root container. Everything ties back to a family_id.
- People are members of a family.
- Entities (trusts, LLCs, etc.) belong to a family.
- Roles bridge People <-> Entities (trustee, beneficiary, etc.).
- Relationships bridge People <-> People (spouse, parent_of).
- Documents belong to a family and may be tied to a person and/or entity.
- Extractions are structured facts pulled from documents — the bridge
  between the RAG layer and structured data, with citations back to
  source page + bbox.

PII fields (SSN, EIN) are stored encrypted as BLOBs. Use the model
properties (.ssn, .tax_id) to read/write plaintext transparently.

Why no foreign-key enforcement on SQLite by default:
SQLite ships with FK enforcement OFF unless you turn it on. We can
enable it via a connection-level PRAGMA in database.py later. For now,
relationships are defined and queryable — we just won't get cascade
deletes until we flip that switch. That's a Step 1.5 concern, not Step 1.
"""

from datetime import datetime, date
from typing import Optional

from sqlmodel import SQLModel, Field
from sqlalchemy import Column, LargeBinary

from db.crypto import encrypt_field, decrypt_field


# =============================================================================
# Users & access control
# =============================================================================

class User(SQLModel, table=True):
    """An authenticated user — admin, advisor, or client."""
    __tablename__ = "users"

    id: Optional[int] = Field(default=None, primary_key=True)
    email: str = Field(unique=True, index=True)
    password_hash: str  # bcrypt hash
    full_name: str
    role: str  # 'admin' | 'advisor' | 'client'
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None


class UserFamilyAccess(SQLModel, table=True):
    """Grants a user access to a family. Powers the future collaboration layer.

    A family always has exactly one owner (the primary advisor). Editors
    can read and modify; viewers (e.g. attorneys, CPAs, the family itself)
    can only read.
    """
    __tablename__ = "user_family_access"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id", index=True)
    family_id: int = Field(foreign_key="families.id", index=True)
    access_level: str  # 'owner' | 'editor' | 'viewer'
    granted_by_user_id: Optional[int] = Field(default=None, foreign_key="users.id")
    granted_at: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# Family graph
# =============================================================================

class Family(SQLModel, table=True):
    """A family unit — the root container for everything else."""
    __tablename__ = "families"

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str  # "The Smith Family"
    advisor_user_id: int = Field(foreign_key="users.id", index=True)
    notes: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Person(SQLModel, table=True):
    """An individual member of a family."""
    __tablename__ = "people"

    id: Optional[int] = Field(default=None, primary_key=True)
    family_id: int = Field(foreign_key="families.id", index=True)

    first_name: str
    last_name: str
    middle_name: Optional[str] = None
    preferred_name: Optional[str] = None  # what they actually go by

    dob: Optional[date] = None
    is_deceased: bool = False
    date_of_death: Optional[date] = None

    email: Optional[str] = None
    phone: Optional[str] = None

    # SSN stored as encrypted blob — use the .ssn property to read/write plaintext
    ssn_encrypted: Optional[bytes] = Field(
        default=None,
        sa_column=Column(LargeBinary),
    )

    notes: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # ---- transparent PII access ----

    @property
    def ssn(self) -> Optional[str]:
        return decrypt_field(self.ssn_encrypted)

    @ssn.setter
    def ssn(self, plaintext: Optional[str]) -> None:
        self.ssn_encrypted = encrypt_field(plaintext)

    @property
    def full_name(self) -> str:
        parts = [self.first_name, self.middle_name, self.last_name]
        return " ".join(p for p in parts if p)

    @property
    def display_name(self) -> str:
        """Preferred name when set, otherwise full name."""
        return self.preferred_name or self.full_name


class Entity(SQLModel, table=True):
    """A legal entity owned by or tied to the family.

    Trusts, LLCs, partnerships, S-corps, foundations, etc. The
    entity_type is the broad category; sub_type captures the
    important nuance (revocable vs irrevocable trust, GRAT vs ILIT,
    single-member vs multi-member LLC, etc.).
    """
    __tablename__ = "entities"

    id: Optional[int] = Field(default=None, primary_key=True)
    family_id: int = Field(foreign_key="families.id", index=True)

    name: str  # "Smith Family Revocable Trust"

    entity_type: str
    # Common entity_type values:
    #   'trust' | 'llc' | 'partnership' | 'corporation'
    #   | 'foundation' | 'donor_advised_fund' | 'other'

    sub_type: Optional[str] = None
    # Common sub_type values:
    #   trust:    'revocable' | 'irrevocable' | 'grat' | 'ilit'
    #             | 'crat' | 'crut' | 'qprt' | 'dynasty' | 'special_needs'
    #   llc:      'single_member' | 'multi_member' | 'series'
    #   corp:     'c_corp' | 's_corp' | 'b_corp'

    jurisdiction: Optional[str] = None  # e.g. 'NC', 'DE', 'NV'
    formation_date: Optional[date] = None
    termination_date: Optional[date] = None

    # EIN / Tax ID — encrypted
    tax_id_encrypted: Optional[bytes] = Field(
        default=None,
        sa_column=Column(LargeBinary),
    )

    notes: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def tax_id(self) -> Optional[str]:
        return decrypt_field(self.tax_id_encrypted)

    @tax_id.setter
    def tax_id(self, plaintext: Optional[str]) -> None:
        self.tax_id_encrypted = encrypt_field(plaintext)


class Role(SQLModel, table=True):
    """A person's role in an entity (trustee, beneficiary, member, etc.).

    A person can hold multiple roles in the same entity (e.g. grantor +
    trustee + beneficiary of their own revocable trust). Each role gets
    its own row.
    """
    __tablename__ = "roles"

    id: Optional[int] = Field(default=None, primary_key=True)
    person_id: int = Field(foreign_key="people.id", index=True)
    entity_id: int = Field(foreign_key="entities.id", index=True)

    role_type: str
    # Common role_type values:
    #   trust:  'grantor' | 'settlor' | 'trustee' | 'co_trustee'
    #           | 'successor_trustee' | 'trust_protector'
    #           | 'beneficiary' | 'contingent_beneficiary'
    #           | 'remainder_beneficiary'
    #   llc:    'member' | 'manager' | 'managing_member'
    #   corp:   'shareholder' | 'officer' | 'director'

    start_date: Optional[date] = None
    end_date: Optional[date] = None
    is_active: bool = True

    # For beneficiaries / shareholders / members
    interest_percentage: Optional[float] = None

    notes: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Relationship(SQLModel, table=True):
    """A person-to-person relationship (spouse, parent_of, sibling).

    For directional relationships (parent_of), person_a is the "from"
    side: A is parent of B. For symmetric relationships (spouse,
    sibling), direction doesn't matter but we still store one row.

    Inactive relationships (divorced, deceased) get end_date set and
    is_active=False rather than being deleted — historical context
    matters for estate planning.
    """
    __tablename__ = "relationships"

    id: Optional[int] = Field(default=None, primary_key=True)
    person_a_id: int = Field(foreign_key="people.id", index=True)
    person_b_id: int = Field(foreign_key="people.id", index=True)

    relationship_type: str
    # Values: 'spouse' (symmetric) | 'ex_spouse' (symmetric)
    #       | 'parent_of' (a is parent of b)
    #       | 'sibling' (symmetric)
    #       | 'guardian_of' (a is guardian of b)

    start_date: Optional[date] = None  # marriage date, birth date, etc.
    end_date: Optional[date] = None    # divorce, death, etc.
    is_active: bool = True
    notes: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# Documents & extractions — the bridge to the RAG layer
# =============================================================================

class Document(SQLModel, table=True):
    """A document uploaded for a family.

    file_path points to the actual file on disk (your existing per-client
    folder structure). file_hash is the sha256 — used to detect duplicate
    uploads.

    A document can be tied to a primary person and/or entity (e.g. John's
    will → person_id=John; the trust agreement → entity_id=Trust).
    """
    __tablename__ = "documents"

    id: Optional[int] = Field(default=None, primary_key=True)
    family_id: int = Field(foreign_key="families.id", index=True)
    person_id: Optional[int] = Field(default=None, foreign_key="people.id")
    entity_id: Optional[int] = Field(default=None, foreign_key="entities.id")

    file_path: str
    file_hash: str = Field(index=True)  # sha256 hex digest
    original_filename: str
    file_size_bytes: Optional[int] = None
    mime_type: Optional[str] = None

    doc_type: str
    # Common doc_type values:
    #   'will' | 'trust_agreement' | 'trust_amendment'
    #   | 'tax_return_1040' | 'tax_return_1041' | 'k1'
    #   | 'insurance_policy' | 'beneficiary_designation'
    #   | 'deed' | 'financial_statement' | 'brokerage_statement'
    #   | 'operating_agreement' | 'bylaws'
    #   | 'power_of_attorney' | 'healthcare_directive'
    #   | 'other'

    doc_year: Optional[int] = None        # e.g. 2024 for a 2024 tax return
    doc_date: Optional[date] = None       # date the doc was signed/effective

    # Pipeline status flags
    indexed_in_vectorstore: bool = False  # has it been chunked + embedded?
    extraction_status: str = "pending"
    # 'pending' | 'processing' | 'complete' | 'failed'

    category: Optional[str] = None
    # 'investments' | 'estate_planning' | 'tax_planning'
    # | 'insurance_planning' | 'business_planning'

    notes: Optional[str] = None
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)
    uploaded_by_user_id: Optional[int] = Field(default=None, foreign_key="users.id")


class Extraction(SQLModel, table=True):
    """A structured fact pulled from a document.

    This is the bridge between the RAG layer and the structured data
    layer. Every fact the system surfaces should ideally land here, with
    a pointer back to the exact page (and ideally bbox) where it came
    from. That's what makes the document viewer's click-to-cite feature
    work.

    PII handling: if is_pii=True, field_value is base64(encrypted_blob)
    rather than plaintext. Use the .value property to read.
    """
    __tablename__ = "extractions"

    id: Optional[int] = Field(default=None, primary_key=True)
    document_id: int = Field(foreign_key="documents.id", index=True)

    # ---- where in the source doc? ----
    page_number: Optional[int] = None
    bbox_json: Optional[str] = None       # '{"x":0,"y":0,"w":0,"h":0}'
    text_snippet: Optional[str] = None    # surrounding context for the UI

    # ---- what did we extract? ----
    extraction_type: str
    # Values: 'ssn' | 'ein' | 'role' | 'date' | 'amount'
    #       | 'party_name' | 'address' | 'account_number'
    #       | 'beneficiary_designation' | 'distribution_term'

    field_key: str            # human-readable label, e.g. "Trustee"
    field_value: str          # value (or base64-encrypted if PII)
    is_pii: bool = False

    confidence: float = 1.0   # 0.0–1.0; manual = 1.0, regex high, LLM varies

    # ---- optional links into the family graph ----
    person_id: Optional[int] = Field(default=None, foreign_key="people.id", index=True)
    entity_id: Optional[int] = Field(default=None, foreign_key="entities.id", index=True)

    # ---- provenance ----
    extracted_by: str = "regex"  # 'regex' | 'llm' | 'manual'
    verified_by_user_id: Optional[int] = Field(default=None, foreign_key="users.id")
    verified_at: Optional[datetime] = None

    created_at: datetime = Field(default_factory=datetime.utcnow)





# ════════════════════════════════════════════════════════════════════
# Advisory Team
# ════════════════════════════════════════════════════════════════════
 
class AdvisoryTeamMember(SQLModel, table=True):
    """A professional helping the family — attorney, CPA, etc.
 
    Scoped to a family. The same human may appear as separate rows on
    different families (a CPA might serve five families; that's five
    rows). That's intentional — each family relationship is its own
    record with its own engagement notes.
    """
    __tablename__ = "advisory_team_members"
 
    id: Optional[int] = Field(default=None, primary_key=True)
    family_id: int = Field(foreign_key="families.id", index=True)
 
    role: str
    # Allowed values (validated in application code, not DB):
    #   'wealth_strategist' | 'tax_strategist'
    #   | 'estate_planning_attorney' | 'business_attorney'
    #   | 'general_counsel' | 'm_and_a_attorney'
    #   | 'insurance_broker' | 'family_coach'
    #   | 'investment_banker' | 'other'
 
    full_name: str
    firm: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    notes: Optional[str] = None
 
    is_active: bool = True
    engaged_at: Optional[date] = None    # when they started helping
    departed_at: Optional[date] = None   # when (if) they stopped
 
    # If/when this team member also has a user account in Angel,
    # link them by setting this. Optional, nullable, doesn't block
    # the team member from existing as a contact-only record.
    user_id: Optional[int] = Field(default=None, foreign_key="users.id")
 
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
 
 
# ════════════════════════════════════════════════════════════════════
# Tasks
# ════════════════════════════════════════════════════════════════════
 
class Task(SQLModel, table=True):
    """A work item — for the family or the advisory team.
 
    Tasks are scoped to a family. Assignment is to EITHER a Person
    (family member) OR an AdvisoryTeamMember — exactly one of the
    two foreign keys is set, indicated by `assignee_type`.
    """
    __tablename__ = "tasks"
 
    id: Optional[int] = Field(default=None, primary_key=True)
    family_id: int = Field(foreign_key="families.id", index=True)
 
    title: str  # "Header" in your spec; we use 'title' for SQL friendliness
    description: Optional[str] = None
 
    # Assignee — exactly one of these is set
    assignee_type: Optional[str] = None
    # 'person' | 'team_member' | None (unassigned)
    assigned_person_id: Optional[int] = Field(
        default=None, foreign_key="people.id"
    )
    assigned_team_member_id: Optional[int] = Field(
        default=None, foreign_key="advisory_team_members.id"
    )
 
    # Who created the task — for "assigned by" display and audit
    created_by_user_id: int = Field(foreign_key="users.id")
 
    due_date: Optional[date] = None
 
    status: str = "open"
    # 'open' | 'in_progress' | 'blocked' | 'complete' | 'archived'
 
    priority: Optional[str] = None
    # 'low' | 'normal' | 'high' | 'urgent' | None
 
    completed_at: Optional[datetime] = None
 
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
 
 
class TaskComment(SQLModel, table=True):
    """A comment on a task — thread of updates and discussion."""
    __tablename__ = "task_comments"
 
    id: Optional[int] = Field(default=None, primary_key=True)
    task_id: int = Field(foreign_key="tasks.id", index=True)
    author_user_id: int = Field(foreign_key="users.id")
 
    body: str
 
    # If this comment represents an automated event ("status changed
    # to in_progress", "assignee changed from X to Y"), use this. UI
    # can render system comments differently from human ones.
    is_system: bool = False
 
    created_at: datetime = Field(default_factory=datetime.utcnow)
    edited_at: Optional[datetime] = None
 
 
class TaskDocumentLink(SQLModel, table=True):
    """Many-to-many: a task can reference multiple documents,
    and a document can be referenced by multiple tasks."""
    __tablename__ = "task_document_links"
 
    id: Optional[int] = Field(default=None, primary_key=True)
    task_id: int = Field(foreign_key="tasks.id", index=True)
    document_id: int = Field(foreign_key="documents.id", index=True)
    attached_by_user_id: int = Field(foreign_key="users.id")
    attached_at: datetime = Field(default_factory=datetime.utcnow)
 
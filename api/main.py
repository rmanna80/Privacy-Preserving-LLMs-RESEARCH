"""
api/main.py — Angel local API (the FastAPI bridge).

Runs alongside the existing Streamlit app, on localhost, sharing the same
database and the same AuthSystem. Nothing about the existing app changes;
this is a parallel, secured access layer that a future frontend
(React, or a Tauri/Electron desktop app) will talk to.

Run it (from the project root, in its own terminal):
    uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload

Interactive docs:  http://127.0.0.1:8000/docs

Security posture (appliance model):
  - Binds to localhost; not exposed to the public internet.
  - JWT auth: the token proves who you are. Endpoints scope data to the
    authenticated identity — a client cannot ask for someone else's data
    by changing a parameter, because there is no identity parameter.
  - Defense in depth: every family-scoped route runs through the
    `get_owned_family` dependency, which re-checks that the authenticated
    advisor owns the requested family before ANY nested data is returned.
    Non-owners get 404 (we don't even confirm the family exists).
  - PII (SSN / EIN) is never returned by read endpoints — only a boolean
    "on file" flag.
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import (
    FastAPI, Depends, HTTPException, status, UploadFile, File, Form,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from ui.auth import AuthSystem
from db.repositories import (
    ensure_db_user,
    list_families_for_advisor,
    get_family,
    list_people_in_family,
    list_entities_in_family,
    list_relationships_in_family,
    list_roles_for_entity,
    list_documents_for_family,
    get_person, get_entity, get_role,
    create_person, update_person, delete_person,
    create_entity, update_entity, delete_entity,
    create_role, delete_role,
    category_label,
    compute_file_hash, find_document_by_hash, create_document,
    DOCUMENT_CATEGORIES,
)

from api.security import (
    create_access_token,
    get_current_user,
    require_advisor,
    AuthedUser,
)
from api.schemas import (
    TokenResponse, MeResponse, FamilyOut,
    PersonOut, EntityOut, RoleOut, RelationshipOut,
    PersonCreate, PersonUpdate, EntityCreate, EntityUpdate, RoleCreate,
    DocumentOut, ChatRequest, ChatResponse,
)


app = FastAPI(
    title="Angel API",
    description="Local API for Angel — Database for Family Wealth.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", "http://127.0.0.1:3000",
        "http://localhost:5173", "http://127.0.0.1:5173",
        "tauri://localhost",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "angel-api"}


# ─────────────────────────────────────────────────────────────────────
# Auth
# ─────────────────────────────────────────────────────────────────────

@app.post("/auth/login", response_model=TokenResponse)
def login(form: OAuth2PasswordRequestForm = Depends()) -> TokenResponse:
    auth = AuthSystem()
    if auth.is_locked_out(form.username):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many failed attempts. Please wait and try again.",
        )
    user = auth.authenticate(form.username, form.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )
    role = user.role.value if hasattr(user.role, "value") else str(user.role)
    token = create_access_token(username=user.username, role=role)
    return TokenResponse(
        access_token=token, role=role,
        name=user.client_name or user.username,
    )


@app.get("/me", response_model=MeResponse)
def me(user: AuthedUser = Depends(get_current_user)) -> MeResponse:
    auth = AuthSystem()
    db_user = auth.get_user(user.username)
    name = db_user.client_name if db_user else user.username
    return MeResponse(username=user.username, role=user.role, name=name)


# ─────────────────────────────────────────────────────────────────────
# Shared helpers / dependencies
# ─────────────────────────────────────────────────────────────────────

def _resolve_advisor_db_id(user: AuthedUser) -> int:
    """Map the authenticated user to their wealth.db id, server-side —
    derived from the token's identity, never from a client parameter."""
    auth = AuthSystem()
    auth_user = auth.get_user(user.username)
    if auth_user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authenticated user no longer exists.",
        )
    return ensure_db_user(auth_user)


def get_owned_family(
    family_id: int,
    user: AuthedUser = Depends(require_advisor),
):
    """THE authorization gate for every family-scoped route.

    Resolves the advisor from the token, loads the family, and confirms
    ownership. Returns the family object if (and only if) the
    authenticated advisor owns it; otherwise 404 — we don't reveal that
    a family exists to someone who doesn't own it.

    Every nested endpoint depends on this, so the ownership check is
    written once and can't be forgotten on a new route.
    """
    advisor_db_id = _resolve_advisor_db_id(user)
    family = get_family(family_id)
    if family is None or family.advisor_user_id != advisor_db_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Family not found.",
        )
    return family


def _family_to_out(family) -> FamilyOut:
    return FamilyOut(
        id=family.id,
        name=family.name,
        notes=family.notes,
        people_count=len(list_people_in_family(family.id)),
        entities_count=len(list_entities_in_family(family.id)),
        documents_count=len(list_documents_for_family(family.id)),
    )


# ─────────────────────────────────────────────────────────────────────
# Families
# ─────────────────────────────────────────────────────────────────────

@app.get("/families", response_model=list[FamilyOut])
def list_my_families(user: AuthedUser = Depends(require_advisor)) -> list[FamilyOut]:
    advisor_db_id = _resolve_advisor_db_id(user)
    families = list_families_for_advisor(advisor_db_id)
    return [_family_to_out(f) for f in families]


@app.get("/families/{family_id}", response_model=FamilyOut)
def get_my_family(family=Depends(get_owned_family)) -> FamilyOut:
    return _family_to_out(family)


# ─────────────────────────────────────────────────────────────────────
# Family graph — people / entities / roles / relationships
# All gated by get_owned_family, so a non-owner can't reach any of them.
# ─────────────────────────────────────────────────────────────────────

@app.get("/families/{family_id}/people", response_model=list[PersonOut])
def list_family_people(family=Depends(get_owned_family)) -> list[PersonOut]:
    out = []
    for p in list_people_in_family(family.id):
        out.append(PersonOut(
            id=p.id,
            display_name=p.display_name,
            full_name=p.full_name,
            first_name=p.first_name,
            last_name=p.last_name,
            middle_name=getattr(p, "middle_name", None),
            preferred_name=getattr(p, "preferred_name", None),
            dob=getattr(p, "dob", None),
            email=getattr(p, "email", None),
            phone=getattr(p, "phone", None),
            is_deceased=getattr(p, "is_deceased", False),
            date_of_death=getattr(p, "date_of_death", None),
            has_ssn=getattr(p, "ssn_encrypted", None) is not None,
        ))
    return out


@app.get("/families/{family_id}/entities", response_model=list[EntityOut])
def list_family_entities(family=Depends(get_owned_family)) -> list[EntityOut]:
    out = []
    for e in list_entities_in_family(family.id):
        out.append(EntityOut(
            id=e.id,
            name=e.name,
            entity_type=e.entity_type,
            sub_type=getattr(e, "sub_type", None),
            jurisdiction=getattr(e, "jurisdiction", None),
            formation_date=getattr(e, "formation_date", None),
            termination_date=getattr(e, "termination_date", None),
            has_tax_id=getattr(e, "tax_id_encrypted", None) is not None,
        ))
    return out


@app.get("/families/{family_id}/roles", response_model=list[RoleOut])
def list_family_roles(family=Depends(get_owned_family)) -> list[RoleOut]:
    """Roles across all of this family's entities, with the person's name
    resolved so a frontend doesn't have to join client-side."""
    people_by_id = {p.id: p for p in list_people_in_family(family.id)}
    out = []
    for e in list_entities_in_family(family.id):
        for r in list_roles_for_entity(e.id):
            person = people_by_id.get(r.person_id)
            person_name = person.display_name if person else f"Person {r.person_id}"
            out.append(RoleOut(
                id=r.id,
                person_id=r.person_id,
                person_name=person_name,
                entity_id=r.entity_id,
                role_type=r.role_type,
                interest_percentage=getattr(r, "interest_percentage", None),
                is_active=getattr(r, "is_active", True),
                start_date=getattr(r, "start_date", None),
                end_date=getattr(r, "end_date", None),
                notes=getattr(r, "notes", None),
            ))
    return out


@app.get("/families/{family_id}/relationships", response_model=list[RelationshipOut])
def list_family_relationships(family=Depends(get_owned_family)) -> list[RelationshipOut]:
    people_by_id = {p.id: p for p in list_people_in_family(family.id)}

    def name_for(pid):
        p = people_by_id.get(pid)
        return p.display_name if p else f"Person {pid}"

    out = []
    for rel in list_relationships_in_family(family.id):
        out.append(RelationshipOut(
            id=rel.id,
            person_a_id=rel.person_a_id,
            person_a_name=name_for(rel.person_a_id),
            person_b_id=rel.person_b_id,
            person_b_name=name_for(rel.person_b_id),
            relationship_type=rel.relationship_type,
            start_date=getattr(rel, "start_date", None),
            notes=getattr(rel, "notes", None),
        ))
    return out


# ─────────────────────────────────────────────────────────────────────
# Serialization helpers (shared by read + write responses)
# ─────────────────────────────────────────────────────────────────────

def _person_to_out(p) -> PersonOut:
    return PersonOut(
        id=p.id,
        display_name=p.display_name,
        full_name=p.full_name,
        first_name=p.first_name,
        last_name=p.last_name,
        middle_name=getattr(p, "middle_name", None),
        preferred_name=getattr(p, "preferred_name", None),
        dob=getattr(p, "dob", None),
        email=getattr(p, "email", None),
        phone=getattr(p, "phone", None),
        is_deceased=getattr(p, "is_deceased", False),
        date_of_death=getattr(p, "date_of_death", None),
        has_ssn=getattr(p, "ssn_encrypted", None) is not None,
    )


def _entity_to_out(e) -> EntityOut:
    return EntityOut(
        id=e.id,
        name=e.name,
        entity_type=e.entity_type,
        sub_type=getattr(e, "sub_type", None),
        jurisdiction=getattr(e, "jurisdiction", None),
        formation_date=getattr(e, "formation_date", None),
        termination_date=getattr(e, "termination_date", None),
        has_tax_id=getattr(e, "tax_id_encrypted", None) is not None,
    )


# ─────────────────────────────────────────────────────────────────────
# Sub-resource ownership checks (the second gate on writes by ID)
#
# get_owned_family already proved the advisor owns {family_id}. These
# additionally prove the person/entity/role actually belongs to THAT
# family — so Adam can't mutate Jake's records by guessing an ID, even
# while passing one of his own family ids in the path.
# ─────────────────────────────────────────────────────────────────────

def _person_in_family_or_404(person_id: int, family):
    p = get_person(person_id)
    if p is None or p.family_id != family.id:
        raise HTTPException(status_code=404, detail="Person not found.")
    return p


def _entity_in_family_or_404(entity_id: int, family):
    e = get_entity(entity_id)
    if e is None or e.family_id != family.id:
        raise HTTPException(status_code=404, detail="Entity not found.")
    return e


# ─────────────────────────────────────────────────────────────────────
# People — writes
# ─────────────────────────────────────────────────────────────────────

@app.post("/families/{family_id}/people", response_model=PersonOut, status_code=201)
def create_family_person(body: PersonCreate, family=Depends(get_owned_family)) -> PersonOut:
    p = create_person(
        family_id=family.id,
        first_name=body.first_name,
        last_name=body.last_name,
        middle_name=body.middle_name,
        preferred_name=body.preferred_name,
        dob=body.dob,
        email=body.email,
        phone=body.phone,
        ssn=body.ssn,
        is_deceased=body.is_deceased,
        date_of_death=body.date_of_death,
        notes=body.notes,
    )
    return _person_to_out(p)


@app.patch("/families/{family_id}/people/{person_id}", response_model=PersonOut)
def update_family_person(
    person_id: int, body: PersonUpdate, family=Depends(get_owned_family)
) -> PersonOut:
    _person_in_family_or_404(person_id, family)
    # Only pass through fields the client actually set.
    changes = body.model_dump(exclude_unset=True)
    updated = update_person(person_id, **changes)
    if updated is None:
        raise HTTPException(status_code=404, detail="Person not found.")
    return _person_to_out(updated)


@app.delete("/families/{family_id}/people/{person_id}", status_code=204)
def delete_family_person(person_id: int, family=Depends(get_owned_family)):
    _person_in_family_or_404(person_id, family)
    delete_person(person_id)
    return None


# ─────────────────────────────────────────────────────────────────────
# Entities — writes
# ─────────────────────────────────────────────────────────────────────

@app.post("/families/{family_id}/entities", response_model=EntityOut, status_code=201)
def create_family_entity(body: EntityCreate, family=Depends(get_owned_family)) -> EntityOut:
    e = create_entity(
        family_id=family.id,
        name=body.name,
        entity_type=body.entity_type,
        sub_type=body.sub_type,
        jurisdiction=body.jurisdiction,
        formation_date=body.formation_date,
        termination_date=body.termination_date,
        tax_id=body.tax_id,
        notes=body.notes,
    )
    return _entity_to_out(e)


@app.patch("/families/{family_id}/entities/{entity_id}", response_model=EntityOut)
def update_family_entity(
    entity_id: int, body: EntityUpdate, family=Depends(get_owned_family)
) -> EntityOut:
    _entity_in_family_or_404(entity_id, family)
    changes = body.model_dump(exclude_unset=True)
    updated = update_entity(entity_id, **changes)
    if updated is None:
        raise HTTPException(status_code=404, detail="Entity not found.")
    return _entity_to_out(updated)


@app.delete("/families/{family_id}/entities/{entity_id}", status_code=204)
def delete_family_entity(entity_id: int, family=Depends(get_owned_family)):
    _entity_in_family_or_404(entity_id, family)
    delete_entity(entity_id)
    return None


# ─────────────────────────────────────────────────────────────────────
# Roles — writes
#
# A role links a person to an entity. Both must belong to THIS family —
# otherwise an advisor could attach one of their people to another
# family's entity (or vice versa). We check both ends.
# ─────────────────────────────────────────────────────────────────────

@app.post("/families/{family_id}/roles", response_model=RoleOut, status_code=201)
def create_family_role(body: RoleCreate, family=Depends(get_owned_family)) -> RoleOut:
    person = _person_in_family_or_404(body.person_id, family)
    _entity_in_family_or_404(body.entity_id, family)
    r = create_role(
        person_id=body.person_id,
        entity_id=body.entity_id,
        role_type=body.role_type,
        start_date=body.start_date,
        end_date=body.end_date,
        is_active=body.is_active,
        interest_percentage=body.interest_percentage,
        notes=body.notes,
    )
    return RoleOut(
        id=r.id,
        person_id=r.person_id,
        person_name=person.display_name,
        entity_id=r.entity_id,
        role_type=r.role_type,
        interest_percentage=getattr(r, "interest_percentage", None),
        is_active=getattr(r, "is_active", True),
        start_date=getattr(r, "start_date", None),
        end_date=getattr(r, "end_date", None),
        notes=getattr(r, "notes", None),
    )


@app.delete("/families/{family_id}/roles/{role_id}", status_code=204)
def delete_family_role(role_id: int, family=Depends(get_owned_family)):
    role = get_role(role_id)
    if role is None:
        raise HTTPException(status_code=404, detail="Role not found.")
    # The role's entity must belong to this owned family.
    _entity_in_family_or_404(role.entity_id, family)
    delete_role(role_id)
    return None


# ─────────────────────────────────────────────────────────────────────
# Documents — list (read). Gated by family ownership.
#
# Internal fields (file_path on disk, file_hash) are never serialized —
# DocumentOut omits them entirely.
# ─────────────────────────────────────────────────────────────────────

def _document_to_out(d) -> DocumentOut:
    return DocumentOut(
        id=d.id,
        original_filename=d.original_filename,
        category=getattr(d, "category", None),
        category_label=category_label(getattr(d, "category", None)),
        doc_type=d.doc_type,
        doc_year=getattr(d, "doc_year", None),
        file_size_bytes=getattr(d, "file_size_bytes", None),
        mime_type=getattr(d, "mime_type", None),
        notes=getattr(d, "notes", None),
        person_id=getattr(d, "person_id", None),
        entity_id=getattr(d, "entity_id", None),
        indexed_in_vectorstore=bool(getattr(d, "indexed_in_vectorstore", False)),
        extraction_status=getattr(d, "extraction_status", None),
        uploaded_at=getattr(d, "uploaded_at", None),
    )


@app.get("/families/{family_id}/documents", response_model=list[DocumentOut])
def list_family_documents(
    family=Depends(get_owned_family),
    category: str | None = None,
    include_archived: bool = False,
) -> list[DocumentOut]:
    """List this family's documents. Optional ?category= filter and
    ?include_archived=true (archived are hidden by default, matching the
    repository default)."""
    docs = list_documents_for_family(
        family.id, category=category, include_archived=include_archived
    )
    return [_document_to_out(d) for d in docs]


# ─────────────────────────────────────────────────────────────────────
# Documents — upload (write). Calls into ai_core for indexing.
#
# Disk-path helpers are replicated here (not imported from the Streamlit
# documents.py, which pulls in `streamlit`). They're pure path logic.
# ─────────────────────────────────────────────────────────────────────

import mimetypes
from datetime import datetime


def _family_docs_dir(advisor_username: str, family_id: int) -> Path:
    root = Path("data/advisors") / advisor_username / "families" / str(family_id) / "docs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _safe_filename(original: str) -> str:
    name = original
    for ch in [" ", "/", "\\", ":", "*", "?", '"', "<", ">", "|"]:
        name = name.replace(ch, "_")
    return name


@app.post("/families/{family_id}/documents", response_model=DocumentOut, status_code=201)
async def upload_family_document(
    family=Depends(get_owned_family),
    user: AuthedUser = Depends(require_advisor),
    file: UploadFile = File(...),
    category: str = Form(...),
    doc_type: str = Form(...),
    person_id: int | None = Form(None),
    entity_id: int | None = Form(None),
    doc_year: int | None = Form(None),
    notes: str | None = Form(None),
) -> DocumentOut:
    """Upload a PDF to a family.

    Security:
      - family ownership enforced by get_owned_family
      - if person_id / entity_id are given, each must belong to THIS family
      - category must be a known category key
    Behavior:
      - dedup by file hash (409 if the same file is already on file)
      - saves to disk, creates the DB row, then reindexes the family
      - if indexing fails (e.g. Ollama down), the document is still saved;
        indexed_in_vectorstore stays false and the response reflects that
    """
    # Validate category
    if category not in DOCUMENT_CATEGORIES:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown category '{category}'.",
        )

    # Validate optional links belong to this family (same gate as roles)
    if person_id is not None:
        _person_in_family_or_404(person_id, family)
    if entity_id is not None:
        _entity_in_family_or_404(entity_id, family)

    # Only PDFs (matches the current uploader)
    fname = file.filename or "upload.pdf"
    if not fname.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=422,
            detail="Only PDF files are accepted.",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(
            status_code=422,
            detail="Uploaded file is empty.",
        )

    file_hash = compute_file_hash(file_bytes)

    # Dedup — same exact file already in this family's library?
    existing = find_document_by_hash(family.id, file_hash)
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"This file is already in the library "
                f"(as '{existing.original_filename}')."
            ),
        )

    # Save to disk — advisor username comes from the verified token
    docs_dir = _family_docs_dir(user.username, family.id)
    final_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_safe_filename(fname)}"
    file_path = docs_dir / final_name
    try:
        file_path.write_bytes(file_bytes)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not save file: {e}",
        )

    # Create the DB row
    try:
        from db.repositories import ensure_db_user as _ensure
        advisor_db_id = _resolve_advisor_db_id(user)
        doc = create_document(
            family_id=family.id,
            file_path=str(file_path),
            file_hash=file_hash,
            original_filename=fname,
            file_size_bytes=len(file_bytes),
            mime_type=mimetypes.guess_type(fname)[0] or "application/pdf",
            category=category,
            doc_type=doc_type,
            person_id=person_id,
            entity_id=entity_id,
            doc_year=doc_year,
            notes=notes,
            uploaded_by_user_id=advisor_db_id,
        )
    except Exception as e:
        # Roll back the on-disk file so we don't leave an orphan
        try:
            file_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not save document record: {e}",
        )

    # Reindex the family so chat can see the new doc. Best-effort: if the
    # AI stack is unavailable, the document is still saved.
    # NOTE: this is synchronous and can be slow on large libraries; moving
    # it to a background task is a future optimization.
    try:
        from ai_core.family_qa import reindex_family
        reindex_family(family.id, verbose=False)
        # reflect the fresh index state
        doc = _refresh_document(doc.id) or doc
    except Exception:
        # Indexing failed — doc remains saved, indexed flag stays false.
        pass

    return _document_to_out(doc)


def _refresh_document(document_id: int):
    try:
        from db.repositories import get_document
        return get_document(document_id)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────
# Chat — ask Angel about a family's documents. Calls into ai_core.
# ─────────────────────────────────────────────────────────────────────

@app.post("/families/{family_id}/chat", response_model=ChatResponse)
def chat_with_family(
    body: ChatRequest,
    family=Depends(get_owned_family),
    user: AuthedUser = Depends(require_advisor),
) -> ChatResponse:
    """Ask a question scoped to this family's documents.

    Advisors query their own families authorized=True with an AUTHORIZED
    disclosure mode (matching the Streamlit app's advisor behavior).
    """
    question = (body.question or "").strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question must not be empty.")

    try:
        from ai_core.family_qa import FamilyQASystem
        system = FamilyQASystem(family_id=family.id, verbose=False)
        system.index_documents(force_rebuild=False)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"AI system unavailable: {e}",
        )

    has_index = getattr(system, "vector_store", None) is not None
    if not has_index:
        return ChatResponse(
            answer=(
                "No documents have been indexed for this family yet. "
                "Upload documents first, then ask again."
            ),
            documents_indexed=False,
        )

    try:
        from ai_core.privacy_policy import DisclosureMode
        answer = system.ask(
            question,
            disclosure_mode=DisclosureMode.AUTHORIZED,
            authorized=True,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not answer the question: {e}",
        )

    return ChatResponse(answer=str(answer), documents_indexed=True)
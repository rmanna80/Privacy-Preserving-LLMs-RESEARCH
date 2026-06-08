"""
ui/components/documents.py — Documents page for an advisor (sub-phase 4a).

Per-family document upload, categorization, and recent-uploads list.

Sub-phase 4a scope (intentionally narrow):
  - Upload one PDF at a time
  - Pick category (5 options) + doc type
  - Optional link to a person or entity in the family
  - File hash dedup (catches re-uploads of the same file)
  - File saved to disk under data/advisors/<advisor>/families/<family_id>/
  - Document row created in wealth.db
  - Show a simple list of uploaded docs grouped by category

NOT in 4a (coming in 4b/4c/4d):
  - Chroma reindex on upload (so chat won't see new uploads yet)
  - PDF viewer
  - Edit / archive UI
  - Subfolders
  - Reminder cadences
  - Client portal documents view

Why we save to a per-family folder rather than per-client:
  Family is the right scope. The existing per-client folders from the
  legacy RAG pipeline still work for the old chat path; the new path
  uses family folders, which the upcoming family-scoped chat (sub-phase
  4c) will use too.
"""

from __future__ import annotations

import mimetypes
from datetime import datetime
from pathlib import Path
from typing import Optional

import streamlit as st

from db.repositories import (
    list_people_in_family,
    list_entities_in_family,
    get_family,
    ensure_db_user,
    # Document layer (newly added)
    DOCUMENT_CATEGORIES,
    CATEGORY_LABELS,
    DOC_TYPES_BY_CATEGORY,
    compute_file_hash,
    category_label,
    find_document_by_hash,
    create_document,
    list_documents_by_category,
    list_documents_for_family,
    archive_document,
)


# ─────────────────────────────────────────────────────────────────────
# Path helpers
# ─────────────────────────────────────────────────────────────────────

def _family_docs_dir(advisor_username: str, family_id: int) -> Path:
    """Where this family's documents live on disk."""
    root = Path("data/advisors") / advisor_username / "families" / str(family_id) / "docs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _safe_filename(original: str) -> str:
    """Make a filename safe for the filesystem.

    We don't want spaces, slashes, colons, etc. Replace anything sketchy
    with underscores. Preserve the extension.
    """
    name = original
    for ch in [" ", "/", "\\", ":", "*", "?", "\"", "<", ">", "|"]:
        name = name.replace(ch, "_")
    return name


# ─────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────

def render_documents_page(family_id: int, user) -> None:
    family = get_family(family_id)
    if family is None:
        st.error("Family not found.")
        return

    advisor_db_id = ensure_db_user(user)

    st.markdown("### Documents")
    st.caption(
        "Every document tied to this family — categorized, hashed, and "
        "ready for the AI to query (chat integration arrives in the "
        "next phase)."
    )

    # ---- Upload form ----
    with st.expander("➕ Upload New Document", expanded=True):
        _render_upload_form(family_id, user, advisor_db_id)

    st.markdown("---")

    # ---- Recently uploaded — flat list ----
    st.markdown("#### Recent Uploads")
    recent = list_documents_for_family(family_id)[:5]
    if recent:
        for d in recent:
            _render_doc_row_compact(d)
    else:
        st.caption("_No documents uploaded yet._")

    st.markdown("---")

    # ---- Grouped by category ----
    st.markdown("#### By Category")
    grouped = list_documents_by_category(family_id)

    for cat_key in DOCUMENT_CATEGORIES:
        docs = grouped.get(cat_key, [])
        with st.expander(
            f"{CATEGORY_LABELS[cat_key]}  ·  {len(docs)} document{'s' if len(docs) != 1 else ''}",
            expanded=False,
        ):
            if not docs:
                st.caption("_None yet._")
            else:
                for d in docs:
                    _render_doc_row_detailed(d, family_id)


# ─────────────────────────────────────────────────────────────────────
# Upload form
# ─────────────────────────────────────────────────────────────────────

def _render_upload_form(family_id: int, user, advisor_db_id: int) -> None:
    people = list_people_in_family(family_id)
    entities = list_entities_in_family(family_id)

    with st.form("upload_doc_form", clear_on_submit=True):
        # Category — drives the doc_type options below
        col_cat, col_type = st.columns(2)
        with col_cat:
            cat_labels = [CATEGORY_LABELS[c] for c in DOCUMENT_CATEGORIES]
            picked_cat_label = st.selectbox(
                "Category *",
                cat_labels,
                key="upload_category",
            )
            category = DOCUMENT_CATEGORIES[cat_labels.index(picked_cat_label)]

        with col_type:
            type_options = DOC_TYPES_BY_CATEGORY.get(category, [("other", "Other")])
            type_labels = [t[1] for t in type_options]
            picked_type_label = st.selectbox(
                "Document Type *",
                type_labels,
                key="upload_doc_type",
            )
            doc_type = type_options[type_labels.index(picked_type_label)][0]

        # Optional links
        col_person, col_entity = st.columns(2)
        with col_person:
            person_options = [("— None —", None)] + [
                (p.display_name, p.id) for p in people
            ]
            person_labels = [o[0] for o in person_options]
            picked_person = st.selectbox(
                "Link to Person (optional)",
                person_labels,
                index=0,
            )
            person_id = person_options[person_labels.index(picked_person)][1]

        with col_entity:
            entity_options = [("— None —", None)] + [
                (e.name, e.id) for e in entities
            ]
            entity_labels = [o[0] for o in entity_options]
            picked_entity = st.selectbox(
                "Link to Entity (optional)",
                entity_labels,
                index=0,
            )
            entity_id = entity_options[entity_labels.index(picked_entity)][1]

        # Optional doc year + notes
        col_year, _ = st.columns(2)
        with col_year:
            doc_year = st.number_input(
                "Document Year (optional)",
                min_value=1900,
                max_value=2100,
                value=datetime.now().year,
                step=1,
            )

        notes = st.text_area(
            "Notes (optional)",
            placeholder="Context about this document — what it covers, why it matters…",
        )

        # The file itself
        uploaded = st.file_uploader(
            "Document file *",
            type=["pdf"],
            accept_multiple_files=False,
        )

        col_submit, col_cancel = st.columns(2)
        with col_submit:
            submitted = st.form_submit_button(
                "Upload",
                type="primary",
                use_container_width=True,
            )
        with col_cancel:
            cancelled = st.form_submit_button("Clear", use_container_width=True)

        if cancelled:
            st.rerun()

        if submitted:
            if uploaded is None:
                st.error("Please choose a PDF file to upload.")
                return

            _process_upload(
                family_id=family_id,
                user=user,
                advisor_db_id=advisor_db_id,
                uploaded=uploaded,
                category=category,
                doc_type=doc_type,
                person_id=person_id,
                entity_id=entity_id,
                doc_year=int(doc_year) if doc_year else None,
                notes=notes.strip() or None,
            )


def _process_upload(
    family_id: int,
    user,
    advisor_db_id: int,
    uploaded,
    category: str,
    doc_type: str,
    person_id: Optional[int],
    entity_id: Optional[int],
    doc_year: Optional[int],
    notes: Optional[str],
) -> None:
    """Save the uploaded file to disk + create the DB row, with dedup."""
    file_bytes = uploaded.getvalue()
    file_hash = compute_file_hash(file_bytes)

    # Dedup — has this exact file already been uploaded for this family?
    existing = find_document_by_hash(family_id, file_hash)
    if existing is not None:
        st.warning(
            f"⚠️ This file is already in the library "
            f"(uploaded {existing.uploaded_at.strftime('%Y-%m-%d %H:%M')}, "
            f"as '{existing.original_filename}'). Skipping duplicate."
        )
        return

    # Save to disk
    docs_dir = _family_docs_dir(user.username, family_id)
    safe_name = _safe_filename(uploaded.name)
    # Prefix with a timestamp so two docs with the same name don't collide.
    final_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_name}"
    file_path = docs_dir / final_name

    try:
        file_path.write_bytes(file_bytes)
    except Exception as e:
        st.error(f"Could not save file to disk: {e}")
        return

    # Create DB row
    try:
        doc = create_document(
            family_id=family_id,
            file_path=str(file_path),
            file_hash=file_hash,
            original_filename=uploaded.name,
            file_size_bytes=len(file_bytes),
            mime_type=mimetypes.guess_type(uploaded.name)[0] or "application/pdf",
            category=category,
            doc_type=doc_type,
            person_id=person_id,
            entity_id=entity_id,
            doc_year=doc_year,
            notes=notes,
            uploaded_by_user_id=advisor_db_id,
        )
    except Exception as e:
        # If the DB write fails, remove the file we just saved so we don't
        # have an orphan on disk.
        try:
            file_path.unlink(missing_ok=True)
        except Exception:
            pass
        st.error(f"Could not save document record: {e}")
        return

    st.success(
        f"✅ Uploaded **{uploaded.name}** to "
        f"_{CATEGORY_LABELS[category]}_."
    )

    # Trigger reindex so the chat immediately picks up this new doc
    with st.spinner("Indexing for AI chat…"):
        try:
            from ai_core.family_qa import reindex_family
            reindex_family(family_id, verbose=False)
            # Invalidate the cached QA system so chat reloads with the new index
            if st.session_state.get("qa_owner") == f"family::{family_id}":
                st.session_state.qa_system = None
                st.session_state.qa_owner = None
            st.caption(
                f"🤖 Indexed for AI — ask Angel about this document in Chat History."
            )
        except Exception as e:
            st.warning(
                f"Document saved, but AI indexing failed: {e}. "
                f"Chat may not see this document until reindex succeeds."
            )

    st.rerun()


# ─────────────────────────────────────────────────────────────────────
# Document rows
# ─────────────────────────────────────────────────────────────────────

def _render_doc_row_compact(doc) -> None:
    """Single-line row used in the Recent Uploads section."""
    with st.container(border=True):
        c1, c2, c3 = st.columns([4, 3, 2])
        with c1:
            st.markdown(f"**{doc.original_filename}**")
            st.caption(f"{category_label(doc.category)} · {doc.doc_type}")
        with c2:
            uploaded_at = doc.uploaded_at.strftime("%Y-%m-%d %H:%M")
            st.caption(f"Uploaded {uploaded_at}")
            if doc.doc_year:
                st.caption(f"Doc year: {doc.doc_year}")
        with c3:
            size_kb = (doc.file_size_bytes or 0) / 1024
            st.caption(f"{size_kb:.0f} KB")


def _render_doc_row_detailed(doc, family_id: int) -> None:
    """Per-category row, slightly richer with archive control."""
    with st.container(border=True):
        c1, c2, c3 = st.columns([5, 3, 2])
        with c1:
            st.markdown(f"**{doc.original_filename}**")
            sub_bits = [doc.doc_type]
            if doc.doc_year:
                sub_bits.append(f"Year {doc.doc_year}")
            if doc.notes:
                sub_bits.append(doc.notes[:60] + ("…" if len(doc.notes) > 60 else ""))
            st.caption(" · ".join(sub_bits))
        with c2:
            uploaded_at = doc.uploaded_at.strftime("%Y-%m-%d %H:%M")
            st.caption(f"📅 {uploaded_at}")
            if doc.person_id:
                from db.repositories import get_person
                p = get_person(doc.person_id)
                if p:
                    st.caption(f"👥 {p.display_name}")
            if doc.entity_id:
                from db.repositories import get_entity
                e = get_entity(doc.entity_id)
                if e:
                    st.caption(f"🏛️ {e.name}")
        with c3:
            indexed_emoji = "✅" if doc.indexed_in_vectorstore else "⏳"
            st.caption(f"Indexed: {indexed_emoji}")
            if st.button("Archive", key=f"archive_doc_{doc.id}", use_container_width=True):
                archive_document(doc.id)
                st.rerun()

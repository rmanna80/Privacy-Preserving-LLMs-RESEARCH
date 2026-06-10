"""
ui/components/extractions.py — per-document extraction panel (4d.1 + 4d.2).

Shows extracted facts for one document, with:
  - ✨ Run AI Extraction button (calls ai_core.extractor.run_extraction)
  - Per-fact verify ✓ (with optional inline correction) / reject ✗
  - Manual "Add Fact" form using the canonical field schema
  - Verified facts shown with a green check + timestamp

Flow: the Documents page sets st.session_state.extraction_doc_id when
the advisor clicks "Extractions" on a document row, then calls
render_extraction_panel(doc_id, user) at the bottom of the page.
"""

from __future__ import annotations

from typing import Optional

import streamlit as st

from ai_core.extraction_schema import fields_for_doc_type, field_def
from db.repositories import (
    get_document,
    ensure_db_user,
    list_extractions_for_document,
    create_extraction,
    verify_extraction,
    reject_extraction,
    extraction_plain_value,
)


def render_extraction_panel(document_id: int, user) -> None:
    doc = get_document(document_id)
    if doc is None:
        st.error("Document not found.")
        return

    advisor_db_id = ensure_db_user(user)

    st.markdown(f"#### ✨ Extractions — {doc.original_filename}")
    st.caption(
        f"Structured facts from this {doc.doc_type.replace('_', ' ')}. "
        "AI proposes, you verify. Only verified facts feed the rest of Angel."
    )

    # ── Run AI extraction ────────────────────────────────────────────
    col_run, col_close = st.columns([3, 1])
    with col_run:
        if st.button(
            "✨ Run AI Extraction",
            key=f"run_extract_{doc.id}",
            type="primary",
        ):
            with st.spinner("Reading document and extracting facts (local AI)…"):
                from ai_core.extractor import run_extraction
                result = run_extraction(doc.id)
            if result["proposed"]:
                st.success(
                    f"Proposed {result['proposed']} new fact(s). Review below."
                )
            if result["skipped"]:
                st.caption(
                    f"{result['skipped']} field(s) already had extractions — skipped."
                )
            for err in result["errors"]:
                st.warning(err)
            st.rerun()
    with col_close:
        if st.button(
            "Close",
            key=f"close_extract_{doc.id}",
            use_container_width=True,
        ):
            st.session_state.extraction_doc_id = None
            st.rerun()

    # ── Existing extractions ─────────────────────────────────────────
    extractions = list_extractions_for_document(doc.id)
    proposed = [e for e in extractions if e.verified_at is None]
    verified = [e for e in extractions if e.verified_at is not None]

    if proposed:
        st.markdown("**Awaiting review**")
        for e in proposed:
            _render_proposed_row(e, doc, advisor_db_id)

    if verified:
        st.markdown("**Verified**")
        for e in verified:
            _render_verified_row(e, doc)

    if not extractions:
        st.info(
            "No extractions yet. Run AI Extraction above, or add a fact "
            "manually below."
        )

    # ── Manual add ───────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("➕ Add Fact Manually", expanded=False):
        _render_manual_form(doc, advisor_db_id)


def _render_proposed_row(e, doc, advisor_db_id: int) -> None:
    fdef = field_def(doc.doc_type, e.field_key) or {}
    label = fdef.get("label", e.field_key)
    value = extraction_plain_value(e)

    with st.container(border=True):
        c1, c2 = st.columns([5, 3])
        with c1:
            pii_tag = " 🔒" if e.is_pii else ""
            st.markdown(f"**{label}**{pii_tag}")
            corrected = st.text_input(
                "Value (edit to correct before verifying)",
                value=value,
                key=f"extract_val_{e.id}",
                label_visibility="collapsed",
            )
            meta = [f"AI confidence {e.confidence:.0%}", f"via {e.extracted_by}"]
            if e.page_number:
                meta.append(f"page {e.page_number}")
            st.caption(" · ".join(meta))
            if e.text_snippet:
                snippet = e.text_snippet
                st.caption(
                    f"“{snippet[:160]}…”" if len(snippet) > 160 else f"“{snippet}”"
                )
        with c2:
            if st.button(
                "✓ Verify",
                key=f"verify_{e.id}",
                use_container_width=True,
                type="primary",
            ):
                corrected_val = corrected if corrected != value else None
                verify_extraction(e.id, advisor_db_id, corrected_value=corrected_val)
                st.rerun()
            if st.button(
                "✗ Reject",
                key=f"reject_{e.id}",
                use_container_width=True,
            ):
                reject_extraction(e.id)
                st.rerun()


def _render_verified_row(e, doc) -> None:
    fdef = field_def(doc.doc_type, e.field_key) or {}
    label = fdef.get("label", e.field_key)
    value = extraction_plain_value(e)

    with st.container(border=True):
        c1, c2 = st.columns([6, 2])
        with c1:
            pii_tag = " 🔒" if e.is_pii else ""
            st.markdown(f"✅ **{label}**{pii_tag}: {value}")
            when = e.verified_at.strftime("%Y-%m-%d %H:%M") if e.verified_at else ""
            st.caption(f"Verified {when}")
        with c2:
            if st.button(
                "Remove",
                key=f"unverify_{e.id}",
                use_container_width=True,
            ):
                reject_extraction(e.id)
                st.rerun()


def _render_manual_form(doc, advisor_db_id: int) -> None:
    fields = fields_for_doc_type(doc.doc_type)
    labels = [f["label"] for f in fields]

    with st.form(f"manual_extract_{doc.id}", clear_on_submit=True):
        picked_label = st.selectbox("Field", labels)
        fdef = fields[labels.index(picked_label)]
        value = st.text_input("Value")
        page = st.number_input("Page (optional)", min_value=0, value=0, step=1)

        if st.form_submit_button("Add & Verify", type="primary"):
            if not value.strip():
                st.error("Value is required.")
            else:
                e = create_extraction(
                    document_id=doc.id,
                    field_key=fdef["key"],
                    field_value=value.strip(),
                    extraction_type=fdef["value_type"],
                    is_pii=fdef["is_pii"],
                    page_number=int(page) if page else None,
                    extracted_by="manual",
                    confidence=1.0,
                )
                verify_extraction(e.id, advisor_db_id)
                st.rerun()

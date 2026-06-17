"""
ui/components/promotion_panel.py — promote verified extractions to graph data.

Renders, for one document, the verified extractions that CAN become
structured data (roles, people, entity attributes), each with a confirm
step. The advisor picks/confirms the Person and Entity, then promotes.

This is the Model B flow: AI extracts → advisor verifies the fact →
advisor confirms the structured link → real Role/Person/Entity rows get
created, which the Family Tree / Org Ownership / Entities pages render.

Called from the extraction panel via a "Promote verified facts →" button
that sets st.session_state.promotion_doc_id.
"""

from __future__ import annotations

from typing import Optional

import streamlit as st

from ai_core.extraction_schema import field_def
from ai_core.promotion import (
    classify_field,
    is_promotable,
    split_names,
    suggest_person_match,
    split_first_last,
    promote_role,
    ensure_person,
    update_entity_attr,
    create_entity_for_family,
    mark_extraction_promoted,
    is_promoted,
    ROLE_FIELD_MAP,
    ENTITY_ATTR_MAP,
)
from db.repositories import (
    get_document,
    get_family,
    ensure_db_user,
    list_extractions_for_document,
    list_people_in_family,
    list_entities_in_family,
    extraction_plain_value,
)


def render_promotion_panel(document_id: int, user) -> None:
    doc = get_document(document_id)
    if doc is None:
        st.error("Document not found.")
        return

    family_id = doc.family_id
    advisor_db_id = ensure_db_user(user)

    st.markdown(f"#### 🔗 Promote to Family Data — {doc.original_filename}")
    st.caption(
        "Turn verified facts into real records. Each promotion creates "
        "trustee/beneficiary roles, people, or entity details that show up "
        "on the Family Tree, Org Ownership, and Entities pages."
    )

    if st.button("Close", key=f"close_promo_{doc.id}"):
        st.session_state.promotion_doc_id = None
        st.rerun()

    extractions = list_extractions_for_document(doc.id)
    verified = [e for e in extractions if e.verified_at is not None]
    promotable = [e for e in verified if is_promotable(e.field_key)]

    if not verified:
        st.info("No verified facts yet. Verify some extractions first.")
        return
    if not promotable:
        st.info(
            "None of the verified facts map to structured data. "
            "(Things like AGI or dates stay as display-only facts.)"
        )
        return

    people = list_people_in_family(family_id)
    entities = list_entities_in_family(family_id)

    for e in promotable:
        _render_promotion_row(e, doc, family_id, people, entities, advisor_db_id)


def _render_promotion_row(e, doc, family_id, people, entities, advisor_db_id):
    fdef = field_def(doc.doc_type, e.field_key) or {}
    label = fdef.get("label", e.field_key)
    value = extraction_plain_value(e)
    kind = classify_field(e.field_key)

    with st.container(border=True):
        if is_promoted(e):
            st.markdown(f"✅ **{label}**: {value}  — _promoted_")
            return

        st.markdown(f"**{label}**: {value}")
        st.caption(f"Maps to: {kind.replace('_', ' ')}")

        if kind == "role":
            _render_role_promotion(e, value, family_id, people, entities, advisor_db_id)
        elif kind == "person":
            _render_person_promotion(e, value, family_id, people, advisor_db_id)
        elif kind == "entity_attr":
            _render_entity_attr_promotion(e, value, family_id, entities, advisor_db_id)


def _render_role_promotion(e, value, family_id, people, entities, advisor_db_id):
    """Role: link a Person to an Entity with a role_type. The value may
    contain multiple names (e.g. two co-trustees)."""
    role_type = ROLE_FIELD_MAP.get(e.field_key, "beneficiary")
    names = split_names(value)

    st.caption(f"Role type: **{role_type.replace('_', ' ')}**")

    # Entity picker (which entity does this role attach to?)
    if not entities:
        st.warning(
            "No entities exist for this family yet. Create one to attach "
            "this role to (e.g. the trust this document describes)."
        )
        new_entity_name = st.text_input(
            "New entity name",
            value=_guess_entity_name(e, doc_default="Family Trust"),
            key=f"new_entity_{e.id}",
        )
        entity_choice = ("__new__", new_entity_name)
    else:
        ent_labels = [f"{en.name}" for en in entities] + ["➕ Create new entity…"]
        picked = st.selectbox("Entity", ent_labels, key=f"entity_pick_{e.id}")
        if picked == "➕ Create new entity…":
            new_entity_name = st.text_input(
                "New entity name", value="Family Trust",
                key=f"new_entity_{e.id}",
            )
            entity_choice = ("__new__", new_entity_name)
        else:
            chosen = entities[ent_labels.index(picked)]
            entity_choice = ("existing", chosen.id)

    # One person picker per name in the value
    person_choices = []
    for idx, name in enumerate(names):
        suggested_id = suggest_person_match(name, people)
        opts = [f"{p.display_name}" for p in people] + [f"➕ Create '{name}'"]
        default_idx = len(opts) - 1  # default to create
        if suggested_id is not None:
            for i, p in enumerate(people):
                if p.id == suggested_id:
                    default_idx = i
                    break
        picked = st.selectbox(
            f"Person for '{name}'",
            opts,
            index=default_idx,
            key=f"person_pick_{e.id}_{idx}",
        )
        if picked.startswith("➕ Create"):
            person_choices.append(("__new__", name))
        else:
            person_choices.append(("existing", people[opts.index(picked)].id))

    if st.button("🔗 Promote this role", key=f"promote_role_{e.id}", type="primary"):
        # Resolve entity
        if entity_choice[0] == "__new__":
            ent = create_entity_for_family(
                family_id=family_id, name=entity_choice[1] or "Family Trust"
            )
            entity_id = ent.id
        else:
            entity_id = entity_choice[1]

        # Resolve each person and create a role
        created = 0
        for choice, payload in person_choices:
            if choice == "__new__":
                first, last = split_first_last(payload)
                p = ensure_person(family_id=family_id, first_name=first, last_name=last)
                pid = p.id
            else:
                pid = payload
            promote_role(
                family_id=family_id, person_id=pid,
                entity_id=entity_id, role_type=role_type,
            )
            created += 1

        mark_extraction_promoted(e.id)
        st.success(f"Created {created} role link(s).")
        st.rerun()


def _render_person_promotion(e, value, family_id, people, advisor_db_id):
    """Person: ensure each named person exists in the family."""
    names = split_names(value)
    for idx, name in enumerate(names):
        suggested_id = suggest_person_match(name, people)
        if suggested_id is not None:
            match = next((p for p in people if p.id == suggested_id), None)
            st.caption(f"'{name}' looks like existing **{match.display_name}** — already in family.")
        else:
            st.caption(f"'{name}' is not yet in the family.")

    if st.button("👤 Add missing people", key=f"promote_person_{e.id}", type="primary"):
        added = 0
        for name in names:
            if suggest_person_match(name, people) is None:
                first, last = split_first_last(name)
                ensure_person(family_id=family_id, first_name=first, last_name=last)
                added += 1
        mark_extraction_promoted(e.id)
        st.success(f"Added {added} new person(s)." if added else "All already present.")
        st.rerun()


def _render_entity_attr_promotion(e, value, family_id, entities, advisor_db_id):
    """Entity attribute: set a field on an entity."""
    attr = ENTITY_ATTR_MAP.get(e.field_key, "name")

    if not entities:
        st.warning("No entities exist yet. Create one, then set this attribute.")
        new_name = st.text_input(
            "New entity name",
            value=value if attr == "name" else "Family Trust",
            key=f"attr_new_entity_{e.id}",
        )
        if st.button("Create entity & set", key=f"promote_attr_new_{e.id}", type="primary"):
            ent = create_entity_for_family(family_id=family_id, name=new_name)
            if attr != "name":
                update_entity_attr(entity_id=ent.id, attr=attr, value=value)
            mark_extraction_promoted(e.id)
            st.success("Entity created and attribute set.")
            st.rerun()
        return

    ent_labels = [en.name for en in entities]
    picked = st.selectbox("Apply to entity", ent_labels, key=f"attr_entity_{e.id}")
    chosen = entities[ent_labels.index(picked)]
    st.caption(f"Will set **{attr}** = {value} on {chosen.name}")

    if st.button("🏛️ Set attribute", key=f"promote_attr_{e.id}", type="primary"):
        update_entity_attr(entity_id=chosen.id, attr=attr, value=value)
        mark_extraction_promoted(e.id)
        st.success(f"Updated {chosen.name}.")
        st.rerun()


def _guess_entity_name(e, doc_default: str) -> str:
    """If a trust_name was extracted on the same doc, suggest it."""
    return doc_default

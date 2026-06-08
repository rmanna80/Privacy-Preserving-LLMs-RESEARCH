"""
Family / People / Entities / Roles management.

Originally rendered a Families tab with sub-tabs (Step 2 and 3a). Now
its sub-renderers are imported by advisor_shell.py and used as standalone
pages in the sidebar workspace.

The render_family_manager() function below is kept for backward
compatibility with the original tab-based layout.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

import streamlit as st

from db.models import Person, Family, Entity, Role
from db.repositories import (
    ensure_db_user,
    list_families_for_advisor,
    get_family,
    create_family,
    update_family,
    delete_family,
    list_people_in_family,
    get_person,
    create_person,
    update_person,
    delete_person,
    list_relationships_in_family,
    create_relationship,
    delete_relationship,
    list_entities_in_family,
    get_entity,
    create_entity,
    update_entity,
    delete_entity,
    list_roles_for_entity,
    create_role,
    delete_role,
)


# ---------------------------------------------------------------------------
# Constants — dropdown choices
# ---------------------------------------------------------------------------

RELATIONSHIP_TYPES = [
    "spouse",
    "parent_of",
    "sibling",
    "ex_spouse",
    "guardian_of",
]

ENTITY_TYPES = [
    "trust",
    "llc",
    "partnership",
    "corporation",
    "foundation",
    "donor_advised_fund",
    "holding_company",
    "other",
]

ENTITY_SUBTYPE_HINTS = {
    "trust": "revocable, irrevocable, GRAT, ILIT, CRAT, CRUT, QPRT, dynasty, special_needs",
    "llc": "single_member, multi_member, series",
    "partnership": "general, limited, LP, LLP",
    "corporation": "C_corp, S_corp, B_corp",
    "foundation": "private_foundation, family_foundation",
}

ROLE_TYPES = [
    "grantor",
    "settlor",
    "trustee",
    "co_trustee",
    "successor_trustee",
    "trust_protector",
    "beneficiary",
    "contingent_beneficiary",
    "remainder_beneficiary",
    "member",
    "managing_member",
    "manager",
    "shareholder",
    "officer",
    "director",
    "principal",
    "power_of_attorney",
    "executor",
    "guardian",
]


# ===========================================================================
# Public entry point — kept for backward compatibility
# ===========================================================================

def render_family_manager(user) -> None:
    """Original tab-based entry point. The new advisor_shell.py uses the
    individual _render_*_tab functions below as standalone pages instead."""
    st.markdown("# Families")
    st.caption(
        "The structured backbone. Every person, role, trust, and "
        "document hangs off a family."
    )

    advisor_db_id = ensure_db_user(user)
    families = list_families_for_advisor(advisor_db_id)

    _render_family_picker(advisor_db_id, families)

    family_id = st.session_state.get("selected_family_id")
    if family_id is None:
        return

    family = get_family(family_id)
    if family is None:
        st.session_state.selected_family_id = None
        st.rerun()

    st.markdown("---")
    tab_people, tab_rels, tab_entities, tab_roles, tab_settings = st.tabs(
        ["👥 People", "🔗 Relationships", "🏛️ Entities", "👤 Roles", "⚙️ Settings"]
    )
    with tab_people:
        _render_people_tab(family_id)
    with tab_rels:
        _render_relationships_tab(family_id)
    with tab_entities:
        _render_entities_tab(family_id)
    with tab_roles:
        _render_roles_tab(family_id)
    with tab_settings:
        _render_settings_tab(family)


# ===========================================================================
# Family picker / creator
# ===========================================================================

def _render_family_picker(advisor_db_id: int, families: list[Family]) -> None:
    col_pick, col_new = st.columns([4, 1])

    with col_pick:
        if families:
            labels_to_id = {f.name: f.id for f in families}
            current_id = st.session_state.get("selected_family_id")
            if current_id is None or current_id not in labels_to_id.values():
                current_id = families[0].id
                st.session_state.selected_family_id = current_id

            current_label = next(
                lbl for lbl, fid in labels_to_id.items() if fid == current_id
            )
            label_list = list(labels_to_id.keys())
            selected_label = st.selectbox(
                "Select a family",
                label_list,
                index=label_list.index(current_label),
            )
            st.session_state.selected_family_id = labels_to_id[selected_label]
        else:
            st.info("No families yet. Click **+ New Family** to create your first one.")

    with col_new:
        st.write("")
        st.write("")
        if st.button("➕ New Family", use_container_width=True):
            st.session_state.show_new_family_form = True

    if st.session_state.get("show_new_family_form", False):
        with st.expander("Create New Family", expanded=True):
            with st.form("new_family_form", clear_on_submit=True):
                name = st.text_input("Family Name *", placeholder="e.g. The Smith Family")
                notes = st.text_area("Notes (optional)")
                col_a, col_b = st.columns(2)
                with col_a:
                    submitted = st.form_submit_button(
                        "Create", use_container_width=True, type="primary"
                    )
                with col_b:
                    cancelled = st.form_submit_button("Cancel", use_container_width=True)

                if submitted:
                    if not name.strip():
                        st.error("Family name is required.")
                    else:
                        family = create_family(
                            name=name.strip(),
                            advisor_user_id=advisor_db_id,
                            notes=notes.strip() or None,
                        )
                        st.session_state.selected_family_id = family.id
                        st.session_state.show_new_family_form = False
                        st.rerun()
                elif cancelled:
                    st.session_state.show_new_family_form = False
                    st.rerun()


# ===========================================================================
# People tab
# ===========================================================================

def _render_people_tab(family_id: int) -> None:
    people = list_people_in_family(family_id)

    st.subheader("People")

    if people:
        for person in people:
            _render_person_card(person)
    else:
        st.info("No people yet. Add the first family member below.")

    pending_delete_id = st.session_state.get("confirm_delete_person_id")
    if pending_delete_id is not None:
        target = get_person(pending_delete_id)
        if target is None:
            st.session_state.confirm_delete_person_id = None
        else:
            with st.expander(f"⚠️ Confirm deletion of {target.full_name}?", expanded=True):
                st.warning(
                    "This deletes the person, plus all their roles and "
                    "relationships. Documents linked to this person are "
                    "NOT deleted (they stay on the family)."
                )
                col_y, col_n = st.columns(2)
                with col_y:
                    if st.button(
                        "Yes, delete",
                        key=f"confirm_yes_{pending_delete_id}",
                        type="primary",
                        use_container_width=True,
                    ):
                        delete_person(pending_delete_id)
                        st.session_state.confirm_delete_person_id = None
                        st.session_state.editing_person_id = None
                        st.rerun()
                with col_n:
                    if st.button(
                        "Cancel",
                        key=f"confirm_no_{pending_delete_id}",
                        use_container_width=True,
                    ):
                        st.session_state.confirm_delete_person_id = None
                        st.rerun()

    st.markdown("---")
    editing_id = st.session_state.get("editing_person_id")
    if editing_id is not None:
        editing_person = get_person(editing_id)
        if editing_person is None:
            st.session_state.editing_person_id = None
            st.rerun()
        st.subheader(f"✏️ Edit: {editing_person.full_name}")
        _render_person_form(family_id, editing_person)
    else:
        with st.expander("➕ Add Person", expanded=False):
            _render_person_form(family_id, None)


def _render_person_card(person: Person) -> None:
    with st.container(border=True):
        col_info, col_ssn, col_actions = st.columns([3, 2, 2])

        with col_info:
            name = person.display_name
            if person.is_deceased:
                name += " ⚰️"
            st.markdown(f"**{name}**")
            bits = []
            if person.dob:
                bits.append(f"DOB: {person.dob.isoformat()}")
            if person.email:
                bits.append(person.email)
            if person.phone:
                bits.append(person.phone)
            if bits:
                st.caption(" · ".join(bits))

        with col_ssn:
            show_key = f"show_ssn_{person.id}"
            show = st.toggle("Show SSN", key=show_key, value=False)
            if person.ssn_encrypted is None:
                st.code("— no SSN on file —")
            elif show:
                st.code(person.ssn or "—")
            else:
                st.code(_mask_id(person.ssn))

        with col_actions:
            if st.button("Edit", key=f"edit_{person.id}", use_container_width=True):
                st.session_state.editing_person_id = person.id
                st.rerun()
            if st.button("Delete", key=f"del_{person.id}", use_container_width=True):
                st.session_state.confirm_delete_person_id = person.id
                st.rerun()


def _render_person_form(family_id: int, existing: Optional[Person]) -> None:
    is_edit = existing is not None
    form_key = f"person_form_edit_{existing.id}" if is_edit else "person_form_new"

    with st.form(form_key, clear_on_submit=not is_edit):
        c1, c2, c3 = st.columns(3)
        with c1:
            first_name = st.text_input("First Name *", value=existing.first_name if is_edit else "")
        with c2:
            middle_name = st.text_input("Middle Name", value=(existing.middle_name or "") if is_edit else "")
        with c3:
            last_name = st.text_input("Last Name *", value=existing.last_name if is_edit else "")

        c4, c5 = st.columns(2)
        with c4:
            preferred_name = st.text_input("Preferred Name", value=(existing.preferred_name or "") if is_edit else "")
        with c5:
            dob = st.date_input(
                "Date of Birth",
                value=existing.dob if (is_edit and existing.dob) else None,
                min_value=date(1900, 1, 1),
                max_value=date.today(),
                format="YYYY-MM-DD",
            )

        c6, c7 = st.columns(2)
        with c6:
            email = st.text_input("Email", value=(existing.email or "") if is_edit else "")
        with c7:
            phone = st.text_input("Phone", value=(existing.phone or "") if is_edit else "")

        if is_edit:
            ssn_action = st.radio("SSN", ["Keep current", "Update", "Clear"], horizontal=True, index=0)
            new_ssn = ""
            if ssn_action == "Update":
                new_ssn = st.text_input("New SSN (xxx-xx-xxxx)", type="password")
        else:
            ssn_action = "New"
            new_ssn = st.text_input("SSN (xxx-xx-xxxx)", type="password")

        c8, c9 = st.columns(2)
        with c8:
            is_deceased = st.checkbox("Deceased", value=existing.is_deceased if is_edit else False)
        with c9:
            dod = st.date_input(
                "Date of Death",
                value=existing.date_of_death if (is_edit and existing.date_of_death) else None,
                min_value=date(1900, 1, 1),
                max_value=date.today(),
                format="YYYY-MM-DD",
            )
            if not is_deceased:
                st.caption("Check 'Deceased' to record this date.")

        notes = st.text_area("Notes", value=(existing.notes or "") if is_edit else "")

        col_save, col_cancel = st.columns(2)
        with col_save:
            submitted = st.form_submit_button("💾 Save", use_container_width=True, type="primary")
        with col_cancel:
            cancelled = st.form_submit_button("Cancel", use_container_width=True)

        if cancelled:
            if is_edit:
                st.session_state.editing_person_id = None
            st.rerun()

        if submitted:
            if not first_name.strip() or not last_name.strip():
                st.error("First and last name are required.")
                return

            common = {
                "first_name": first_name.strip(),
                "last_name": last_name.strip(),
                "middle_name": middle_name.strip() or None,
                "preferred_name": preferred_name.strip() or None,
                "dob": dob if dob else None,
                "email": email.strip() or None,
                "phone": phone.strip() or None,
                "is_deceased": is_deceased,
                "date_of_death": dod if (is_deceased and dod) else None,
                "notes": notes.strip() or None,
            }

            if is_edit:
                update_kwargs = dict(common)
                if ssn_action == "Update":
                    update_kwargs["ssn"] = new_ssn.strip() or None
                elif ssn_action == "Clear":
                    update_kwargs["ssn"] = None
                update_person(existing.id, **update_kwargs)
                st.session_state.editing_person_id = None
            else:
                create_person(family_id=family_id, ssn=new_ssn.strip() or None, **common)
            st.rerun()


# ===========================================================================
# Relationships tab
# ===========================================================================

def _render_relationships_tab(family_id: int) -> None:
    people = list_people_in_family(family_id)
    relationships = list_relationships_in_family(family_id)

    st.subheader("Relationships")

    if len(people) < 2:
        st.info("You need at least two people in the family to define a relationship.")
        return

    person_by_id = {p.id: p for p in people}

    if relationships:
        for rel in relationships:
            a = person_by_id.get(rel.person_a_id)
            b = person_by_id.get(rel.person_b_id)
            if not a or not b:
                continue
            with st.container(border=True):
                col_text, col_action = st.columns([5, 1])
                with col_text:
                    st.markdown(_format_relationship(a, b, rel.relationship_type))
                    sub = []
                    if rel.start_date:
                        sub.append(f"since {rel.start_date.isoformat()}")
                    if rel.notes:
                        sub.append(rel.notes)
                    if sub:
                        st.caption(" · ".join(sub))
                with col_action:
                    if st.button("Delete", key=f"del_rel_{rel.id}", use_container_width=True):
                        delete_relationship(rel.id)
                        st.rerun()
    else:
        st.info("No relationships defined yet.")

    st.markdown("---")
    with st.expander("➕ Add Relationship", expanded=not relationships):
        person_options = {p.display_name: p.id for p in people}
        with st.form("new_relationship_form", clear_on_submit=True):
            c1, c2, c3 = st.columns([2, 2, 2])
            with c1:
                a_label = st.selectbox("Person A", list(person_options.keys()), key="rel_a")
            with c2:
                rel_type = st.selectbox("Relationship Type", RELATIONSHIP_TYPES, key="rel_type")
            with c3:
                b_label = st.selectbox("Person B", list(person_options.keys()), key="rel_b")

            if rel_type == "parent_of":
                st.caption("ℹ️ Direction matters: A is the parent of B.")

            start_dt = st.date_input(
                "Start Date (optional — e.g. marriage date)",
                value=None,
                min_value=date(1900, 1, 1),
                max_value=date.today(),
                format="YYYY-MM-DD",
            )
            notes = st.text_input("Notes (optional)")

            if st.form_submit_button("Create", type="primary", use_container_width=True):
                a_id = person_options[a_label]
                b_id = person_options[b_label]
                if a_id == b_id:
                    st.error("Please pick two different people.")
                else:
                    create_relationship(
                        person_a_id=a_id,
                        person_b_id=b_id,
                        relationship_type=rel_type,
                        start_date=start_dt if start_dt else None,
                        notes=notes.strip() or None,
                    )
                    st.rerun()


# ===========================================================================
# Entities tab (Step 3a)
# ===========================================================================

def _render_entities_tab(family_id: int) -> None:
    entities = list_entities_in_family(family_id)

    st.subheader("Entities")
    st.caption("Trusts, LLCs, partnerships, corporations, foundations, etc.")

    if entities:
        for entity in entities:
            _render_entity_card(entity)
    else:
        st.info("No entities yet. Add the first one below.")

    pending_delete_id = st.session_state.get("confirm_delete_entity_id")
    if pending_delete_id is not None:
        target = get_entity(pending_delete_id)
        if target is None:
            st.session_state.confirm_delete_entity_id = None
        else:
            with st.expander(f"⚠️ Confirm deletion of {target.name}?", expanded=True):
                st.warning(
                    "This deletes the entity and all its roles. Documents "
                    "linked to this entity will have the link cleared but "
                    "remain on the family."
                )
                col_y, col_n = st.columns(2)
                with col_y:
                    if st.button(
                        "Yes, delete",
                        key=f"confirm_ent_yes_{pending_delete_id}",
                        type="primary",
                        use_container_width=True,
                    ):
                        delete_entity(pending_delete_id)
                        st.session_state.confirm_delete_entity_id = None
                        st.session_state.editing_entity_id = None
                        st.rerun()
                with col_n:
                    if st.button(
                        "Cancel",
                        key=f"confirm_ent_no_{pending_delete_id}",
                        use_container_width=True,
                    ):
                        st.session_state.confirm_delete_entity_id = None
                        st.rerun()

    st.markdown("---")
    editing_id = st.session_state.get("editing_entity_id")
    if editing_id is not None:
        editing_entity = get_entity(editing_id)
        if editing_entity is None:
            st.session_state.editing_entity_id = None
            st.rerun()
        st.subheader(f"✏️ Edit: {editing_entity.name}")
        _render_entity_form(family_id, editing_entity)
    else:
        with st.expander("➕ Add Entity", expanded=not entities):
            _render_entity_form(family_id, None)


def _render_entity_card(entity: Entity) -> None:
    with st.container(border=True):
        col_info, col_id, col_actions = st.columns([3, 2, 2])

        with col_info:
            icon = _entity_icon(entity.entity_type)
            st.markdown(f"{icon}  **{entity.name}**")
            bits = [entity.entity_type]
            if entity.sub_type:
                bits.append(entity.sub_type)
            if entity.jurisdiction:
                bits.append(entity.jurisdiction)
            if entity.formation_date:
                bits.append(f"formed {entity.formation_date.isoformat()}")
            if entity.termination_date:
                bits.append(f"terminated {entity.termination_date.isoformat()}")
            st.caption(" · ".join(bits))

        with col_id:
            show_key = f"show_taxid_{entity.id}"
            show = st.toggle("Show Tax ID", key=show_key, value=False)
            if entity.tax_id_encrypted is None:
                st.code("— no Tax ID on file —")
            elif show:
                st.code(entity.tax_id or "—")
            else:
                st.code(_mask_id(entity.tax_id))

        with col_actions:
            if st.button("Edit", key=f"edit_ent_{entity.id}", use_container_width=True):
                st.session_state.editing_entity_id = entity.id
                st.rerun()
            if st.button("Delete", key=f"del_ent_{entity.id}", use_container_width=True):
                st.session_state.confirm_delete_entity_id = entity.id
                st.rerun()


def _render_entity_form(family_id: int, existing: Optional[Entity]) -> None:
    is_edit = existing is not None
    form_key = f"entity_form_edit_{existing.id}" if is_edit else "entity_form_new"

    with st.form(form_key, clear_on_submit=not is_edit):
        name = st.text_input(
            "Entity Name *",
            value=existing.name if is_edit else "",
            placeholder="e.g. The Smith Family Revocable Trust",
        )

        c1, c2 = st.columns(2)
        with c1:
            entity_type = st.selectbox(
                "Entity Type *",
                ENTITY_TYPES,
                index=ENTITY_TYPES.index(existing.entity_type)
                if (is_edit and existing.entity_type in ENTITY_TYPES)
                else 0,
            )
        with c2:
            sub_type_hint = ENTITY_SUBTYPE_HINTS.get(entity_type, "")
            sub_type = st.text_input(
                "Sub-type",
                value=(existing.sub_type or "") if is_edit else "",
                placeholder=f"e.g. {sub_type_hint}" if sub_type_hint else "",
            )

        c3, c4 = st.columns(2)
        with c3:
            jurisdiction = st.text_input(
                "Jurisdiction (state or country)",
                value=(existing.jurisdiction or "") if is_edit else "",
                placeholder="e.g. NC, DE, Delaware",
            )
        with c4:
            formation_date = st.date_input(
                "Formation Date",
                value=existing.formation_date if (is_edit and existing.formation_date) else None,
                min_value=date(1900, 1, 1),
                max_value=date.today(),
                format="YYYY-MM-DD",
            )

        termination_date = st.date_input(
            "Termination Date (optional)",
            value=existing.termination_date if (is_edit and existing.termination_date) else None,
            min_value=date(1900, 1, 1),
            max_value=date.today(),
            format="YYYY-MM-DD",
        )

        if is_edit:
            tax_id_action = st.radio(
                "Tax ID (EIN)",
                ["Keep current", "Update", "Clear"],
                horizontal=True,
                index=0,
            )
            new_tax_id = ""
            if tax_id_action == "Update":
                new_tax_id = st.text_input("New Tax ID (xx-xxxxxxx)", type="password")
        else:
            tax_id_action = "New"
            new_tax_id = st.text_input("Tax ID / EIN (xx-xxxxxxx)", type="password")

        notes = st.text_area("Notes", value=(existing.notes or "") if is_edit else "")

        col_save, col_cancel = st.columns(2)
        with col_save:
            submitted = st.form_submit_button("💾 Save", use_container_width=True, type="primary")
        with col_cancel:
            cancelled = st.form_submit_button("Cancel", use_container_width=True)

        if cancelled:
            if is_edit:
                st.session_state.editing_entity_id = None
            st.rerun()

        if submitted:
            if not name.strip():
                st.error("Entity name is required.")
                return

            common = {
                "name": name.strip(),
                "entity_type": entity_type,
                "sub_type": sub_type.strip() or None,
                "jurisdiction": jurisdiction.strip() or None,
                "formation_date": formation_date if formation_date else None,
                "termination_date": termination_date if termination_date else None,
                "notes": notes.strip() or None,
            }

            if is_edit:
                update_kwargs = dict(common)
                if tax_id_action == "Update":
                    update_kwargs["tax_id"] = new_tax_id.strip() or None
                elif tax_id_action == "Clear":
                    update_kwargs["tax_id"] = None
                update_entity(existing.id, **update_kwargs)
                st.session_state.editing_entity_id = None
            else:
                create_entity(family_id=family_id, tax_id=new_tax_id.strip() or None, **common)
            st.rerun()


# ===========================================================================
# Roles tab (Step 3a)
# ===========================================================================

def _render_roles_tab(family_id: int) -> None:
    people = list_people_in_family(family_id)
    entities = list_entities_in_family(family_id)

    st.subheader("Roles")
    st.caption(
        "Who plays what role in each entity. A person can hold multiple "
        "roles in the same entity (e.g. grantor + trustee + beneficiary)."
    )

    if not people or not entities:
        st.info(
            "Add at least one person AND one entity to assign roles. "
            f"Currently: {len(people)} people, {len(entities)} entities."
        )
        return

    person_by_id = {p.id: p for p in people}

    for entity in entities:
        roles = list_roles_for_entity(entity.id)
        with st.container(border=True):
            icon = _entity_icon(entity.entity_type)
            sub = entity.sub_type or entity.entity_type
            st.markdown(f"### {icon}  {entity.name}")
            st.caption(f"{sub}" + (f" · {entity.jurisdiction}" if entity.jurisdiction else ""))

            if roles:
                for role in roles:
                    person = person_by_id.get(role.person_id)
                    if person is None:
                        continue
                    col_role, col_action = st.columns([6, 1])
                    with col_role:
                        bits = [f"**{person.display_name}**", f"_{role.role_type}_"]
                        if role.interest_percentage is not None:
                            bits.append(f"{role.interest_percentage:.2f}%")
                        if role.start_date:
                            bits.append(f"since {role.start_date.isoformat()}")
                        if role.end_date:
                            bits.append(f"until {role.end_date.isoformat()}")
                        if not role.is_active:
                            bits.append("⚠️ inactive")
                        st.markdown(" · ".join(bits))
                        if role.notes:
                            st.caption(role.notes)
                    with col_action:
                        if st.button(
                            "Delete",
                            key=f"del_role_{role.id}",
                            use_container_width=True,
                        ):
                            delete_role(role.id)
                            st.rerun()
            else:
                st.caption("_No roles assigned yet._")

    st.markdown("---")
    with st.expander("➕ Add Role", expanded=False):
        person_options = {p.display_name: p.id for p in people}
        entity_options = {e.name: e.id for e in entities}

        with st.form("new_role_form", clear_on_submit=True):
            c1, c2 = st.columns(2)
            with c1:
                p_label = st.selectbox("Person", list(person_options.keys()), key="role_person")
            with c2:
                e_label = st.selectbox("Entity", list(entity_options.keys()), key="role_entity")

            c3, c4 = st.columns(2)
            with c3:
                role_type = st.selectbox("Role", ROLE_TYPES, key="role_type")
            with c4:
                interest_pct = st.number_input(
                    "Interest % (optional)",
                    min_value=0.0,
                    max_value=100.0,
                    value=0.0,
                    step=0.01,
                    format="%.2f",
                )

            c5, c6 = st.columns(2)
            with c5:
                start_dt = st.date_input(
                    "Start Date (optional)",
                    value=None,
                    min_value=date(1900, 1, 1),
                    max_value=date.today(),
                    format="YYYY-MM-DD",
                )
            with c6:
                end_dt = st.date_input(
                    "End Date (optional)",
                    value=None,
                    min_value=date(1900, 1, 1),
                    max_value=date.today(),
                    format="YYYY-MM-DD",
                )

            notes = st.text_input("Notes (optional)")

            if st.form_submit_button("Create Role", type="primary", use_container_width=True):
                create_role(
                    person_id=person_options[p_label],
                    entity_id=entity_options[e_label],
                    role_type=role_type,
                    start_date=start_dt if start_dt else None,
                    end_date=end_dt if end_dt else None,
                    interest_percentage=interest_pct if interest_pct > 0 else None,
                    notes=notes.strip() or None,
                )
                st.rerun()


# ===========================================================================
# Settings tab
# ===========================================================================

def _render_settings_tab(family: Family) -> None:
    st.subheader("Family Settings")

    with st.form("family_settings_form"):
        name = st.text_input("Family Name", value=family.name)
        notes = st.text_area("Notes", value=family.notes or "")
        if st.form_submit_button("💾 Save", type="primary"):
            update_family(
                family.id,
                name=name.strip() or family.name,
                notes=notes.strip() or None,
            )
            st.rerun()

    st.markdown("---")
    with st.expander("⚠️ Danger Zone"):
        st.warning(
            "Deleting a family permanently removes its people, "
            "relationships, entities, roles, document records, and "
            "extractions. Files on disk are NOT touched. Cannot be undone."
        )
        confirm_text = st.text_input(
            f"Type the family name to confirm:  **{family.name}**",
            key=f"confirm_del_fam_text_{family.id}",
        )
        if st.button(
            "Delete Family Permanently",
            type="primary",
            key=f"confirm_del_fam_btn_{family.id}",
        ):
            if confirm_text.strip() == family.name:
                delete_family(family.id)
                st.session_state.selected_family_id = None
                st.rerun()
            else:
                st.error("Family name doesn't match. Not deleted.")


# ===========================================================================
# Helpers
# ===========================================================================

def _mask_id(value: Optional[str]) -> str:
    """Mask an SSN or EIN, showing only the last 4 digits."""
    if not value:
        return "—"
    digits = "".join(c for c in value if c.isdigit())
    if len(digits) >= 4:
        return f"XXX-XX-{digits[-4:]}"
    return "XXX-XX-XXXX"


def _format_relationship(a: Person, b: Person, rel_type: str) -> str:
    a_name = a.display_name
    b_name = b.display_name
    if rel_type == "spouse":
        return f"**{a_name}**  ↔  **{b_name}**  · spouses"
    if rel_type == "ex_spouse":
        return f"**{a_name}**  ↔  **{b_name}**  · former spouses"
    if rel_type == "parent_of":
        return f"**{a_name}**  →  **{b_name}**  · parent of"
    if rel_type == "guardian_of":
        return f"**{a_name}**  →  **{b_name}**  · guardian of"
    if rel_type == "sibling":
        return f"**{a_name}**  ↔  **{b_name}**  · siblings"
    return f"**{a_name}**  ↔  **{b_name}**  · {rel_type}"


def _entity_icon(entity_type: str) -> str:
    return {
        "trust": "🏛️",
        "llc": "🏢",
        "partnership": "🤝",
        "corporation": "🏢",
        "foundation": "💝",
        "donor_advised_fund": "💝",
        "holding_company": "🏗️",
    }.get(entity_type, "📋")
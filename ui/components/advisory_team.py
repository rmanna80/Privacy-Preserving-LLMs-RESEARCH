"""
Advisory Team page — the professionals helping a family.

Renders the list of advisory team members for a family, with a
hub-and-spoke SVG visualization (the family at the center, advisors
around the edge — the Screenshot #1 look from the product spec).

Below the visual is the team roster with edit/remove controls and the
add-team-member form.
"""

from __future__ import annotations

import math
from typing import Optional

import streamlit as st

from db.repositories import (
    list_team_members,
    get_team_member,
    create_team_member,
    update_team_member,
    delete_team_member,
    ADVISORY_ROLES,
    get_family,
)
from db.models import AdvisoryTeamMember


# ─────────────────────────────────────────────────────────────────────
# Role metadata — labels and icons used in dropdowns & the visual
# ─────────────────────────────────────────────────────────────────────

# Display label + emoji icon per role.
ROLE_META: dict[str, tuple[str, str]] = {
    "wealth_strategist":        ("Wealth Strategist",        "💼"),
    "tax_strategist":           ("Tax Strategist",           "📊"),
    "estate_planning_attorney": ("Estate Planning Attorney", "🏛️"),
    "business_attorney":        ("Business Attorney",        "⚖️"),
    "general_counsel":          ("General Counsel",          "📋"),
    "m_and_a_attorney":         ("M&A Attorney",             "🤝"),
    "insurance_broker":         ("Insurance Broker",         "🛡️"),
    "family_coach":             ("Family Coach",             "🧭"),
    "investment_banker":        ("Investment Banker",        "🏦"),
    "other":                    ("Other",                    "👤"),
}


def role_label(role: str) -> str:
    return ROLE_META.get(role, (role.replace("_", " ").title(), "👤"))[0]


def role_icon(role: str) -> str:
    return ROLE_META.get(role, ("", "👤"))[1]


# ─────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────

def render_advisory_team_page(family_id: int) -> None:
    family = get_family(family_id)
    if family is None:
        st.error("Family not found.")
        return

    st.markdown("### Advisory Team")
    st.caption(
        "The professionals helping this family. Add wealth strategists, "
        "tax strategists, estate attorneys, and other key advisors."
    )

    members = list_team_members(family_id, active_only=True)

    # Hub-and-spoke visual at the top
    if members:
        _render_hub_and_spoke(family.name, members)
        st.markdown("---")

    # Member cards / list
    if members:
        st.markdown("**Team Roster**")
        for m in members:
            _render_member_card(m)
    else:
        st.info(
            "No advisors yet for this family. Add the first team member below."
        )

    # Pending-delete confirmation
    pending_delete = st.session_state.get("confirm_delete_team_member_id")
    if pending_delete is not None:
        target = get_team_member(pending_delete)
        if target is None:
            st.session_state.confirm_delete_team_member_id = None
        else:
            with st.expander(
                f"⚠️ Remove {target.full_name} from the advisory team?",
                expanded=True,
            ):
                st.warning(
                    "This removes them from this family's roster. Any tasks "
                    "currently assigned to them will become unassigned."
                )
                col_y, col_n = st.columns(2)
                with col_y:
                    if st.button(
                        "Yes, remove",
                        key=f"confirm_tm_yes_{pending_delete}",
                        type="primary",
                        use_container_width=True,
                    ):
                        delete_team_member(pending_delete)
                        st.session_state.confirm_delete_team_member_id = None
                        st.session_state.editing_team_member_id = None
                        st.rerun()
                with col_n:
                    if st.button(
                        "Cancel",
                        key=f"confirm_tm_no_{pending_delete}",
                        use_container_width=True,
                    ):
                        st.session_state.confirm_delete_team_member_id = None
                        st.rerun()

    # Add / edit form
    st.markdown("---")
    editing_id = st.session_state.get("editing_team_member_id")
    if editing_id is not None:
        editing = get_team_member(editing_id)
        if editing is None:
            st.session_state.editing_team_member_id = None
            st.rerun()
        st.subheader(f"✏️ Edit: {editing.full_name}")
        _render_member_form(family_id, editing)
    else:
        with st.expander("➕ Add Team Member", expanded=not members):
            _render_member_form(family_id, None)


# ─────────────────────────────────────────────────────────────────────
# Hub-and-spoke visual
# ─────────────────────────────────────────────────────────────────────

# Brand colors — match the theme
COL_BG = "#F8F4EC"
COL_HUB_FILL = "#0B1E3F"
COL_HUB_STROKE = "#0B1E3F"
COL_HUB_TEXT = "#F8F4EC"
COL_SPOKE_LINE = "#A8884D"
COL_NODE_FILL = "#FFFFFF"
COL_NODE_STROKE = "#A8884D"
COL_NODE_TEXT = "#0B1E3F"
COL_LABEL_TEXT = "#5F7494"


def _render_hub_and_spoke(family_name: str, members: list[AdvisoryTeamMember]) -> None:
    """SVG hub-and-spoke: family at center, advisors around the edge.

    Uses Streamlit components.html so SVG renders properly (lesson from
    the family tree — st.markdown strips SVG even with unsafe_allow_html).
    """
    import streamlit.components.v1 as components

    svg = _build_hub_and_spoke_svg(family_name, members)

    # Fixed aspect: the SVG sizes itself, iframe just hosts it.
    components.html(
        f"""
        <div style="background:{COL_BG}; border-radius:12px; padding:16px;
                    height:480px; box-sizing:border-box;
                    display:flex; align-items:center; justify-content:center;">
          {svg}
        </div>
        """,
        height=500,
        scrolling=False,
    )


def _build_hub_and_spoke_svg(
    family_name: str,
    members: list[AdvisoryTeamMember],
) -> str:
    """Radial layout. Family in center, members arranged in a circle.

    Spacing scales with member count — many members get a larger ring.
    """
    n = len(members)
    width = 700
    height = 460
    cx, cy = width / 2, height / 2

    # Ring radius scales with member count
    if n <= 4:
        radius = 150
    elif n <= 8:
        radius = 170
    else:
        radius = 190

    hub_r = 60      # central family circle
    node_r = 34     # member circles

    parts: list[str] = []

    # Background
    parts.append(
        f'<rect x="0" y="0" width="{width}" height="{height}" '
        f'fill="{COL_BG}" rx="12" ry="12"/>'
    )

    # Spokes (drawn first, so circles sit on top)
    for i, m in enumerate(members):
        angle = -math.pi / 2 + (2 * math.pi * i / max(n, 1))
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        # Trim spoke so it doesn't visually pierce the hub or node
        from_x = cx + hub_r * math.cos(angle)
        from_y = cy + hub_r * math.sin(angle)
        to_x = x - node_r * math.cos(angle)
        to_y = y - node_r * math.sin(angle)
        parts.append(
            f'<line x1="{from_x:.1f}" y1="{from_y:.1f}" '
            f'x2="{to_x:.1f}" y2="{to_y:.1f}" '
            f'stroke="{COL_SPOKE_LINE}" stroke-width="1.5" '
            f'stroke-dasharray="3 3"/>'
        )

    # Central hub — the family
    parts.append(
        f'<circle cx="{cx}" cy="{cy}" r="{hub_r}" '
        f'fill="{COL_HUB_FILL}" stroke="{COL_HUB_STROKE}" stroke-width="2"/>'
    )
    # Family name inside the hub — break into up to 2 lines for long names
    family_lines = _split_for_hub(family_name)
    if len(family_lines) == 1:
        parts.append(
            f'<text x="{cx}" y="{cy + 5}" text-anchor="middle" '
            f'font-family="Playfair Display, Georgia, serif" '
            f'font-size="14" font-weight="600" fill="{COL_HUB_TEXT}">'
            f'{_escape_xml(family_lines[0])}</text>'
        )
    else:
        parts.append(
            f'<text x="{cx}" y="{cy - 4}" text-anchor="middle" '
            f'font-family="Playfair Display, Georgia, serif" '
            f'font-size="13" font-weight="600" fill="{COL_HUB_TEXT}">'
            f'{_escape_xml(family_lines[0])}</text>'
            f'<text x="{cx}" y="{cy + 14}" text-anchor="middle" '
            f'font-family="Playfair Display, Georgia, serif" '
            f'font-size="13" font-weight="600" fill="{COL_HUB_TEXT}">'
            f'{_escape_xml(family_lines[1])}</text>'
        )

    # Member nodes
    for i, m in enumerate(members):
        angle = -math.pi / 2 + (2 * math.pi * i / max(n, 1))
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)

        # Circle
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{node_r}" '
            f'fill="{COL_NODE_FILL}" stroke="{COL_NODE_STROKE}" stroke-width="1.5"/>'
        )

        # Icon inside the circle
        icon = role_icon(m.role)
        parts.append(
            f'<text x="{x:.1f}" y="{y + 8:.1f}" text-anchor="middle" '
            f'font-size="22">{icon}</text>'
        )

        # Label below the node (role)
        label_y = y + node_r + 16
        # Push label outward to avoid overlapping the hub at the top/bottom
        outward_x = x + (12 * math.cos(angle) if abs(math.cos(angle)) < 0.3 else 0)
        parts.append(
            f'<text x="{outward_x:.1f}" y="{label_y:.1f}" text-anchor="middle" '
            f'font-family="Inter, sans-serif" font-size="11" '
            f'font-weight="600" fill="{COL_NODE_TEXT}">'
            f'{_escape_xml(role_label(m.role))}</text>'
        )

        # Name underneath the role
        name = m.full_name
        if len(name) > 26:
            name = name[:25] + "…"
        parts.append(
            f'<text x="{outward_x:.1f}" y="{label_y + 14:.1f}" text-anchor="middle" '
            f'font-family="Inter, sans-serif" font-size="10" '
            f'fill="{COL_LABEL_TEXT}">{_escape_xml(name)}</text>'
        )

    body = "\n".join(parts)
    return f"""
    <svg xmlns="http://www.w3.org/2000/svg"
         viewBox="0 0 {width} {height}"
         preserveAspectRatio="xMidYMid meet"
         width="100%" height="100%"
         style="background:{COL_BG}; border-radius:12px; display:block;">
      {body}
    </svg>
    """


def _split_for_hub(name: str) -> list[str]:
    """Split family name into up to 2 lines for display inside the hub."""
    if len(name) <= 14:
        return [name]
    # Try to split on a space near the middle
    mid = len(name) // 2
    best = None
    for i, ch in enumerate(name):
        if ch == " ":
            if best is None or abs(i - mid) < abs(best - mid):
                best = i
    if best is None:
        return [name[:14] + "…"]
    return [name[:best], name[best + 1:]]


def _escape_xml(s: str) -> str:
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
         .replace('"', "&quot;")
    )


# ─────────────────────────────────────────────────────────────────────
# Member card + form
# ─────────────────────────────────────────────────────────────────────

def _render_member_card(m: AdvisoryTeamMember) -> None:
    with st.container(border=True):
        col_info, col_contact, col_actions = st.columns([3, 3, 2])

        with col_info:
            icon = role_icon(m.role)
            st.markdown(f"{icon}  **{m.full_name}**")
            bits = [role_label(m.role)]
            if m.firm:
                bits.append(m.firm)
            st.caption(" · ".join(bits))

        with col_contact:
            if m.email:
                st.markdown(f"📧 {m.email}")
            if m.phone:
                st.markdown(f"📞 {m.phone}")
            if not m.email and not m.phone:
                st.caption("_No contact info on file_")

        with col_actions:
            if st.button("Edit", key=f"edit_tm_{m.id}", use_container_width=True):
                st.session_state.editing_team_member_id = m.id
                st.rerun()
            if st.button("Remove", key=f"del_tm_{m.id}", use_container_width=True):
                st.session_state.confirm_delete_team_member_id = m.id
                st.rerun()

        if m.notes:
            st.caption(f"📝 {m.notes}")


def _render_member_form(
    family_id: int,
    existing: Optional[AdvisoryTeamMember],
) -> None:
    is_edit = existing is not None
    form_key = (
        f"team_member_form_edit_{existing.id}"
        if is_edit
        else "team_member_form_new"
    )

    with st.form(form_key, clear_on_submit=not is_edit):
        c1, c2 = st.columns(2)
        with c1:
            full_name = st.text_input(
                "Full Name *",
                value=existing.full_name if is_edit else "",
                placeholder="e.g. Jane Carlson",
            )
        with c2:
            role_options = [r for r in ADVISORY_ROLES]
            role_labels_for_options = [role_label(r) for r in role_options]
            current_role_idx = (
                role_options.index(existing.role) if is_edit else 0
            )
            picked_label = st.selectbox(
                "Role *",
                role_labels_for_options,
                index=current_role_idx,
            )
            role = role_options[role_labels_for_options.index(picked_label)]

        firm = st.text_input(
            "Firm",
            value=(existing.firm or "") if is_edit else "",
            placeholder="e.g. Carlson Estate Law",
        )

        c3, c4 = st.columns(2)
        with c3:
            email = st.text_input(
                "Email",
                value=(existing.email or "") if is_edit else "",
            )
        with c4:
            phone = st.text_input(
                "Phone",
                value=(existing.phone or "") if is_edit else "",
            )

        notes = st.text_area(
            "Notes",
            value=(existing.notes or "") if is_edit else "",
            placeholder="Engagement scope, billing rate, key contacts…",
        )

        col_save, col_cancel = st.columns(2)
        with col_save:
            submitted = st.form_submit_button(
                "💾 Save", use_container_width=True, type="primary"
            )
        with col_cancel:
            cancelled = st.form_submit_button(
                "Cancel", use_container_width=True
            )

        if cancelled:
            if is_edit:
                st.session_state.editing_team_member_id = None
            st.rerun()

        if submitted:
            if not full_name.strip():
                st.error("Full name is required.")
                return

            common = dict(
                role=role,
                full_name=full_name.strip(),
                firm=firm.strip() or None,
                email=email.strip() or None,
                phone=phone.strip() or None,
                notes=notes.strip() or None,
            )

            if is_edit:
                update_team_member(existing.id, **common)
                st.session_state.editing_team_member_id = None
            else:
                create_team_member(family_id=family_id, **common)

            st.rerun()

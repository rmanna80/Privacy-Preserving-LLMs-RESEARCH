"""
ui/components/org_ownership.py — Organizational Ownership chart (cleaner layout).

Layer 1: the Person ↔ Entity role graph. Each entity is a node; the people
who hold roles in it are listed beneath it.

This version fixes the overlapping-connector problem from v1. Instead of
curved lines from a single shared point with floating midpoint labels
(which stacked on top of each other), we use a clean "comb":
  - a single vertical spine drops from each entity
  - a short horizontal stub connects the spine to each person box
  - the ROLE label sits inside a small tag ABOVE each person's name,
    so it never floats on a line or overlaps anything

Rendering: SVG via streamlit.components.v1.html (st.markdown strips SVG).
"""

from __future__ import annotations

import streamlit as st

from db.repositories import (
    get_family,
    list_entities_in_family,
    list_roles_for_entity,
    get_person,
)


# Brand palette
COL_BG = "#F8F4EC"
COL_ENTITY_FILL = "#0B1E3F"
COL_ENTITY_TEXT = "#F8F4EC"
COL_ENTITY_SUB = "#9AA8C0"
COL_PERSON_FILL = "#FFFFFF"
COL_PERSON_STROKE = "#22385F"
COL_PERSON_TEXT = "#0B1E3F"
COL_LINE = "#A8884D"
COL_ROLE_TEXT = "#A8884D"

# Layout constants
ENTITY_W = 220
ENTITY_H = 70
PERSON_W = 190
PERSON_H = 56          # taller: holds role tag + name
LANE_GAP = 60          # gap between lanes
PERSON_GAP_Y = 74      # vertical gap between people in a lane
TOP_PAD = 40
SIDE_PAD = 40
SPINE_DROP = 50        # how far the spine drops before first person
SPINE_X_OFFSET = 0     # spine runs down lane center


ROLE_LABELS = {
    "grantor": "Grantor", "settlor": "Settlor", "trustee": "Trustee",
    "co_trustee": "Co-Trustee", "successor_trustee": "Successor Trustee",
    "trust_protector": "Trust Protector", "beneficiary": "Beneficiary",
    "contingent_beneficiary": "Contingent Beneficiary",
    "remainder_beneficiary": "Remainder Beneficiary",
    "member": "Member", "manager": "Manager",
    "managing_member": "Managing Member", "shareholder": "Shareholder",
    "officer": "Officer", "director": "Director", "owner": "Owner",
    "insured": "Insured",
}


def role_label(role_type: str) -> str:
    return ROLE_LABELS.get(role_type, (role_type or "").replace("_", " ").title())


def _escape(s: str) -> str:
    return (
        (s or "").replace("&", "&amp;").replace("<", "&lt;")
        .replace(">", "&gt;").replace('"', "&quot;")
    )


def render_org_ownership(family_id: int) -> None:
    family = get_family(family_id)
    if family is None:
        st.error("Family not found.")
        return

    st.markdown("### Organizational Ownership")
    st.caption(
        "Who holds which roles across this family's entities. Populated "
        "from the People & Roles you define or promote from documents."
    )

    entities = list_entities_in_family(family_id)
    if not entities:
        st.info(
            "No entities yet. Add them in **Key People & Orgs → Entities**, "
            "or promote them from a document in **Documents → Extractions**."
        )
        return

    # Gather (entity, [(role, person), ...]) and de-duplicate identical
    # role+person pairs so the same fact promoted twice doesn't double-draw.
    entity_blocks = []
    for ent in entities:
        roles = list_roles_for_entity(ent.id)
        seen = set()
        people_in_roles = []
        for r in roles:
            key = (r.person_id, r.role_type)
            if key in seen:
                continue
            seen.add(key)
            p = get_person(r.person_id)
            if p is not None:
                people_in_roles.append((r, p))
        entity_blocks.append((ent, people_in_roles))

    svg, width, height = _build_svg(entity_blocks)

    # import streamlit.components.v1 as components
    # iframe_height = min(1000, max(360, height + 30))
    # components.html(
    #     f'<div style="background:{COL_BG}; border-radius:12px; padding:8px; '
    #     f'overflow-x:auto;">{svg}</div>',
    #     height=iframe_height,
    #     scrolling=True,
    # )

    import streamlit.components.v1 as components

    # Scale the SVG to fit the iframe width while keeping aspect ratio,
    # so the whole chart is visible without zooming/scrolling. We cap the
    # display height and let the SVG shrink to fit.
    display_height = min(620, height + 20)
    components.html(
        f'''
        <div style="background:{COL_BG}; border-radius:12px; padding:8px;
                    width:100%; height:{display_height}px; box-sizing:border-box;
                    display:flex; align-items:flex-start; justify-content:center;
                    overflow:auto;">
          <div style="width:100%; max-width:{width}px;">{svg}</div>
        </div>
        ''',
        height=display_height + 16,
        scrolling=False,
    )

    st.markdown("---")
    col_legend, col_dl = st.columns([3, 1])
    with col_legend:
        st.markdown("**How to read this**")
        st.caption(
            "Navy boxes are entities. White boxes are people; the gold tag "
            "above each name is the role they hold in that entity. A person "
            "can appear under several entities."
        )
    with col_dl:
        st.download_button(
            "⬇ Download SVG", data=svg,
            file_name=f"org_ownership_family_{family_id}.svg",
            mime="image/svg+xml",
        )


def _build_svg(entity_blocks) -> tuple[str, int, int]:
    n = len(entity_blocks)
    lane_w = max(ENTITY_W, PERSON_W)
    total_width = SIDE_PAD * 2 + n * lane_w + (n - 1) * LANE_GAP

    max_people = max((len(p) for _, p in entity_blocks), default=0)
    total_height = (
        TOP_PAD + ENTITY_H + SPINE_DROP
        + max(0, max_people) * PERSON_GAP_Y + 30
    )

    parts = [
        f'<rect x="0" y="0" width="{total_width}" height="{total_height}" '
        f'fill="{COL_BG}" rx="12" ry="12"/>'
    ]

    for lane_idx, (entity, people) in enumerate(entity_blocks):
        lane_left = SIDE_PAD + lane_idx * (lane_w + LANE_GAP)
        lane_center = lane_left + lane_w / 2

        ent_x = lane_center - ENTITY_W / 2
        ent_y = TOP_PAD
        parts.append(_entity_node(entity, ent_x, ent_y))

        if not people:
            continue

        # Spine: a single vertical line from entity bottom down past the
        # last person's vertical center.
        spine_x = lane_center
        spine_top = ent_y + ENTITY_H
        first_person_y = ent_y + ENTITY_H + SPINE_DROP
        last_person_y = first_person_y + (len(people) - 1) * PERSON_GAP_Y
        spine_bottom = last_person_y + PERSON_H / 2
        parts.append(
            f'<line x1="{spine_x:.1f}" y1="{spine_top:.1f}" '
            f'x2="{spine_x:.1f}" y2="{spine_bottom:.1f}" '
            f'stroke="{COL_LINE}" stroke-width="1.4"/>'
        )

        for i, (role, person) in enumerate(people):
            py = first_person_y + i * PERSON_GAP_Y
            px = lane_center - PERSON_W / 2
            # Horizontal stub from spine to the top-center of the box
            box_center_y = py + PERSON_H / 2
            parts.append(
                f'<line x1="{spine_x:.1f}" y1="{box_center_y:.1f}" '
                f'x2="{spine_x:.1f}" y2="{box_center_y:.1f}" '
                f'stroke="{COL_LINE}" stroke-width="1.4"/>'
            )
            parts.append(_person_node(person, role, px, py))

    body = "".join(parts)
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {total_width} {total_height}" '
        f'preserveAspectRatio="xMidYMid meet" width="100%" height="100%" '
        f'style="background:{COL_BG}; border-radius:12px; display:block;">'
        f'{body}</svg>'
    )
    return svg, total_width, total_height


def _entity_node(entity, x: float, y: float) -> str:
    cx = x + ENTITY_W / 2
    name_lines = _wrap(entity.name, 26)
    sub = (entity.sub_type or entity.entity_type or "").replace("_", " ").title()

    if len(name_lines) == 1:
        name_svg = (
            f'<text x="{cx:.1f}" y="{y + 30:.1f}" text-anchor="middle" '
            f'font-family="Playfair Display, Georgia, serif" font-size="15" '
            f'font-weight="600" fill="{COL_ENTITY_TEXT}">{_escape(name_lines[0])}</text>'
        )
        sub_y = y + 50
    else:
        name_svg = (
            f'<text x="{cx:.1f}" y="{y + 24:.1f}" text-anchor="middle" '
            f'font-family="Playfair Display, Georgia, serif" font-size="14" '
            f'font-weight="600" fill="{COL_ENTITY_TEXT}">{_escape(name_lines[0])}</text>'
            f'<text x="{cx:.1f}" y="{y + 41:.1f}" text-anchor="middle" '
            f'font-family="Playfair Display, Georgia, serif" font-size="14" '
            f'font-weight="600" fill="{COL_ENTITY_TEXT}">{_escape(name_lines[1])}</text>'
        )
        sub_y = y + 58

    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{ENTITY_W}" height="{ENTITY_H}" '
        f'rx="8" ry="8" fill="{COL_ENTITY_FILL}"/>'
        f'{name_svg}'
        f'<text x="{cx:.1f}" y="{sub_y:.1f}" text-anchor="middle" '
        f'font-family="Inter, sans-serif" font-size="10" letter-spacing="1" '
        f'fill="{COL_ENTITY_SUB}">{_escape(sub.upper())}</text>'
    )


def _person_node(person, role, x: float, y: float) -> str:
    cx = x + PERSON_W / 2
    name = person.display_name
    if len(name) > 24:
        name = name[:23] + "…"
    role_txt = role_label(role.role_type)

    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{PERSON_W}" height="{PERSON_H}" '
        f'rx="7" ry="7" fill="{COL_PERSON_FILL}" stroke="{COL_PERSON_STROKE}" '
        f'stroke-width="1.4"/>'
        # role tag (gold, small, uppercase) on the first line
        f'<text x="{cx:.1f}" y="{y + 20:.1f}" text-anchor="middle" '
        f'font-family="Inter, sans-serif" font-size="9" font-weight="700" '
        f'letter-spacing="0.8" fill="{COL_ROLE_TEXT}">{_escape(role_txt.upper())}</text>'
        # name on the second line
        f'<text x="{cx:.1f}" y="{y + 40:.1f}" text-anchor="middle" '
        f'font-family="Inter, sans-serif" font-size="13" font-weight="500" '
        f'fill="{COL_PERSON_TEXT}">{_escape(name)}</text>'
    )


def _wrap(name: str, max_chars: int) -> list[str]:
    if len(name) <= max_chars:
        return [name]
    mid = len(name) // 2
    best = None
    for i, ch in enumerate(name):
        if ch == " " and (best is None or abs(i - mid) < abs(best - mid)):
            best = i
    if best is None:
        return [name[:max_chars] + "…"]
    return [name[:best], name[best + 1:]]
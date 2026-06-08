"""
Advisor shell — sidebar-nav workspace for advisors.

Replaces the tab-based layout in ui/app.py for the advisor role.

Architecture:
  - Family selector lives in the top bar — global context for every page.
  - Left sidebar is the primary nav, grouped: Overview/Chat, Family,
    Documents, Wealth, Workspace.
  - Each nav item maps to one render_*() function below.
  - Pages read st.session_state.selected_family_id to know what to display.

Navigation:
  We render ONE option_menu per section. The currently selected page lives
  in st.session_state.advisor_nav_selected. Each group has its own widget
  key — only the group containing the currently selected page shows a
  highlight; the others render unselected (default_index=0 but visually
  no highlight since their item is not the global selection).

  We rely on the option_menu's manual_select parameter to clear highlights
  on groups whose items aren't currently active.
"""

from __future__ import annotations

from typing import Optional

import streamlit as st
# from streamlit_option_menu import option_menu

from db.repositories import (
    ensure_db_user,
    list_families_for_advisor,
    get_family,
    create_family,
    list_people_in_family,
    list_entities_in_family,
    list_relationships_in_family,
    list_roles_for_entity,
)

from ui.components.family_manager import (
    _render_people_tab,
    _render_relationships_tab,
    _render_entities_tab,
    _render_roles_tab,
    _render_settings_tab,
)


# ===========================================================================
# Nav structure
# ===========================================================================

NAV_STRUCTURE = [
    {"section": None, "items": [
        ("Overview", "house"),
        ("Chat", "chat-dots"),
    ]},
    {"section": "Family", "items": [
        ("People", "people"),
        ("Relationships", "diagram-3"),
        ("Entities", "bank"),
        ("Roles", "person-badge"),
        ("Family Tree", "tree"),
    ]},
    {"section": "Documents", "items": [
        ("Documents", "file-earmark-text"),
        ("Extractions", "magic"),
        ("Document Viewer", "search"),
    ]},
    {"section": "Wealth", "items": [
        ("Balance Sheet", "cash-stack"),
        ("EstateFlow", "diagram-2"),
        ("Analytics", "graph-up"),
    ]},
    {"section": "Workspace", "items": [
        ("Team", "people-fill"),
        ("Audit Log", "clipboard-check"),
        ("Settings", "gear"),
    ]},
]


# ===========================================================================
# Public entry point
# ===========================================================================

def render_advisor_shell(user) -> None:
    advisor_db_id = ensure_db_user(user)
    _render_top_bar(user, advisor_db_id)

    with st.sidebar:
        selected = _render_sidebar_nav()

    family_id = st.session_state.get("selected_family_id")

    needs_family = {
        "Overview", "Chat", "People", "Relationships", "Entities", "Roles",
        "Family Tree", "Documents", "Extractions", "Document Viewer",
        "Balance Sheet", "EstateFlow", "Analytics", "Team", "Audit Log",
        "Settings",
    }

    if selected in needs_family and family_id is None:
        _render_no_family_selected()
        return

    page_fn = _PAGE_ROUTES.get(selected)
    if page_fn is None:
        st.error(f"Unknown page: {selected}")
        return

    page_fn(user, family_id)


# ===========================================================================
# Top bar
# ===========================================================================

def _render_top_bar(user, advisor_db_id: int) -> None:
    families = list_families_for_advisor(advisor_db_id)

    col_family, col_user, col_logout = st.columns([5, 2, 1])

    with col_family:
        if families:
            labels_to_id = {f.name: f.id for f in families}
            current_id = st.session_state.get("selected_family_id")
            if current_id is None or current_id not in labels_to_id.values():
                current_id = families[0].id
                st.session_state.selected_family_id = current_id

            label_list = list(labels_to_id.keys())
            current_label = next(
                lbl for lbl, fid in labels_to_id.items() if fid == current_id
            )
            selected_label = st.selectbox(
                "Family",
                label_list,
                index=label_list.index(current_label),
                key="top_bar_family_selector",
                label_visibility="collapsed",
            )
            new_id = labels_to_id[selected_label]
            if new_id != st.session_state.get("selected_family_id"):
                st.session_state.selected_family_id = new_id
                st.rerun()
        else:
            st.info("No families yet — create one to get started.")

    with col_user:
        name = getattr(user, "client_name", None) or user.username
        role = (
            user.role.value if hasattr(user.role, "value") else str(user.role)
        ).replace("_", " ").title()
        st.markdown(
            f"<div style='text-align:right; padding-top:6px;'>"
            f"<b>{name}</b><br>"
            f"<span style='font-size:0.8em; opacity:0.7;'>{role}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with col_logout:
        st.write("")
        if st.button("Logout", use_container_width=True, key="top_logout"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

    st.markdown(
        "<hr style='margin-top:0.5rem; margin-bottom:1rem;'>",
        unsafe_allow_html=True,
    )


# ===========================================================================
# Sidebar nav — REWRITTEN
# ===========================================================================

def _render_sidebar_nav() -> str:
    """Render all nav sections using native Streamlit buttons.

    Uses native buttons rather than streamlit-option-menu because
    option_menu maintains its own widget state per group that fights
    our global advisor_nav_selected state, causing flicker on rerun.
    """
    if "advisor_nav_selected" not in st.session_state:
        st.session_state.advisor_nav_selected = "Overview"

    current = st.session_state.advisor_nav_selected

    st.markdown("### Workspace")

    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] div.stButton > button {
            width: 100%;
            text-align: left;
            justify-content: flex-start;
            background: transparent;
            border: none;
            padding: 8px 12px;
            margin: 0;
            border-radius: 6px;
            font-size: 14px;
            font-weight: 400;
            color: inherit;
        }
        section[data-testid="stSidebar"] div.stButton > button:hover {
            background: rgba(255,255,255,0.06);
            color: inherit;
        }
        section[data-testid="stSidebar"] div.nav-active div.stButton > button {
            background: rgba(99,102,241,0.20);
            font-weight: 500;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    for group_idx, group in enumerate(NAV_STRUCTURE):
        section_name = group["section"]
        if section_name:
            st.caption(section_name.upper())

        for item_idx, (label, icon_name) in enumerate(group["items"]):
            is_active = (label == current)

            if is_active:
                st.markdown('<div class="nav-active">', unsafe_allow_html=True)

            display_label = f"{_ICON_EMOJI.get(icon_name, '-')}  {label}"

            clicked = st.button(
                display_label,
                key=f"navbtn_{group_idx}_{item_idx}",
                use_container_width=True,
            )

            if is_active:
                st.markdown('</div>', unsafe_allow_html=True)

            if clicked and not is_active:
                st.session_state.advisor_nav_selected = label
                st.rerun()

    return st.session_state.advisor_nav_selected


_ICON_EMOJI = {
    "house": "🏠",
    "chat-dots": "💬",
    "people": "👥",
    "diagram-3": "🔗",
    "bank": "🏛️",
    "person-badge": "👤",
    "tree": "🌳",
    "file-earmark-text": "📄",
    "magic": "✨",
    "search": "🔍",
    "cash-stack": "💰",
    "diagram-2": "🌊",
    "graph-up": "📊",
    "people-fill": "🤝",
    "clipboard-check": "📜",
    "gear": "⚙️",
}


# ===========================================================================
# Empty-state helpers
# ===========================================================================

def _render_no_family_selected() -> None:
    st.markdown("# Welcome 👋")
    st.markdown(
        "You haven't selected (or created) a family yet. Every workspace "
        "in the platform — chat, documents, balance sheet, estate flow — "
        "is scoped to a specific family."
    )
    st.markdown("---")
    st.subheader("Create your first family")

    with st.form("first_family_form", clear_on_submit=False):
        name = st.text_input("Family Name *", placeholder="e.g. The Smith Family")
        notes = st.text_area("Notes (optional)")
        if st.form_submit_button("Create Family", type="primary"):
            if not name.strip():
                st.error("Family name is required.")
            else:
                user = st.session_state.user
                advisor_db_id = ensure_db_user(user)
                family = create_family(
                    name=name.strip(),
                    advisor_user_id=advisor_db_id,
                    notes=notes.strip() or None,
                )
                st.session_state.selected_family_id = family.id
                st.rerun()


def _coming_soon(
    page_title: str,
    description: str,
    what_it_will_do: list[str],
    data_sources: list[str],
    target_step: str,
) -> None:
    st.markdown(f"# {page_title}")
    st.caption(description)
    st.info(f"🚧 Coming in **{target_step}**")

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("What this will do")
        for item in what_it_will_do:
            st.markdown(f"- {item}")
    with col_right:
        st.subheader("Data it'll use")
        for item in data_sources:
            st.markdown(f"- `{item}`")


# ===========================================================================
# Page renderers
# ===========================================================================

def _render_overview(user, family_id: int) -> None:
    family = get_family(family_id)
    if family is None:
        st.error("Selected family not found.")
        return

    people = list_people_in_family(family_id)
    entities = list_entities_in_family(family_id)
    relationships = list_relationships_in_family(family_id)

    st.markdown(f"# {family.name}")
    if family.notes:
        st.caption(family.notes)

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("👥 People", len(people))
    col2.metric("🏛️ Entities", len(entities))
    col3.metric("🔗 Relationships", len(relationships))
    total_roles = sum(len(list_roles_for_entity(e.id)) for e in entities)
    col4.metric("👤 Roles", total_roles)

    st.markdown("---")

    col_people, col_entities = st.columns(2)

    with col_people:
        st.subheader("👥 People")
        if people:
            for p in people[:8]:
                tag = " ⚰️" if p.is_deceased else ""
                st.markdown(f"**{p.display_name}**{tag}")
                sub = []
                if p.dob:
                    sub.append(f"DOB {p.dob.isoformat()}")
                if p.email:
                    sub.append(p.email)
                if sub:
                    st.caption(" · ".join(sub))
            if len(people) > 8:
                st.caption(
                    f"…and {len(people) - 8} more. Open People for the full list."
                )
        else:
            st.info(
                "No people in this family yet. Open **People** to add some."
            )

    with col_entities:
        st.subheader("🏛️ Entities")
        if entities:
            for e in entities[:8]:
                sub = e.sub_type or e.entity_type
                if e.jurisdiction:
                    sub += f" · {e.jurisdiction}"
                st.markdown(f"**{e.name}**")
                st.caption(sub)
            if len(entities) > 8:
                st.caption(f"…and {len(entities) - 8} more.")
        else:
            st.info(
                "No entities (trusts, LLCs) in this family yet. "
                "Open **Entities** to add some."
            )

    st.markdown("---")
    with st.container(border=True):
        st.markdown("### 🤖 Advisor Copilot")
        st.caption(
            "Coming in Step 7+. This panel will surface AI-generated "
            "observations across this family — missing beneficiary "
            "designations, stale trust amendments, gift tax exposure, "
            "opportunities for estate strategies — based on documents "
            "and structure already loaded."
        )


def _render_chat(user, family_id: int) -> None:
    st.markdown("# 💬 Chat")
    st.caption(
        "Ask questions across this family's documents. Family-scoped "
        "chat lands when documents are linked to the family graph (Step 3b)."
    )
    st.markdown("---")
    st.info(
        "**Bridging note:** chat is currently driven by the original "
        "advisor-client model. We'll migrate it to read from the family "
        "graph once Documents are wired in."
    )


def _render_people(user, family_id: int) -> None:
    st.markdown("# 👥 People")
    st.caption("Family members. Encrypted SSN, deceased tracking, contact details.")
    st.markdown("---")
    _render_people_tab(family_id)


def _render_relationships(user, family_id: int) -> None:
    st.markdown("# 🔗 Relationships")
    st.caption("How family members are related to each other.")
    st.markdown("---")
    _render_relationships_tab(family_id)


def _render_entities(user, family_id: int) -> None:
    st.markdown("# 🏛️ Entities")
    st.caption("Trusts, LLCs, partnerships, corporations, foundations.")
    st.markdown("---")
    _render_entities_tab(family_id)


def _render_roles(user, family_id: int) -> None:
    st.markdown("# 👤 Roles")
    st.caption("Who plays what role in which entity.")
    st.markdown("---")
    _render_roles_tab(family_id)


def _render_family_tree(user, family_id: int) -> None:
    _coming_soon(
        page_title="🌳 Family Tree",
        description="Visual map of family relationships across generations.",
        what_it_will_do=[
            "Interactive node-link diagram of people and relationships",
            "Color-coding for living / deceased, spouses, lineage",
            "Click a person to jump to their People page entry",
            "Show entity ownership lines (who is trustee/beneficiary of what)",
            "Export as PNG/SVG for client presentations",
        ],
        data_sources=[
            "people (the nodes)",
            "relationships (the edges)",
            "roles (entity ownership overlay)",
        ],
        target_step="Step 6 — this is the page that pushes us past Streamlit",
    )


def _render_documents(user, family_id: int) -> None:
    _coming_soon(
        page_title="📄 Documents",
        description="Every document tied to this family, indexed and searchable.",
        what_it_will_do=[
            "Upload PDFs with metadata (doc type, year, who/what it belongs to)",
            "Tag documents to a specific person or entity in the family",
            "Auto-detect duplicates via file hash",
            "Trigger reindexing into Chroma after upload",
            "Show extraction status (pending / processing / complete / failed)",
        ],
        data_sources=[
            "documents (DB row per file)",
            "data/advisors/.../*.pdf (files on disk)",
            "vectorstore/chroma_db (embeddings)",
        ],
        target_step="Step 3b — next implementation",
    )


def _render_extractions(user, family_id: int) -> None:
    _coming_soon(
        page_title="✨ Extractions",
        description="Structured facts the AI has pulled from documents.",
        what_it_will_do=[
            "Show every extracted SSN, EIN, role, date, amount per document",
            "Link each fact to its source page + bounding box",
            "Advisor can verify, edit, or reject extractions",
            "Verified facts feed Balance Sheet, EstateFlow, and the family tree overlay",
            "Audit trail of who verified what when",
        ],
        data_sources=[
            "extractions (one row per fact)",
            "documents (provenance)",
            "people / entities (the targets being enriched)",
        ],
        target_step="Step 4 — depends on Documents (3b) first",
    )


def _render_doc_viewer(user, family_id: int) -> None:
    _coming_soon(
        page_title="🔍 Document Viewer",
        description="Read a document with citations and highlighted extractions.",
        what_it_will_do=[
            "Render PDF pages in-app (no download required)",
            "Click an extracted fact → jump to that page with the source text highlighted",
            "Side panel listing every extraction on the visible page",
            "Search within the document",
            "Page-level annotations advisors can add",
        ],
        data_sources=[
            "documents (the PDF file)",
            "extractions (bbox coordinates per fact)",
        ],
        target_step="Step 5 — the moment the product feels like a platform, not a chatbot",
    )


def _render_balance_sheet(user, family_id: int) -> None:
    _coming_soon(
        page_title="💰 Balance Sheet",
        description="Aggregated asset view across the family.",
        what_it_will_do=[
            "Roll up assets and liabilities across every entity in the family",
            "Inside-estate vs outside-estate breakdown",
            "Group by asset class, by entity, or by beneficial owner",
            "Track changes over time (snapshots)",
            "Export to PDF for client-ready reports",
        ],
        data_sources=[
            "entities + roles (ownership chains)",
            "extractions (asset values from statements & tax returns)",
            "manual entries (advisor-edited fields)",
        ],
        target_step="Step 7 — needs Extractions (Step 4) populated first",
    )


def _render_estateflow(user, family_id: int) -> None:
    _coming_soon(
        page_title="🌊 EstateFlow",
        description="How wealth moves across people, entities, and generations.",
        what_it_will_do=[
            "Sankey-style diagram of asset flow at death/incapacity",
            "Show beneficiary chains across multiple trusts",
            "Scenario testing: 'what if X predeceases Y'",
            "Highlight tax exposure at each transfer point",
            "Compare with/without proposed planning changes",
        ],
        data_sources=[
            "entities (containers)",
            "roles (beneficiary designations)",
            "relationships (heirs)",
            "extractions (distribution terms from trust docs)",
        ],
        target_step="Step 8 — the headline visualization. Likely a Next.js page.",
    )


def _render_analytics(user, family_id: int) -> None:
    st.markdown("# 📊 Analytics")
    st.caption("Usage analytics for this family / your book.")
    st.markdown("---")
    try:
        from ui.components.analytics_dashboard import render_analytics_dashboard
        render_analytics_dashboard()
    except ImportError:
        st.warning("Analytics dashboard component not found.")


def _render_team(user, family_id: int) -> None:
    _coming_soon(
        page_title="🤝 Team",
        description="Co-advisors, attorneys, and CPAs collaborating on this family.",
        what_it_will_do=[
            "Invite another advisor, attorney, or CPA to this family",
            "Grant viewer / editor / owner access",
            "See who currently has access and when it was granted",
            "Revoke access; track all access changes in the audit log",
            "Shared comments / notes per document or entity",
        ],
        data_sources=[
            "users (the invitees)",
            "user_family_access (grants — schema already exists)",
        ],
        target_step="Step 9 — collaboration unlocks the multi-disciplinary workflow",
    )


def _render_audit_log(user, family_id: int) -> None:
    _coming_soon(
        page_title="📜 Audit Log",
        description="Who saw what PII, when. Compliance-grade access record.",
        what_it_will_do=[
            "Log every SSN/EIN view with user + timestamp",
            "Log every document view, edit, extraction verification",
            "Filterable by user, date range, action type",
            "Export for compliance review (SOC 2 / RIA audits)",
            "Tamper-evident hashes on each entry",
        ],
        data_sources=[
            "pii_access_log (new table)",
            "users (who did it)",
            "documents / people / entities (the targets)",
        ],
        target_step="Step 10 — required before any real RIA deployment",
    )


def _render_settings(user, family_id: int) -> None:
    family = get_family(family_id)
    if family is None:
        st.error("Selected family not found.")
        return
    st.markdown("# ⚙️ Family Settings")
    st.markdown("---")
    _render_settings_tab(family)


# ===========================================================================
# Route table
# ===========================================================================

_PAGE_ROUTES = {
    "Overview": _render_overview,
    "Chat": _render_chat,
    "People": _render_people,
    "Relationships": _render_relationships,
    "Entities": _render_entities,
    "Roles": _render_roles,
    "Family Tree": _render_family_tree,
    "Documents": _render_documents,
    "Extractions": _render_extractions,
    "Document Viewer": _render_doc_viewer,
    "Balance Sheet": _render_balance_sheet,
    "EstateFlow": _render_estateflow,
    "Analytics": _render_analytics,
    "Team": _render_team,
    "Audit Log": _render_audit_log,
    "Settings": _render_settings,
}
"""
Angel — advisor workspace.

Restructured to match the product spec exactly:

  Top-level advisor nav (left sidebar):
      🏠  Families            — list of all clients
      ✓   My Tasks            — tasks assigned across all families
      🤝  Partnership Circle  — chat with advisory team
      📊  Reports             — generated reports / summaries
      💬  Chat History        — Angel AI past sessions
      ↩   Logout

  Inside a selected family, the right pane shows the per-family nav:
      ✓   Tasks
      👥  Key People & Orgs
      🌳  Family Tree
      🏢  Organizational Ownership
      🤝  Advisory Team
      📁  Documents

  A chat bar lives near the top of every page.

Architecture:
  - Top-level nav lives in the left sidebar.
  - Per-family nav appears as a secondary strip when a family is selected.
  - Each leaf maps to one render_*() function.
  - "Coming soon" stubs describe what will live there until built.
"""

from __future__ import annotations

from typing import Optional

import streamlit as st

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
from ui.components.family_tree import render_family_tree
from ui.components.advisory_team import render_advisory_team_page
from ui.components.tasks import render_tasks_page, render_my_tasks_page
from ui.components.documents import render_documents_page
from ui.components.org_ownership import render_org_ownership
from ui.theme import inject_theme, render_brand_header, Color


# ─────────────────────────────────────────────────────────────────────
# Nav structure
# ─────────────────────────────────────────────────────────────────────

# Top-level advisor nav (always visible in left sidebar)
TOP_NAV = [
    ("Families", "🏠"),
    ("My Tasks", "✓"),
    ("Partnership Circle", "🤝"),
    ("Reports", "📊"),
    ("Angel", "💬"),
]

# Per-family secondary nav (visible when a family is selected)
FAMILY_NAV = [
    ("Overview", "🏛️"),
    ("Tasks", "✓"),
    ("Key People & Orgs", "👥"),
    ("Family Tree", "🌳"),
    ("Org Ownership", "🏢"),
    ("Advisory Team", "🤝"),
    ("Documents", "📁"),
]


# ─────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────

def render_advisor_shell(user) -> None:
    """Top-level advisor experience."""
    inject_theme()
    advisor_db_id = ensure_db_user(user)

    # ---- Left sidebar: brand + top-level nav ----
    with st.sidebar:
        render_brand_header()
        selected_top = _render_top_nav()
        _render_user_card(user)

    # ---- Main content area ----
    # _render_chat_bar(user)

    if selected_top == "Families":
        _render_families_view(advisor_db_id)
    elif selected_top == "My Tasks":
        _render_my_tasks_view(user)
    elif selected_top == "Partnership Circle":
        _render_partnership_circle_view(user)
    elif selected_top == "Reports":
        _render_reports_view(user)
    elif selected_top == "Angel":
        _render_chat_history_view(user)


# ─────────────────────────────────────────────────────────────────────
# Sidebar — top-level nav
# ─────────────────────────────────────────────────────────────────────

def _render_top_nav() -> str:
    """Render the top-level advisor nav. Returns selected label."""
    if "advisor_top_nav" not in st.session_state:
        st.session_state.advisor_top_nav = "Families"

    current = st.session_state.advisor_top_nav

    for label, icon in TOP_NAV:
        is_active = (label == current)

        if is_active:
            st.markdown('<div class="nav-active">', unsafe_allow_html=True)
        clicked = st.button(
            f"{icon}   {label}",
            key=f"topnav_{label}",
            use_container_width=True,
        )
        if is_active:
            st.markdown('</div>', unsafe_allow_html=True)

        if clicked and not is_active:
            st.session_state.advisor_top_nav = label
            # Reset family selection when leaving the Families view
            if label != "Families":
                st.session_state.show_family_detail = False
            st.rerun()

    st.markdown("---")
    return st.session_state.advisor_top_nav


def _render_user_card(user) -> None:
    """Bottom of the sidebar: user identity + logout."""
    st.markdown("---")
    name = getattr(user, "client_name", None) or user.username
    role = (
        user.role.value if hasattr(user.role, "value") else str(user.role)
    ).replace("_", " ").title()

    st.markdown(
        f"""
        <div style='padding: 8px 14px; line-height: 1.4;'>
          <div style='font-weight: 600; font-size: 0.9rem; color: {Color.TEXT_ON_DARK};'>{name}</div>
          <div style='font-size: 0.75rem; color: {Color.TEXT_MUTED_ON_DARK};'>{role}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("↩  Logout", key="sidebar_logout", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


# ─────────────────────────────────────────────────────────────────────
# Top chat bar — Angel AI question input
# ─────────────────────────────────────────────────────────────────────

def _render_chat_bar(user) -> None:
    """The 'ask Angel anything' input that sits at the top of every page."""
    _, col_chat, _ = st.columns([1, 5, 1])
    with col_chat:
        st.text_input(
            "Ask Angel",
            placeholder="✨ Ask Angel anything about your families or documents…",
            label_visibility="collapsed",
            key="advisor_chatbar",
        )
    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# Families view — list of clients, then drill-down
# ─────────────────────────────────────────────────────────────────────

def _render_families_view(advisor_db_id: int) -> None:
    show_detail = st.session_state.get("show_family_detail", False)
    selected_family_id = st.session_state.get("selected_family_id")

    if show_detail and selected_family_id:
        _render_family_detail(selected_family_id)
    else:
        _render_families_list(advisor_db_id)


def _render_families_list(advisor_db_id: int) -> None:
    """Card grid of every family this advisor manages."""
    st.markdown("# Families")
    st.caption("Your families and their key information at a glance.")
    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

    families = list_families_for_advisor(advisor_db_id)

    col_search, col_new = st.columns([4, 1])
    with col_search:
        st.text_input(
            "Search families",
            placeholder="Search by family name…",
            label_visibility="collapsed",
            key="family_search",
        )
    with col_new:
        if st.button("➕ New Family", use_container_width=True, type="primary"):
            st.session_state.show_new_family_form = True

    if st.session_state.get("show_new_family_form", False):
        with st.expander("Create New Family", expanded=True):
            with st.form("new_family_form_v2", clear_on_submit=True):
                name = st.text_input(
                    "Family Name *",
                    placeholder="e.g. The Smith Family",
                )
                notes = st.text_area("Notes")
                col_a, col_b = st.columns(2)
                with col_a:
                    submitted = st.form_submit_button(
                        "Create", use_container_width=True, type="primary"
                    )
                with col_b:
                    cancelled = st.form_submit_button(
                        "Cancel", use_container_width=True
                    )

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
                        st.session_state.show_family_detail = True
                        st.session_state.show_new_family_form = False
                        st.rerun()
                elif cancelled:
                    st.session_state.show_new_family_form = False
                    st.rerun()

    # Apply search filter
    query = (st.session_state.get("family_search") or "").strip().lower()
    if query:
        families = [f for f in families if query in f.name.lower()]

    st.markdown(
        "<div style='height:0.5rem; border-bottom:1px solid "
        "rgba(201,169,97,0.12); margin-bottom:1.5rem;'></div>",
        unsafe_allow_html=True,
    )

    if not families:
        st.info(
            "No families to show. Click **➕ New Family** above to start "
            "onboarding your first client."
        )
        return

    # Grid of family cards (2 per row)
    for i in range(0, len(families), 2):
        items_this_row = min(2, len(families) - i)
        cols = st.columns(2)
        for j in range(items_this_row):
            with cols[j]:
                _render_family_card(families[i + j])


def _render_family_card(family) -> None:
    """One family card — calm, borderless stats, gold accent."""
    people = list_people_in_family(family.id)
    entities = list_entities_in_family(family.id)
    doc_count = 0
    try:
        from db.repositories import list_documents_for_family
        doc_count = len(list_documents_for_family(family.id))
    except Exception:
        doc_count = 0

    sub = ""
    if family.notes:
        sub = family.notes[:90] + ("…" if len(family.notes) > 90 else "")

    st.markdown(
        f"""
        <div class="angel-fam-card">
          <div class="angel-fam-accent"></div>
          <div class="angel-fam-body">
            <div class="angel-fam-name">{family.name}</div>
            <div class="angel-fam-sub">{sub or "&nbsp;"}</div>
            <div class="angel-fam-stats">
              <div class="angel-fam-stat">
                <div class="angel-fam-stat-label">People</div>
                <div class="angel-fam-stat-value">{len(people)}</div>
              </div>
              <div class="angel-fam-stat">
                <div class="angel-fam-stat-label">Entities</div>
                <div class="angel-fam-stat-value">{len(entities)}</div>
              </div>
              <div class="angel-fam-stat">
                <div class="angel-fam-stat-label">Documents</div>
                <div class="angel-fam-stat-value">{doc_count}</div>
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Real Streamlit button, styled gold + flush via the wrapper class
    st.markdown('<div class="angel-fam-open">', unsafe_allow_html=True)
    if st.button("Open →", key=f"open_fam_{family.id}",
                 use_container_width=True):
        st.session_state.selected_family_id = family.id
        st.session_state.show_family_detail = True
        st.session_state.family_detail_section = "Overview"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# Family detail — secondary nav appears
# ─────────────────────────────────────────────────────────────────────

def _render_family_detail(family_id: int) -> None:
    family = get_family(family_id)
    if family is None:
        st.session_state.show_family_detail = False
        st.session_state.selected_family_id = None
        st.rerun()

    # Breadcrumb
    col_back, col_title = st.columns([1, 9])
    with col_back:
        if st.button("← Families", key="back_to_families"):
            st.session_state.show_family_detail = False
            st.rerun()
    with col_title:
        st.markdown(f"# {family.name}")

    # Secondary nav strip
    if "family_detail_section" not in st.session_state:
        st.session_state.family_detail_section = "Overview"

    current_section = st.session_state.family_detail_section

    nav_cols = st.columns(len(FAMILY_NAV))
    for col, (label, icon) in zip(nav_cols, FAMILY_NAV):
        with col:
            is_active = (label == current_section)
            if is_active:
                st.markdown('<div class="nav-active">', unsafe_allow_html=True)
            clicked = st.button(
                f"{icon}  {label}",
                key=f"famnav_{label}",
                use_container_width=True,
            )
            if is_active:
                st.markdown('</div>', unsafe_allow_html=True)

            if clicked and not is_active:
                st.session_state.family_detail_section = label
                st.rerun()

    st.markdown("---")

    # Route within family detail
    if current_section == "Overview":
        _render_family_overview(family, family_id)
    elif current_section == "Tasks":
        _render_tasks_family_scope(family_id)
    elif current_section == "Key People & Orgs":
        _render_key_people_orgs(family_id)
    elif current_section == "Family Tree":
        _render_family_tree_page(family_id)
    elif current_section == "Org Ownership":
        _render_org_ownership(family_id)
    elif current_section == "Advisory Team":
        _render_advisory_team(family_id)
    elif current_section == "Documents":
        _render_documents_family_scope(family_id)


def _render_family_overview(family, family_id: int) -> None:
    """At-a-glance dashboard for the selected family."""
    if family.notes:
        st.caption(family.notes)

    people = list_people_in_family(family_id)
    entities = list_entities_in_family(family_id)
    relationships = list_relationships_in_family(family_id)
    total_roles = sum(len(list_roles_for_entity(e.id)) for e in entities)

    _overview_stat_strip(
        len(people), len(entities), len(relationships), total_roles
    )

    st.markdown("---")
    col_people, col_entities = st.columns(2)

    with col_people:
        st.markdown("### People")
        if people:
            for p in people[:6]:
                tag = " ⚰️" if p.is_deceased else ""
                st.markdown(f"**{p.display_name}**{tag}")
                if p.dob:
                    st.caption(f"DOB {p.dob.isoformat()}")
            if len(people) > 6:
                st.caption(f"…and {len(people) - 6} more.")
        else:
            st.caption("_None yet. Add via Key People & Orgs._")

    with col_entities:
        st.markdown("### Entities")
        if entities:
            for e in entities[:6]:
                sub = e.sub_type or e.entity_type
                if e.jurisdiction:
                    sub += f" · {e.jurisdiction}"
                st.markdown(f"**{e.name}**")
                st.caption(sub)
            if len(entities) > 6:
                st.caption(f"…and {len(entities) - 6} more.")
        else:
            st.caption("_None yet. Add via Key People & Orgs._")

    st.markdown("---")
    with st.container(border=True):
        st.markdown("### ✨ Angel Copilot")
        st.caption(
            "Soon: AI-surfaced observations for this family — missing "
            "beneficiary designations, stale trust amendments, gift tax "
            "exposure, planning opportunities. Powered by your local LLM."
        )


def _overview_stat_strip(people_n, entities_n, rel_n, roles_n) -> None:
    st.markdown(
        f"""
        <div class="angel-stat-strip">
          <div class="angel-stat-cell"><div class="lbl">People</div>
            <div class="val">{people_n}</div></div>
          <div class="angel-stat-cell"><div class="lbl">Entities</div>
            <div class="val">{entities_n}</div></div>
          <div class="angel-stat-cell"><div class="lbl">Relationships</div>
            <div class="val">{rel_n}</div></div>
          <div class="angel-stat-cell"><div class="lbl">Roles</div>
            <div class="val">{roles_n}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_key_people_orgs(family_id: int) -> None:
    """The combined People + Entities + Relationships + Roles view."""
    st.markdown("### Key People & Organizations")
    st.caption(
        "Family members, business entities, charitable entities, and the "
        "relationships and roles connecting them."
    )

    sub_people, sub_rel, sub_ent, sub_roles = st.tabs(
        ["👥 People", "🔗 Relationships", "🏛️ Entities", "👤 Roles"]
    )
    with sub_people:
        _render_people_tab(family_id)
    with sub_rel:
        _render_relationships_tab(family_id)
    with sub_ent:
        _render_entities_tab(family_id)
    with sub_roles:
        _render_roles_tab(family_id)


# ─────────────────────────────────────────────────────────────────────
# Coming-soon page renderer
# ─────────────────────────────────────────────────────────────────────

def _coming_soon(
    title: str,
    description: str,
    will_do: list[str],
    phase: str,
) -> None:
    st.markdown(f"### {title}")
    st.caption(description)

    st.info(f"🚧 Coming in **{phase}**")

    st.markdown("**This page will let you:**")
    for item in will_do:
        st.markdown(f"- {item}")


# Per-family stubs
def _render_tasks_family_scope(family_id: int) -> None:
    user = st.session_state.get("user")
    render_tasks_page(family_id, user)


def _render_family_tree_page(family_id: int) -> None:
    render_family_tree(family_id)


def _render_org_ownership(family_id: int) -> None:
    render_org_ownership(family_id)


def _render_advisory_team(family_id: int) -> None:
    render_advisory_team_page(family_id)


def _render_documents_family_scope(family_id: int) -> None:
    user = st.session_state.get("user")
    render_documents_page(family_id, user)


# Top-level stubs
def _render_my_tasks_view(user) -> None:
    render_my_tasks_page(user)


def _render_partnership_circle_view(user) -> None:
    _coming_soon(
        title="Partnership Circle",
        description="Direct collaboration with the advisory team across all families.",
        will_do=[
            "Direct-message any team member (attorney, CPA, insurance broker, etc.)",
            "Create project threads scoped to a strategy or family",
            "Share documents in conversations (with access control)",
            "Tag colleagues to pull them into a thread",
            "See all your active conversations and projects in one place",
        ],
        phase="Phase 6 — Collaboration",
    )


def _render_reports_view(user) -> None:
    _coming_soon(
        title="Reports",
        description="Generate polished reports and summaries from family data.",
        will_do=[
            "Generate an Estate Report (à la Wealth.com format) per family",
            "Balance Sheet roll-up across every entity",
            "Estate Plan Flow Chart (death-of-John / death-of-Jane scenarios)",
            "Decision Makers summary (trustees, executors, agents)",
            "Custom reports filtered by date, family, or document type",
            "Export to PDF, ready to send to clients",
        ],
        phase="Phase 5 — Reports & Visualizations",
    )


def _render_chat_history_view(user) -> None:
    """Advisor Chat History — pick a family, then chat about its documents."""
    from ui.chat_bridge import (
        ensure_chat_session_state,
        get_or_build_qa,
        load_chat_for_context,
        save_chat_for_context,
        chat_context_key,
    )
    from ui.components.chat_interface import render_chat_interface
    from db.repositories import list_families_for_advisor, ensure_db_user

    ensure_chat_session_state()
    advisor_db_id = ensure_db_user(user)

    st.markdown("# Chat")
    st.caption(
        "Ask Angel about any of your families' documents. Pick a family "
        "below — chat is scoped to that family's library."
    )
    st.markdown("---")

    families = list_families_for_advisor(advisor_db_id)
    if not families:
        st.info("You have no families yet. Create one first.")
        return

    family_options = {f.name: f.id for f in families}
    labels = list(family_options.keys())

    # Default to the family currently selected elsewhere in the UI
    existing_family_id = st.session_state.get("selected_family_id")
    default_idx = 0
    if existing_family_id is not None:
        for i, label in enumerate(labels):
            if family_options[label] == existing_family_id:
                default_idx = i
                break

    picked_label = st.selectbox(
        "Chat about which family?",
        labels,
        index=default_idx,
    )
    family_id = family_options[picked_label]

    # When the user switches families, invalidate the cached QA + history
    prev_family_id = st.session_state.get("chat_active_family_id")
    if prev_family_id != family_id:
        if prev_family_id is not None:
            save_chat_for_context(user, family_id=prev_family_id)
        st.session_state.chat_active_family_id = family_id
        st.session_state.qa_system = None
        st.session_state.qa_owner = None
        load_chat_for_context(user, family_id=family_id)
        st.rerun()

    expected_key = chat_context_key(user, family_id=family_id)
    if st.session_state.get("current_chat_id") != expected_key:
        load_chat_for_context(user, family_id=family_id)

    qa_system = get_or_build_qa(user, family_id=family_id)

    if qa_system is None:
        st.warning("Could not initialize family AI.")
        return

    if qa_system.vector_store is None:
        st.info(
            f"📭 No documents have been indexed for {picked_label} yet. "
            f"Upload documents in the Documents tab first."
        )
        return

    if st.button("➕ New Chat", key="advisor_new_chat"):
        st.session_state.chat_history = []
        key = chat_context_key(user, family_id=family_id)
        if key in st.session_state.chat_histories:
            del st.session_state.chat_histories[key]
        st.session_state.current_chat_id = None
        st.rerun()

    render_chat_interface(user, qa_system)
    save_chat_for_context(user, family_id=family_id)
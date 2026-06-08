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
    ("Chat History", "💬"),
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
    elif selected_top == "Chat History":
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
    # with col_label:
    #     st.markdown(
    #         f"<div style='text-align:right; padding-top:8px; "
    #         f"font-size:0.8rem; color:{Color.GOLD_500};'>"
    #         f"✨ Angel AI"
    #         f"</div>",
    #         unsafe_allow_html=True,
    #     )
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
        if st.button("➕ New Family", use_container_width=True):
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

    st.markdown("---")

    if not families:
        st.info(
            "You haven't added any families yet. Click **➕ New Family** "
            "above to start onboarding your first client."
        )
        return

    # Grid of family cards (2 per row, no phantom right column on odd counts)
    for i in range(0, len(families), 2):
        # Only create as many columns as we have cards in this row
        items_this_row = min(2, len(families) - i)
        if items_this_row == 1:
            # Render in left half only — keeps card the same width as the
            # 2-up layout instead of stretching across the full page.
            left, right = st.columns(2)
            with left:
                _render_family_card(families[i])
        else:
            cols = st.columns(2)
            for j, col in enumerate(cols):
                with col:
                    _render_family_card(families[i + j])


def _render_family_card(family) -> None:
    """One family card in the list view."""
    people = list_people_in_family(family.id)
    entities = list_entities_in_family(family.id)

    with st.container(border=True):
        st.markdown(f"### {family.name}")
        if family.notes:
            st.caption(family.notes[:120] + ("…" if len(family.notes) > 120 else ""))

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("People", len(people))
        col_b.metric("Entities", len(entities))
        col_c.metric("Documents", 0)  # placeholder until Documents wired

        if st.button(
            "Open →",
            key=f"open_fam_{family.id}",
            use_container_width=True,
        ):
            st.session_state.selected_family_id = family.id
            st.session_state.show_family_detail = True
            st.session_state.family_detail_section = "Overview"
            st.rerun()


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

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("People", len(people))
    c2.metric("Entities", len(entities))
    c3.metric("Relationships", len(relationships))
    c4.metric("Roles", total_roles)

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
    # people = list_people_in_family(family_id)
    # relationships = list_relationships_in_family(family_id)

    # st.markdown("### Family Tree")
    # st.caption(
    #     "Visual map of family relationships across generations. "
    #     "Pulled from People & Relationships defined under Key People & Orgs."
    # )

    # if not people:
    #     st.info("Add people in **Key People & Orgs → People** to see them here.")
    #     return

    # st.info(
    #     f"🚧 Editorial family tree visualization coming in **Phase 3**. "
    #     f"Today: {len(people)} people, {len(relationships)} relationships "
    #     f"in this family's graph. The polished tree (like the Anderson "
    #     f"family mockup) will land once we move past Streamlit for "
    #     f"this page."
    # )

    # # Lightweight preview: list people grouped by deceased/living
    # living = [p for p in people if not p.is_deceased]
    # deceased = [p for p in people if p.is_deceased]

    # col_l, col_d = st.columns(2)
    # with col_l:
    #     st.markdown("**Living**")
    #     for p in living:
    #         st.markdown(f"- {p.display_name}")
    # with col_d:
    #     st.markdown("**Deceased**")
    #     for p in deceased:
    #         st.markdown(f"- {p.display_name} ⚰️")


def _render_org_ownership(family_id: int) -> None:
    _coming_soon(
        title="Organizational Ownership",
        description="A flow-tree view of who owns what across the family's entities.",
        will_do=[
            "Show ownership percentages between people and entities",
            "Show entity-to-entity ownership (parent / subsidiary chains)",
            "Color-code by entity type (trust, LLC, corp, foundation)",
            "Click any entity to jump to its detail page",
            "Export the chart as PNG for client meetings",
        ],
        phase="Phase 3 — Family Tree work (next visual phase)",
    )


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
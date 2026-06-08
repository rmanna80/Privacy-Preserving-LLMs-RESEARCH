"""
Angel — client portal.

Separate experience from the advisor side. Clients (the family themselves)
see their own data with read-only or limited edit permissions and a
simpler nav surface.

Nav per the product spec:
  ✓   Tasks
  👥  Key People & Organizations
  🌳  Family Tree
  🏢  Organizational Ownership
  🤝  Advisory Team
  📁  Documents
  💬  Chat History
  ↩   Logout

Plus the same Angel AI chat bar at the top.

A client is scoped to ONE family (the family they belong to). No family
selector — that's an advisor concept.
"""

from __future__ import annotations

import streamlit as st

from db.repositories import (
    list_families_for_advisor,
    ensure_db_user,
    get_family,
    list_people_in_family,
    list_entities_in_family,
)
from ui.theme import inject_theme, render_brand_header, Color


CLIENT_NAV = [
    ("Tasks", "✓"),
    ("Key People & Orgs", "👥"),
    ("Family Tree", "🌳"),
    ("Org Ownership", "🏢"),
    ("Advisory Team", "🤝"),
    ("Documents", "📁"),
    ("Chat History", "💬"),
]


def render_client_portal(user) -> None:
    """Top-level client experience."""
    inject_theme()

    family_id = _resolve_client_family(user)

    with st.sidebar:
        render_brand_header()
        selected = _render_client_nav()
        _render_user_card(user)

    # _render_chat_bar(user)

    if family_id is None:
        _render_no_family_assigned(user)
        return

    family = get_family(family_id)
    if family is None:
        _render_no_family_assigned(user)
        return

    st.markdown(f"# {family.name}")

    if selected == "Tasks":
        _coming_soon(
            "Tasks",
            "Tasks your advisory team has assigned to you and your family.",
            [
                "See every task assigned to you by your advisors",
                "Add comments, upload requested documents",
                "Mark tasks complete and track progress",
                "Get reminders for upcoming due dates",
            ],
        )
    elif selected == "Key People & Orgs":
        _render_key_people_readonly(family_id)
    elif selected == "Family Tree":
        _coming_soon(
            "Family Tree",
            "Visual map of your family across generations.",
            [
                "See your family tree as an editorial visualization",
                "Click any person to see their key details",
                "Print or export for family records",
            ],
        )
    elif selected == "Org Ownership":
        _coming_soon(
            "Organizational Ownership",
            "How your family's entities are owned and connected.",
            [
                "Flow chart of ownership across trusts, LLCs, and other entities",
                "See who benefits from what",
                "Understand the structure your advisors have built",
            ],
        )
    elif selected == "Advisory Team":
        _coming_soon(
            "Advisory Team",
            "The professionals helping your family.",
            [
                "See your full advisory team at a glance",
                "Contact information for each advisor",
                "Visual hub-and-spoke layout showing who's helping with what",
            ],
        )
    elif selected == "Documents":
        _coming_soon(
            "Documents",
            "Your family's documents, organized.",
            [
                "Browse documents by category (Investments, Estate, Tax, "
                "Insurance, Business)",
                "Upload new documents your advisor has requested",
                "Search across all your documents",
                "Chat with Angel AI about any of them",
            ],
        )
    elif selected == "Chat History":
        _render_client_chat(user, family_id)


# ─────────────────────────────────────────────────────────────────────
# Nav, top bar, identity
# ─────────────────────────────────────────────────────────────────────

def _render_client_nav() -> str:
    if "client_nav" not in st.session_state:
        st.session_state.client_nav = "Family Tree"

    current = st.session_state.client_nav

    for label, icon in CLIENT_NAV:
        is_active = (label == current)
        if is_active:
            st.markdown('<div class="nav-active">', unsafe_allow_html=True)
        clicked = st.button(
            f"{icon}   {label}",
            key=f"clientnav_{label}",
            use_container_width=True,
        )
        if is_active:
            st.markdown('</div>', unsafe_allow_html=True)

        if clicked and not is_active:
            st.session_state.client_nav = label
            st.rerun()

    return st.session_state.client_nav


def _render_user_card(user) -> None:
    st.markdown("---")
    name = getattr(user, "client_name", None) or user.username

    st.markdown(
        f"""
        <div style='padding: 8px 14px; line-height: 1.4;'>
          <div style='font-weight: 600; font-size: 0.9rem; color: {Color.TEXT_ON_DARK};'>{name}</div>
          <div style='font-size: 0.75rem; color: {Color.TEXT_MUTED_ON_DARK};'>Family Member</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("↩  Logout", key="client_sidebar_logout", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


def _render_chat_bar(user) -> None:
    _, col_chat, _ = st.columns([1, 5, 1])
    with col_chat:
        st.text_input(
            "Ask Angel",
            placeholder="✨ Ask Angel anything about your family or documents…",
            label_visibility="collapsed",
            key="client_chatbar",
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
# Family resolution for clients
# ─────────────────────────────────────────────────────────────────────

def _resolve_client_family(user) -> int | None:
    """Find which family a client user belongs to.

    Today: stub. A client's link to a family will be wired in Phase 1
    when we extend the schema with Person ↔ User association.

    For now we pick the first family the system can find, which lets
    clients at least see the portal shell with real data. Once Phase 1
    lands, replace this with a proper join.
    """
    # If session already has it from auth or a prior interaction
    if st.session_state.get("client_family_id"):
        return st.session_state.client_family_id

    # Stub: ask the DB for ANY family the system knows about, just so the
    # demo portal has data. Replace with real client→family lookup later.
    from db.database import get_session
    from db.models import Family
    from sqlmodel import select

    with get_session() as s:
        first = s.exec(select(Family)).first()
        if first is not None:
            st.session_state.client_family_id = first.id
            return first.id
    return None


# ─────────────────────────────────────────────────────────────────────
# Read-only views for clients
# ─────────────────────────────────────────────────────────────────────

def _render_key_people_readonly(family_id: int) -> None:
    """Clients see people and entities but cannot edit them — that's the
    advisor's job. (Editing will be advisor-only via permissions.)"""
    st.markdown("### Key People & Organizations")
    st.caption("Your family members and the entities tied to them.")

    people = list_people_in_family(family_id)
    entities = list_entities_in_family(family_id)

    col_p, col_e = st.columns(2)

    with col_p:
        st.markdown("**👥 Family Members**")
        if people:
            for p in people:
                with st.container(border=True):
                    tag = " ⚰️" if p.is_deceased else ""
                    st.markdown(f"**{p.display_name}**{tag}")
                    bits = []
                    if p.dob:
                        bits.append(f"DOB {p.dob.isoformat()}")
                    if p.email:
                        bits.append(p.email)
                    if bits:
                        st.caption(" · ".join(bits))
        else:
            st.info("No family members listed yet.")

    with col_e:
        st.markdown("**🏛️ Entities**")
        if entities:
            for e in entities:
                with st.container(border=True):
                    st.markdown(f"**{e.name}**")
                    sub = e.sub_type or e.entity_type
                    if e.jurisdiction:
                        sub += f" · {e.jurisdiction}"
                    st.caption(sub)
        else:
            st.info("No entities listed yet.")

    st.caption(
        "_Information here is maintained by your advisory team. "
        "Contact them if you need updates._"
    )


def _render_no_family_assigned(user) -> None:
    st.markdown("# Welcome")
    st.warning(
        "Your account isn't linked to a family yet. Please contact your "
        "advisor — they'll get you set up. Once linked, you'll see your "
        "family tree, documents, advisory team, and more right here."
    )


def _coming_soon(title: str, description: str, will_do: list[str]) -> None:
    st.markdown(f"### {title}")
    st.caption(description)
    st.info("🚧 Coming soon — this section is part of the Angel client portal roadmap.")
    st.markdown("**What you'll be able to do:**")
    for item in will_do:
        st.markdown(f"- {item}")


def _render_client_chat(user, family_id: int) -> None:
    """Render the chat interface for a client."""
    from ui.chat_bridge import (
        ensure_chat_session_state,
        get_or_build_qa,
        load_chat_for_context,
        save_chat_for_context,
        chat_context_key,
    )
    from ui.components.chat_interface import render_chat_interface

    ensure_chat_session_state()

    st.markdown("### Chat")
    st.caption(
        "Ask Angel anything about your documents. Your conversations are "
        "private to you — they never leave your advisor's local system."
    )
    st.markdown("---")

    qa_system = get_or_build_qa(user)

    if qa_system is None:
        st.warning("No AI system is ready yet. Contact your advisor.")
        return

    if qa_system.vector_store is None:
        st.info(
            "📭 No documents are loaded yet. Your advisor needs to upload "
            "your documents before you can ask questions about them."
        )
        return

    # CRITICAL: Only load chat history from chat_histories dict when we
    # first enter this context (or after a New Chat). On subsequent reruns
    # within the same context, session_state.chat_history is the source of
    # truth — we must NOT overwrite it with the stale chat_histories dict,
    # or every answer the user sees will disappear on the next rerun.
    expected_key = chat_context_key(user)
    if st.session_state.get("current_chat_id") != expected_key:
        load_chat_for_context(user)

    # New Chat button — clears the live thread and resets context
    if st.button("➕ New Chat", key="client_new_chat"):
        # Wipe both the live history AND the saved context, otherwise the
        # next render's load_chat_for_context will restore the old thread.
        st.session_state.chat_history = []
        key = chat_context_key(user)
        if key in st.session_state.chat_histories:
            del st.session_state.chat_histories[key]
        st.session_state.current_chat_id = None
        st.rerun()
    render_chat_interface(user, qa_system)

    # Save AFTER rendering so chat_histories dict stays in sync with the
    # latest chat_history. Important for context switches and "go back to
    # chat history" flows.
    save_chat_for_context(user)
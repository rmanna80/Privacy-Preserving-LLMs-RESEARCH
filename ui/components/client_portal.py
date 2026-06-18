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
    ("Angel", "💬"),
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
        from ui.components.client_views import render_client_tasks
        render_client_tasks(user, family_id)
    elif selected == "Key People & Orgs":
        _render_key_people_readonly(family_id)
    elif selected == "Family Tree":
        from ui.components.family_tree import render_family_tree
        render_family_tree(family_id)
    elif selected == "Org Ownership":
        from ui.components.org_ownership import render_org_ownership
        render_org_ownership(family_id)
    elif selected == "Advisory Team":
        from ui.components.client_views import render_client_advisory_team
        render_client_advisory_team(family_id)
    elif selected == "Documents":
        from ui.components.client_views import render_client_documents
        render_client_documents(family_id)
    elif selected == "Angel":
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
    """Find which family a client belongs to.

    Interim resolution (until the full User<->Person identity link in the
    next phase):
      1. If a Person in any family has an email matching this user's
         login email, use that Person's family.
      2. Else, if the auth system maps this client to an advisor, use the
         most recently created family owned by that advisor.
      3. Else fall back to None (shows the 'no family' message).
    """
    if st.session_state.get("client_family_id"):
        return st.session_state.client_family_id

    from db.database import get_session
    from db.models import Family, Person, User as DBUser
    from sqlmodel import select

    login_email = (getattr(user, "username", "") or "").lower()

    with get_session() as s:
        # 1) Match a Person by email
        if login_email:
            person = s.exec(
                select(Person).where(Person.email == login_email)
            ).first()
            if person is not None:
                st.session_state.client_family_id = person.family_id
                return person.family_id

        # 2) Match by advisor ownership — most recent family this client's
        #    advisor owns. Requires the client's advisor to be resolvable.
        #    We match the DB user by email, then find families they can access.
        db_user = s.exec(
            select(DBUser).where(DBUser.email == login_email)
        ).first()
        if db_user is not None:
            fam = s.exec(
                select(Family)
                .where(Family.advisor_user_id == db_user.id)
                .order_by(Family.created_at.desc())
            ).first()
            if fam is not None:
                st.session_state.client_family_id = fam.id
                return fam.id

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
    """Render the chat interface for a client, scoped to their family."""
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
        "Ask Angel anything about your family's documents. Your "
        "conversations stay private to your family — they never leave "
        "your advisor's local system."
    )
    st.markdown("---")

    # Family-scoped QA — uses documents from the new Documents table,
    # tagged with this family_id.
    qa_system = get_or_build_qa(user, family_id=family_id)

    if qa_system is None:
        st.warning("No AI system is ready yet. Contact your advisor.")
        return

    if qa_system.vector_store is None:
        st.info(
            "📭 No documents have been uploaded for your family yet. "
            "Your advisor needs to upload documents before you can ask "
            "questions about them."
        )
        return

    # Only load when context changes (otherwise rerun wipes the live thread)
    expected_key = chat_context_key(user, family_id=family_id)
    if st.session_state.get("current_chat_id") != expected_key:
        load_chat_for_context(user, family_id=family_id)

    if st.button("➕ New Chat", key="client_new_chat"):
        st.session_state.chat_history = []
        key = chat_context_key(user, family_id=family_id)
        if key in st.session_state.chat_histories:
            del st.session_state.chat_histories[key]
        st.session_state.current_chat_id = None
        st.rerun()

    render_chat_interface(user, qa_system)
    save_chat_for_context(user, family_id=family_id)
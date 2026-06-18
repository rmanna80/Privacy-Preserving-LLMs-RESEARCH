"""
ui/components/client_views.py — read-only client-portal views.

Clients see their family's data but cannot edit it (that's the advisor's
job). Exceptions per product decision:
  - Tasks (1c): a client sees tasks ASSIGNED TO THEM, can comment, and can
    mark them complete. They do NOT see internal advisor tasks or tasks
    assigned to other family members / team members.
  - Documents: view + download only. No upload, no archive.
  - Advisory Team: view the team (visual + contact cards), no editing.

Client→Person resolution: we reuse the same email-match approach the
client portal already uses to find the family. To scope tasks to "me",
we find the Person in this family whose email matches the login email.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Optional

import streamlit as st

from db.repositories import (
    get_family,
    list_people_in_family,
    list_team_members,
    list_documents_by_category,
    list_documents_for_family,
    list_task_comments,
    add_task_comment,
    update_task,
    ensure_db_user,
    get_person,
    get_team_member,
    CATEGORY_LABELS,
    DOCUMENT_CATEGORIES,
)


# ─────────────────────────────────────────────────────────────────────
# Helper: which Person is this logged-in client?
# ─────────────────────────────────────────────────────────────────────

def _resolve_client_person_id(user, family_id: int) -> Optional[int]:
    """Find the Person record matching the logged-in client's email."""
    login_email = (getattr(user, "username", "") or "").lower()
    if not login_email:
        return None
    for p in list_people_in_family(family_id):
        if (p.email or "").lower() == login_email:
            return p.id
    return None


# ─────────────────────────────────────────────────────────────────────
# TASKS (1c) — my tasks, comment + complete
# ─────────────────────────────────────────────────────────────────────

def render_client_tasks(user, family_id: int) -> None:
    st.markdown("### My Tasks")
    st.caption(
        "Tasks your advisory team has assigned to you. Add updates or mark "
        "them complete as you go."
    )

    user_db_id = ensure_db_user(user)
    my_person_id = _resolve_client_person_id(user, family_id)

    # Pull this family's tasks, keep only those assigned to me (as a person)
    from db.repositories import list_tasks_for_family
    all_tasks = list_tasks_for_family(family_id, include_complete=True)
    my_tasks = [
        t for t in all_tasks
        if t.assignee_type == "person" and t.assigned_person_id == my_person_id
        and my_person_id is not None
    ]

    if my_person_id is None:
        st.info(
            "We couldn't match your login to a family member record, so we "
            "can't show your assigned tasks yet. Your advisor can link your "
            "account."
        )
        return

    if not my_tasks:
        st.info("You have no tasks assigned right now. 🎉")
        return

    open_tasks = [t for t in my_tasks if t.status != "complete"]
    done_tasks = [t for t in my_tasks if t.status == "complete"]

    for t in open_tasks:
        _render_client_task_card(t, user_db_id, expanded=False)

    if done_tasks:
        with st.expander(f"✅ Completed ({len(done_tasks)})", expanded=False):
            for t in done_tasks:
                _render_client_task_card(t, user_db_id, expanded=False, done=True)


def _render_client_task_card(task, user_db_id, *, expanded, done=False):
    with st.container(border=True):
        c1, c2 = st.columns([5, 2])
        with c1:
            title_style = "text-decoration:line-through; opacity:0.6;" if done else ""
            st.markdown(
                f"<div style='font-weight:600; {title_style}'>{task.title}</div>",
                unsafe_allow_html=True,
            )
            if task.description:
                st.caption(task.description)
        with c2:
            if task.due_date:
                days = (task.due_date - date.today()).days
                if not done and days < 0:
                    st.markdown(
                        f"<span style='color:#C66666;'>Overdue "
                        f"({task.due_date.isoformat()})</span>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.caption(f"Due {task.due_date.isoformat()}")

        # Expand for comments + complete
        key = f"client_task_open_{task.id}"
        if st.button("View / Update" if not done else "View",
                     key=f"toggle_{key}", use_container_width=True):
            st.session_state[key] = not st.session_state.get(key, False)
            st.rerun()

        if st.session_state.get(key, False):
            st.markdown("---")
            # Comments
            comments = list_task_comments(task.id)
            if comments:
                for c in comments:
                    from db.database import get_session
                    from db.models import User as DBUser
                    author = f"User #{c.author_user_id}"
                    with get_session() as s:
                        u = s.get(DBUser, c.author_user_id)
                        if u:
                            author = u.full_name or u.email
                    st.markdown(f"**{author}** · {c.created_at.strftime('%Y-%m-%d %H:%M')}")
                    st.markdown(c.body)
            else:
                st.caption("_No comments yet._")

            new_comment = st.text_area("Add an update", key=f"client_comment_{task.id}",
                                       height=70)
            cc1, cc2 = st.columns(2)
            with cc1:
                if st.button("Post update", key=f"client_post_{task.id}",
                             use_container_width=True):
                    if new_comment.strip():
                        add_task_comment(task.id, user_db_id, new_comment.strip())
                        st.rerun()
            with cc2:
                if not done:
                    if st.button("✅ Mark complete", key=f"client_done_{task.id}",
                                 use_container_width=True, type="primary"):
                        update_task(task.id, status="complete")
                        st.rerun()


# ─────────────────────────────────────────────────────────────────────
# ADVISORY TEAM — read-only
# ─────────────────────────────────────────────────────────────────────

def render_client_advisory_team(family_id: int) -> None:
    from ui.components.advisory_team import (
        _build_hub_and_spoke_svg, role_icon, role_label,
    )

    family = get_family(family_id)
    st.markdown("### Advisory Team")
    st.caption("The professionals helping your family.")

    members = list_team_members(family_id, active_only=True)
    if not members:
        st.info("Your advisory team hasn't been set up yet.")
        return

    # Reuse the hub-and-spoke visual (read-only — no edit controls)
    import streamlit.components.v1 as components
    svg = _build_hub_and_spoke_svg(family.name, members)
    components.html(
        f'<div style="background:#F8F4EC; border-radius:12px; padding:16px; '
        f'height:480px; box-sizing:border-box; display:flex; '
        f'align-items:center; justify-content:center;">{svg}</div>',
        height=500, scrolling=False,
    )

    st.markdown("---")
    st.markdown("**Contact Information**")
    for m in members:
        with st.container(border=True):
            st.markdown(f"{role_icon(m.role)}  **{m.full_name}** — {role_label(m.role)}")
            bits = []
            if m.firm:
                bits.append(m.firm)
            if m.email:
                bits.append(f"📧 {m.email}")
            if m.phone:
                bits.append(f"📞 {m.phone}")
            if bits:
                st.caption(" · ".join(bits))


# ─────────────────────────────────────────────────────────────────────
# DOCUMENTS — read-only view + download
# ─────────────────────────────────────────────────────────────────────

def render_client_documents(family_id: int) -> None:
    st.markdown("### Documents")
    st.caption("Your family's documents, organized by category.")

    grouped = list_documents_by_category(family_id)
    total = sum(len(v) for v in grouped.values())
    if total == 0:
        st.info("No documents have been uploaded for your family yet.")
        return

    for cat_key in DOCUMENT_CATEGORIES:
        docs = grouped.get(cat_key, [])
        if not docs:
            continue
        with st.expander(f"{CATEGORY_LABELS[cat_key]} · {len(docs)}", expanded=False):
            for d in docs:
                with st.container(border=True):
                    c1, c2 = st.columns([4, 2])
                    with c1:
                        st.markdown(f"**{d.original_filename}**")
                        sub = [d.doc_type.replace('_', ' ').title()]
                        if d.doc_year:
                            sub.append(str(d.doc_year))
                        st.caption(" · ".join(sub))
                    with c2:
                        # Download button if the file still exists on disk
                        try:
                            p = Path(d.file_path)
                            if p.exists():
                                st.download_button(
                                    "⬇ Download",
                                    data=p.read_bytes(),
                                    file_name=d.original_filename,
                                    mime=d.mime_type or "application/pdf",
                                    key=f"client_dl_{d.id}",
                                    use_container_width=True,
                                )
                            else:
                                st.caption("_File unavailable_")
                        except Exception:
                            st.caption("_File unavailable_")

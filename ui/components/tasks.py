"""
Tasks pages.

Two entry points exposed:
  - render_tasks_page(family_id, user)    — per-family Tasks view
                                            (under Family detail nav)
  - render_my_tasks_page(user)            — All tasks assigned to me,
                                            across every family
                                            (top-level nav)

Architecture
------------
Each task is rendered as a card. Clicking 'Open' on a card expands its
detail (description, comments thread, status controls). The card list
and the detail share one rerender model — no modal popups, no nested
forms, no fragile state.

Status workflow: open → in_progress → complete. 'blocked' and 'archived'
are reachable from any state via direct selectbox change.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

import streamlit as st

from db.models import Task, Person, AdvisoryTeamMember, User
from db.repositories import (
    list_tasks_for_family,
    list_tasks_assigned_to_user,
    get_task,
    create_task,
    update_task,
    archive_task,
    hard_delete_task,
    list_task_comments,
    add_task_comment,
    list_people_in_family,
    list_team_members,
    get_family,
    get_person,
    get_team_member,
    ensure_db_user,
    list_families_for_advisor,
    VALID_TASK_STATUSES,
    VALID_TASK_PRIORITIES,
)

from ui.components.advisory_team import role_icon, role_label


# ─────────────────────────────────────────────────────────────────────
# Status & priority display helpers
# ─────────────────────────────────────────────────────────────────────

STATUS_META = {
    "open":        ("Open",         "🔵"),
    "in_progress": ("In Progress",  "🟡"),
    "blocked":     ("Blocked",      "🔴"),
    "complete":    ("Complete",     "✅"),
    "archived":    ("Archived",     "📦"),
}

PRIORITY_META = {
    None:     ("—",      ""),
    "low":    ("Low",    "⬇️"),
    "normal": ("Normal", "•"),
    "high":   ("High",   "⬆️"),
    "urgent": ("Urgent", "🔥"),
}


def status_label(s: str) -> str:
    return STATUS_META.get(s, (s, ""))[0]


def status_icon(s: str) -> str:
    return STATUS_META.get(s, ("", "•"))[1]


def priority_label(p: Optional[str]) -> str:
    return PRIORITY_META.get(p, ("—", ""))[0]


def priority_icon(p: Optional[str]) -> str:
    return PRIORITY_META.get(p, ("", ""))[1]


# ─────────────────────────────────────────────────────────────────────
# Public — per-family Tasks page
# ─────────────────────────────────────────────────────────────────────

def render_tasks_page(family_id: int, user) -> None:
    """Tasks scoped to one family. Renders under Family detail → Tasks."""
    family = get_family(family_id)
    if family is None:
        st.error("Family not found.")
        return

    advisor_db_id = ensure_db_user(user)

    st.markdown("### Tasks")
    st.caption(
        "Tasks for this family — assignments, due dates, conversations. "
        "Assign to a family member or a member of the advisory team."
    )

    # Filter controls
    col_show_complete, col_show_archived, _ = st.columns([2, 2, 4])
    with col_show_complete:
        show_complete = st.checkbox("Show completed", value=False, key="tasks_show_complete")
    with col_show_archived:
        show_archived = st.checkbox("Show archived", value=False, key="tasks_show_archived")

    tasks = list_tasks_for_family(
        family_id,
        include_archived=show_archived,
        include_complete=show_complete,
    )

    # Active expanded task (so detail renders below)
    expanded_id = st.session_state.get("expanded_task_id")

    if tasks:
        for t in tasks:
            _render_task_card(t, family_id, expanded=(t.id == expanded_id), user_db_id=advisor_db_id)
    else:
        st.info(
            "No tasks yet. Use the form below to create the first one."
        )

    st.markdown("---")
    _render_create_task_form(family_id, advisor_db_id)


# ─────────────────────────────────────────────────────────────────────
# Public — top-level "My Tasks" page
# ─────────────────────────────────────────────────────────────────────

def render_my_tasks_page(user) -> None:
    """All open tasks assigned to this user across every family they touch."""
    advisor_db_id = ensure_db_user(user)

    st.markdown("### My Tasks")
    st.caption(
        "Every task assigned to you, across all the families you manage."
    )

    tasks = list_tasks_assigned_to_user(advisor_db_id)

    if not tasks:
        st.info(
            "No tasks assigned to you right now. Tasks assigned to you "
            "from any family you work with will show up here."
        )
        return

    # Group by family for clarity
    families_by_id = {
        f.id: f for f in list_families_for_advisor(advisor_db_id)
    }

    grouped: dict[int, list[Task]] = {}
    for t in tasks:
        grouped.setdefault(t.family_id, []).append(t)

    for fid, fam_tasks in grouped.items():
        family = families_by_id.get(fid) or get_family(fid)
        if family is None:
            continue
        st.markdown(f"#### {family.name}")
        for t in fam_tasks:
            _render_task_card(
                t,
                fid,
                expanded=(t.id == st.session_state.get("expanded_task_id")),
                user_db_id=advisor_db_id,
                show_family_label=False,
            )
        st.markdown("---")


# ─────────────────────────────────────────────────────────────────────
# Task card — collapsed and expanded states
# ─────────────────────────────────────────────────────────────────────

def _render_task_card(
    task: Task,
    family_id: int,
    *,
    expanded: bool,
    user_db_id: int,
    show_family_label: bool = True,
) -> None:
    with st.container(border=True):
        col_status, col_main, col_meta, col_action = st.columns([1, 5, 3, 2])

        with col_status:
            st.markdown(
                f"<div style='font-size:1.4rem; text-align:center;'>"
                f"{status_icon(task.status)}"
                f"</div>",
                unsafe_allow_html=True,
            )

        with col_main:
            title_style = (
                "text-decoration:line-through; opacity:0.6;"
                if task.status == "complete"
                else ""
            )
            st.markdown(
                f"<div style='font-weight:600; {title_style}'>{task.title}</div>",
                unsafe_allow_html=True,
            )
            meta_bits = [status_label(task.status)]
            if task.priority:
                meta_bits.append(
                    f"{priority_icon(task.priority)} {priority_label(task.priority)}"
                )
            assignee_label = _assignee_display(task)
            if assignee_label:
                meta_bits.append(f"→ {assignee_label}")
            st.caption(" · ".join(meta_bits))

        with col_meta:
            if task.due_date:
                today = date.today()
                days = (task.due_date - today).days
                overdue = days < 0 and task.status != "complete"
                color = "#C66666" if overdue else "#5F7494"
                pretext = "Overdue" if overdue else (
                    "Due today" if days == 0 else f"Due in {days}d" if days > 0 else f"{-days}d ago"
                )
                st.markdown(
                    f"<div style='color:{color}; font-size:0.85rem;'>"
                    f"{pretext}<br>"
                    f"<span style='font-size:0.75rem; opacity:0.7;'>"
                    f"{task.due_date.isoformat()}"
                    f"</span></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.caption("_No due date_")

        with col_action:
            btn_label = "Close" if expanded else "Open"
            if st.button(btn_label, key=f"toggle_task_{task.id}", use_container_width=True):
                if expanded:
                    st.session_state.expanded_task_id = None
                else:
                    st.session_state.expanded_task_id = task.id
                st.rerun()

        if expanded:
            st.markdown("---")
            _render_task_detail(task, family_id, user_db_id)


def _render_task_detail(task: Task, family_id: int, user_db_id: int) -> None:
    # Description editable inline
    new_desc = st.text_area(
        "Description",
        value=task.description or "",
        key=f"task_desc_{task.id}",
        height=100,
    )

    # Status + priority + assignee + due in one row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        status_options = ["open", "in_progress", "blocked", "complete", "archived"]
        new_status = st.selectbox(
            "Status",
            status_options,
            index=status_options.index(task.status),
            key=f"task_status_{task.id}",
            format_func=lambda s: f"{status_icon(s)} {status_label(s)}",
        )
    with c2:
        priority_options = ["", "low", "normal", "high", "urgent"]
        current_pri_idx = priority_options.index(task.priority or "")
        new_priority_raw = st.selectbox(
            "Priority",
            priority_options,
            index=current_pri_idx,
            key=f"task_priority_{task.id}",
            format_func=lambda p: (
                "— None —" if p == "" else f"{priority_icon(p)} {priority_label(p)}"
            ),
        )
        new_priority = new_priority_raw if new_priority_raw else None
    with c3:
        new_due = st.date_input(
            "Due date",
            value=task.due_date,
            key=f"task_due_{task.id}",
            format="YYYY-MM-DD",
        )
    with c4:
        # Assignee picker — both Person and TeamMember options in one list
        assignee_pick = _assignee_picker(task, family_id)

    col_save, col_archive, col_delete = st.columns(3)
    with col_save:
        if st.button("💾 Save changes", key=f"task_save_{task.id}", use_container_width=True, type="primary"):
            kwargs = dict(
                description=new_desc.strip() or None,
                status=new_status,
                priority=new_priority,
                due_date=new_due if new_due else None,
            )
            # Apply assignment
            if assignee_pick is None:
                kwargs["assigned_person_id"] = None
                kwargs["assigned_team_member_id"] = None
            elif assignee_pick[0] == "person":
                kwargs["assigned_person_id"] = assignee_pick[1]
                kwargs["assigned_team_member_id"] = None
            elif assignee_pick[0] == "team_member":
                kwargs["assigned_team_member_id"] = assignee_pick[1]
                kwargs["assigned_person_id"] = None
            update_task(task.id, **kwargs)
            st.rerun()
    with col_archive:
        if task.status != "archived":
            if st.button("📦 Archive", key=f"task_archive_{task.id}", use_container_width=True):
                archive_task(task.id)
                st.session_state.expanded_task_id = None
                st.rerun()
    with col_delete:
        confirm_key = f"confirm_delete_task_{task.id}"
        if st.session_state.get(confirm_key, False):
            if st.button("⚠️ Confirm permanent delete", key=f"task_delete_{task.id}", use_container_width=True, type="primary"):
                hard_delete_task(task.id)
                st.session_state.expanded_task_id = None
                st.session_state[confirm_key] = False
                st.rerun()
        else:
            if st.button("🗑️ Delete", key=f"task_delete_{task.id}", use_container_width=True):
                st.session_state[confirm_key] = True
                st.rerun()

    # Comments thread
    st.markdown("---")
    st.markdown("**💬 Comments**")
    comments = list_task_comments(task.id)
    if comments:
        for c in comments:
            with st.container(border=True):
                # Resolve author
                author_label = f"User #{c.author_user_id}"
                from db.database import get_session
                from sqlmodel import select
                with get_session() as s:
                    u = s.get(User, c.author_user_id)
                    if u:
                        author_label = u.full_name or u.email
                col_author, col_time = st.columns([3, 2])
                with col_author:
                    prefix = "⚙️ " if c.is_system else ""
                    st.markdown(f"{prefix}**{author_label}**")
                with col_time:
                    st.caption(c.created_at.strftime("%Y-%m-%d %H:%M"))
                st.markdown(c.body)
    else:
        st.caption("_No comments yet._")

    # New comment
    new_comment = st.text_area(
        "Add a comment",
        key=f"new_comment_{task.id}",
        placeholder="Update the team on progress, blockers, decisions…",
        height=80,
    )
    if st.button("Post", key=f"post_comment_{task.id}"):
        if new_comment.strip():
            add_task_comment(task.id, user_db_id, new_comment.strip())
            st.rerun()
        else:
            st.warning("Empty comment.")


# ─────────────────────────────────────────────────────────────────────
# New-task form
# ─────────────────────────────────────────────────────────────────────

def _render_create_task_form(family_id: int, user_db_id: int) -> None:
    with st.expander("➕ Create Task", expanded=False):
        with st.form("new_task_form", clear_on_submit=True):
            title = st.text_input("Title *", placeholder="e.g. Review revised trust amendment")
            description = st.text_area(
                "Description",
                placeholder="Context, links, what 'done' looks like…",
            )

            c1, c2, c3 = st.columns(3)
            with c1:
                priority_options = ["", "low", "normal", "high", "urgent"]
                pri_raw = st.selectbox(
                    "Priority",
                    priority_options,
                    index=2,  # normal
                    format_func=lambda p: (
                        "— None —" if p == "" else f"{priority_icon(p)} {priority_label(p)}"
                    ),
                )
                priority = pri_raw if pri_raw else None
            with c2:
                due = st.date_input(
                    "Due date",
                    value=None,
                    format="YYYY-MM-DD",
                )
            with c3:
                assignee_pick = _new_task_assignee_picker(family_id)

            col_create, col_cancel = st.columns(2)
            with col_create:
                submitted = st.form_submit_button(
                    "Create Task", type="primary", use_container_width=True
                )
            with col_cancel:
                cancelled = st.form_submit_button("Cancel", use_container_width=True)

            if cancelled:
                st.rerun()

            if submitted:
                if not title.strip():
                    st.error("Title is required.")
                    return
                kwargs = dict(
                    family_id=family_id,
                    title=title.strip(),
                    description=description.strip() or None,
                    created_by_user_id=user_db_id,
                    priority=priority,
                    due_date=due if due else None,
                )
                if assignee_pick is not None:
                    if assignee_pick[0] == "person":
                        kwargs["assigned_person_id"] = assignee_pick[1]
                    elif assignee_pick[0] == "team_member":
                        kwargs["assigned_team_member_id"] = assignee_pick[1]
                create_task(**kwargs)
                st.rerun()


# ─────────────────────────────────────────────────────────────────────
# Assignee picker — shared logic
# ─────────────────────────────────────────────────────────────────────

def _build_assignee_options(family_id: int) -> list[tuple[str, Optional[tuple[str, int]]]]:
    """Return list of (display_label, (assignee_type, id)) for picker.
    First element is the 'Unassigned' option."""
    people = list_people_in_family(family_id)
    members = list_team_members(family_id)

    options: list[tuple[str, Optional[tuple[str, int]]]] = [
        ("— Unassigned —", None)
    ]
    if people:
        for p in people:
            options.append((f"👥 {p.display_name}", ("person", p.id)))
    if members:
        for m in members:
            options.append(
                (f"{role_icon(m.role)} {m.full_name} ({role_label(m.role)})",
                 ("team_member", m.id))
            )
    return options


def _assignee_picker(task: Task, family_id: int) -> Optional[tuple[str, int]]:
    """Picker for editing an existing task. Returns (type, id) or None."""
    options = _build_assignee_options(family_id)
    labels = [o[0] for o in options]

    # Find current selection
    current_idx = 0
    if task.assignee_type == "person" and task.assigned_person_id:
        for i, (_, val) in enumerate(options):
            if val == ("person", task.assigned_person_id):
                current_idx = i
                break
    elif task.assignee_type == "team_member" and task.assigned_team_member_id:
        for i, (_, val) in enumerate(options):
            if val == ("team_member", task.assigned_team_member_id):
                current_idx = i
                break

    picked = st.selectbox(
        "Assignee",
        labels,
        index=current_idx,
        key=f"task_assignee_{task.id}",
    )
    return options[labels.index(picked)][1]


def _new_task_assignee_picker(family_id: int) -> Optional[tuple[str, int]]:
    """Picker for the new-task form."""
    options = _build_assignee_options(family_id)
    labels = [o[0] for o in options]
    picked = st.selectbox("Assignee", labels, index=0)
    return options[labels.index(picked)][1]


def _assignee_display(task: Task) -> str:
    """Plain-text display name for a task's current assignee."""
    if task.assignee_type == "person" and task.assigned_person_id:
        p = get_person(task.assigned_person_id)
        return p.display_name if p else "(unknown person)"
    if task.assignee_type == "team_member" and task.assigned_team_member_id:
        m = get_team_member(task.assigned_team_member_id)
        if m:
            return f"{m.full_name} ({role_label(m.role)})"
        return "(unknown team member)"
    return ""

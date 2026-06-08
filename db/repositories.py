"""
Repository layer — clean Python functions for all DB queries.

The UI never opens a session directly; it calls these functions, which
return either ORM objects or simple values.

Step 2  added: families, people, relationships.
Step 3a adds:  entities, roles.
Step 3b will add: documents, extractions.
"""

from __future__ import annotations

from datetime import datetime, date
from typing import Optional, Any

from sqlmodel import select
from sqlalchemy import or_

from db.database import get_session
from db.models import (
    User,
    Family,
    Person,
    Entity,
    Role,
    Relationship,
    Document,
    Extraction,
    UserFamilyAccess,
)


# Sentinel for "argument not provided" — distinct from None ("set to null")
_UNCHANGED: Any = object()


# ============================================================================
# Bridge: existing auth -> wealth.db users
# ============================================================================

def ensure_db_user(auth_user) -> int:
    """Upsert a wealth.db User row matching the existing auth user; return its id."""
    email = auth_user.username
    full_name = getattr(auth_user, "client_name", None) or email
    role = (
        auth_user.role.value
        if hasattr(auth_user.role, "value")
        else str(auth_user.role)
    )

    with get_session() as s:
        existing = s.exec(select(User).where(User.email == email)).first()
        if existing is not None:
            updated = False
            if existing.full_name != full_name:
                existing.full_name = full_name
                updated = True
            if existing.role != role:
                existing.role = role
                updated = True
            if updated:
                s.add(existing)
                s.flush()
            return existing.id

        new_user = User(
            email=email,
            password_hash="bridge::external_auth",
            full_name=full_name,
            role=role,
        )
        s.add(new_user)
        s.flush()
        return new_user.id


# ============================================================================
# Families
# ============================================================================

def list_families_for_advisor(advisor_user_id: int) -> list[Family]:
    with get_session() as s:
        return list(
            s.exec(
                select(Family)
                .where(Family.advisor_user_id == advisor_user_id)
                .order_by(Family.name)
            ).all()
        )


def get_family(family_id: int) -> Optional[Family]:
    with get_session() as s:
        return s.get(Family, family_id)


def create_family(
    name: str,
    advisor_user_id: int,
    notes: Optional[str] = None,
) -> Family:
    with get_session() as s:
        family = Family(
            name=name,
            advisor_user_id=advisor_user_id,
            notes=notes,
        )
        s.add(family)
        s.flush()
        s.refresh(family)
        return family


def update_family(
    family_id: int,
    name: Optional[str] = None,
    notes: Any = _UNCHANGED,
) -> Optional[Family]:
    with get_session() as s:
        family = s.get(Family, family_id)
        if family is None:
            return None
        if name is not None:
            family.name = name
        if notes is not _UNCHANGED:
            family.notes = notes
        family.updated_at = datetime.utcnow()
        s.add(family)
        s.flush()
        s.refresh(family)
        return family


def delete_family(family_id: int) -> bool:
    """Delete a family and all dependents (manual cascade).

    Files on disk are NOT touched — only DB rows. Removing on-disk PDFs
    is the responsibility of the document manager.
    """
    with get_session() as s:
        family = s.get(Family, family_id)
        if family is None:
            return False

        people = list(s.exec(select(Person).where(Person.family_id == family_id)).all())
        person_ids = [p.id for p in people]

        if person_ids:
            for rel in s.exec(
                select(Relationship).where(
                    or_(
                        Relationship.person_a_id.in_(person_ids),
                        Relationship.person_b_id.in_(person_ids),
                    )
                )
            ).all():
                s.delete(rel)
            for role in s.exec(
                select(Role).where(Role.person_id.in_(person_ids))
            ).all():
                s.delete(role)
            for p in people:
                s.delete(p)

        for ent in s.exec(select(Entity).where(Entity.family_id == family_id)).all():
            for role in s.exec(select(Role).where(Role.entity_id == ent.id)).all():
                s.delete(role)
            s.delete(ent)

        for doc in s.exec(select(Document).where(Document.family_id == family_id)).all():
            for ext in s.exec(
                select(Extraction).where(Extraction.document_id == doc.id)
            ).all():
                s.delete(ext)
            s.delete(doc)

        for access in s.exec(
            select(UserFamilyAccess).where(UserFamilyAccess.family_id == family_id)
        ).all():
            s.delete(access)

        s.delete(family)
        return True


# ============================================================================
# People
# ============================================================================

def list_people_in_family(family_id: int) -> list[Person]:
    with get_session() as s:
        return list(
            s.exec(
                select(Person)
                .where(Person.family_id == family_id)
                .order_by(Person.last_name, Person.first_name)
            ).all()
        )


def get_person(person_id: int) -> Optional[Person]:
    with get_session() as s:
        return s.get(Person, person_id)


def create_person(
    family_id: int,
    first_name: str,
    last_name: str,
    middle_name: Optional[str] = None,
    preferred_name: Optional[str] = None,
    dob: Optional[date] = None,
    email: Optional[str] = None,
    phone: Optional[str] = None,
    ssn: Optional[str] = None,
    is_deceased: bool = False,
    date_of_death: Optional[date] = None,
    notes: Optional[str] = None,
) -> Person:
    with get_session() as s:
        person = Person(
            family_id=family_id,
            first_name=first_name,
            last_name=last_name,
            middle_name=middle_name,
            preferred_name=preferred_name,
            dob=dob,
            email=email,
            phone=phone,
            is_deceased=is_deceased,
            date_of_death=date_of_death,
            notes=notes,
        )
        if ssn:
            person.ssn = ssn
        s.add(person)
        s.flush()
        s.refresh(person)
        return person


def update_person(
    person_id: int,
    *,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
    middle_name: Any = _UNCHANGED,
    preferred_name: Any = _UNCHANGED,
    dob: Any = _UNCHANGED,
    email: Any = _UNCHANGED,
    phone: Any = _UNCHANGED,
    ssn: Any = _UNCHANGED,
    is_deceased: Optional[bool] = None,
    date_of_death: Any = _UNCHANGED,
    notes: Any = _UNCHANGED,
) -> Optional[Person]:
    with get_session() as s:
        person = s.get(Person, person_id)
        if person is None:
            return None

        if first_name is not None:
            person.first_name = first_name
        if last_name is not None:
            person.last_name = last_name
        if middle_name is not _UNCHANGED:
            person.middle_name = middle_name
        if preferred_name is not _UNCHANGED:
            person.preferred_name = preferred_name
        if dob is not _UNCHANGED:
            person.dob = dob
        if email is not _UNCHANGED:
            person.email = email
        if phone is not _UNCHANGED:
            person.phone = phone
        if is_deceased is not None:
            person.is_deceased = is_deceased
        if date_of_death is not _UNCHANGED:
            person.date_of_death = date_of_death
        if notes is not _UNCHANGED:
            person.notes = notes
        if ssn is not _UNCHANGED:
            person.ssn = ssn

        person.updated_at = datetime.utcnow()
        s.add(person)
        s.flush()
        s.refresh(person)
        return person


def delete_person(person_id: int) -> bool:
    with get_session() as s:
        person = s.get(Person, person_id)
        if person is None:
            return False

        for rel in s.exec(
            select(Relationship).where(
                or_(
                    Relationship.person_a_id == person_id,
                    Relationship.person_b_id == person_id,
                )
            )
        ).all():
            s.delete(rel)

        for role in s.exec(select(Role).where(Role.person_id == person_id)).all():
            s.delete(role)

        s.delete(person)
        return True


# ============================================================================
# Relationships
# ============================================================================

def list_relationships_in_family(family_id: int) -> list[Relationship]:
    with get_session() as s:
        people_ids = [
            p.id
            for p in s.exec(select(Person).where(Person.family_id == family_id)).all()
        ]
        if not people_ids:
            return []
        return list(
            s.exec(
                select(Relationship).where(
                    Relationship.person_a_id.in_(people_ids)
                    & Relationship.person_b_id.in_(people_ids)
                )
            ).all()
        )


def create_relationship(
    person_a_id: int,
    person_b_id: int,
    relationship_type: str,
    start_date: Optional[date] = None,
    notes: Optional[str] = None,
) -> Relationship:
    with get_session() as s:
        rel = Relationship(
            person_a_id=person_a_id,
            person_b_id=person_b_id,
            relationship_type=relationship_type,
            start_date=start_date,
            notes=notes,
        )
        s.add(rel)
        s.flush()
        s.refresh(rel)
        return rel


def delete_relationship(relationship_id: int) -> bool:
    with get_session() as s:
        rel = s.get(Relationship, relationship_id)
        if rel is None:
            return False
        s.delete(rel)
        return True


# ============================================================================
# Entities (Step 3a)
# ============================================================================

def list_entities_in_family(family_id: int) -> list[Entity]:
    with get_session() as s:
        return list(
            s.exec(
                select(Entity)
                .where(Entity.family_id == family_id)
                .order_by(Entity.name)
            ).all()
        )


def get_entity(entity_id: int) -> Optional[Entity]:
    with get_session() as s:
        return s.get(Entity, entity_id)


def create_entity(
    family_id: int,
    name: str,
    entity_type: str,
    sub_type: Optional[str] = None,
    jurisdiction: Optional[str] = None,
    formation_date: Optional[date] = None,
    termination_date: Optional[date] = None,
    tax_id: Optional[str] = None,
    notes: Optional[str] = None,
) -> Entity:
    with get_session() as s:
        entity = Entity(
            family_id=family_id,
            name=name,
            entity_type=entity_type,
            sub_type=sub_type,
            jurisdiction=jurisdiction,
            formation_date=formation_date,
            termination_date=termination_date,
            notes=notes,
        )
        if tax_id:
            entity.tax_id = tax_id  # encryption via property setter
        s.add(entity)
        s.flush()
        s.refresh(entity)
        return entity


def update_entity(
    entity_id: int,
    *,
    name: Optional[str] = None,
    entity_type: Optional[str] = None,
    sub_type: Any = _UNCHANGED,
    jurisdiction: Any = _UNCHANGED,
    formation_date: Any = _UNCHANGED,
    termination_date: Any = _UNCHANGED,
    tax_id: Any = _UNCHANGED,
    notes: Any = _UNCHANGED,
) -> Optional[Entity]:
    with get_session() as s:
        entity = s.get(Entity, entity_id)
        if entity is None:
            return None

        if name is not None:
            entity.name = name
        if entity_type is not None:
            entity.entity_type = entity_type
        if sub_type is not _UNCHANGED:
            entity.sub_type = sub_type
        if jurisdiction is not _UNCHANGED:
            entity.jurisdiction = jurisdiction
        if formation_date is not _UNCHANGED:
            entity.formation_date = formation_date
        if termination_date is not _UNCHANGED:
            entity.termination_date = termination_date
        if notes is not _UNCHANGED:
            entity.notes = notes
        if tax_id is not _UNCHANGED:
            entity.tax_id = tax_id  # encryption / clearing via property setter

        entity.updated_at = datetime.utcnow()
        s.add(entity)
        s.flush()
        s.refresh(entity)
        return entity


def delete_entity(entity_id: int) -> bool:
    """Delete an entity and all its roles. Documents linked to this entity
    have their entity_id set to NULL (the document itself stays)."""
    with get_session() as s:
        entity = s.get(Entity, entity_id)
        if entity is None:
            return False

        for role in s.exec(select(Role).where(Role.entity_id == entity_id)).all():
            s.delete(role)

        for doc in s.exec(select(Document).where(Document.entity_id == entity_id)).all():
            doc.entity_id = None
            s.add(doc)

        s.delete(entity)
        return True


# ============================================================================
# Roles (Step 3a)
# ============================================================================

def list_roles_for_entity(entity_id: int) -> list[Role]:
    with get_session() as s:
        return list(
            s.exec(
                select(Role)
                .where(Role.entity_id == entity_id)
                .order_by(Role.role_type)
            ).all()
        )


def list_roles_for_person(person_id: int) -> list[Role]:
    with get_session() as s:
        return list(
            s.exec(
                select(Role)
                .where(Role.person_id == person_id)
                .order_by(Role.created_at)
            ).all()
        )


def get_role(role_id: int) -> Optional[Role]:
    with get_session() as s:
        return s.get(Role, role_id)


def create_role(
    person_id: int,
    entity_id: int,
    role_type: str,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    is_active: bool = True,
    interest_percentage: Optional[float] = None,
    notes: Optional[str] = None,
) -> Role:
    with get_session() as s:
        role = Role(
            person_id=person_id,
            entity_id=entity_id,
            role_type=role_type,
            start_date=start_date,
            end_date=end_date,
            is_active=is_active,
            interest_percentage=interest_percentage,
            notes=notes,
        )
        s.add(role)
        s.flush()
        s.refresh(role)
        return role


def delete_role(role_id: int) -> bool:
    with get_session() as s:
        role = s.get(Role, role_id)
        if role is None:
            return False
        s.delete(role)
        return True
    


# ════════════════════════════════════════════════════════════════════
# Advisory Team
# ════════════════════════════════════════════════════════════════════
 
# Canonical role list — keep in sync with the UI dropdown.
ADVISORY_ROLES = [
    "wealth_strategist",
    "tax_strategist",
    "estate_planning_attorney",
    "business_attorney",
    "general_counsel",
    "m_and_a_attorney",
    "insurance_broker",
    "family_coach",
    "investment_banker",
    "other",
]
 
 
def list_team_members(family_id: int, *, active_only: bool = True) -> list:
    from db.models import AdvisoryTeamMember
    with get_session() as s:
        stmt = (
            select(AdvisoryTeamMember)
            .where(AdvisoryTeamMember.family_id == family_id)
        )
        if active_only:
            stmt = stmt.where(AdvisoryTeamMember.is_active == True)  # noqa: E712
        stmt = stmt.order_by(AdvisoryTeamMember.role, AdvisoryTeamMember.full_name)
        return list(s.exec(stmt).all())
 
 
def get_team_member(member_id: int):
    from db.models import AdvisoryTeamMember
    with get_session() as s:
        return s.get(AdvisoryTeamMember, member_id)
 
 
def create_team_member(
    family_id: int,
    role: str,
    full_name: str,
    firm: Optional[str] = None,
    email: Optional[str] = None,
    phone: Optional[str] = None,
    notes: Optional[str] = None,
    engaged_at: Optional[date] = None,
    user_id: Optional[int] = None,
):
    from db.models import AdvisoryTeamMember
    if role not in ADVISORY_ROLES:
        raise ValueError(
            f"Unknown advisory role '{role}'. Must be one of: {ADVISORY_ROLES}"
        )
    with get_session() as s:
        m = AdvisoryTeamMember(
            family_id=family_id,
            role=role,
            full_name=full_name,
            firm=firm,
            email=email,
            phone=phone,
            notes=notes,
            engaged_at=engaged_at,
            user_id=user_id,
        )
        s.add(m)
        s.flush()
        s.refresh(m)
        return m
 
 
def update_team_member(
    member_id: int,
    *,
    role: Optional[str] = None,
    full_name: Optional[str] = None,
    firm: Any = _UNCHANGED,
    email: Any = _UNCHANGED,
    phone: Any = _UNCHANGED,
    notes: Any = _UNCHANGED,
    is_active: Optional[bool] = None,
    engaged_at: Any = _UNCHANGED,
    departed_at: Any = _UNCHANGED,
):
    from db.models import AdvisoryTeamMember
    if role is not None and role not in ADVISORY_ROLES:
        raise ValueError(f"Unknown advisory role '{role}'.")
    with get_session() as s:
        m = s.get(AdvisoryTeamMember, member_id)
        if m is None:
            return None
        if role is not None:
            m.role = role
        if full_name is not None:
            m.full_name = full_name
        if firm is not _UNCHANGED:
            m.firm = firm
        if email is not _UNCHANGED:
            m.email = email
        if phone is not _UNCHANGED:
            m.phone = phone
        if notes is not _UNCHANGED:
            m.notes = notes
        if is_active is not None:
            m.is_active = is_active
        if engaged_at is not _UNCHANGED:
            m.engaged_at = engaged_at
        if departed_at is not _UNCHANGED:
            m.departed_at = departed_at
        m.updated_at = datetime.utcnow()
        s.add(m)
        s.flush()
        s.refresh(m)
        return m
 
 
def delete_team_member(member_id: int) -> bool:
    """Hard delete — fine here since team members have no critical history."""
    from db.models import AdvisoryTeamMember
    with get_session() as s:
        m = s.get(AdvisoryTeamMember, member_id)
        if m is None:
            return False
        s.delete(m)
        return True
 
 
# ════════════════════════════════════════════════════════════════════
# Tasks
# ════════════════════════════════════════════════════════════════════
 
VALID_TASK_STATUSES = ["open", "in_progress", "blocked", "complete", "archived"]
VALID_TASK_PRIORITIES = [None, "low", "normal", "high", "urgent"]
 
 
def list_tasks_for_family(
    family_id: int,
    *,
    include_archived: bool = False,
    include_complete: bool = True,
) -> list:
    """List tasks for a family, with sensible defaults for the main task view."""
    from db.models import Task
    with get_session() as s:
        stmt = select(Task).where(Task.family_id == family_id)
        if not include_archived:
            stmt = stmt.where(Task.status != "archived")
        if not include_complete:
            stmt = stmt.where(Task.status != "complete")
        # Order: incomplete tasks first by due date (NULLs last), then by created.
        stmt = stmt.order_by(
            Task.status.asc(),
            Task.due_date.asc().nullslast() if hasattr(Task.due_date, 'asc') else Task.due_date,
            Task.created_at.desc(),
        )
        return list(s.exec(stmt).all())
 
 
def list_tasks_assigned_to_user(user_id: int) -> list:
    """All tasks where the assignee is a Person linked to this user, OR
    where the assignee is a team_member linked to this user.
 
    Used by the advisor's 'My Tasks' top-nav view.
    """
    from db.models import Task, Person, AdvisoryTeamMember
    with get_session() as s:
        # Find person rows linked to this user (by email match through User table)
        from db.models import User
        u = s.get(User, user_id)
        if u is None:
            return []
 
        # Person rows that share email with this user
        person_ids = [
            p.id for p in s.exec(
                select(Person).where(Person.email == u.email)
            ).all()
        ]
        # AdvisoryTeamMember rows directly linked to this user_id
        tm_ids = [
            m.id for m in s.exec(
                select(AdvisoryTeamMember).where(
                    AdvisoryTeamMember.user_id == user_id
                )
            ).all()
        ]
 
        if not person_ids and not tm_ids:
            return []
 
        clauses = []
        if person_ids:
            clauses.append(Task.assigned_person_id.in_(person_ids))
        if tm_ids:
            clauses.append(Task.assigned_team_member_id.in_(tm_ids))
 
        stmt = (
            select(Task)
            .where(or_(*clauses))
            .where(Task.status != "archived")
            .order_by(Task.due_date.asc().nullslast(), Task.created_at.desc())
        )
        return list(s.exec(stmt).all())
 
 
def get_task(task_id: int):
    from db.models import Task
    with get_session() as s:
        return s.get(Task, task_id)
 
 
def create_task(
    family_id: int,
    title: str,
    created_by_user_id: int,
    *,
    description: Optional[str] = None,
    assigned_person_id: Optional[int] = None,
    assigned_team_member_id: Optional[int] = None,
    due_date: Optional[date] = None,
    priority: Optional[str] = None,
    status: str = "open",
):
    from db.models import Task
    if status not in VALID_TASK_STATUSES:
        raise ValueError(f"Invalid status '{status}'. Must be in {VALID_TASK_STATUSES}")
    if priority not in VALID_TASK_PRIORITIES:
        raise ValueError(f"Invalid priority '{priority}'.")
    if assigned_person_id is not None and assigned_team_member_id is not None:
        raise ValueError(
            "A task can be assigned to a person OR a team member, not both."
        )
 
    assignee_type = None
    if assigned_person_id is not None:
        assignee_type = "person"
    elif assigned_team_member_id is not None:
        assignee_type = "team_member"
 
    with get_session() as s:
        t = Task(
            family_id=family_id,
            title=title,
            description=description,
            created_by_user_id=created_by_user_id,
            assignee_type=assignee_type,
            assigned_person_id=assigned_person_id,
            assigned_team_member_id=assigned_team_member_id,
            due_date=due_date,
            priority=priority,
            status=status,
        )
        s.add(t)
        s.flush()
        s.refresh(t)
        return t
 
 
def update_task(
    task_id: int,
    *,
    title: Optional[str] = None,
    description: Any = _UNCHANGED,
    assigned_person_id: Any = _UNCHANGED,
    assigned_team_member_id: Any = _UNCHANGED,
    due_date: Any = _UNCHANGED,
    status: Optional[str] = None,
    priority: Any = _UNCHANGED,
):
    from db.models import Task
    if status is not None and status not in VALID_TASK_STATUSES:
        raise ValueError(f"Invalid status '{status}'.")
    if priority is not _UNCHANGED and priority not in VALID_TASK_PRIORITIES:
        raise ValueError(f"Invalid priority '{priority}'.")
 
    with get_session() as s:
        t = s.get(Task, task_id)
        if t is None:
            return None
 
        if title is not None:
            t.title = title
        if description is not _UNCHANGED:
            t.description = description
        if due_date is not _UNCHANGED:
            t.due_date = due_date
        if priority is not _UNCHANGED:
            t.priority = priority
 
        # Assignment: changing either side updates assignee_type accordingly
        if assigned_person_id is not _UNCHANGED:
            t.assigned_person_id = assigned_person_id
            if assigned_person_id is not None:
                t.assigned_team_member_id = None
                t.assignee_type = "person"
            else:
                t.assignee_type = None if t.assigned_team_member_id is None else t.assignee_type
        if assigned_team_member_id is not _UNCHANGED:
            t.assigned_team_member_id = assigned_team_member_id
            if assigned_team_member_id is not None:
                t.assigned_person_id = None
                t.assignee_type = "team_member"
            else:
                t.assignee_type = None if t.assigned_person_id is None else t.assignee_type
 
        # Status transitions: stamp completed_at when entering 'complete'
        if status is not None:
            previous = t.status
            t.status = status
            if status == "complete" and previous != "complete":
                t.completed_at = datetime.utcnow()
            if status != "complete":
                t.completed_at = None
 
        t.updated_at = datetime.utcnow()
        s.add(t)
        s.flush()
        s.refresh(t)
        return t
 
 
def archive_task(task_id: int) -> bool:
    """Soft-delete: mark archived. Tasks should rarely be hard-deleted."""
    result = update_task(task_id, status="archived")
    return result is not None
 
 
def hard_delete_task(task_id: int) -> bool:
    """Actually drop the row — use only when you really mean it.
 
    Also drops associated comments and document links.
    """
    from db.models import Task, TaskComment, TaskDocumentLink
    with get_session() as s:
        t = s.get(Task, task_id)
        if t is None:
            return False
        for c in s.exec(select(TaskComment).where(TaskComment.task_id == task_id)).all():
            s.delete(c)
        for link in s.exec(select(TaskDocumentLink).where(TaskDocumentLink.task_id == task_id)).all():
            s.delete(link)
        s.delete(t)
        return True
 
 
# ════════════════════════════════════════════════════════════════════
# Task comments
# ════════════════════════════════════════════════════════════════════
 
def list_task_comments(task_id: int) -> list:
    from db.models import TaskComment
    with get_session() as s:
        return list(
            s.exec(
                select(TaskComment)
                .where(TaskComment.task_id == task_id)
                .order_by(TaskComment.created_at.asc())
            ).all()
        )
 
 
def add_task_comment(
    task_id: int,
    author_user_id: int,
    body: str,
    *,
    is_system: bool = False,
):
    from db.models import TaskComment
    with get_session() as s:
        c = TaskComment(
            task_id=task_id,
            author_user_id=author_user_id,
            body=body,
            is_system=is_system,
        )
        s.add(c)
        s.flush()
        s.refresh(c)
        return c
 
 
def delete_task_comment(comment_id: int) -> bool:
    from db.models import TaskComment
    with get_session() as s:
        c = s.get(TaskComment, comment_id)
        if c is None:
            return False
        s.delete(c)
        return True
 
 
# ════════════════════════════════════════════════════════════════════
# Task ↔ Document links
# ════════════════════════════════════════════════════════════════════
 
def list_documents_attached_to_task(task_id: int) -> list:
    from db.models import TaskDocumentLink, Document
    with get_session() as s:
        link_rows = s.exec(
            select(TaskDocumentLink).where(TaskDocumentLink.task_id == task_id)
        ).all()
        doc_ids = [link.document_id for link in link_rows]
        if not doc_ids:
            return []
        return list(s.exec(select(Document).where(Document.id.in_(doc_ids))).all())
 
 
def attach_document_to_task(
    task_id: int,
    document_id: int,
    attached_by_user_id: int,
):
    from db.models import TaskDocumentLink
    with get_session() as s:
        # Don't create duplicates
        existing = s.exec(
            select(TaskDocumentLink).where(
                (TaskDocumentLink.task_id == task_id)
                & (TaskDocumentLink.document_id == document_id)
            )
        ).first()
        if existing:
            return existing
 
        link = TaskDocumentLink(
            task_id=task_id,
            document_id=document_id,
            attached_by_user_id=attached_by_user_id,
        )
        s.add(link)
        s.flush()
        s.refresh(link)
        return link
 
 
def detach_document_from_task(task_id: int, document_id: int) -> bool:
    from db.models import TaskDocumentLink
    with get_session() as s:
        link = s.exec(
            select(TaskDocumentLink).where(
                (TaskDocumentLink.task_id == task_id)
                & (TaskDocumentLink.document_id == document_id)
            )
        ).first()
        if link is None:
            return False
        s.delete(link)
        return True
 
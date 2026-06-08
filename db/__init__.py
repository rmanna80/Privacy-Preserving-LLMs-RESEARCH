"""
Relational data layer for the Family Wealth Intelligence Platform.

This module provides SQLite-backed storage for families, people, entities,
documents, and extracted facts. It complements the existing RAG pipeline
in ai_core/ — vector search remains the same, but now we have a relational
spine for structured queries (who's the trustee?), visualizations
(family tree), and citations (extraction → source page + bbox).

Public API:
    from db import init_db, get_session
    from db import Family, Person, Entity, Role, Relationship
    from db import Document, Extraction, User, UserFamilyAccess
"""

from db.database import engine, get_session, init_db
from db.models import (
   Family,
    Person,
    Entity,
    Role,
    Relationship,
    Document,
    Extraction,
    User,
    UserFamilyAccess,
    Task,
    TaskComment,
    TaskDocumentLink,
    AdvisoryTeamMember,
)
from db.crypto import encrypt_field, decrypt_field

__all__ = [
    # connection
     "engine",
    "get_session",
    "init_db",
    "Family",
    "Person",
    "Entity",
    "Role",
    "Relationship",
    "Document",
    "Extraction",
    "User",
    "UserFamilyAccess",
    "Task",
    "TaskComment",
    "TaskDocumentLink",
    "AdvisoryTeamMember",
    "encrypt_field",
    "decrypt_field",
]
from db import get_session
from db.models import Document, Entity, Person
from sqlmodel import select

with get_session() as s:
    docs = s.exec(select(Document)).all()
    for d in docs:
        ent = s.get(Entity, d.entity_id) if d.entity_id else None
        per = s.get(Person, d.person_id) if d.person_id else None
        print(f"{d.original_filename}")
        print(f"   linked entity: {ent.name if ent else '(none)'}")
        print(f"   linked person: {per.full_name if per else '(none)'}")
    print()
    print("Entities in DB:")
    for e in s.exec(select(Entity)).all():
        print(f"  {e.id}: {e.name} ({e.entity_type})")

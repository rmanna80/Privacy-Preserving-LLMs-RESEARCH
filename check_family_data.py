from db import get_session
from db.models import Family, Person, Document
from sqlmodel import select

with get_session() as s:
    for fam in s.exec(select(Family)).all():
        print(f"\n=== Family {fam.id}: {fam.name} ===")
        people = s.exec(select(Person).where(Person.family_id == fam.id)).all()
        print("  People:", ", ".join(p.full_name for p in people) or "(none)")
        docs = s.exec(select(Document).where(Document.family_id == fam.id)).all()
        print("  Documents:", ", ".join(d.original_filename for d in docs) or "(none)")

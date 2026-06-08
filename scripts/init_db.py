"""
Initialize the wealth.db database. Idempotent — safe to run repeatedly.

Usage:
    python -m scripts.init_db              # just create empty tables
    python -m scripts.init_db --seed       # also insert a demo family
    python -m scripts.init_db --reset      # drop all data, recreate empty
                                           # (does NOT touch the encryption key)

If you need an absolutely clean slate (including the master key —
which makes any existing PII unrecoverable), delete:
    data/wealth.db
    data/.master.key
...and run this script again.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date

import bcrypt
from sqlmodel import SQLModel, select

from db.database import engine, init_db, get_session
from db.models import (
    User,
    Family,
    Person,
    Entity,
    Role,
    Relationship,
)


def hash_password(plaintext: str) -> str:
    """Hash a password with bcrypt."""
    return bcrypt.hashpw(plaintext.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def reset_schema() -> None:
    """Drop all tables and recreate them. Destructive."""
    print("⚠️  Dropping all tables...")
    SQLModel.metadata.drop_all(engine)
    print("Recreating schema...")
    init_db()
    print("Schema reset complete.")


def seed_demo_data() -> None:
    """Insert one demo family with two spouses and a revocable trust."""
    with get_session() as s:
        existing = s.exec(
            select(User).where(User.email == "advisor@demo.com")
        ).first()
        if existing:
            print("Demo data already present — skipping seed.")
            return

        # ---- demo advisor ----
        advisor = User(
            email="advisor@demo.com",
            password_hash=hash_password("demo1234"),
            full_name="Demo Advisor",
            role="advisor",
        )
        s.add(advisor)
        s.flush()  # populate advisor.id

        # ---- family ----
        family = Family(name="The Demo Family", advisor_user_id=advisor.id)
        s.add(family)
        s.flush()

        # ---- two spouses ----
        john = Person(
            family_id=family.id,
            first_name="John",
            last_name="Demo",
            dob=date(1965, 4, 12),
            email="john@example.test",
        )
        john.ssn = "123-45-6789"  # encrypted via the property setter

        sally = Person(
            family_id=family.id,
            first_name="Sally",
            last_name="Demo",
            dob=date(1968, 8, 22),
            email="sally@example.test",
        )
        sally.ssn = "987-65-4321"

        s.add_all([john, sally])
        s.flush()

        # ---- their marriage ----
        s.add(
            Relationship(
                person_a_id=john.id,
                person_b_id=sally.id,
                relationship_type="spouse",
                start_date=date(1990, 6, 15),
            )
        )

        # ---- a revocable trust ----
        trust = Entity(
            family_id=family.id,
            name="The Demo Family Revocable Trust",
            entity_type="trust",
            sub_type="revocable",
            jurisdiction="NC",
            formation_date=date(2015, 1, 1),
        )
        trust.tax_id = "12-3456789"
        s.add(trust)
        s.flush()

        # Both spouses are co-grantors and co-trustees
        for person in (john, sally):
            s.add(
                Role(
                    person_id=person.id,
                    entity_id=trust.id,
                    role_type="grantor",
                )
            )
            s.add(
                Role(
                    person_id=person.id,
                    entity_id=trust.id,
                    role_type="co_trustee",
                )
            )

        print(f"✅ Seeded demo family (id={family.id}) with 2 people and 1 trust.")
        print()
        print("   Login credentials:")
        print("     email:    advisor@demo.com")
        print("     password: demo1234")


def verify_install() -> None:
    """Quick smoke test — read back what we just wrote."""
    print()
    print("Verifying install...")
    with get_session() as s:
        users = s.exec(select(User)).all()
        families = s.exec(select(Family)).all()
        people = s.exec(select(Person)).all()
        print(f"  users:    {len(users)}")
        print(f"  families: {len(families)}")
        print(f"  people:   {len(people)}")

        # Round-trip an encrypted field to verify crypto layer works
        if people:
            p = people[0]
            decrypted = p.ssn
            print(f"  PII round-trip:  {p.full_name} → SSN reads as: {decrypted}")
    print("✅ All good.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed",
        action="store_true",
        help="Insert demo family / advisor data after creating tables",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate all tables. Destroys all DB data.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Print a summary of DB contents after operations",
    )
    args = parser.parse_args()

    if args.reset:
        reset_schema()
    else:
        print("Creating database schema (idempotent)...")
        init_db()
        print("✅ Schema ready at data/wealth.db")

    if args.seed:
        print()
        print("Seeding demo data...")
        seed_demo_data()

    if args.verify or args.seed:
        verify_install()

    return 0


if __name__ == "__main__":
    sys.exit(main())
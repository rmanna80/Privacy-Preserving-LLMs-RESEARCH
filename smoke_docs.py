from db import get_session, Family
from db.repositories import (
    compute_file_hash,
    find_document_by_hash,
    create_document,
    list_documents_for_family,
    list_documents_by_category,
    archive_document,
    hard_delete_document,
    DOCUMENT_CATEGORIES,
)
from sqlmodel import select

with get_session() as s:
    family = s.exec(select(Family)).first()
    if not family:
        print("No family found — create one in the UI first.")
        raise SystemExit(0)
    family_id = family.id
    advisor_id = family.advisor_user_id

print(f"Testing on family: {family.name} (id={family_id})")

# Pretend we have a file
fake_bytes = b"This is a fake PDF for testing purposes."
file_hash = compute_file_hash(fake_bytes)
print(f"  Computed hash: {file_hash[:16]}...")

# No dedup match yet
existing = find_document_by_hash(family_id, file_hash)
assert existing is None, "Unexpected: file already exists by hash"
print(f"  Dedup check: no existing match (expected)")

# Create the document
doc = create_document(
    family_id=family_id,
    file_path="/tmp/fake_test.pdf",
    file_hash=file_hash,
    original_filename="test_will.pdf",
    file_size_bytes=len(fake_bytes),
    mime_type="application/pdf",
    category="estate_planning",
    doc_type="will",
    doc_year=2024,
    notes="Smoke test fake document",
    uploaded_by_user_id=advisor_id,
)
print(f"  Created document id={doc.id} category={doc.category}")

# Dedup should now hit
existing = find_document_by_hash(family_id, file_hash)
assert existing is not None and existing.id == doc.id
print(f"  Dedup check: existing match found (expected)")

# List by family
docs = list_documents_for_family(family_id)
print(f"  Documents in family: {len(docs)}")

# List by category
grouped = list_documents_by_category(family_id)
for cat, docs in grouped.items():
    if docs:
        print(f"    {cat}: {len(docs)} doc(s)")

# Archive
archived = archive_document(doc.id)
assert archived, "Archive failed"
print(f"  Archived doc")

# Clean up the test row
hard_delete_document(doc.id, delete_file=False)
print(f"  Hard-deleted doc")

print("OK - documents layer works end-to-end.")

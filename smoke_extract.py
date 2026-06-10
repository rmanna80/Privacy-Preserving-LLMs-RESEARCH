from db import get_session, Family
from db.repositories import (
    list_documents_for_family,
    create_extraction, list_extractions_for_document,
    verify_extraction, reject_extraction,
    list_verified_extractions_for_family, extraction_plain_value,
)
from sqlmodel import select

with get_session() as s:
    family = s.exec(select(Family)).first()
    family_id = family.id
    advisor_id = family.advisor_user_id

docs = list_documents_for_family(family_id)
if not docs:
    print("No documents — upload one in the UI first, then re-run.")
    raise SystemExit(0)
doc = docs[0]
print(f"Testing on document: {doc.original_filename} (id={doc.id})")

# Create a manual extraction
e = create_extraction(
    document_id=doc.id, field_key="trust_name",
    field_value="The Smoke Test Trust", extraction_type="text",
    extracted_by="manual", confidence=1.0,
)
print(f"  Created extraction id={e.id}")

# Create a PII extraction (encrypted at rest)
e_pii = create_extraction(
    document_id=doc.id, field_key="trust_ein",
    field_value="12-3456789", extraction_type="text",
    is_pii=True, extracted_by="manual",
)
print(f"  PII stored encrypted: {e_pii.field_value[:20]}...")
print(f"  PII decrypts to: {extraction_plain_value(e_pii)}")
assert extraction_plain_value(e_pii) == "12-3456789"

# Verify with a correction
verify_extraction(e.id, advisor_id, corrected_value="The Corrected Trust")
verified = list_verified_extractions_for_family(family_id)
print(f"  Verified extractions in family: {len(verified)}")
assert any(x.field_value == "The Corrected Trust" for x in verified)

# List + clean up
all_e = list_extractions_for_document(doc.id)
print(f"  Total extractions on doc: {len(all_e)}")
for x in all_e:
    reject_extraction(x.id)
print("  Cleaned up.")
print("OK - extraction layer works end-to-end.")

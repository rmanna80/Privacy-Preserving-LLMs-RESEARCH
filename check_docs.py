from db.models import Family, Document

# Family should NOT have category
family_fields = Family.__fields__ if hasattr(Family, "__fields__") else Family.model_fields
assert "category" not in family_fields, "BUG: category is still on Family"
print("Family: no category field ✓")

# Document SHOULD have category
doc_fields = Document.__fields__ if hasattr(Document, "__fields__") else Document.model_fields
assert "category" in doc_fields, "BUG: category is missing from Document"
print("Document: has category field ✓")
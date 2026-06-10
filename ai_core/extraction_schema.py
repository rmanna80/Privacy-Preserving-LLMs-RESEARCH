"""
ai_core/extraction_schema.py — canonical extraction fields per doc_type.

This is the single source of truth for WHAT we extract. Used by:
  - The manual "Add Fact" form (field dropdown)
  - The AI extractor (prompt construction)
  - Downstream pages that consume extracted data (they can rely on
    stable field_key values)

Design: schema-based extraction (decision: option B). Every extractable
fact has a stable field_key. The AI is asked for exactly these fields —
nothing open-ended — so downstream consumers (Family Tree, Entities,
future Reports) can join on field_key reliably.

Each field is a dict:
  key        — stable identifier stored in Extraction.field_key
  label      — human-readable name for the UI
  hint       — guidance injected into the AI prompt
  is_pii     — if True, the value is encrypted at rest
  value_type — 'text' | 'date' | 'yes_no' | 'number' | 'list'
               (list values are stored as "; "-joined text in v1)
"""

from __future__ import annotations


EXTRACTION_FIELDS_BY_DOC_TYPE: dict[str, list[dict]] = {

    # ──────────────────────────────────────────────────────────────
    "trust_agreement": [
        {"key": "trust_name",            "label": "Trust Name",
         "hint": "The formal name of the trust, e.g. 'The Smith Family Revocable Trust'.",
         "is_pii": False, "value_type": "text"},
        {"key": "trust_type",            "label": "Trust Type",
         "hint": "revocable, irrevocable, GRAT, ILIT, CRUT, dynasty, etc.",
         "is_pii": False, "value_type": "text"},
        {"key": "trustor",               "label": "Trustor(s) / Settlor(s)",
         "hint": "Who created the trust. May be one person or a married couple (joint).",
         "is_pii": False, "value_type": "list"},
        {"key": "creation_date",         "label": "Creation Date",
         "hint": "The date the trust was signed/established.",
         "is_pii": False, "value_type": "date"},
        {"key": "jurisdiction",          "label": "Governing State",
         "hint": "The state whose law governs the trust, e.g. California, Delaware.",
         "is_pii": False, "value_type": "text"},
        {"key": "current_trustee",       "label": "Current Trustee(s)",
         "hint": "Who currently serves as trustee. May be multiple people or a corporate trustee.",
         "is_pii": False, "value_type": "list"},
        {"key": "successor_trustee",     "label": "Successor Trustee(s)",
         "hint": "Who takes over as trustee if the current trustee cannot serve, in order.",
         "is_pii": False, "value_type": "list"},
        {"key": "beneficiaries_current", "label": "Current Beneficiaries",
         "hint": "Who currently benefits from the trust.",
         "is_pii": False, "value_type": "list"},
        {"key": "beneficiaries_remainder","label": "Remainder Beneficiaries",
         "hint": "Who receives the trust assets upon termination.",
         "is_pii": False, "value_type": "list"},
        {"key": "income_distribution",   "label": "Income Distribution Terms",
         "hint": "How/when income must be distributed, e.g. 'mandatory, at least quarterly'.",
         "is_pii": False, "value_type": "text"},
        {"key": "principal_distribution","label": "Principal Distribution Standard",
         "hint": "Standard for principal distributions, e.g. HEMS (health, education, maintenance, support).",
         "is_pii": False, "value_type": "text"},
        {"key": "termination_event",     "label": "Trust Termination Event",
         "hint": "When the trust ends, e.g. 'beneficiary attains age 35', 'death of surviving spouse'.",
         "is_pii": False, "value_type": "text"},
        {"key": "gst_exempt",            "label": "GST Tax Exempt?",
         "hint": "Is any portion expected to be generation-skipping-transfer tax exempt? yes or no.",
         "is_pii": False, "value_type": "yes_no"},
        {"key": "spendthrift_clause",    "label": "Spendthrift Clause?",
         "hint": "Does the trust contain a spendthrift clause? yes or no.",
         "is_pii": False, "value_type": "yes_no"},
        {"key": "no_contest_clause",     "label": "No-Contest Clause?",
         "hint": "Does the trust contain a no-contest (in terrorem) clause? yes or no.",
         "is_pii": False, "value_type": "yes_no"},
        {"key": "trust_ein",             "label": "Trust EIN",
         "hint": "The trust's tax identification number if stated (format XX-XXXXXXX).",
         "is_pii": True,  "value_type": "text"},
    ],

    # ──────────────────────────────────────────────────────────────
    "tax_return_1040": [
        {"key": "tax_year",              "label": "Tax Year",
         "hint": "The year this return covers, e.g. 2024.",
         "is_pii": False, "value_type": "number"},
        {"key": "filing_status",         "label": "Filing Status",
         "hint": "single, married filing jointly, married filing separately, head of household.",
         "is_pii": False, "value_type": "text"},
        {"key": "primary_taxpayer",      "label": "Primary Taxpayer",
         "hint": "Name of the primary taxpayer on the return.",
         "is_pii": False, "value_type": "text"},
        {"key": "spouse_name",           "label": "Spouse Name",
         "hint": "Name of the spouse, if filing jointly.",
         "is_pii": False, "value_type": "text"},
        {"key": "adjusted_gross_income", "label": "Adjusted Gross Income",
         "hint": "AGI from line 11 (Form 1040). Numeric dollar amount.",
         "is_pii": False, "value_type": "number"},
        {"key": "total_tax",             "label": "Total Tax",
         "hint": "Total tax liability. Numeric dollar amount.",
         "is_pii": False, "value_type": "number"},
        {"key": "taxpayer_ssn",          "label": "Taxpayer SSN",
         "hint": "Primary taxpayer's social security number if visible (format XXX-XX-XXXX).",
         "is_pii": True,  "value_type": "text"},
    ],

    # ──────────────────────────────────────────────────────────────
    "will": [
        {"key": "testator",              "label": "Testator",
         "hint": "The person whose will this is.",
         "is_pii": False, "value_type": "text"},
        {"key": "execution_date",        "label": "Execution Date",
         "hint": "The date the will was signed.",
         "is_pii": False, "value_type": "date"},
        {"key": "executor_primary",      "label": "Primary Executor",
         "hint": "Who is named executor / personal representative.",
         "is_pii": False, "value_type": "text"},
        {"key": "executor_alternate",    "label": "Alternate Executor",
         "hint": "Who serves if the primary executor cannot.",
         "is_pii": False, "value_type": "text"},
        {"key": "guardian_minors",       "label": "Guardian for Minor Children",
         "hint": "Who is named guardian for any minor children.",
         "is_pii": False, "value_type": "text"},
        {"key": "specific_bequests",     "label": "Specific Bequests",
         "hint": "Specific gifts to named people or charities, with amounts/items.",
         "is_pii": False, "value_type": "list"},
        {"key": "residuary_beneficiaries","label": "Residuary Beneficiaries",
         "hint": "Who receives the residue of the estate, and in what shares.",
         "is_pii": False, "value_type": "list"},
    ],

    # ──────────────────────────────────────────────────────────────
    "insurance_policy": [
        {"key": "carrier",               "label": "Insurance Carrier",
         "hint": "The insurance company issuing the policy.",
         "is_pii": False, "value_type": "text"},
        {"key": "policy_type",           "label": "Policy Type",
         "hint": "term life, whole life, universal life, umbrella, etc.",
         "is_pii": False, "value_type": "text"},
        {"key": "insured",               "label": "Insured Person",
         "hint": "Whose life/property is insured.",
         "is_pii": False, "value_type": "text"},
        {"key": "owner",                 "label": "Policy Owner",
         "hint": "Who owns the policy (may differ from insured, e.g. an ILIT).",
         "is_pii": False, "value_type": "text"},
        {"key": "death_benefit",         "label": "Death Benefit / Coverage Amount",
         "hint": "The face amount or coverage limit. Numeric dollar amount.",
         "is_pii": False, "value_type": "number"},
        {"key": "beneficiary_primary",   "label": "Primary Beneficiary",
         "hint": "Who receives the benefit.",
         "is_pii": False, "value_type": "list"},
        {"key": "policy_number",         "label": "Policy Number",
         "hint": "The policy identifier.",
         "is_pii": True,  "value_type": "text"},
    ],
}

# Fallback fields offered for any doc_type not listed above.
GENERIC_FIELDS: list[dict] = [
    {"key": "document_title",  "label": "Document Title",
     "hint": "The title or heading of the document.",
     "is_pii": False, "value_type": "text"},
    {"key": "parties",         "label": "Parties Involved",
     "hint": "The people or entities this document concerns.",
     "is_pii": False, "value_type": "list"},
    {"key": "effective_date",  "label": "Effective / Signing Date",
     "hint": "The date the document was signed or takes effect.",
     "is_pii": False, "value_type": "date"},
    {"key": "key_terms",       "label": "Key Terms",
     "hint": "The most important provisions or facts in this document.",
     "is_pii": False, "value_type": "list"},
]


def fields_for_doc_type(doc_type: str) -> list[dict]:
    """Return the canonical field list for a doc_type, falling back to
    the generic set for unknown types."""
    return EXTRACTION_FIELDS_BY_DOC_TYPE.get(doc_type, GENERIC_FIELDS)


def field_def(doc_type: str, field_key: str) -> dict | None:
    """Look up one field definition by key."""
    for f in fields_for_doc_type(doc_type):
        if f["key"] == field_key:
            return f
    return None

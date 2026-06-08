from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Pattern, Tuple


class DisclosureMode(Enum):
    OPEN = "open"
    AUTHORIZED = "authorized"
    REDACTED = "redacted"


class SensitivityType(Enum):
    DIRECT_IDENTIFIER = "direct_identifier"
    FINANCIAL_IDENTIFIER = "financial_identifier"
    QUASI_IDENTIFIER = "quasi_identifier"
    UNRESTRICTED = "unrestricted"


@dataclass(frozen=True)
class SensitiveMatch:
    label: str
    sensitivity: SensitivityType
    start: int
    end: int
    value: str


@dataclass(frozen=True)
class PolicyDecision:
    original_text: str
    output_text: str
    mode: DisclosureMode
    grounded: bool
    authorized: bool
    allowed: bool
    sensitive_matches: Tuple[SensitiveMatch, ...] = field(default_factory=tuple)
    reasons: Tuple[str, ...] = field(default_factory=tuple)


class PrivacyPolicy:
    """
    Central privacy policy engine for controlling disclosure of sensitive content.

    Design goals:
    - separate sensitive-content detection from disclosure decisions
    - support more than just SSNs
    - return structured decisions
    - keep a compatibility `enforce()` method so older code keeps working
    """

    SENSITIVE_PATTERNS: Dict[str, Tuple[SensitivityType, Pattern[str]]] = {
        "ssn": (
            SensitivityType.DIRECT_IDENTIFIER,
            re.compile(r"\b\d{3}[- ]?\d{2}[- ]?\d{4}\b"),
        ),
        "ein": (
            SensitivityType.FINANCIAL_IDENTIFIER,
            re.compile(r"\b\d{2}[- ]?\d{7}\b"),
        ),
        "account_number": (
            SensitivityType.FINANCIAL_IDENTIFIER,
            re.compile(r"\b(?:account|acct)\s*[:#-]?\s*\d{4,17}\b", re.IGNORECASE),
        ),
        "routing_number": (
            SensitivityType.FINANCIAL_IDENTIFIER,
            re.compile(r"\b(?:routing)\s*[:#-]?\s*\d{9}\b", re.IGNORECASE),
        ),
        "dob": (
            SensitivityType.QUASI_IDENTIFIER,
            re.compile(
                r"\b(?:dob|date\s+of\s+birth)\s*[:#-]?\s*"
                r"(?:\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2})\b",
                re.IGNORECASE,
            ),
        ),
    }

    DEFAULT_PLACEHOLDERS: Dict[str, str] = {
        "ssn": "[SSN]",
        "ein": "[EIN]",
        "account_number": "[ACCOUNT_NUMBER]",
        "routing_number": "[ROUTING_NUMBER]",
        "dob": "[DOB]",
    }

    @classmethod
    def detect_sensitive_fields(cls, text: str) -> List[SensitiveMatch]:
        """
        Scan text for known sensitive patterns and return structured matches.
        """
        if not text:
            return []

        matches: List[SensitiveMatch] = []

        for label, (sensitivity, pattern) in cls.SENSITIVE_PATTERNS.items():
            for match in pattern.finditer(text):
                matches.append(
                    SensitiveMatch(
                        label=label,
                        sensitivity=sensitivity,
                        start=match.start(),
                        end=match.end(),
                        value=match.group(0),
                    )
                )

        matches.sort(key=lambda item: (item.start, item.end))
        return cls._deduplicate_overlaps(matches)

    @classmethod
    def has_sensitive_content(cls, text: str) -> bool:
        """
        Convenience check for whether any sensitive content is present.
        """
        return bool(cls.detect_sensitive_fields(text))

    @classmethod
    def evaluate_disclosure(
        cls,
        text: str,
        mode: DisclosureMode,
        grounded: bool,
        authorized: bool = False,
    ) -> PolicyDecision:
        """
        Evaluate whether text can be disclosed under the current privacy mode.

        Rules:
        - OPEN: return original text
        - AUTHORIZED:
            - if grounded and authorized, return original text
            - otherwise return masked text
        - REDACTED: always return masked text if sensitive content is detected
        """
        sensitive_matches = tuple(cls.detect_sensitive_fields(text))
        reasons: List[str] = []

        if mode == DisclosureMode.OPEN:
            reasons.append("Open mode allows disclosure without masking.")
            return PolicyDecision(
                original_text=text,
                output_text=text,
                mode=mode,
                grounded=grounded,
                authorized=authorized,
                allowed=True,
                sensitive_matches=sensitive_matches,
                reasons=tuple(reasons),
            )

        masked_text = cls.mask_sensitive_fields(text, list(sensitive_matches))

        if mode == DisclosureMode.AUTHORIZED:
            if authorized and grounded:
                reasons.append(
                    "Authorized mode allows grounded disclosure for authorized users."
                )
                return PolicyDecision(
                    original_text=text,
                    output_text=text,
                    mode=mode,
                    grounded=grounded,
                    authorized=authorized,
                    allowed=True,
                    sensitive_matches=sensitive_matches,
                    reasons=tuple(reasons),
                )

            if not authorized:
                reasons.append("User is not authorized for sensitive disclosure.")
            if not grounded:
                reasons.append("Answer is not sufficiently grounded in retrieved evidence.")

            return PolicyDecision(
                original_text=text,
                output_text=masked_text,
                mode=mode,
                grounded=grounded,
                authorized=authorized,
                allowed=False,
                sensitive_matches=sensitive_matches,
                reasons=tuple(reasons),
            )

        if mode == DisclosureMode.REDACTED:
            reasons.append("Redacted mode always masks detected sensitive content.")
            return PolicyDecision(
                original_text=text,
                output_text=masked_text,
                mode=mode,
                grounded=grounded,
                authorized=authorized,
                allowed=False,
                sensitive_matches=sensitive_matches,
                reasons=tuple(reasons),
            )

        reasons.append("Unknown mode received; returning original text unchanged.")
        return PolicyDecision(
            original_text=text,
            output_text=text,
            mode=mode,
            grounded=grounded,
            authorized=authorized,
            allowed=True,
            sensitive_matches=sensitive_matches,
            reasons=tuple(reasons),
        )

    @classmethod
    def mask_sensitive_fields(
        cls,
        text: str,
        matches: List[SensitiveMatch] | None = None,
    ) -> str:
        """
        Replace sensitive spans with placeholders.
        """
        if not text:
            return text

        matches = matches or cls.detect_sensitive_fields(text)
        if not matches:
            return text

        parts: List[str] = []
        cursor = 0

        for item in matches:
            if item.start < cursor:
                continue

            parts.append(text[cursor:item.start])
            parts.append(cls.DEFAULT_PLACEHOLDERS.get(item.label, "[REDACTED]"))
            cursor = item.end

        parts.append(text[cursor:])
        return "".join(parts)

    @classmethod
    def enforce(
        cls,
        text: str,
        mode: DisclosureMode,
        grounded: bool,
        authorized: bool = False,
    ) -> str:
        """
        Backward-compatible wrapper for older code paths.
        Returns only the output text.
        """
        decision = cls.evaluate_disclosure(
            text=text,
            mode=mode,
            grounded=grounded,
            authorized=authorized,
        )
        return decision.output_text

    @staticmethod
    def _deduplicate_overlaps(matches: List[SensitiveMatch]) -> List[SensitiveMatch]:
        """
        Resolve overlapping regex matches by keeping the longest one.
        """
        if not matches:
            return []

        deduped: List[SensitiveMatch] = []
        current = matches[0]

        for item in matches[1:]:
            if item.start < current.end:
                current_len = current.end - current.start
                item_len = item.end - item.start
                if item_len > current_len:
                    current = item
            else:
                deduped.append(current)
                current = item

        deduped.append(current)
        return deduped
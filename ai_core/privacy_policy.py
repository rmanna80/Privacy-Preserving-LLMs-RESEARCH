# ai_core/privacy_policy.py

import re
from enum import Enum


class DisclosureMode(Enum):
    OPEN = "open"
    AUTHORIZED = "authorized"
    REDACTED = "redacted"


class PrivacyPolicy:
    SSN_PATTERN = re.compile(r"\b\d{3}[- ]?\d{2}[- ]?\d{4}\b")

    @staticmethod
    def enforce(
        text: str,
        mode: DisclosureMode,
        grounded: bool,
        authorized: bool = False,
    ) -> str:
        """
        Apply privacy policy to model output.

        - OPEN: always return text
        - AUTHORIZED: return text only if grounded
        - REDACTED: always mask sensitive data
        """
        if mode == DisclosureMode.OPEN:
            return text

        if mode == DisclosureMode.AUTHORIZED:
            if authorized and grounded:
                return text
            return PrivacyPolicy.SSN_PATTERN.sub("[SSN]", text)

        if mode == DisclosureMode.REDACTED:
            return PrivacyPolicy.SSN_PATTERN.sub("[SSN]", text)

        return text

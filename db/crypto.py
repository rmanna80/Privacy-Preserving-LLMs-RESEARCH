"""
Field-level encryption for PII (SSN, EIN, account numbers).

v1 design: a single master key stored in data/.master.key, generated on
first run. AES-GCM 256 via the `cryptography` library.

Why this approach for v1:
- One key for the whole DB — simple, no user-key management yet.
- The key file lives outside the repo (gitignored). If an attacker gets
  the DB but not the key file, PII stays encrypted.
- Honest limitation: the running app process can read both, so this
  protects against exfiltrated DB backups, not against full host
  compromise. That's a reasonable v1 posture.

Future (v2):
- Per-user keys derived from password via Argon2.
- Re-encryption tooling for key rotation.
- Optionally: KMS / HSM integration for enterprise deployments.

Storage format:
    [12-byte nonce][ciphertext+tag]   stored as bytes (BLOB column)
"""

import os
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

KEY_PATH = Path("data/.master.key")
KEY_PATH.parent.mkdir(parents=True, exist_ok=True)

_NONCE_BYTES = 12  # AES-GCM standard


def _load_or_create_key() -> bytes:
    """Load the master key, generating it on first run."""
    if KEY_PATH.exists():
        key = KEY_PATH.read_bytes()
        if len(key) != 32:
            raise RuntimeError(
                f"Master key at {KEY_PATH} is corrupted (expected 32 bytes, "
                f"got {len(key)}). If this is a fresh install, delete the "
                f"file and let it regenerate. WARNING: doing so will make "
                f"any existing encrypted data unreadable."
            )
        return key

    key = AESGCM.generate_key(bit_length=256)
    KEY_PATH.write_bytes(key)
    # Restrict permissions on POSIX. Windows ignores chmod; protect via
    # NTFS ACLs / folder permissions in deployment.
    try:
        os.chmod(KEY_PATH, 0o600)
    except (PermissionError, OSError):
        pass
    return key


_KEY = _load_or_create_key()
_AESGCM = AESGCM(_KEY)


def encrypt_field(plaintext: str | None) -> bytes | None:
    """Encrypt a string. Returns None for None/empty input.

    The returned bytes are nonce || ciphertext_with_tag.
    """
    if not plaintext:
        return None
    nonce = os.urandom(_NONCE_BYTES)
    ciphertext = _AESGCM.encrypt(nonce, plaintext.encode("utf-8"), None)
    return nonce + ciphertext


def decrypt_field(blob: bytes | None) -> str | None:
    """Decrypt a stored field. Returns None for None/empty input."""
    if not blob:
        return None
    if len(blob) < _NONCE_BYTES + 16:  # nonce + min GCM tag
        raise ValueError("Encrypted field is too short to be valid.")
    nonce, ciphertext = blob[:_NONCE_BYTES], blob[_NONCE_BYTES:]
    return _AESGCM.decrypt(nonce, ciphertext, None).decode("utf-8")
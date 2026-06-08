from __future__ import annotations

import hashlib
import hmac
import json
import logging
import secrets
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class UserRole(Enum):
    SUPER_ADMIN = "super_admin"
    ADVISOR = "advisor"
    CLIENT = "client"


@dataclass
class User:
    username: str
    role: UserRole
    client_name: str = ""
    advisor_id: Optional[str] = None
    session_token: Optional[str] = None

    def is_admin(self) -> bool:
        return self.role == UserRole.SUPER_ADMIN

    def is_advisor(self) -> bool:
        return self.role == UserRole.ADVISOR

    def is_client(self) -> bool:
        return self.role == UserRole.CLIENT


class AuthSystem:
    _failed_attempts: Dict[str, list] = {}

    MAX_ATTEMPTS = 5
    LOCKOUT_WINDOW = 300  # seconds

    PBKDF2_ITERATIONS = 200_000
    SALT_BYTES = 16
    MIN_PASSWORD_LENGTH = 8

    def __init__(self, users_file: str | None = None, seed_demo_users: bool = True):
        project_root = Path(__file__).resolve().parents[1]
        self.users_file = Path(users_file) if users_file else project_root / "data" / "users.json"
        self.users_file.parent.mkdir(parents=True, exist_ok=True)

        if seed_demo_users:
            self._init_default_users()
        elif not self.users_file.exists():
            self._save({})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_username(username: str) -> str:
        return username.strip().lower()

    def _load(self) -> dict:
        if not self.users_file.exists():
            return {}
        return json.loads(self.users_file.read_text(encoding="utf-8"))

    def _save(self, data: dict) -> None:
        self.users_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _hash_password(self, password: str, salt: Optional[str] = None) -> Dict[str, str]:
        """
        Create a PBKDF2 password hash record.
        """
        salt = salt or secrets.token_hex(self.SALT_BYTES)
        derived = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(salt),
            self.PBKDF2_ITERATIONS,
        )
        return {
            "scheme": "pbkdf2_sha256",
            "salt": salt,
            "hash": derived.hex(),
        }

    def _verify_password(self, password: str, stored_record: Any) -> tuple[bool, Optional[Dict[str, str]]]:
        """
        Verify either:
        - new PBKDF2 records
        - legacy plain SHA-256 hex strings from the old demo version

        Returns:
        - success flag
        - optional upgraded hash record to save back
        """
        if isinstance(stored_record, dict):
            scheme = stored_record.get("scheme")
            if scheme != "pbkdf2_sha256":
                return False, None

            salt = stored_record.get("salt", "")
            expected_hash = stored_record.get("hash", "")
            candidate = self._hash_password(password, salt=salt)
            ok = hmac.compare_digest(candidate["hash"], expected_hash)
            return ok, None

        if isinstance(stored_record, str):
            legacy_sha256 = hashlib.sha256(password.encode("utf-8")).hexdigest()
            ok = hmac.compare_digest(legacy_sha256, stored_record)
            if ok:
                upgraded = self._hash_password(password)
                return True, upgraded
            return False, None

        return False, None

    def _password_is_valid(self, password: str) -> bool:
        return len(password) >= self.MIN_PASSWORD_LENGTH

    def _is_locked_out(self, username: str) -> bool:
        now = time.time()
        recent = [t for t in self._failed_attempts.get(username, []) if now - t < self.LOCKOUT_WINDOW]
        self._failed_attempts[username] = recent
        return len(recent) >= self.MAX_ATTEMPTS

    def _record_failure(self, username: str) -> None:
        self._failed_attempts.setdefault(username, []).append(time.time())

    def _clear_failures(self, username: str) -> None:
        self._failed_attempts.pop(username, None)

    def _make_user(self, username: str, record: dict) -> User:
        return User(
            username=username,
            role=UserRole(record["role"]),
            client_name=record.get("name", username),
            advisor_id=record.get("advisor_id"),
            session_token=None,
        )

    # ------------------------------------------------------------------
    # Seed data
    # ------------------------------------------------------------------
    def _init_default_users(self) -> None:
        if self.users_file.exists():
            return

        default = {
            "admin@demo.com": {
                "password": self._hash_password("admin12345"),
                "role": "super_admin",
                "name": "System Administrator",
                "advisor_id": None,
            },
            "advisor.adam@demo.com": {
                "password": self._hash_password("advisor12345"),
                "role": "advisor",
                "name": "Adam Advisor",
                "advisor_id": None,
            },
            "advisor.jake@demo.com": {
                "password": self._hash_password("advisor45678"),
                "role": "advisor",
                "name": "Jake Advisor",
                "advisor_id": None,
            },
            "john.smith@demo.com": {
                "password": self._hash_password("client12345"),
                "role": "client",
                "name": "John Smith",
                "advisor_id": "advisor.adam@demo.com",
            },
            "sally.smith@demo.com": {
                "password": self._hash_password("client12345"),
                "role": "client",
                "name": "Sally Smith",
                "advisor_id": "advisor.adam@demo.com",
            },
            "peter.professor@demo.com": {
                "password": self._hash_password("client12345"),
                "role": "client",
                "name": "Peter Professor",
                "advisor_id": "advisor.jake@demo.com",
            },
        }
        self._save(default)

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------
    def authenticate(self, username: str, password: str) -> Optional[User]:
        username = self._normalize_username(username)

        if self._is_locked_out(username):
            logger.warning("Locked-out login attempt for %s", username)
            return None

        try:
            users = self._load()
            record = users.get(username)
            if not record:
                self._record_failure(username)
                return None

            ok, upgraded_record = self._verify_password(password, record.get("password"))
            if not ok:
                self._record_failure(username)
                return None

            if upgraded_record is not None:
                record["password"] = upgraded_record
                users[username] = record
                self._save(users)

            self._clear_failures(username)

            user = self._make_user(username, record)
            user.session_token = secrets.token_hex(32)
            return user

        except Exception as exc:
            logger.exception("Authentication error for %s: %s", username, exc)
            return None

    def is_locked_out(self, username: str) -> bool:
        return self._is_locked_out(self._normalize_username(username))

    # ------------------------------------------------------------------
    # User queries
    # ------------------------------------------------------------------
    def get_all_advisors(self) -> List[User]:
        users = self._load()
        return [
            self._make_user(username, record)
            for username, record in users.items()
            if record.get("role") == UserRole.ADVISOR.value
        ]

    def get_clients_for_advisor(self, advisor_username: str) -> List[User]:
        advisor_username = self._normalize_username(advisor_username)
        users = self._load()
        return [
            self._make_user(username, record)
            for username, record in users.items()
            if record.get("role") == UserRole.CLIENT.value
            and record.get("advisor_id") == advisor_username
        ]

    def get_all_clients(self) -> List[User]:
        users = self._load()
        return [
            self._make_user(username, record)
            for username, record in users.items()
            if record.get("role") == UserRole.CLIENT.value
        ]

    def get_user(self, username: str) -> Optional[User]:
        username = self._normalize_username(username)
        try:
            record = self._load().get(username)
            if not record:
                return None
            return self._make_user(username, record)
        except Exception:
            logger.exception("Failed to load user %s", username)
            return None

    # ------------------------------------------------------------------
    # Admin CRUD
    # ------------------------------------------------------------------
    def create_user(
        self,
        username: str,
        password: str,
        role: str,
        name: str,
        advisor_id: Optional[str] = None,
    ) -> bool:
        try:
            username = self._normalize_username(username)
            role = role.strip().lower()

            if role not in {r.value for r in UserRole}:
                logger.warning("Rejected create_user with invalid role: %s", role)
                return False

            if not self._password_is_valid(password):
                logger.warning("Rejected create_user for %s due to weak password", username)
                return False

            users = self._load()
            if username in users:
                return False

            if role == UserRole.CLIENT.value and advisor_id:
                advisor_id = self._normalize_username(advisor_id)
                advisor = users.get(advisor_id)
                if not advisor or advisor.get("role") != UserRole.ADVISOR.value:
                    logger.warning("Rejected client creation with invalid advisor_id: %s", advisor_id)
                    return False

            users[username] = {
                "password": self._hash_password(password),
                "role": role,
                "name": name,
                "advisor_id": advisor_id,
            }
            self._save(users)
            return True

        except Exception as exc:
            logger.exception("create_user error: %s", exc)
            return False

    def delete_user(self, username: str) -> bool:
        try:
            username = self._normalize_username(username)
            users = self._load()
            if username not in users:
                return False
            del users[username]
            self._save(users)
            return True
        except Exception:
            logger.exception("delete_user failed for %s", username)
            return False

    def reassign_client(self, client_username: str, new_advisor_username: str) -> bool:
        """
        Move a client from one advisor to another, verifying the new advisor exists.
        """
        try:
            client_username = self._normalize_username(client_username)
            new_advisor_username = self._normalize_username(new_advisor_username)

            users = self._load()
            client = users.get(client_username)
            advisor = users.get(new_advisor_username)

            if not client or client.get("role") != UserRole.CLIENT.value:
                return False
            if not advisor or advisor.get("role") != UserRole.ADVISOR.value:
                return False

            client["advisor_id"] = new_advisor_username
            users[client_username] = client
            self._save(users)
            return True

        except Exception:
            logger.exception(
                "reassign_client failed for client=%s advisor=%s",
                client_username,
                new_advisor_username,
            )
            return False

    # ------------------------------------------------------------------
    # Document directories
    # ------------------------------------------------------------------
    @staticmethod
    def _root() -> Path:
        return Path(__file__).resolve().parents[1]

    def get_user_documents_dir(self, user: User) -> Path:
        if user.is_advisor():
            path = self._root() / "data" / "advisors" / user.username / "docs"
            path.mkdir(parents=True, exist_ok=True)
            return path

        if user.is_client():
            return self.get_client_documents_dir(user.username)

        path = self._root() / "data" / "admin_docs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_client_documents_dir(self, client_username: str) -> Path:
        """
        Docs stored under:
        data/advisors/<advisor_id>/clients/<client_username>/
        """
        client_username = self._normalize_username(client_username)
        users = self._load()
        advisor_id = users.get(client_username, {}).get("advisor_id") or "unassigned"
        path = self._root() / "data" / "advisors" / advisor_id / "clients" / client_username
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_vectorstore_dir(self, username: str) -> Path:
        username = self._normalize_username(username)
        path = self._root() / "vectorstore" / username / "chroma_db"
        path.mkdir(parents=True, exist_ok=True)
        return path
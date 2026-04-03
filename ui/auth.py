# ui/auth.py
"""
3-tier authentication system:
  SUPER_ADMIN  →  manages advisors, sees everything
  ADVISOR      →  manages their own clients only, no overlap
  CLIENT       →  read-only chat, sees only their own docs
"""

import json
import hashlib
import secrets
import time
from pathlib import Path
from typing import Optional, List, Dict
from enum import Enum


class UserRole(Enum):
    SUPER_ADMIN = "super_admin"
    ADVISOR     = "advisor"
    CLIENT      = "client"


class User:
    def __init__(
        self,
        username: str,
        role: UserRole,
        client_name: str = "",
        advisor_id: Optional[str] = None,
        session_token: Optional[str] = None,
    ):
        self.username      = username
        self.role          = role
        self.client_name   = client_name or username
        self.advisor_id    = advisor_id
        self.session_token = session_token

    def is_admin(self)   -> bool: return self.role == UserRole.SUPER_ADMIN
    def is_advisor(self) -> bool: return self.role == UserRole.ADVISOR
    def is_client(self)  -> bool: return self.role == UserRole.CLIENT


class AuthSystem:
    _failed_attempts: Dict[str, list] = {}
    MAX_ATTEMPTS   = 5
    LOCKOUT_WINDOW = 300  # seconds

    def __init__(self, users_file: str = None):
        PROJECT_ROOT = Path(__file__).resolve().parents[1]
        self.users_file = Path(users_file) if users_file else PROJECT_ROOT / "data" / "users.json"
        self.users_file.parent.mkdir(parents=True, exist_ok=True)
        self._init_default_users()

    # ── Internal helpers ───────────────────────────────────────────────────

    def _hash_password(self, password: str) -> str:
        return hashlib.sha256(password.encode()).hexdigest()

    def _load(self) -> dict:
        return json.loads(self.users_file.read_text())

    def _save(self, data: dict) -> None:
        self.users_file.write_text(json.dumps(data, indent=2))

    def _is_locked_out(self, username: str) -> bool:
        now     = time.time()
        recent  = [t for t in self._failed_attempts.get(username, []) if now - t < self.LOCKOUT_WINDOW]
        self._failed_attempts[username] = recent
        return len(recent) >= self.MAX_ATTEMPTS

    def _record_failure(self, username: str) -> None:
        self._failed_attempts.setdefault(username, []).append(time.time())

    def _clear_failures(self, username: str) -> None:
        self._failed_attempts.pop(username, None)

    # ── Seed data ──────────────────────────────────────────────────────────

    def _init_default_users(self):
        if self.users_file.exists():
            return
        default = {
            "admin@demo.com": {
                "password":   self._hash_password("admin123"),
                "role":       "super_admin",
                "name":       "System Administrator",
                "advisor_id": None,
            },
            "advisor.adam@demo.com": {
                "password":   self._hash_password("advisor123"),
                "role":       "advisor",
                "name":       "Adam Advisor",
                "advisor_id": None,
            },
            "advisor.jake@demo.com": {
                "password":   self._hash_password("advisor456"),
                "role":       "advisor",
                "name":       "Jake Advisor",
                "advisor_id": None,
            },
            "john.smith@demo.com": {
                "password":   self._hash_password("client123"),
                "role":       "client",
                "name":       "John Smith",
                "advisor_id": "advisor.adam@demo.com",
            },
            "sally.smith@demo.com": {
                "password":   self._hash_password("client123"),
                "role":       "client",
                "name":       "Sally Smith",
                "advisor_id": "advisor.adam@demo.com",
            },
            "peter.professor@demo.com": {
                "password":   self._hash_password("client123"),
                "role":       "client",
                "name":       "Peter Professor",
                "advisor_id": "advisor.jake@demo.com",
            },
        }
        self._save(default)

    # ── Authentication ─────────────────────────────────────────────────────

    def authenticate(self, username: str, password: str) -> Optional[User]:
        username = username.strip().lower()
        if self._is_locked_out(username):
            return None
        try:
            users = self._load()
            if username not in users:
                self._record_failure(username)
                return None
            udata = users[username]
            if udata["password"] != self._hash_password(password):
                self._record_failure(username)
                return None
            self._clear_failures(username)
            return User(
                username=username,
                role=UserRole(udata["role"]),
                client_name=udata.get("name", username),
                advisor_id=udata.get("advisor_id"),
                session_token=secrets.token_hex(32),
            )
        except Exception as e:
            print(f"[Auth] Error: {e}")
            return None

    def is_locked_out(self, username: str) -> bool:
        return self._is_locked_out(username.strip().lower())

    # ── User queries ───────────────────────────────────────────────────────

    def get_all_advisors(self) -> List[User]:
        users = self._load()
        return [
            User(username=u, role=UserRole(d["role"]), client_name=d.get("name", u))
            for u, d in users.items() if d["role"] == "advisor"
        ]

    def get_clients_for_advisor(self, advisor_username: str) -> List[User]:
        """Returns ONLY clients assigned to this advisor — no cross-advisor leakage."""
        users = self._load()
        return [
            User(
                username=u,
                role=UserRole(d["role"]),
                client_name=d.get("name", u),
                advisor_id=d.get("advisor_id"),
            )
            for u, d in users.items()
            if d["role"] == "client" and d.get("advisor_id") == advisor_username
        ]

    def get_all_clients(self) -> List[User]:
        """Super-admin only: all clients across all advisors."""
        users = self._load()
        return [
            User(
                username=u,
                role=UserRole(d["role"]),
                client_name=d.get("name", u),
                advisor_id=d.get("advisor_id"),
            )
            for u, d in users.items() if d["role"] == "client"
        ]

    def get_user(self, username: str) -> Optional[User]:
        try:
            d = self._load().get(username)
            if not d:
                return None
            return User(
                username=username,
                role=UserRole(d["role"]),
                client_name=d.get("name", username),
                advisor_id=d.get("advisor_id"),
            )
        except Exception:
            return None

    # ── Admin CRUD ─────────────────────────────────────────────────────────

    def create_user(self, username: str, password: str, role: str, name: str,
                    advisor_id: Optional[str] = None) -> bool:
        try:
            users = self._load()
            if username in users:
                return False
            users[username] = {
                "password":   self._hash_password(password),
                "role":       role,
                "name":       name,
                "advisor_id": advisor_id,
            }
            self._save(users)
            return True
        except Exception as e:
            print(f"[Auth] create_user error: {e}")
            return False

    def delete_user(self, username: str) -> bool:
        try:
            users = self._load()
            if username not in users:
                return False
            del users[username]
            self._save(users)
            return True
        except Exception:
            return False

    def reassign_client(self, client_username: str, new_advisor_username: str) -> bool:
        """Super-admin: move a client from one advisor to another."""
        try:
            users = self._load()
            if client_username not in users:
                return False
            users[client_username]["advisor_id"] = new_advisor_username
            self._save(users)
            return True
        except Exception:
            return False

    # ── Document directories ───────────────────────────────────────────────

    @staticmethod
    def _root() -> Path:
        return Path(__file__).resolve().parents[1]

    def get_user_documents_dir(self, user: User) -> Path:
        if user.is_advisor():
            path = self._root() / "data" / "advisors" / user.username / "docs"
            path.mkdir(parents=True, exist_ok=True)
            return path
        elif user.is_client():
            return self.get_client_documents_dir(user.username)
        else:
            path = self._root() / "data" / "admin_docs"
            path.mkdir(parents=True, exist_ok=True)
            return path

    def get_client_documents_dir(self, client_username: str) -> Path:
        """
        Docs stored under the advisor's folder tree:
        data/advisors/<advisor_id>/clients/<client_id>/
        This physically prevents cross-advisor access.
        """
        users = self._load()
        advisor_id = users.get(client_username, {}).get("advisor_id") or "unassigned"
        path = self._root() / "data" / "advisors" / advisor_id / "clients" / client_username
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_vectorstore_dir(self, username: str) -> Path:
        path = self._root() / "vectorstore" / username / "chroma_db"
        path.mkdir(parents=True, exist_ok=True)
        return path
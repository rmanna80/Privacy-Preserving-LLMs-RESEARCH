# ui/auth.py

import json
import hashlib
from pathlib import Path
from typing import Optional, List
from enum import Enum


class UserRole(Enum):
    ADVISOR = "advisor"
    CLIENT = "client"


class User:
    def __init__(self, username: str, role: UserRole, client_name: str = None):
        self.username = username
        self.role = role
        self.client_name = client_name


class AuthSystem:
    def __init__(self, users_file: str = None):
        PROJECT_ROOT = Path(__file__).resolve().parents[1]
        if users_file is None:
            self.users_file = PROJECT_ROOT / "data" / "users.json"
        else:
            self.users_file = Path(users_file)
        self.users_file.parent.mkdir(parents=True, exist_ok=True)
        self._init_default_users()

    def _hash_password(self, password: str) -> str:
        return hashlib.sha256(password.encode()).hexdigest()

    def _init_default_users(self):
        """Create default users if file doesn't exist"""
        if not self.users_file.exists():
            default_users = {
                "advisor@demo.com": {
                    "password": self._hash_password("advisor123"),
                    "role": "advisor",
                    "name": "Financial Advisor"
                },
                "john.smith@demo.com": {
                    "password": self._hash_password("client123"),
                    "role": "client",
                    "name": "John Smith"
                },
                "sally.smith@demo.com": {
                    "password": self._hash_password("client123"),
                    "role": "client",
                    "name": "Sally Smith"
                }
            }
            self.users_file.write_text(json.dumps(default_users, indent=2))

    def authenticate(self, username: str, password: str) -> Optional[User]:
        """Authenticate user and return User object if successful"""
        try:
            users = json.loads(self.users_file.read_text())
            if username not in users:
                return None
            user_data = users[username]
            if user_data["password"] != self._hash_password(password):
                return None
            return User(
                username=username,
                role=UserRole(user_data["role"]),
                client_name=user_data.get("name", username)
            )
        except Exception as e:
            print(f"Auth error: {e}")
            return None

    def get_all_clients(self) -> List[User]:
        """Return all client users (for advisor to select when uploading docs)"""
        try:
            users = json.loads(self.users_file.read_text())
            return [
                User(
                    username=uname,
                    role=UserRole(udata["role"]),
                    client_name=udata.get("name", uname)
                )
                for uname, udata in users.items()
                if udata["role"] == "client"
            ]
        except Exception:
            return []

    def get_user_documents_dir(self, user: User) -> Path:
        """Get the documents directory for a specific user"""
        PROJECT_ROOT = Path(__file__).resolve().parents[1]
        if user.role == UserRole.ADVISOR:
            # Advisor's own general doc pool (not used for chat anymore)
            path = PROJECT_ROOT / "data" / "raw_pdfs"
            path.mkdir(parents=True, exist_ok=True)
            return path
        else:
            return self.get_client_documents_dir(user.username)

    def get_client_documents_dir(self, client_username: str) -> Path:
        """Get the documents directory for a specific client (used by advisor uploads)"""
        PROJECT_ROOT = Path(__file__).resolve().parents[1]
        client_dir = PROJECT_ROOT / "data" / "client_docs" / client_username
        client_dir.mkdir(parents=True, exist_ok=True)
        return client_dir
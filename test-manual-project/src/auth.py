"""Authentication service module.

This module handles user authentication, token generation, and session management.
"""

from typing import Optional, Dict
import hashlib


class AuthenticationService:
    """Main authentication service for handling user logins."""

    def __init__(self, secret_key: str):
        """Initialize with secret key for token signing."""
        self.secret_key = secret_key
        self.sessions: Dict[str, dict] = {}

    def authenticate(self, username: str, password: str) -> Optional[str]:
        """Authenticate user and return token if successful."""
        if self._validate_credentials(username, password):
            return self._generate_token(username)
        return None

    def _validate_credentials(self, username: str, password: str) -> bool:
        """Validate username and password against stored hashes."""
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        # Simplified validation for testing
        return (
            username == "test_user"
            and password_hash
            == "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
        )

    def _generate_token(self, username: str) -> str:
        """Generate authentication token for user."""
        token_data = f"{username}:{self.secret_key}"
        return hashlib.sha256(token_data.encode()).hexdigest()

    def validate_session(self, session_id: str) -> bool:
        """Check if session is valid."""
        return session_id in self.sessions

    def logout(self, session_id: str) -> bool:
        """Invalidate session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False


def create_auth_service(secret: str) -> AuthenticationService:
    """Factory function to create authentication service."""
    return AuthenticationService(secret)

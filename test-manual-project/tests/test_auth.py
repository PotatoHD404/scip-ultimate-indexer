"""Tests for authentication service."""

import pytest
from src.auth import AuthenticationService, create_auth_service


class TestAuthenticationService:
    """Test cases for authentication service."""

    def test_init(self):
        """Test service initialization."""
        service = AuthenticationService("test_secret")
        assert service.secret_key == "test_secret"
        assert service.sessions == {}

    def test_authenticate_success(self):
        """Test successful authentication."""
        service = AuthenticationService("secret")
        token = service.authenticate("test_user", "password")
        assert token is not None
        assert len(token) == 64  # SHA256 hex length

    def test_authenticate_failure(self):
        """Test failed authentication."""
        service = AuthenticationService("secret")
        token = service.authenticate("wrong_user", "wrong_pass")
        assert token is None

    def test_session_management(self):
        """Test session creation and validation."""
        service = AuthenticationService("secret")
        session_id = "test_session_123"
        service.sessions[session_id] = {"user": "test"}
        assert service.validate_session(session_id) is True
        assert service.logout(session_id) is True
        assert service.validate_session(session_id) is False


def test_factory_function():
    """Test factory function creates service correctly."""
    service = create_auth_service("my_secret")
    assert isinstance(service, AuthenticationService)
    assert service.secret_key == "my_secret"

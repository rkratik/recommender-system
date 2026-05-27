"""Tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient
from api.server import app

client = TestClient(app)


class TestAPI:
    """Test suite for API endpoints."""
    
    def test_root(self):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        assert "name" in response.json()
    
    def test_health(self):
        """Test health endpoint."""
        response = client.get("/health/")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data

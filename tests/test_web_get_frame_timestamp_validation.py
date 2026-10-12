"""get_frame rejects NaN, inf, and negative timestamps."""

from __future__ import annotations

import math

import pytest
from fastapi.testclient import TestClient

from gomoveshift.web.app import create_app


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


@pytest.mark.parametrize(
    "ts",
    [
        float("nan"),
        float("inf"),
        -float("inf"),
        -1.0,
        -0.5,
    ],
    ids=["nan", "+inf", "-inf", "neg-1", "neg-0.5"],
)
def test_invalid_timestamp_returns_422(client: TestClient, ts: float) -> None:
    # Use a path that would otherwise hit _resolve_safe_path's 403 — the
    # validation runs BEFORE path resolution so 422 is the expected
    # status regardless of whether the video exists.
    ts_param = "NaN" if math.isnan(ts) else repr(ts)
    response = client.get(f"/api/frame/some/video.mp4?timestamp={ts_param}")
    assert response.status_code == 422, response.text
    assert "non-negative finite number" in response.text


def test_zero_timestamp_does_not_422(client: TestClient) -> None:
    """Zero is the documented default and must NOT trip the validation."""
    response = client.get("/api/frame/nonexistent/video.mp4?timestamp=0")
    # Will 403 (path) or 404 (file missing) but not 422 (validation).
    assert response.status_code != 422

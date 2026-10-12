"""Web API surface hardening.

Two security properties:

1. ``_resolve_safe_path`` builds its allowlist ONLY from the configured
   ``input_folder`` / ``output_folder`` — NOT a workspace-root seed that
   would expose every sibling project. A path under a sibling project is
   rejected; a path under a configured folder is accepted. When neither
   folder is configured the resolver raises a clear server-config error
   rather than silently rejecting everything (or, with the old seed,
   over-permitting the whole workspace).
2. CORS ``allow_origins`` is loopback-only by default and only widened
   when ``allow_remote`` is threaded into ``create_app``.
"""

from __future__ import annotations

from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from gomoveshift.config import Config
from gomoveshift.web.app import _build_allowed_roots, create_app


def _config_with_folders(tmp_path) -> Config:
    config = Config()
    config.paths.input_folder = str(tmp_path / "input")
    config.paths.output_folder = str(tmp_path / "output")
    (tmp_path / "input").mkdir()
    (tmp_path / "output").mkdir()
    return config


# --- _build_allowed_roots ---------------------------------------------------


def test_allowed_roots_exclude_workspace_seed(tmp_path) -> None:
    """The allowlist is exactly the two configured folders — no
    workspace seed that would expose sibling projects."""
    config = _config_with_folders(tmp_path)
    roots = _build_allowed_roots(config)
    assert roots == [
        (tmp_path / "input").resolve(),
        (tmp_path / "output").resolve(),
    ]
    assert all(str(r).startswith(str(tmp_path)) for r in roots)


def test_allowed_roots_empty_when_both_unset() -> None:
    config = Config()  # input_folder == output_folder == ""
    assert _build_allowed_roots(config) == []


# --- _resolve_safe_path -----------------------------------------------------


def _resolver(config: Config):
    # ``_resolve_safe_path`` is a closure created in ``create_app``; reach it
    # by exercising the route layer through the resolver the app installs.
    app = create_app(config=config)
    # The closure is captured on the route handlers; call it via a known
    # route that resolves a path (``/api/analysis`` does not, ``/api/frame``
    # does). Easiest stable handle: re-derive via TestClient behaviour.
    return TestClient(app)


def test_rejects_path_under_sibling_project(tmp_path) -> None:
    config = _config_with_folders(tmp_path)
    client = _resolver(config)
    # A path under a sibling workspace project must NOT resolve — timestamp
    # validation passes (0), file-existence is irrelevant because path
    # containment rejects first with 403.
    resp = client.get("/api/frame/srv/other-project/secret.mp4?timestamp=0")
    assert resp.status_code == 403, resp.text
    assert "allowed directories" in resp.text


def test_accepts_path_under_configured_input_folder(tmp_path) -> None:
    config = _config_with_folders(tmp_path)
    video = tmp_path / "input" / "clip.mp4"
    video.write_text("not really a video")
    client = _resolver(config)
    # Inside the allowlist → containment passes. The file exists but is not a
    # decodable video, so we get past the 403/404 gates (status != 403/422).
    resp = client.get(f"/api/frame/{video}?timestamp=0")
    assert resp.status_code != 403, resp.text
    assert resp.status_code != 422, resp.text


def test_resolve_raises_clear_error_when_no_roots_configured() -> None:
    """Both folders unset → path resolution fails loud (server config
    error), never silently rejecting/permitting."""
    config = Config()
    client = TestClient(create_app(config=config), raise_server_exceptions=False)
    resp = client.get("/api/frame/some/video.mp4?timestamp=0")
    assert resp.status_code == 500, resp.text
    assert "no allowed directories configured" in resp.text.lower()


# --- CORS -------------------------------------------------------------------


def _cors_origins(app) -> list[str]:
    for mw in app.user_middleware:
        if mw.cls is CORSMiddleware:
            return mw.kwargs["allow_origins"]
    raise AssertionError("CORS middleware not installed")


def test_cors_loopback_only_by_default(tmp_path) -> None:
    app = create_app(config=_config_with_folders(tmp_path))
    origins = _cors_origins(app)
    assert origins == ["http://127.0.0.1:8765", "http://localhost:8765"]
    assert "*" not in origins


def test_cors_widened_only_under_allow_remote(tmp_path) -> None:
    app = create_app(config=_config_with_folders(tmp_path), allow_remote=True)
    origins = _cors_origins(app)
    assert "*" in origins

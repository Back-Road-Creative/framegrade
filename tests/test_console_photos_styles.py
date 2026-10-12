"""The console's `styles` control lists what exists and refuses what does not.

Before this, `styles` was a free-text field with no list and no check, while the
`look` field beside it got a dropdown and a ValueError on an unknown name. An
operator typed style names blind and learned they were wrong at develop time.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from gomoveshift.console import photos
from gomoveshift.console.app import create_app
from gomoveshift.pipeline.photos.styles import STYLES


def _client() -> TestClient:
    return TestClient(create_app())


def test_build_photos_argv_develop_known_styles_pass_through_in_order():
    argv = photos.build_photos_argv(
        "develop", paths=["/mnt/card/a.jpg"], out="/tmp/develop-out", styles="vivid, mono"
    )
    assert argv[argv.index("--styles") + 1] == "vivid,mono"


def test_build_photos_argv_develop_unknown_style_is_refused_by_name():
    with pytest.raises(ValueError, match="not-a-style"):
        photos.build_photos_argv(
            "develop", paths=["/mnt/card/a.jpg"], out="/tmp/develop-out", styles="vivid,not-a-style"
        )


def test_photos_tab_lists_every_registered_style_as_an_option():
    resp = _client().get("/photos")
    assert resp.status_code == 200
    for name in STYLES:
        assert f'<option value="{name}"' in resp.text


def test_photos_run_develop_unknown_style_is_400(tmp_path, monkeypatch):
    monkeypatch.setattr(photos, "resolved_looks", lambda: {})
    (tmp_path / "a.jpg").write_bytes(b"x")
    resp = _client().post(
        "/photos/run",
        data={
            "action": "develop",
            "paths": str(tmp_path / "a.jpg"),
            "out": str(tmp_path / "out"),
            "styles": "moon",
        },
    )
    assert resp.status_code == 400
    assert "moon" in resp.text

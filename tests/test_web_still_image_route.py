"""Still-image route for the photo develop review gate.

The review UI could only decode frames out of a video (``/api/frame``), so a
developed still had no surface. ``/api/image`` serves one — but through the
SAME ``_resolve_safe_path`` containment check, widened only by the photo
roots (journey image tree + the configured photo output/resized bases) and
only for explicitly-allowed image extensions. It is not a file server:

* outside every allowed root → 403
* ``../`` traversal out of an allowed root → 403
* a non-image extension inside an allowed root → refused, never served
* the video route's allowlist is unchanged (photo roots do NOT widen it)
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from gomoveshift.config import Config
from gomoveshift.web.app import create_app

PNG_BYTES = bytes.fromhex(
    "89504e470d0a1a0a0000000d494844520000000100000001080600000"
    "01f15c4890000000a49444154789c6300010000050001"
    "0d0a2db40000000049454e44ae426082"
)


def _config(tmp_path) -> Config:
    config = Config()
    config.paths.input_folder = str(tmp_path / "input")
    config.paths.output_folder = str(tmp_path / "output")
    config.photos.site_root = str(tmp_path / "site")
    config.photos.output_dir = str(tmp_path / "photo_out")
    config.photos.resized_base = str(tmp_path / "photo_resized")
    for sub in ("input", "output", "photo_out", "photo_resized"):
        (tmp_path / sub).mkdir()
    (tmp_path / "site" / "static" / "images" / "journeys").mkdir(parents=True)
    (tmp_path / "outside").mkdir()
    return config


def _client(config: Config) -> TestClient:
    return TestClient(create_app(config=config), raise_server_exceptions=False)


def test_serves_jpeg_from_photo_output_dir(tmp_path) -> None:
    config = _config(tmp_path)
    image = tmp_path / "photo_out" / "developed.jpg"
    image.write_bytes(b"\xff\xd8\xff\xdb-fake-jpeg-body")
    resp = _client(config).get(f"/api/image/{image}")
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "image/jpeg"
    assert resp.content == b"\xff\xd8\xff\xdb-fake-jpeg-body"


def test_serves_png_from_journeys_tree(tmp_path) -> None:
    config = _config(tmp_path)
    image = tmp_path / "site" / "static" / "images" / "journeys" / "day-01.png"
    image.write_bytes(PNG_BYTES)
    resp = _client(config).get(f"/api/image/{image}")
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "image/png"
    assert resp.content == PNG_BYTES


def test_serves_from_resized_base(tmp_path) -> None:
    config = _config(tmp_path)
    image = tmp_path / "photo_resized" / "ireland" / "shot.jpeg"
    image.parent.mkdir()
    image.write_bytes(b"jpeg")
    resp = _client(config).get(f"/api/image/{image}")
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "image/jpeg"


def test_path_outside_every_allowed_root_is_forbidden(tmp_path) -> None:
    config = _config(tmp_path)
    image = tmp_path / "outside" / "secret.jpg"
    image.write_bytes(b"secret")
    resp = _client(config).get(f"/api/image/{image}")
    assert resp.status_code == 403, resp.text
    assert "allowed directories" in resp.text


def test_traversal_out_of_allowed_root_is_forbidden(tmp_path) -> None:
    config = _config(tmp_path)
    (tmp_path / "outside" / "secret.jpg").write_bytes(b"secret")
    traversal = f"{tmp_path / 'photo_out'}/../outside/secret.jpg"
    resp = _client(config).get(f"/api/image/{traversal}")
    assert resp.status_code == 403, resp.text


def test_symlink_escaping_allowed_root_is_forbidden(tmp_path) -> None:
    config = _config(tmp_path)
    target = tmp_path / "outside" / "secret.jpg"
    target.write_bytes(b"secret")
    link = tmp_path / "photo_out" / "link.jpg"
    link.symlink_to(target)
    resp = _client(config).get(f"/api/image/{link}")
    assert resp.status_code == 403, resp.text


def test_non_image_extension_inside_allowed_root_is_refused(tmp_path) -> None:
    config = _config(tmp_path)
    secret = tmp_path / "photo_out" / "credentials.json"
    secret.write_text('{"token": "shh"}')
    resp = _client(config).get(f"/api/image/{secret}")
    assert resp.status_code == 415, resp.text
    assert "shh" not in resp.text


def test_missing_image_inside_allowed_root_is_404(tmp_path) -> None:
    config = _config(tmp_path)
    resp = _client(config).get(f"/api/image/{tmp_path / 'photo_out' / 'nope.jpg'}")
    assert resp.status_code == 404, resp.text


def test_frame_route_allowlist_not_widened_by_photo_roots(tmp_path) -> None:
    """The video route keeps the narrow input/output allowlist."""
    config = _config(tmp_path)
    clip = tmp_path / "photo_out" / "clip.mp4"
    clip.write_text("not really a video")
    resp = _client(config).get(f"/api/frame/{clip}?timestamp=0")
    assert resp.status_code == 403, resp.text


def test_frame_route_still_serves_video_paths(tmp_path) -> None:
    """Existing /api/frame containment + validation behaviour is unchanged."""
    config = _config(tmp_path)
    clip = tmp_path / "input" / "clip.mp4"
    clip.write_text("not really a video")
    client = _client(config)
    inside = client.get(f"/api/frame/{clip}?timestamp=0")
    assert inside.status_code not in (403, 422), inside.text
    assert client.get(f"/api/frame/{clip}?timestamp=-1").status_code == 422
    outside = client.get(f"/api/frame/{tmp_path / 'outside' / 'clip.mp4'}?timestamp=0")
    assert outside.status_code == 403, outside.text

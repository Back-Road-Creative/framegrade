"""Night projects are reachable through the actual application and CLI."""

from fastapi.testclient import TestClient
from gomoveshift.cli.parser import _build_parser
from gomoveshift.console.app import create_app


def test_cli_accepts_same_worker_invocation_as_console(tmp_path):
    args = _build_parser().parse_args(
        [
            "photos",
            "night",
            "--project",
            "id",
            "--workspace",
            str(tmp_path),
            "--brand",
            "gomoveshift",
        ]
    )
    assert args.photos_command == "night"
    assert args.wait_seconds == 600
    assert args.recipe == "natural"


def test_cli_accepts_durable_cancel_control(tmp_path):
    args = _build_parser().parse_args(
        [
            "photos",
            "night",
            "--project",
            "id",
            "--workspace",
            str(tmp_path),
            "--brand",
            "gomoveshift",
            "--cancel",
        ]
    )
    assert args.cancel is True


def test_real_app_has_private_night_route_and_photo_link(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("GMS_CONSOLE_TOKEN", raising=False)
    monkeypatch.setenv("GMS_CONSOLE_BRAND", "gomoveshift")
    client = TestClient(create_app())
    assert client.get("/photos/night").status_code == 200
    assert "/photos/night" in client.get("/photos").text


def test_cli_accepts_explicit_creation_budget(tmp_path):
    args = _build_parser().parse_args(
        [
            "photos",
            "night",
            "--sources",
            str(tmp_path),
            "--source-root",
            str(tmp_path),
            "--workspace",
            str(tmp_path / "output"),
            "--brand",
            "gomoveshift",
            "--storage-budget-gib",
            "1.5",
        ]
    )
    assert args.storage_budget_gib == "1.5"


def test_cli_accepts_read_only_collection_scan(tmp_path):
    args = _build_parser().parse_args(
        [
            "photos",
            "night",
            "--scan",
            str(tmp_path),
            "--workspace",
            str(tmp_path / "output"),
            "--brand",
            "gomoveshift",
        ]
    )
    assert args.scan == str(tmp_path)
    assert args.project is None
    assert args.sources is None

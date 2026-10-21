"""Own-channel ids no listing returns come from the operator's inputs dir
(``<GMS_INPUTS_DIR>/known_ids.txt``), never from the package: the public tree
ships no channel's inventory."""

from __future__ import annotations

from gomoveshift.cli.handlers import library_hygiene
from gomoveshift.pipeline.site_sync.known_ids import FROZEN_KNOWN_IDS, load_known_ids


def test_package_ships_no_ids():
    assert FROZEN_KNOWN_IDS == ()


def test_absent_file_yields_the_empty_default(monkeypatch, tmp_path):
    monkeypatch.setenv("GMS_INPUTS_DIR", str(tmp_path))
    assert load_known_ids() == ()


def test_reads_one_id_per_line_skipping_blanks_comments_and_repeats(monkeypatch, tmp_path):
    (tmp_path / "known_ids.txt").write_text(
        "# private videos no listing returns\nAAAAAAAAAAA\n\n  BBBBBBBBBBB  # unlisted\nAAAAAAAAAAA\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("GMS_INPUTS_DIR", str(tmp_path))
    assert load_known_ids() == ("AAAAAAAAAAA", "BBBBBBBBBBB")


def test_list_uploads_passes_the_input_ids_to_enumeration(monkeypatch, tmp_path):
    (tmp_path / "known_ids.txt").write_text("CCCCCCCCCCC\n", encoding="utf-8")
    monkeypatch.setenv("GMS_INPUTS_DIR", str(tmp_path))
    monkeypatch.setattr(library_hygiene, "ledger_video_ids", lambda: ["DDDDDDDDDDD"])
    seen = {}

    def fake_enumerate(service, known_ids):
        seen["ids"] = list(known_ids)
        return []

    monkeypatch.setattr(library_hygiene, "enumerate_uploads", fake_enumerate)
    assert library_hygiene.list_uploads(object()) == []
    assert seen["ids"] == ["DDDDDDDDDDD", "CCCCCCCCCCC"]

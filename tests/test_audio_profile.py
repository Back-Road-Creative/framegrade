"""Per-brand music-bed audio profile + licence ledger.

A Route Story film may carry a licensed music bed under the attenuated
source audio -- but only when the exact bed file is a recorded entry in
the brand's licence ledger, and never for GoMoveShift, whose near-silent
attenuate-only audio contract refuses a bed outright.
"""

from __future__ import annotations

import json

import pytest

from gomoveshift.pipeline.audio_profile import (
    AudioProfile,
    GmsAudioRefused,
    LicenceLedger,
    UnlicensedBedError,
    load_audio_profile,
    resolve_bed,
)


def test_gomoveshift_slug_is_refused_regardless_of_config(tmp_path):
    brands_dir = tmp_path / "brands"
    slug_dir = brands_dir / "gomoveshift"
    slug_dir.mkdir(parents=True)
    (slug_dir / "brand.yaml").write_text("audio:\n  enabled: true\n  bed_path: bed.mp3\n")

    with pytest.raises(GmsAudioRefused):
        load_audio_profile("gomoveshift", brands_dir=brands_dir)


def test_gomoveshift_refused_even_with_no_brand_yaml_at_all(tmp_path):
    with pytest.raises(GmsAudioRefused):
        load_audio_profile("gomoveshift", brands_dir=tmp_path)


def test_missing_brand_yaml_falls_back_to_disabled_profile(tmp_path):
    profile = load_audio_profile("no-such-brand", brands_dir=tmp_path)
    assert profile.enabled is False
    assert profile.bed_path is None


def test_missing_audio_block_falls_back_to_disabled(tmp_path):
    brands_dir = tmp_path / "brands"
    slug_dir = brands_dir / "noaudio"
    slug_dir.mkdir(parents=True)
    (slug_dir / "brand.yaml").write_text("disabled_tabs: [drives]\n")

    profile = load_audio_profile("noaudio", brands_dir=brands_dir)
    assert profile.enabled is False


def test_audio_block_is_read_with_defaults_for_omitted_fields(tmp_path):
    brands_dir = tmp_path / "brands"
    slug_dir = brands_dir / "waypost"
    slug_dir.mkdir(parents=True)
    (slug_dir / "brand.yaml").write_text("audio:\n  enabled: true\n  bed_path: music/bed.mp3\n")

    profile = load_audio_profile("waypost", brands_dir=brands_dir)
    assert profile.enabled is True
    assert profile.bed_path == slug_dir / "music/bed.mp3"
    assert profile.gain_db == -18.0
    assert profile.fade_out_s == 2.0
    assert profile.ledger_path == slug_dir / "music" / "ledger.json"


def test_audio_block_explicit_gain_and_fade_override_defaults(tmp_path):
    brands_dir = tmp_path / "brands"
    slug_dir = brands_dir / "waypost"
    slug_dir.mkdir(parents=True)
    (slug_dir / "brand.yaml").write_text(
        "audio:\n  enabled: true\n  bed_path: music/bed.mp3\n  gain_db: -22.5\n  fade_out_s: 3.5\n"
    )

    profile = load_audio_profile("waypost", brands_dir=brands_dir)
    assert profile.gain_db == -22.5
    assert profile.fade_out_s == 3.5


def test_licence_ledger_loads_entries_keyed_by_file(tmp_path):
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "file": "bed.mp3",
                        "title": "Open Road",
                        "source": "pixabay",
                        "licence": "Pixabay Content License",
                        "url": "https://pixabay.com/music/x",
                        "added": "2026-09-25",
                    }
                ]
            }
        )
    )

    ledger = LicenceLedger.load(ledger_path)
    entry = ledger.find("bed.mp3")
    assert entry["source"] == "pixabay"
    assert entry["licence"] == "Pixabay Content License"


def test_licence_ledger_missing_file_returns_no_entry(tmp_path):
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(json.dumps({"entries": []}))

    ledger = LicenceLedger.load(ledger_path)
    assert ledger.find("bed.mp3") is None


def test_licence_ledger_missing_file_on_disk_is_empty(tmp_path):
    ledger = LicenceLedger.load(tmp_path / "does-not-exist.json")
    assert ledger.find("bed.mp3") is None


def test_resolve_bed_returns_the_matching_entry(tmp_path):
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "file": "bed.mp3",
                        "title": "Open Road",
                        "source": "pixabay",
                        "licence": "Pixabay Content License",
                        "url": "https://pixabay.com/music/x",
                        "added": "2026-09-25",
                    }
                ]
            }
        )
    )
    ledger = LicenceLedger.load(ledger_path)
    profile = AudioProfile(enabled=True, bed_path=tmp_path / "bed.mp3", ledger_path=ledger_path)

    entry = resolve_bed(profile, ledger)
    assert entry["title"] == "Open Road"


def test_resolve_bed_raises_when_bed_is_unlicensed(tmp_path):
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(json.dumps({"entries": []}))
    ledger = LicenceLedger.load(ledger_path)
    profile = AudioProfile(enabled=True, bed_path=tmp_path / "bed.mp3", ledger_path=ledger_path)

    with pytest.raises(UnlicensedBedError):
        resolve_bed(profile, ledger)


def test_audio_profile_does_not_import_publishing():
    import ast
    from pathlib import Path

    import gomoveshift.pipeline.audio_profile as mod

    tree = ast.parse(Path(mod.__file__).read_text())
    names = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)] + [
        a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names
    ]
    assert not [n for n in names if "publishing" in n]

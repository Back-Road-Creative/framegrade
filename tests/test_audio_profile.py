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


def test_gomoveshift_slug_is_refused_regardless_of_config(tmp_path, monkeypatch):
    import gomoveshift.publishing.brand_policy as brand_policy_mod

    brands_dir = tmp_path / "brands"
    slug_dir = brands_dir / "gomoveshift"
    slug_dir.mkdir(parents=True)
    (slug_dir / "brand.yaml").write_text("audio:\n  enabled: true\n  bed_path: bed.mp3\n")
    monkeypatch.setattr(brand_policy_mod, "_SHIPPED_BRANDS_DIR", brands_dir)

    with pytest.raises(GmsAudioRefused):
        load_audio_profile("gomoveshift")


def test_gomoveshift_refused_even_with_no_brand_yaml_at_all():
    with pytest.raises(GmsAudioRefused):
        load_audio_profile("gomoveshift")


def test_missing_brand_yaml_falls_back_to_disabled_profile():
    profile = load_audio_profile("no-such-brand")
    assert profile.enabled is False
    assert profile.bed_path is None


def test_missing_audio_block_falls_back_to_disabled(tmp_path, monkeypatch):
    import gomoveshift.publishing.brand_policy as brand_policy_mod

    brands_dir = tmp_path / "brands"
    slug_dir = brands_dir / "noaudio"
    slug_dir.mkdir(parents=True)
    (slug_dir / "brand.yaml").write_text("disabled_tabs: [drives]\n")
    monkeypatch.setattr(brand_policy_mod, "_SHIPPED_BRANDS_DIR", brands_dir)

    profile = load_audio_profile("noaudio")
    assert profile.enabled is False


def test_audio_block_is_read_with_defaults_for_omitted_fields(tmp_path, monkeypatch):
    import gomoveshift.publishing.brand_policy as brand_policy_mod

    brands_dir = tmp_path / "brands"
    slug_dir = brands_dir / "waypost"
    slug_dir.mkdir(parents=True)
    (slug_dir / "brand.yaml").write_text("audio:\n  enabled: true\n  bed_path: music/bed.mp3\n")
    monkeypatch.setattr(brand_policy_mod, "_SHIPPED_BRANDS_DIR", brands_dir)

    profile = load_audio_profile("waypost")
    assert profile.enabled is True
    assert profile.bed_path == slug_dir / "music/bed.mp3"
    assert profile.gain_db == -18.0
    assert profile.fade_out_s == 2.0
    assert profile.ledger_path == slug_dir / "music" / "ledger.json"


def test_audio_block_explicit_gain_and_fade_override_defaults(tmp_path, monkeypatch):
    import gomoveshift.publishing.brand_policy as brand_policy_mod

    brands_dir = tmp_path / "brands"
    slug_dir = brands_dir / "waypost"
    slug_dir.mkdir(parents=True)
    (slug_dir / "brand.yaml").write_text(
        "audio:\n  enabled: true\n  bed_path: music/bed.mp3\n  gain_db: -22.5\n  fade_out_s: 3.5\n"
    )
    monkeypatch.setattr(brand_policy_mod, "_SHIPPED_BRANDS_DIR", brands_dir)

    profile = load_audio_profile("waypost")
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

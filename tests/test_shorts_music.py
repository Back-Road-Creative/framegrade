"""Licensed music on Facebook and Instagram Shorts (docs/pipeline/188)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from gomoveshift.publishing import shorts_music as sm

TEST = sm.ShortsMusicTest(date(2026, 11, 27), 45, ("facebook", "instagram"))
IN_TEST = lambda: date(2026, 12, 1)  # noqa: E731


def _ledger(folder: Path, *names: str, complete: bool = True) -> None:
    entries = []
    for n in names:
        (folder / n).write_bytes(b"mp3")
        entry = {
            "file": n,
            "title": n,
            "source": "pixabay",
            "licence": "Pixabay Content License",
            "url": "https://example.invalid/t",
            "added": "2026-11-27",
        }
        if not complete:
            entry.pop("licence")
        entries.append(entry)
    (folder / "ledger.json").write_text(json.dumps({"entries": entries}))


def _record(tmp_path: Path, project="p", index=None, fmt="short"):
    src = tmp_path / "short.mp4"
    src.write_bytes(b"muted")
    if index is None:
        index = next(i for i in range(100) if sm.music_arm(project, i))
    rec = SimpleNamespace(
        project_name=project,
        short_index=index,
        format=fmt,
        media=SimpleNamespace(duration_seconds=20.0),
    )
    return rec, src


def _fake_make(calls):
    def make(path, track, duration):
        calls.append((path, track.path.name, duration))
        out = path.with_name(path.stem + ".music.mp4")
        out.write_bytes(b"with-music")
        return out

    return make


def test_parse_accepts_meta_platforms_only():
    assert sm.parse_shorts_music(None) is None
    t = sm.parse_shorts_music(
        {"start": "2026-11-27", "stop_after_days": 45, "platforms": ["facebook"]}
    )
    assert t.platforms == ("facebook",) and t.is_active(date(2026, 11, 27))
    assert not t.is_active(date(2027, 1, 11))
    with pytest.raises(ValueError, match="platforms"):
        sm.parse_shorts_music(
            {"start": "2026-11-27", "stop_after_days": 45, "platforms": ["youtube"]}
        )
    with pytest.raises(ValueError, match="volume"):
        sm.parse_shorts_music(
            {"start": "2026-11-27", "stop_after_days": 4, "platforms": ["facebook"], "volume": 1}
        )


def test_arm_is_stable_and_roughly_half():
    arms = [sm.music_arm("drive", i) for i in range(400)]
    assert arms == [sm.music_arm("drive", i) for i in range(400)]
    assert 150 < sum(arms) < 250


def test_music_half_gets_the_copy_and_a_receipt(tmp_path):
    music = tmp_path / "music"
    music.mkdir()
    _ledger(music, "a.mp3", "b.mp3")
    rec, src = _record(tmp_path)
    calls = []
    read = sm.with_music(
        "facebook",
        lambda r: src,
        test=TEST,
        folder=music,
        storage_dir=tmp_path / "seo",
        today=IN_TEST,
        make=_fake_make(calls),
    )
    assert read(rec) == b"with-music"
    assert calls and calls[0][2] == 20.0
    assert sm.read_receipts(tmp_path / "seo") == {("p", rec.short_index, "facebook")}
    row = json.loads((tmp_path / "seo" / sm.RECEIPTS_FILE).read_text())
    assert row["licence"] == "Pixabay Content License"


@pytest.mark.parametrize(
    "platform,today",
    [("youtube", IN_TEST), ("tiktok", IN_TEST), ("facebook", lambda: date(2026, 11, 26))],
)
def test_other_platforms_and_outside_the_window_stay_muted(tmp_path, platform, today):
    music = tmp_path / "music"
    music.mkdir()
    _ledger(music, "a.mp3")
    rec, src = _record(tmp_path)
    calls = []
    read = sm.with_music(
        platform,
        lambda r: src,
        test=TEST,
        folder=music,
        storage_dir=tmp_path,
        today=today,
        make=_fake_make(calls),
    )
    assert read(rec) == b"muted" and not calls


def test_silent_half_and_non_shorts_stay_muted(tmp_path):
    music = tmp_path / "music"
    music.mkdir()
    _ledger(music, "a.mp3")
    off = next(i for i in range(100) if not sm.music_arm("p", i))
    calls = []
    for rec_kwargs in ({"index": off}, {"fmt": "photo"}):
        rec, src = _record(tmp_path, **rec_kwargs)
        read = sm.with_music(
            "instagram",
            lambda r: src,
            test=TEST,
            folder=music,
            storage_dir=tmp_path,
            today=IN_TEST,
            make=_fake_make(calls),
        )
        assert read(rec) == b"muted"
    assert not calls


def test_no_licensed_track_posts_muted_with_no_receipt(tmp_path):
    music = tmp_path / "music"
    music.mkdir()
    _ledger(music, "a.mp3", complete=False)
    rec, src = _record(tmp_path)
    read = sm.with_music(
        "facebook",
        lambda r: src,
        test=TEST,
        folder=music,
        storage_dir=tmp_path,
        today=IN_TEST,
        make=_fake_make([]),
    )
    assert read(rec) == b"muted"
    assert sm.read_receipts(tmp_path) == set()


def test_a_failed_copy_never_blocks_the_post(tmp_path):
    music = tmp_path / "music"
    music.mkdir()
    _ledger(music, "a.mp3")
    rec, src = _record(tmp_path)

    def boom(*a):
        raise sm.MusicRefused("nope")

    read = sm.with_music(
        "facebook",
        lambda r: src,
        test=TEST,
        folder=music,
        storage_dir=tmp_path,
        today=IN_TEST,
        make=boom,
    )
    assert read(rec) == b"muted" and sm.read_receipts(tmp_path) == set()


def test_a_short_with_its_own_audio_is_refused(tmp_path):
    src = tmp_path / "s.mp4"
    src.write_bytes(b"x")
    track = sm.Track(tmp_path / "a.mp3", {})
    ran = []
    with pytest.raises(sm.MusicRefused, match="own audio"):
        sm.ensure_music_copy(src, track, 20.0, probe=lambda p: 1, run=ran.append)
    with pytest.raises(sm.MusicRefused, match="probe"):
        sm.ensure_music_copy(src, track, 20.0, probe=lambda p: None, run=ran.append)
    assert not ran


def test_the_copy_keeps_the_video_and_only_adds_the_track(tmp_path):
    src = tmp_path / "s.mp4"
    src.write_bytes(b"x")
    track = sm.Track(tmp_path / "a.mp3", {})
    seen = []

    def run(argv):
        seen.append(argv)
        Path(argv[-1]).write_bytes(b"out")

    out = sm.ensure_music_copy(src, track, 20.0, probe=lambda p: 0, run=run)
    argv = seen[0]
    assert out.read_bytes() == b"out" and out.name.startswith("s.music-")
    assert argv[argv.index("-c:v") + 1] == "copy"
    assert ["-map", "0:v:0", "-map", "1:a:0"] == argv[argv.index("-map") : argv.index("-map") + 4]
    assert argv[argv.index("-c:a") + 1] == "aac" and argv[argv.index("-ar") + 1] == "48000"
    # Reused, not re-encoded.
    sm.ensure_music_copy(src, track, 20.0, probe=lambda p: 0, run=run)
    assert len(seen) == 1


def test_ledger_rejects_paths_and_missing_files(tmp_path):
    (tmp_path / "ledger.json").write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "file": "../x.mp3",
                        "title": "t",
                        "source": "s",
                        "licence": "l",
                        "url": "u",
                        "added": "d",
                    },
                    {
                        "file": "gone.mp3",
                        "title": "t",
                        "source": "s",
                        "licence": "l",
                        "url": "u",
                        "added": "d",
                    },
                ]
            }
        )
    )
    assert sm.licensed_tracks(tmp_path) == []
    assert sm.licensed_tracks(None) == []


def test_music_dir_defaults_beside_the_host_brands_dir(monkeypatch, tmp_path):
    monkeypatch.delenv("GMS_MUSIC_DIR", raising=False)
    monkeypatch.setenv("GMS_BRANDS_DIR", str(tmp_path / "brands"))
    assert sm.music_dir("gomoveshift") == tmp_path / "music" / "gomoveshift"
    monkeypatch.setenv("GMS_MUSIC_DIR", str(tmp_path / "m"))
    assert sm.music_dir("gomoveshift") == tmp_path / "m"
    monkeypatch.delenv("GMS_MUSIC_DIR")
    monkeypatch.delenv("GMS_BRANDS_DIR")
    assert sm.music_dir("gomoveshift") is None


def test_growth_review_groups_only_shorts_that_really_had_music():
    from gomoveshift.tracking.growth_review import music_grouper

    on = next(i for i in range(100) if sm.music_arm("p", i))
    off = next(i for i in range(100) if not sm.music_arm("p", i))
    other_on = next(i for i in range(on + 1, 200) if sm.music_arm("p", i))
    group = music_grouper(TEST, {("p", on, "facebook")})
    d = date(2026, 12, 1)
    assert group("p", on, "facebook", d) == "music:on"
    assert group("p", other_on, "facebook", d) is None
    assert group("p", off, "instagram", d) == "music:off"
    assert group("p", off, "youtube", d) is None
    assert group("p", off, "facebook", date(2026, 11, 1)) is None


def test_the_shipped_gomoveshift_brand_declares_the_test():
    from gomoveshift.publishing.brand_policy import load_brand_policy

    test = load_brand_policy("gomoveshift").shorts_music
    assert test is not None and set(test.platforms) == {"facebook", "instagram"}

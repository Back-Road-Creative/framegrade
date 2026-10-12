"""Tests for photo selector with journey rotation."""

import textwrap
from pathlib import Path

import pytest

from gomoveshift.pipeline.photos.selector import (
    PhotoSelection,
    PhotoSelectionState,
    PhotoSelectionStateError,
    PhotoSelector,
    _extract_photos_from_journey,
    _parse_frontmatter,
)


@pytest.fixture
def sample_frontmatter():
    """Minimal journey frontmatter for testing."""
    return {
        "title": "Example Road Trip - 2025",
        "days": [
            {
                "day_number": 1,
                "date": "2025-05-10",
                "summary": "Day one through the mountains",
                "locations": ["Mockridge Pass", "Example Gulch"],
                "statistics": {"duration_hours": 4.2, "photo_count": 3},
                "photos": [
                    {
                        "location": "Mockridge Pass",
                        "photos": ["photo-001.jpg", "photo-002.jpg"],
                    },
                    {
                        "location": "Example Gulch",
                        "photos": ["photo-003.jpg"],
                    },
                ],
            },
            {
                "day_number": 2,
                "date": "2025-05-11",
                "summary": "Day two at Sample Rocks",
                "locations": ["Sample Rocks"],
                "statistics": {"duration_hours": 5.1},
                "photos": [
                    {
                        "location": "Sample Rocks",
                        "photos": ["photo-010.jpg"],
                    },
                ],
            },
        ],
    }


class TestExtractPhotos:
    def test_extracts_all_photos(self, sample_frontmatter):
        photos = _extract_photos_from_journey(sample_frontmatter, "example-2025")
        assert len(photos) == 4

    def test_preserves_day_info(self, sample_frontmatter):
        photos = _extract_photos_from_journey(sample_frontmatter, "example-2025")
        assert photos[0]["day_number"] == 1
        assert photos[0]["date"] == "2025-05-10"
        assert photos[0]["location"] == "Mockridge Pass"
        assert photos[0]["filename"] == "photo-001.jpg"

    def test_preserves_locations_visited(self, sample_frontmatter):
        photos = _extract_photos_from_journey(sample_frontmatter, "example-2025")
        assert "Mockridge Pass" in photos[0]["locations_visited"]
        assert "Example Gulch" in photos[0]["locations_visited"]

    def test_handles_empty_days(self):
        fm = {"days": [{"day_number": 1, "date": "2025-01-01", "photos": []}]}
        photos = _extract_photos_from_journey(fm, "test")
        assert len(photos) == 0

    def test_handles_no_days(self):
        fm = {}
        photos = _extract_photos_from_journey(fm, "test")
        assert len(photos) == 0


class TestParseFrontmatter:
    def test_parses_yaml_frontmatter(self, tmp_path):
        md = tmp_path / "test.md"
        md.write_text(
            textwrap.dedent("""\
            ---
            title: Test Journey
            photo_count: 5
            ---
            ## Content here
        """)
        )
        fm = _parse_frontmatter(md)
        assert fm["title"] == "Test Journey"
        assert fm["photo_count"] == 5

    def test_returns_none_for_no_frontmatter(self, tmp_path):
        md = tmp_path / "test.md"
        md.write_text("# Just markdown, no frontmatter")
        assert _parse_frontmatter(md) is None


class TestPhotoSelectionStateAtomicSave:
    """State is the posted-photos ledger — a failed write must not corrupt it."""

    def test_failed_save_preserves_previous_state(self, tmp_path, monkeypatch):
        from gomoveshift.pipeline import _atomic_write

        path = tmp_path / "state.json"
        PhotoSelectionState(posted_photos=["journey-a/photo-001.jpg"]).save(path)

        def boom(*args, **kwargs):
            raise OSError("disk full mid-write")

        monkeypatch.setattr(_atomic_write.json, "dump", boom)
        with pytest.raises(OSError):
            PhotoSelectionState(posted_photos=["journey-b/photo-999.jpg"]).save(path)

        assert PhotoSelectionState.load(path).posted_photos == ["journey-a/photo-001.jpg"]
        assert list(tmp_path.glob("*.tmp")) == [], "temp file left behind"


class TestDayOrdering:
    """Module contract: photos flow in sequential day order, not file order."""

    def test_photos_sorted_by_day_number(self):
        fm = {
            "days": [
                {"day_number": 3, "date": "2025-01-03", "photos": [{"photos": ["c.jpg"]}]},
                {"day_number": 1, "date": "2025-01-01", "photos": [{"photos": ["a.jpg"]}]},
                {"day_number": 2, "date": "2025-01-02", "photos": [{"photos": ["b.jpg"]}]},
            ]
        }
        photos = _extract_photos_from_journey(fm, "test")
        assert [p["filename"] for p in photos] == ["a.jpg", "b.jpg", "c.jpg"]

    def test_falls_back_to_date_without_day_number(self):
        fm = {
            "days": [
                {"date": "2025-01-09", "photos": [{"photos": ["late.jpg"]}]},
                {"date": "2025-01-02", "photos": [{"photos": ["early.jpg"]}]},
            ]
        }
        photos = _extract_photos_from_journey(fm, "test")
        assert [p["filename"] for p in photos] == ["early.jpg", "late.jpg"]


class TestPhotoSelectionState:
    def test_save_and_load(self, tmp_path):
        state = PhotoSelectionState(
            posted_photos=["journey1/photo-001.jpg", "journey1/photo-002.jpg"],
            journey_rotation_index=3,
        )
        path = tmp_path / "state.json"
        state.save(path)

        loaded = PhotoSelectionState.load(path)
        assert loaded.posted_photos == state.posted_photos
        assert loaded.journey_rotation_index == 3

    def test_load_missing_returns_empty(self, tmp_path):
        state = PhotoSelectionState.load(tmp_path / "nonexistent.json")
        assert state.posted_photos == []
        assert state.journey_rotation_index == 0


class TestSelectionStateFailClosed:
    """The state file is the posted-photos consumption ledger. A corrupt or
    incompatible file must RAISE, never silently reset dedup history — a silent
    empty state re-offers the entire shipped back catalog as new. Only genuine
    absence (first run) yields fresh state."""

    def test_corrupt_json_raises_named_error(self, tmp_path):
        path = tmp_path / "state.json"
        path.write_text("{not valid json")
        with pytest.raises(PhotoSelectionStateError) as exc_info:
            PhotoSelectionState.load(path)
        msg = str(exc_info.value)
        assert str(path) in msg
        assert "--reset-selection-state" in msg

    def test_incompatible_schema_raises(self, tmp_path):
        path = tmp_path / "state.json"
        path.write_text('{"posted_photos": [], "field_from_newer_schema": 1}')
        with pytest.raises(PhotoSelectionStateError):
            PhotoSelectionState.load(path)

    def test_non_dict_payload_raises(self, tmp_path):
        path = tmp_path / "state.json"
        path.write_text('["not", "a", "dict"]')
        with pytest.raises(PhotoSelectionStateError):
            PhotoSelectionState.load(path)

    def test_absent_file_still_returns_fresh_state(self, tmp_path):
        state = PhotoSelectionState.load(tmp_path / "absent.json")
        assert state.posted_photos == []
        assert state.pending_photos == []

    def test_reset_archives_existing_file(self, tmp_path):
        path = tmp_path / "state.json"
        path.write_text("{corrupt")
        archive = PhotoSelectionState.reset(path)
        assert archive is not None
        assert archive.parent == path.parent
        assert archive.name.startswith("state.json.corrupt-")
        # Archived aside, never deleted — the old ledger stays recoverable.
        assert archive.read_text() == "{corrupt"
        assert not path.exists()
        # The next load starts fresh: the wipe was explicitly accepted.
        assert PhotoSelectionState.load(path).posted_photos == []

    def test_reset_absent_file_is_noop(self, tmp_path):
        assert PhotoSelectionState.reset(tmp_path / "absent.json") is None

    def test_select_batch_raises_before_any_mutation(self, tmp_path):
        """The load failure surfaces before selection touches or rewrites the
        ledger — the corrupt file's bytes must be untouched afterwards."""
        journeys_dir = tmp_path / "content" / "journeys"
        journeys_dir.mkdir(parents=True)
        (journeys_dir / "journey-a.md").write_text(
            textwrap.dedent("""\
            ---
            title: Journey A
            days:
            - day_number: 1
              date: '2025-01-01'
              photos:
              - location: Place A
                photos:
                - photo-001.jpg
            ---
            Content
        """)
        )
        img_dir = tmp_path / "static" / "images" / "journeys" / "journey-a"
        img_dir.mkdir(parents=True)
        (img_dir / "photo-001.jpg").write_bytes(b"fake jpg")

        state_path = tmp_path / "state.json"
        corrupt = "{truncated ledger"
        state_path.write_text(corrupt)

        selector = PhotoSelector(site_root=tmp_path, state_path=state_path)
        with pytest.raises(PhotoSelectionStateError):
            selector.select_batch(count=1)
        assert state_path.read_text() == corrupt


class TestResetSelectionStateFlag:
    """``--reset-selection-state`` is the explicit escape hatch for the
    fail-closed load — registered on the selection-running subcommands."""

    @pytest.mark.parametrize(
        ("subcommand", "extra"),
        [
            ("generate", []),
            ("batch", ["--start", "2025-01-01", "--end", "2025-01-07"]),
        ],
    )
    def test_flag_accepted_default_off(self, subcommand, extra):
        from gomoveshift.cli.parser import _build_parser

        args = _build_parser().parse_args(["photos", subcommand, *extra])
        assert args.reset_selection_state is False

        args = _build_parser().parse_args(["photos", subcommand, *extra, "--reset-selection-state"])
        assert args.reset_selection_state is True


class TestPhotoSelector:
    @pytest.fixture
    def site_root(self, tmp_path):
        """Create a minimal Hugo site structure for testing."""
        journeys_dir = tmp_path / "content" / "journeys"
        journeys_dir.mkdir(parents=True)

        journey1 = journeys_dir / "journey-a.md"
        journey1.write_text(
            textwrap.dedent("""\
            ---
            title: Journey A
            gallery_path: images/journeys/journey-a
            photo_count: 3
            days:
            - day_number: 1
              date: '2025-01-01'
              summary: Day one
              locations: [Place A]
              statistics:
                duration_hours: 3.0
              photos:
              - location: Place A
                photos:
                - photo-001.jpg
                - photo-002.jpg
                - photo-003.jpg
            ---
            Content
        """)
        )

        journey2 = journeys_dir / "journey-b.md"
        journey2.write_text(
            textwrap.dedent("""\
            ---
            title: Journey B
            gallery_path: images/journeys/journey-b
            photo_count: 2
            days:
            - day_number: 1
              date: '2025-02-01'
              summary: Day one of B
              locations: [Place B]
              statistics:
                duration_hours: 2.0
              photos:
              - location: Place B
                photos:
                - photo-001.jpg
                - photo-002.jpg
            ---
            Content
        """)
        )

        for journey_slug in ["journey-a", "journey-b"]:
            img_dir = tmp_path / "static" / "images" / "journeys" / journey_slug
            img_dir.mkdir(parents=True)
            for i in range(1, 4):
                (img_dir / f"photo-00{i}.jpg").write_bytes(b"fake jpg")

        return tmp_path

    def test_loads_journeys(self, site_root, tmp_path):
        selector = PhotoSelector(site_root=site_root, state_path=tmp_path / "state.json")
        assert len(selector.journeys) == 2
        assert selector.get_total_photos() == 5

    def test_select_batch(self, site_root, tmp_path):
        selector = PhotoSelector(site_root=site_root, state_path=tmp_path / "state.json")
        selections = selector.select_batch(count=3)

        assert len(selections) == 3
        assert all(isinstance(s, PhotoSelection) for s in selections)
        for sel in selections:
            assert sel.journey_slug
            assert sel.filename
            assert Path(sel.photo_path).exists()

    def test_no_duplicates_across_batches(self, site_root, tmp_path):
        selector = PhotoSelector(site_root=site_root, state_path=tmp_path / "state.json")

        batch1 = selector.select_batch(count=2)
        batch2 = selector.select_batch(count=2)

        filenames1 = {f"{s.journey_slug}/{s.filename}" for s in batch1}
        filenames2 = {f"{s.journey_slug}/{s.filename}" for s in batch2}

        assert filenames1.isdisjoint(filenames2), "Batches should not overlap"

    def test_wraps_around_when_exhausted(self, site_root, tmp_path):
        selector = PhotoSelector(site_root=site_root, state_path=tmp_path / "state.json")

        batch1 = selector.select_batch(count=5)
        # Publish success clears the holds — only then is the pool genuinely
        # exhausted and eligible for the wrap-around reset. (An unpublished
        # batch's holds block the reset; see TestExhaustedPoolResetGuard.)
        selector.commit_posted(f"{s.journey_slug}/{s.filename}" for s in batch1)

        more = selector.select_batch(count=2)
        assert len(more) == 2


class TestJourneyInterleaving:
    """Tests that round-robin interleaving prevents journey clustering."""

    @pytest.fixture
    def skewed_site(self, tmp_path):
        """Create a site with skewed photo counts across journeys."""
        journeys_dir = tmp_path / "content" / "journeys"
        journeys_dir.mkdir(parents=True)

        # Journey A: 12 photos (dominant)
        photos_a = []
        for i in range(1, 13):
            photos_a.append(f"photo-{i:03d}.jpg")
        journey_a = journeys_dir / "journey-a.md"
        photo_yaml = "\n                - ".join(photos_a)
        journey_a.write_text(
            textwrap.dedent(f"""\
            ---
            title: Journey A
            days:
            - day_number: 1
              date: '2025-01-01'
              summary: Day one
              locations: [Place A]
              photos:
              - location: Place A
                photos:
                - {photo_yaml}
            ---
            Content
        """)
        )

        # Journey B: 4 photos
        journey_b = journeys_dir / "journey-b.md"
        journey_b.write_text(
            textwrap.dedent("""\
            ---
            title: Journey B
            days:
            - day_number: 1
              date: '2025-02-01'
              summary: Day one B
              locations: [Place B]
              photos:
              - location: Place B
                photos:
                - photo-001.jpg
                - photo-002.jpg
                - photo-003.jpg
                - photo-004.jpg
            ---
            Content
        """)
        )

        # Journey C: 1 photo
        journey_c = journeys_dir / "journey-c.md"
        journey_c.write_text(
            textwrap.dedent("""\
            ---
            title: Journey C
            days:
            - day_number: 1
              date: '2025-03-01'
              summary: Day one C
              locations: [Place C]
              photos:
              - location: Place C
                photos:
                - photo-001.jpg
            ---
            Content
        """)
        )

        # Create image files on disk
        for slug, count in [("journey-a", 12), ("journey-b", 4), ("journey-c", 1)]:
            img_dir = tmp_path / "static" / "images" / "journeys" / slug
            img_dir.mkdir(parents=True)
            for i in range(1, count + 1):
                (img_dir / f"photo-{i:03d}.jpg").write_bytes(b"fake jpg")

        return tmp_path

    def test_no_journey_more_than_2_consecutive(self, skewed_site, tmp_path):
        """With skewed photo counts, no journey clusters early in the batch.

        Once smaller journeys exhaust their photos, the dominant journey
        naturally fills remaining slots — that's expected. We check only
        the first N slots where all journeys still have photos available.
        """
        selector = PhotoSelector(site_root=skewed_site, state_path=tmp_path / "state.json")
        # Select only as many as the diverse portion (A:12, B:4, C:1 = 17 total,
        # but B+C exhaust early). Use count=5 so all journeys can contribute.
        selections = selector.select_batch(count=5)

        # Check for runs of >2 consecutive from the same journey
        max_run = 1
        current_run = 1
        for i in range(1, len(selections)):
            if selections[i].journey_slug == selections[i - 1].journey_slug:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 1

        assert max_run <= 2, (
            f"Journey appeared {max_run} times consecutively: "
            f"{[s.journey_slug for s in selections]}"
        )

    def test_all_journeys_represented(self, skewed_site, tmp_path):
        """Every journey with photos appears at least once."""
        selector = PhotoSelector(site_root=skewed_site, state_path=tmp_path / "state.json")
        selections = selector.select_batch(count=10)
        slugs = {s.journey_slug for s in selections}
        assert "journey-a" in slugs
        assert "journey-b" in slugs
        assert "journey-c" in slugs

    def test_single_journey_still_works(self, tmp_path):
        """A site with only one journey still selects normally."""
        journeys_dir = tmp_path / "content" / "journeys"
        journeys_dir.mkdir(parents=True)
        journey = journeys_dir / "solo.md"
        journey.write_text(
            textwrap.dedent("""\
            ---
            title: Solo Journey
            days:
            - day_number: 1
              date: '2025-01-01'
              summary: Day one
              locations: [Solo Place]
              photos:
              - location: Solo Place
                photos:
                - photo-001.jpg
                - photo-002.jpg
                - photo-003.jpg
            ---
            Content
        """)
        )
        img_dir = tmp_path / "static" / "images" / "journeys" / "solo"
        img_dir.mkdir(parents=True)
        for i in range(1, 4):
            (img_dir / f"photo-{i:03d}.jpg").write_bytes(b"fake jpg")

        selector = PhotoSelector(site_root=tmp_path, state_path=tmp_path / "state.json")
        selections = selector.select_batch(count=3)
        assert len(selections) == 3
        assert all(s.journey_slug == "solo" for s in selections)


class TestExhaustionIterativeRetry:
    """Exhaustion retries iteratively instead of recursing."""

    def test_exhaustion_uses_iterative_retry(self, tmp_path):
        """After exhausting all photos, reset + retry works without recursion."""
        journeys_dir = tmp_path / "content" / "journeys"
        journeys_dir.mkdir(parents=True)
        journey = journeys_dir / "journey-a.md"
        journey.write_text(
            textwrap.dedent("""\
            ---
            title: Journey A
            days:
            - day_number: 1
              date: '2025-01-01'
              summary: Day one
              locations: [Place A]
              photos:
              - location: Place A
                photos:
                - photo-001.jpg
                - photo-002.jpg
            ---
            Content
        """)
        )
        img_dir = tmp_path / "static" / "images" / "journeys" / "journey-a"
        img_dir.mkdir(parents=True)
        (img_dir / "photo-001.jpg").write_bytes(b"fake jpg")
        (img_dir / "photo-002.jpg").write_bytes(b"fake jpg")

        state_path = tmp_path / "state.json"

        # Pre-seed state with all photos posted (exhausted)
        state = PhotoSelectionState(
            posted_photos=["journey-a/photo-001.jpg", "journey-a/photo-002.jpg"],
            journey_rotation_index=10,
        )
        state.save(state_path)

        selector = PhotoSelector(site_root=tmp_path, state_path=state_path)
        # Should reset and select without hitting recursion limit
        selections = selector.select_batch(count=2)
        assert len(selections) == 2

        # Verify state was reset then re-populated
        reloaded = PhotoSelectionState.load(state_path)
        assert len(reloaded.posted_photos) == 2

    def test_no_recursion_attribute(self, tmp_path):
        """select_batch does not call itself — _select_from_state is the helper."""
        import inspect

        source = inspect.getsource(PhotoSelector.select_batch)
        # select_batch should not contain "self.select_batch(" — that would be recursion
        assert "self.select_batch(" not in source


class TestShortfallContract:
    """A near-exhausted pool must hard-fail, never ship a thinner week."""

    @pytest.fixture
    def two_photo_site(self, tmp_path):
        journeys_dir = tmp_path / "content" / "journeys"
        journeys_dir.mkdir(parents=True)
        (journeys_dir / "journey-a.md").write_text(
            textwrap.dedent("""\
            ---
            title: Journey A
            days:
            - day_number: 1
              date: '2025-01-01'
              photos:
              - location: Place A
                photos:
                - photo-001.jpg
                - photo-002.jpg
            ---
            Content
        """)
        )
        img_dir = tmp_path / "static" / "images" / "journeys" / "journey-a"
        img_dir.mkdir(parents=True)
        for name in ["photo-001.jpg", "photo-002.jpg"]:
            (img_dir / name).write_bytes(b"fake jpg")
        return tmp_path

    def test_partial_pool_raises(self, two_photo_site, tmp_path):
        state_path = tmp_path / "state.json"
        PhotoSelectionState(posted_photos=["journey-a/photo-001.jpg"]).save(state_path)

        selector = PhotoSelector(site_root=two_photo_site, state_path=state_path)
        with pytest.raises(RuntimeError, match="PHOTO-CONTRACT"):
            selector.select_batch(count=2)

    def test_pool_smaller_than_request_raises_after_reset(self, two_photo_site, tmp_path):
        selector = PhotoSelector(site_root=two_photo_site, state_path=tmp_path / "state.json")
        with pytest.raises(RuntimeError, match="PHOTO-CONTRACT"):
            selector.select_batch(count=5)

    def test_shortfall_allowed_explicitly(self, two_photo_site, tmp_path):
        state_path = tmp_path / "state.json"
        PhotoSelectionState(posted_photos=["journey-a/photo-001.jpg"]).save(state_path)

        selector = PhotoSelector(site_root=two_photo_site, state_path=state_path)
        selections = selector.select_batch(count=2, allow_shortfall=True)
        assert len(selections) == 1


class TestExhaustedPoolResetGuard:
    """The rotation reset fires only on GENUINE exhaustion (every photo posted
    or face-filtered, nothing held, count >= 1). An empty selection from any
    other cause must never wipe the posted-photos ledger or pending holds."""

    @pytest.fixture
    def two_photo_site(self, tmp_path):
        journeys_dir = tmp_path / "content" / "journeys"
        journeys_dir.mkdir(parents=True)
        (journeys_dir / "journey-a.md").write_text(
            textwrap.dedent("""\
            ---
            title: Journey A
            days:
            - day_number: 1
              date: '2025-01-01'
              photos:
              - location: Place A
                photos:
                - photo-001.jpg
                - photo-002.jpg
            ---
            Content
        """)
        )
        img_dir = tmp_path / "static" / "images" / "journeys" / "journey-a"
        img_dir.mkdir(parents=True)
        for name in ["photo-001.jpg", "photo-002.jpg"]:
            (img_dir / name).write_bytes(b"fake jpg")
        return tmp_path

    def test_count_below_one_rejected_without_state_wipe(self, two_photo_site, tmp_path):
        state_path = tmp_path / "state.json"
        PhotoSelectionState(
            posted_photos=["journey-a/photo-001.jpg", "journey-a/photo-002.jpg"],
            pending_photos=["journey-a/photo-002.jpg"],
        ).save(state_path)

        selector = PhotoSelector(site_root=two_photo_site, state_path=state_path)
        for bad_count in (0, -1):
            with pytest.raises(ValueError, match="count"):
                selector.select_batch(count=bad_count)

        reloaded = PhotoSelectionState.load(state_path)
        assert set(reloaded.posted_photos) == {
            "journey-a/photo-001.jpg",
            "journey-a/photo-002.jpg",
        }
        assert reloaded.pending_photos == ["journey-a/photo-002.jpg"]

    def test_pending_holds_block_reset(self, two_photo_site, tmp_path):
        """All photos consumed but one is held by an unpublished batch:
        no wipe (the held photo would be re-offered while a pending week
        already ships it) — the count-match gate raises instead."""
        state_path = tmp_path / "state.json"
        PhotoSelectionState(
            posted_photos=["journey-a/photo-001.jpg", "journey-a/photo-002.jpg"],
            pending_photos=["journey-a/photo-002.jpg"],
        ).save(state_path)

        selector = PhotoSelector(site_root=two_photo_site, state_path=state_path)
        with pytest.raises(RuntimeError, match="PHOTO-CONTRACT"):
            selector.select_batch(count=1)

        reloaded = PhotoSelectionState.load(state_path)
        assert set(reloaded.posted_photos) == {
            "journey-a/photo-001.jpg",
            "journey-a/photo-002.jpg",
        }
        assert reloaded.pending_photos == ["journey-a/photo-002.jpg"]

    def test_missing_files_block_reset(self, two_photo_site, tmp_path):
        """Unposted photos exist but their files are missing on disk (bad
        checkout): not exhaustion — posted history must survive."""
        state_path = tmp_path / "state.json"
        PhotoSelectionState(posted_photos=["journey-a/photo-001.jpg"]).save(state_path)
        (two_photo_site / "static" / "images" / "journeys" / "journey-a" / "photo-002.jpg").unlink()

        selector = PhotoSelector(site_root=two_photo_site, state_path=state_path)
        with pytest.raises(RuntimeError, match="PHOTO-CONTRACT"):
            selector.select_batch(count=1)

        reloaded = PhotoSelectionState.load(state_path)
        assert reloaded.posted_photos == ["journey-a/photo-001.jpg"]

    def test_genuine_exhaustion_still_resets(self, two_photo_site, tmp_path):
        """Every photo posted, nothing held: the intended reset still fires."""
        state_path = tmp_path / "state.json"
        PhotoSelectionState(
            posted_photos=["journey-a/photo-001.jpg", "journey-a/photo-002.jpg"],
            journey_rotation_index=7,
        ).save(state_path)

        selector = PhotoSelector(site_root=two_photo_site, state_path=state_path)
        selections = selector.select_batch(count=2)
        assert len(selections) == 2


class TestFaceFilterPersistence:
    """Tests that face_filtered_photos is persisted in state."""

    def test_face_filtered_persisted_in_state(self, tmp_path):
        """Save/load round-trip preserves face_filtered_photos."""
        state = PhotoSelectionState(
            posted_photos=["journey-a/photo-001.jpg"],
            journey_rotation_index=2,
            face_filtered_photos=["journey-a/photo-003.jpg", "journey-b/photo-001.jpg"],
        )
        path = tmp_path / "state.json"
        state.save(path)

        loaded = PhotoSelectionState.load(path)
        assert set(loaded.face_filtered_photos) == {
            "journey-a/photo-003.jpg",
            "journey-b/photo-001.jpg",
        }

    def test_face_filtered_excluded_next_batch(self, tmp_path):
        """Photo previously skipped for faces stays excluded in next batch."""
        # Pre-seed state with a face-filtered photo
        state_path = tmp_path / "state.json"
        state = PhotoSelectionState(
            posted_photos=[],
            journey_rotation_index=0,
            face_filtered_photos=["journey-a/photo-001.jpg"],
        )
        state.save(state_path)

        # Build site with journey-a having 2 photos
        journeys_dir = tmp_path / "content" / "journeys"
        journeys_dir.mkdir(parents=True)
        journey = journeys_dir / "journey-a.md"
        journey.write_text(
            textwrap.dedent("""\
            ---
            title: Journey A
            days:
            - day_number: 1
              date: '2025-01-01'
              summary: Day one
              locations: [Place A]
              photos:
              - location: Place A
                photos:
                - photo-001.jpg
                - photo-002.jpg
            ---
            Content
        """)
        )
        img_dir = tmp_path / "static" / "images" / "journeys" / "journey-a"
        img_dir.mkdir(parents=True)
        (img_dir / "photo-001.jpg").write_bytes(b"fake jpg")
        (img_dir / "photo-002.jpg").write_bytes(b"fake jpg")

        selector = PhotoSelector(site_root=tmp_path, state_path=state_path)
        # count=1: photo-001 is face-filtered, so only photo-002 can satisfy it
        # (asking for 2 would now hard-fail the count-match gate).
        selections = selector.select_batch(count=1)

        # photo-001 should be skipped (face filtered), only photo-002 available
        filenames = [s.filename for s in selections]
        assert "photo-001.jpg" not in filenames
        assert "photo-002.jpg" in filenames

"""Tests for ContentLedger — scheduling integrity."""

import os
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from gomoveshift.config import Config, PublishSlot
from gomoveshift.pipeline.content_ledger import (
    ContentEntry,
    ContentLedger,
    next_free_superlong_slot,
)

EST = ZoneInfo("America/New_York")


class TestLedgerReload:
    """Ledger must reload from disk before allocating slots."""

    def test_reload_picks_up_entries_from_disk(self, tmp_path):
        """A ledger saved by process A should be visible to process B."""
        ledger_path = tmp_path / "ledger.json"

        # Process A: create and save a ledger with one entry
        ledger_a = ContentLedger()
        ledger_a.add(
            ContentEntry(
                date="2026-03-25",
                time="11:00:00",
                content_type="longform",
                title="Episode 1",
                project_name="episode_01",
                status="scheduled",
                source="pipeline",
            )
        )
        ledger_a.save(path=ledger_path)

        # Process B: fresh empty ledger, reload from disk
        ledger_b = ContentLedger()
        ledger_b.reload(path=ledger_path)
        assert len(ledger_b.entries) == 1
        assert ledger_b.entries[0].title == "Episode 1"

    def test_slot_allocation_skips_occupied_after_reload(self, tmp_path):
        """get_next_available_slot must see entries saved by another process."""
        ledger_path = tmp_path / "ledger.json"

        # Process A occupies Wednesday 2026-03-25 11:00
        ledger_a = ContentLedger()
        ledger_a.add(
            ContentEntry(
                date="2026-03-25",
                time="11:00:00",
                content_type="longform",
                title="Episode 1",
                project_name="episode_01",
                status="scheduled",
                source="pipeline",
            )
        )
        ledger_a.save(path=ledger_path)

        # Process B: load the shared ledger (no-arg via patched default) and
        # allocate the next slot. Must skip 2026-03-25 because it's occupied.
        from unittest.mock import patch

        with patch(
            "gomoveshift.pipeline.content_ledger._default_ledger_path",
            return_value=ledger_path,
        ):
            ledger_b = ContentLedger.load()
            slot = ledger_b.get_next_available_slot(
                content_type="longform",
                after=datetime(2026, 3, 23, 12, 0, tzinfo=EST),
            )
        assert slot is not None
        assert slot["date"] != "2026-03-25", f"Should have skipped occupied 2026-03-25, got: {slot}"


class TestLedgerInstancePinnedPath:
    """One instance ⇔ one file: no-arg reload()/save() reuse load()'s path.

    Regression for the divergence regenerator: a no-arg
    ``reload()`` inside ``get_next_available_slot`` resolved the default
    ledger path and silently re-merged an orphan store into an
    explicitly-loaded ledger, regenerating divergent copies on every run.
    """

    def _seed(self, path, date="2026-03-25", title="Fable Pass Drive"):
        orphan = ContentLedger()
        orphan.add(
            ContentEntry(
                date=date,
                time="11:00:00",
                content_type="longform",
                title=title,
                project_name="conor_pass",
                status="scheduled",
                source="pipeline",
            )
        )
        orphan.save(path=path)

    def test_reload_after_explicit_load_keeps_same_file(
        self, tmp_path, monkeypatch, waypost_lf_days
    ):
        """get_next_available_slot's no-arg reload must NOT merge the
        default store into an explicitly-loaded ledger."""
        from gomoveshift.pipeline import content_ledger as cl

        waypost_lf_days({"ireland": (2,), "usa": (2,)})  # the Wednesday cadence this assumes
        orphan_path = tmp_path / "output" / ".content_ledger.json"
        explicit_path = tmp_path / "seo" / ".content_ledger.json"
        self._seed(orphan_path)
        monkeypatch.setattr(cl, "_default_ledger_path", lambda: orphan_path)

        ledger = ContentLedger.load(path=explicit_path)  # absent → empty, pinned
        slot = ledger.get_next_available_slot(
            content_type="longform",
            after=datetime(2026, 3, 23, 12, 0, tzinfo=EST),
        )
        assert not ledger.entries, "orphan default store leaked into explicit ledger"
        assert slot is not None and slot["date"] == "2026-03-25"

    def test_save_no_arg_writes_back_to_load_path(self, tmp_path, monkeypatch):
        from gomoveshift.pipeline import content_ledger as cl

        orphan_path = tmp_path / "output" / ".content_ledger.json"
        explicit_path = tmp_path / "seo" / ".content_ledger.json"
        monkeypatch.setattr(cl, "_default_ledger_path", lambda: orphan_path)

        ledger = ContentLedger.load(path=explicit_path)
        ledger.add(
            ContentEntry(
                date="2026-04-01",
                time="11:00:00",
                content_type="longform",
                title="Fable Pass Drive",
            )
        )
        ledger.save()  # no arg — must reuse the load path
        assert explicit_path.exists()
        assert not orphan_path.exists()
        assert len(ContentLedger.load(path=explicit_path).entries) == 1

    def test_reload_no_arg_on_inmemory_ledger_is_noop(self, tmp_path, monkeypatch):
        """A ledger never loaded from disk has nothing to reload — it must
        not guess a default file."""
        from gomoveshift.pipeline import content_ledger as cl

        orphan_path = tmp_path / ".content_ledger.json"
        self._seed(orphan_path)
        monkeypatch.setattr(cl, "_default_ledger_path", lambda: orphan_path)

        ledger = ContentLedger()
        ledger.reload()
        assert ledger.entries == []

    @pytest.mark.real_content_ledger
    def test_default_path_resolves_seo_storage(self, tmp_path, monkeypatch):
        """The canonical default is <config.seo.storage_path>/.content_ledger.json
        — never gms_output_dir().

        Opts out of the conftest hermetic-ledger redirect because it tests
        the real resolver; still hermetic via the get_config patch.
        """
        from types import SimpleNamespace

        from gomoveshift.pipeline.content_ledger import _default_ledger_path

        monkeypatch.setattr(
            "gomoveshift.config.get_config",
            lambda config_path=None: SimpleNamespace(
                seo=SimpleNamespace(storage_path=str(tmp_path / "seo"))
            ),
        )
        assert _default_ledger_path() == tmp_path / "seo" / ".content_ledger.json"


class TestLedgerSaveWithLocking:
    """save() must use file locking."""

    def test_save_creates_valid_json(self, tmp_path):
        ledger_path = tmp_path / "ledger.json"
        ledger = ContentLedger()
        # Use a real-looking title — a bare "Test" is now filtered out by the
        # test-entry guard in save() (see TestLedgerTestEntryFilter).
        ledger.add(
            ContentEntry(
                date="2026-04-01",
                time="11:00:00",
                content_type="longform",
                title="Fable Pass Drive",
            )
        )
        ledger.save(path=ledger_path)

        # Verify it can be reloaded
        loaded = ContentLedger.load(path=ledger_path)
        assert len(loaded.entries) == 1
        assert loaded.entries[0].title == "Fable Pass Drive"

    def test_save_writes_group_writable_mode(self, tmp_path):
        """Atomic save (mkstemp + os.replace) must NOT collapse mode to 0600.

        Pre-fix: tempfile.mkstemp's default 0600 carried through os.replace,
        so each save reset the file to owner-only-rw. In a setgid-parent
        workspace where two users share one group, that broke the
        next invocation by the other user (a recurring
        failure). Save must explicitly chmod the final file to
        0664 so group members retain write access regardless of which
        user wrote it.
        """
        import stat

        ledger_path = tmp_path / "ledger.json"
        ledger = ContentLedger()
        ledger.save(path=ledger_path)

        mode = stat.S_IMODE(ledger_path.stat().st_mode)
        assert mode == 0o664, f"expected 0o664 after save, got 0o{mode:o}"

    def test_save_overwrite_preserves_group_writable_mode(self, tmp_path):
        """Repeated saves keep mode 0o664 (no drift on rewrite)."""
        import stat

        ledger_path = tmp_path / "ledger.json"
        ledger = ContentLedger()
        ledger.save(path=ledger_path)
        ledger.add(
            ContentEntry(
                date="2026-04-02",
                time="11:00:00",
                content_type="longform",
                title="Second",
            )
        )
        ledger.save(path=ledger_path)

        mode = stat.S_IMODE(ledger_path.stat().st_mode)
        assert mode == 0o664, f"expected 0o664 after overwrite, got 0o{mode:o}"


class TestLedgerLoadErrorVisibility:
    """Load must not silently swallow real failures.

    Pre-fix, ``except Exception`` returned an empty ledger for PermissionError,
    OSError, TypeError, *anything* — masking real problems as "0 entries loaded"
    and producing silently-degraded metadata. Narrowed to JSONDecodeError
    (the only genuinely recoverable corruption) so operational errors surface.
    """

    def test_permission_error_propagates(self, tmp_path):
        """A ledger that exists but is unreadable must raise, not return empty."""
        ledger_path = tmp_path / "ledger.json"
        ledger_path.write_text('{"entries": []}')
        os.chmod(ledger_path, 0o000)
        try:
            if os.access(ledger_path, os.R_OK):
                pytest.skip("running as root — cannot simulate permission denial")
            with pytest.raises(PermissionError):
                ContentLedger.load(path=ledger_path)
        finally:
            os.chmod(ledger_path, 0o644)

    def test_corrupt_json_raises_and_logs(self, tmp_path, caplog):
        """A JSON-corrupt ledger must raise, not silently degrade to empty —
        an empty in-memory ledger looks like every slot is free, so the very
        next booking (``record_confirmed_upload``) truncates the real file
        down to just that one entry, destroying every historical and
        future-scheduled booking. Log loud, then raise (mirrors the sibling
        ``OSError`` handling six lines above)."""
        ledger_path = tmp_path / "ledger.json"
        ledger_path.write_text("{ this is not json")
        import json
        import logging

        with caplog.at_level(logging.ERROR, logger="gomoveshift.pipeline.content_ledger"):
            with pytest.raises(json.JSONDecodeError):
                ContentLedger.load(path=ledger_path)
        assert any(
            "corrupt" in rec.message.lower() or "jsondecode" in rec.message.lower()
            for rec in caplog.records
        ), f"expected ERROR-level corruption log, got: {[r.message for r in caplog.records]}"

    def test_corrupt_ledger_booking_does_not_truncate_file(self, tmp_path):
        """The actual damage a silently-empty ledger caused: a booking against
        a corrupt file must not ``os.replace`` it down to a fresh
        single-entry ledger. The corrupt bytes on disk must survive."""
        from gomoveshift.pipeline.content_ledger import ContentEntry, record_confirmed_upload

        ledger_path = tmp_path / "ledger.json"
        corrupt_bytes = b"{ this is not json"
        ledger_path.write_bytes(corrupt_bytes)

        ok = record_confirmed_upload(
            [ContentEntry(date="2026-06-21", time="14:00:00", content_type="longform")],
            ledger_path=ledger_path,
        )

        assert ok is False
        assert ledger_path.read_bytes() == corrupt_bytes

    def test_missing_file_returns_empty_without_error(self, tmp_path):
        """A never-written ledger path is a normal first-run state, not an error."""
        loaded = ContentLedger.load(path=tmp_path / "does-not-exist.json")
        assert loaded.entries == []


class TestLedgerAddIdempotency:
    """add() must be idempotent on (date, time, content_type)."""

    def _entry(self, **overrides):
        base = dict(
            date="2026-05-01",
            time="14:00:00",
            content_type="longform",
            title="Ballymock to Glenfable",
            project_name="kerry_lf01",
            status="scheduled",
            source="pipeline",
        )
        base.update(overrides)
        return ContentEntry(**base)

    def test_same_entry_twice_is_idempotent(self):
        """Re-adding identical content neither duplicates nor mutates."""
        ledger = ContentLedger()
        ledger.add(self._entry())
        ledger.add(self._entry())  # exact replay (resume / re-run)

        assert len(ledger.entries) == 1
        assert ledger.entries[0].title == "Ballymock to Glenfable"
        assert ledger.entries[0].project_name == "kerry_lf01"

    def test_conflicting_metadata_from_other_source_is_rejected(self, caplog):
        """A different source cannot clobber a committed title; orig kept + WARN."""
        import logging

        ledger = ContentLedger()
        ledger.add(self._entry())  # source="pipeline"

        with caplog.at_level(logging.WARNING, logger="gomoveshift.pipeline.content_ledger"):
            ledger.add(
                self._entry(
                    title="WRONG County Scenic Drive",
                    project_name="other_project",
                    source="metricool_csv",
                )
            )

        assert len(ledger.entries) == 1
        # Committed fields unchanged — the first writer owns the slot.
        assert ledger.entries[0].title == "Ballymock to Glenfable"
        assert ledger.entries[0].project_name == "kerry_lf01"
        assert ledger.entries[0].source == "pipeline"
        assert any("LEDGER-CONFLICT" in rec.message for rec in caplog.records)

    def test_same_source_refresh_is_allowed(self):
        """The original producer may still refresh its own slot's metadata."""
        ledger = ContentLedger()
        ledger.add(self._entry(title="Old title"))
        ledger.add(self._entry(title="Corrected title"))  # same source="pipeline"

        assert len(ledger.entries) == 1
        assert ledger.entries[0].title == "Corrected title"

    def test_blank_incoming_title_inherits_existing(self):
        """A replay with a blank title must not erase the committed title."""
        ledger = ContentLedger()
        ledger.add(self._entry(title="Ballymock to Glenfable"))
        ledger.add(self._entry(title=""))  # status/lifecycle replay, no new title

        assert len(ledger.entries) == 1
        assert ledger.entries[0].title == "Ballymock to Glenfable"


class TestLedgerTestEntryFilter:
    """save() must drop obvious test-origin entries."""

    def test_test_location_entry_filtered_real_entry_persists(self, tmp_path, caplog):
        import logging

        ledger_path = tmp_path / "ledger.json"
        ledger = ContentLedger()
        ledger.add(
            ContentEntry(
                date="2026-05-02",
                time="14:00:00",
                content_type="longform",
                title="Test Location",  # denylisted sentinel
                source="pipeline",
            )
        )
        ledger.add(
            ContentEntry(
                date="2026-05-03",
                time="14:00:00",
                content_type="longform",
                title="Fable Pass Drive",  # real entry
                project_name="kerry_lf02",
                source="pipeline",
            )
        )

        with caplog.at_level(logging.WARNING, logger="gomoveshift.pipeline.content_ledger"):
            ledger.save(path=ledger_path)

        loaded = ContentLedger.load(path=ledger_path)
        titles = [e.title for e in loaded.entries]
        assert "Test Location" not in titles
        assert "Fable Pass Drive" in titles
        assert len(loaded.entries) == 1
        assert any("LEDGER-TEST-FILTER" in rec.message for rec in caplog.records)

    def test_test_project_prefix_filtered(self, tmp_path):
        ledger_path = tmp_path / "ledger.json"
        ledger = ContentLedger()
        ledger.add(
            ContentEntry(
                date="2026-05-04",
                time="14:00:00",
                content_type="shorts",
                title="Some short",
                project_name="test_fixture_project",  # test_ prefix
                source="pipeline",
            )
        )
        ledger.save(path=ledger_path)

        loaded = ContentLedger.load(path=ledger_path)
        assert loaded.entries == []

    def test_real_entry_with_test_substring_survives(self, tmp_path):
        """Guard is narrow: a legit title CONTAINING 'test' is not dropped."""
        ledger_path = tmp_path / "ledger.json"
        ledger = ContentLedger()
        ledger.add(
            ContentEntry(
                date="2026-05-05",
                time="14:00:00",
                content_type="longform",
                title="Testing the Brakes on the N86",
                project_name="kerry_lf03",
                source="pipeline",
            )
        )
        ledger.save(path=ledger_path)

        loaded = ContentLedger.load(path=ledger_path)
        assert len(loaded.entries) == 1
        assert loaded.entries[0].title == "Testing the Brakes on the N86"

    def test_save_does_not_mutate_in_memory_entries(self, tmp_path):
        """Filtering happens on the serialized copy; in-memory ledger is intact."""
        ledger_path = tmp_path / "ledger.json"
        ledger = ContentLedger()
        ledger.add(
            ContentEntry(
                date="2026-05-06",
                time="14:00:00",
                content_type="longform",
                title="Test",  # denylisted
                source="pipeline",
            )
        )
        ledger.save(path=ledger_path)
        # In-memory entry survives — only the persisted copy is filtered.
        assert len(ledger.entries) == 1


class TestSuperlongSlotOwnTime:
    """A superlong part is booked at its OWN designed slot time — the day's
    ``publish_slots[day].lf_time`` — never the flat long-form default the
    search happens to use internally."""

    def test_uses_the_days_designed_time_not_the_flat_default(self):
        config = Config()
        config.drive.superlong_publish_day = "sat"
        config.drive.publish_slots["sat"] = PublishSlot(lf_time="20:00")
        anchor = datetime(2026, 8, 10, 9, 0, tzinfo=EST)  # a Monday
        slot = next_free_superlong_slot(ContentLedger(), config, "usa", after=anchor)
        assert slot.strftime("%H:%M") == "20:00"


class TestSuperlongMinimumPublishLead:
    """A superlong slot resolved too close to now leaves too little runway
    before its own multi-hour, tens-of-GB upload even starts — the upload
    can overrun the slot it was given. The search must skip a too-thin
    lead rather than return it."""

    def test_thin_lead_pushes_to_the_next_eligible_slot(self):
        config = Config()  # default: sat, 11:00
        anchor = datetime(2026, 8, 15, 9, 0, tzinfo=EST)  # a Saturday, 09:00 — 2h lead to 11:00
        slot = next_free_superlong_slot(ContentLedger(), config, "usa", after=anchor)
        assert slot.date() != anchor.date(), f"a 2h lead was returned as-is: {slot.isoformat()}"
        assert slot.strftime("%H:%M") == "11:00"

"""Tests for the Python preflight validation module.

Validates disk space, OST uniqueness, and input file checks that run
before Phase A in the drive orchestrator.
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest


class TestPreflightDiskSpace:
    """check_disk_space raises on insufficient free space."""

    def test_sufficient_space_passes(self, tmp_path):
        from gomoveshift.pipeline.preflight import check_disk_space

        # Mock shutil.disk_usage to return plenty of space
        usage = type("Usage", (), {"free": 100 * 1_073_741_824})()  # 100 GB
        with patch("shutil.disk_usage", return_value=usage):
            # Should not raise
            check_disk_space(tmp_path, min_free_gb=50)

    def test_insufficient_space_raises(self, tmp_path):
        from gomoveshift.pipeline.preflight import check_disk_space

        usage = type("Usage", (), {"free": 10 * 1_073_741_824})()  # 10 GB
        with patch("shutil.disk_usage", return_value=usage):
            with pytest.raises(RuntimeError, match="disk space"):
                check_disk_space(tmp_path, min_free_gb=50)

    def test_nonexistent_path_walks_up(self, tmp_path):
        from gomoveshift.pipeline.preflight import check_disk_space

        deep = tmp_path / "a" / "b" / "c"  # doesn't exist
        usage = type("Usage", (), {"free": 200 * 1_073_741_824})()
        with patch("shutil.disk_usage", return_value=usage):
            check_disk_space(deep, min_free_gb=50)


class TestSuperlongOutputSpace:
    """check_superlong_output_space scales the output floor to the actual
    day-master bytes/sec x planned superlong part duration, not a fixed
    floor — a 4h part FINAL (~70-90 GB) dwarfs the 50 GB gate."""

    _GB = 1_073_741_824

    def test_insufficient_space_raises_with_gb_shortfall(self, tmp_path):
        from gomoveshift.pipeline.preflight import check_superlong_output_space

        master = tmp_path / "day_master.mp4"
        master.write_bytes(b"x")
        # 10 GB / 3600s master -> ~2.78 MB/s; 4h planned parts x 1.2 safety
        # needs ~40 GB. Only 5 GB free.
        with (
            patch.object(Path, "stat", return_value=type("S", (), {"st_size": 10 * self._GB})()),
            patch("shutil.disk_usage", return_value=type("U", (), {"free": 5 * self._GB})()),
        ):
            with pytest.raises(RuntimeError, match="(?i)output space"):
                check_superlong_output_space(
                    master,
                    tmp_path,
                    total_part_duration_sec=14400.0,
                    probe_fn=lambda p: 3600.0,
                )

    def test_sufficient_space_passes(self, tmp_path):
        from gomoveshift.pipeline.preflight import check_superlong_output_space

        master = tmp_path / "day_master.mp4"
        master.write_bytes(b"x")
        with (
            patch.object(Path, "stat", return_value=type("S", (), {"st_size": 10 * self._GB})()),
            patch("shutil.disk_usage", return_value=type("U", (), {"free": 500 * self._GB})()),
        ):
            check_superlong_output_space(
                master,
                tmp_path,
                total_part_duration_sec=14400.0,
                probe_fn=lambda p: 3600.0,
            )

    def test_reserves_for_raw_concat_alongside_the_final(self, tmp_path):
        """The raw stream-copied concat (sl_NN_segments.mp4) co-exists with
        the final for the whole OST-encode duration but was never priced —
        the gate only reserved one finals-equivalent copy. 70 GB free sits
        between the old (~48 GB) and the correct (~96 GB) requirement: it
        passed the old gate and must raise once the raw-concat term is
        included."""
        from gomoveshift.pipeline.preflight import check_superlong_output_space

        master = tmp_path / "day_master.mp4"
        master.write_bytes(b"x")
        with (
            patch.object(Path, "stat", return_value=type("S", (), {"st_size": 10 * self._GB})()),
            patch("shutil.disk_usage", return_value=type("U", (), {"free": 70 * self._GB})()),
        ):
            with pytest.raises(RuntimeError, match="(?i)output space"):
                check_superlong_output_space(
                    master,
                    tmp_path,
                    total_part_duration_sec=14400.0,
                    probe_fn=lambda p: 3600.0,
                )

    def test_unmeasurable_duration_skips_rather_than_raises(self, tmp_path):
        from gomoveshift.pipeline.preflight import check_superlong_output_space

        master = tmp_path / "day_master.mp4"
        master.write_bytes(b"x")
        with patch("shutil.disk_usage", return_value=type("U", (), {"free": 0})()):
            # probe_fn returning 0 (unmeasurable) must not raise a
            # misleading space error — the master's own integrity is
            # gated elsewhere.
            check_superlong_output_space(
                master, tmp_path, total_part_duration_sec=14400.0, probe_fn=lambda p: 0.0
            )


class TestPreflightInputFiles:
    """validate_input_files checks for missing/empty/small video files."""

    def test_valid_files_pass(self, tmp_path):
        from gomoveshift.pipeline.preflight import validate_input_files

        # Create a mock video file > 100MB
        f = tmp_path / "video.mp4"
        f.touch()
        os.truncate(f, 200 * 1024 * 1024)  # sparse: 200 MB st_size, no disk
        errors = validate_input_files([f])
        assert errors == []

    def test_missing_file_returns_error(self):
        from gomoveshift.pipeline.preflight import validate_input_files

        errors = validate_input_files([Path("/nonexistent/video.mp4")])
        assert len(errors) == 1
        assert "not found" in errors[0].lower()

    def test_empty_file_returns_error(self, tmp_path):
        from gomoveshift.pipeline.preflight import validate_input_files

        f = tmp_path / "empty.mp4"
        f.write_bytes(b"")
        errors = validate_input_files([f])
        assert len(errors) == 1
        assert "empty" in errors[0].lower() or "0 bytes" in errors[0].lower()

    def test_small_file_warns_not_errors(self, tmp_path):
        """A small-but-valid clip must WARN, not hard-fail — ingest drops
        sub-minimum clips, so the run should proceed."""
        from gomoveshift.pipeline.preflight import validate_input_files

        f = tmp_path / "tiny.mp4"
        f.write_bytes(b"\x00" * 1024)  # 1 KB
        warnings: list[str] = []
        errors = validate_input_files([f], min_size_mb=100, warnings=warnings)
        assert errors == []
        assert len(warnings) == 1
        assert "small" in warnings[0].lower()

    def test_small_plus_valid_files_pass(self, tmp_path):
        """A dir with one tiny clip + valid clips returns no error."""
        from gomoveshift.pipeline.preflight import validate_input_files

        tiny = tmp_path / "tiny.mp4"
        tiny.write_bytes(b"\x00" * 1024)
        big = tmp_path / "big.mp4"
        big.touch()
        os.truncate(big, 200 * 1024 * 1024)  # sparse: 200 MB st_size, no disk
        warnings: list[str] = []
        errors = validate_input_files([tiny, big], warnings=warnings)
        assert errors == []
        assert len(warnings) == 1

    def test_empty_source_still_errors(self):
        """The no-files case stays a HARD error even after the small-file
        downgrade."""
        from gomoveshift.pipeline.preflight import validate_input_files

        errors = validate_input_files([])
        assert len(errors) == 1
        assert "no input" in errors[0].lower()

    def test_empty_file_list(self):
        from gomoveshift.pipeline.preflight import validate_input_files

        errors = validate_input_files([])
        assert len(errors) == 1
        assert "no input" in errors[0].lower()

    def test_non_mp4_extension_returns_error(self, tmp_path):
        from gomoveshift.pipeline.preflight import validate_input_files

        f = tmp_path / "video.txt"
        f.touch()
        os.truncate(f, 200 * 1024 * 1024)  # sparse: 200 MB st_size, no disk
        errors = validate_input_files([f])
        assert len(errors) == 1
        assert "extension" in errors[0].lower() or "mp4" in errors[0].lower()


class TestPreflightOSTUniqueness:
    """run_preflight_ost_check raises on duplicate episode titles."""

    def test_unique_titles_pass(self):
        from gomoveshift.pipeline.preflight import run_preflight_ost_check

        titles = ["Title A", "Title B", "Title C"]
        # Should not raise
        run_preflight_ost_check(titles)

    def test_duplicate_titles_raise(self):
        from gomoveshift.pipeline.preflight import run_preflight_ost_check

        titles = ["Title A", "Title B", "Title A"]
        with pytest.raises(RuntimeError, match="duplicate"):
            run_preflight_ost_check(titles)

    def test_duplicate_message_names_the_file_the_loader_actually_reads(self):
        """The fix-it message must name ost_overrides.json — the only filename
        ``ost_overrides.load_overrides`` reads. An operator who authors a
        ``.yaml`` file (the old, wrong text) gets a silently-ignored override
        and the run proceeds to burn duplicate titles."""
        from gomoveshift.pipeline.ost_overrides import _OVERRIDE_FILENAME
        from gomoveshift.pipeline.preflight import run_preflight_ost_check

        titles = ["Title A", "Title B", "Title A"]
        with pytest.raises(RuntimeError) as exc:
            run_preflight_ost_check(titles)
        msg = str(exc.value)
        assert _OVERRIDE_FILENAME in msg
        assert "ost_overrides.yaml" not in msg

    def test_invalid_title_raises(self):
        """Unique-but-garbage titles must fail (gate-bypass guard)."""
        from gomoveshift.pipeline.preflight import run_preflight_ost_check

        with pytest.raises(RuntimeError, match="OST validity"):
            run_preflight_ost_check(["Title A", "TBD"])

    def test_empty_titles_skipped(self):
        """GPS-less episodes legitimately carry '' — never rejected here."""
        from gomoveshift.pipeline.preflight import run_preflight_ost_check

        run_preflight_ost_check(["", "Title A", ""])


class TestRunPreflight:
    """run_preflight orchestrates all checks and raises on first failure."""

    def test_missing_source_dir_raises(self):
        from gomoveshift.pipeline.preflight import run_preflight

        with pytest.raises(RuntimeError, match="Source directory"):
            run_preflight(
                source_dir=Path("/nonexistent/dir"),
                output_dir=Path("/tmp"),
            )

    def test_valid_source_dir_with_files(self, tmp_path):
        from gomoveshift.pipeline.preflight import run_preflight

        src = tmp_path / "source"
        src.mkdir()
        f = src / "CLIP0001.MP4"
        f.touch()
        os.truncate(f, 200 * 1024 * 1024)  # sparse: 200 MB st_size, no disk

        out = tmp_path / "output"

        usage = type("Usage", (), {"free": 200 * 1_073_741_824})()
        with patch("shutil.disk_usage", return_value=usage):
            # Should not raise
            run_preflight(source_dir=src, output_dir=out)

    def test_no_video_files_raises(self, tmp_path):
        from gomoveshift.pipeline.preflight import run_preflight

        src = tmp_path / "source"
        src.mkdir()
        # Create a non-video file
        (src / "readme.txt").write_text("hello")

        out = tmp_path / "output"

        usage = type("Usage", (), {"free": 200 * 1_073_741_824})()
        with patch("shutil.disk_usage", return_value=usage):
            with pytest.raises(RuntimeError, match="(?i)no input|No .MP4"):
                run_preflight(source_dir=src, output_dir=out)


class TestRunPreflightIngestComplete:
    """A resume with Phase A already done won't re-create the per-episode
    concats, so the source×1.5 staging term overstates the real remaining peak.
    ``ingest_complete=True`` treats the full source as already-staged, dropping
    the gate to its floor (no re-ingest will occur, so the full staging term
    is not needed)."""

    _GB = 1_073_741_824

    def _run(self, tmp_path, ingest_complete):
        from unittest.mock import MagicMock

        from gomoveshift.pipeline.preflight import run_preflight

        src = tmp_path / "source"
        src.mkdir()
        f = src / "CLIP0001.MP4"
        f.touch()
        os.truncate(f, 200 * 1024 * 1024)  # sparse: 200 MB st_size, no disk
        out = tmp_path / "output"
        staging = tmp_path / "staging"
        usage = type("Usage", (), {"free": 200 * self._GB})()
        est_mock = MagicMock(return_value=1.0)
        with (
            patch("shutil.disk_usage", return_value=usage),
            patch("gomoveshift.pipeline.preflight.estimate_peak_cache_gb", est_mock),
        ):
            run_preflight(
                source_dir=src,
                output_dir=out,
                staging_dir=staging,
                ingest_complete=ingest_complete,
            )
        return est_mock, f.stat().st_size

    def test_ingest_complete_passes_full_source_as_completed(self, tmp_path):
        est_mock, size = self._run(tmp_path, ingest_complete=True)
        assert est_mock.call_args.kwargs["completed_source_bytes"] == float(size)

    def test_default_passes_zero_completed(self, tmp_path):
        est_mock, _ = self._run(tmp_path, ingest_complete=False)
        assert est_mock.call_args.kwargs["completed_source_bytes"] == 0.0


class TestEstimatePeakCacheGb:
    """``estimate_peak_cache_gb(source_files)`` sizes the intermediate cache
    requirement from raw source file sizes.

    Episode concats persist on staging through the per-day master encode, so
    the TRUE peak is total_source (concats) + the busiest day's master
    working set. The default factor (1.5) is the fail-safe single-day bound
    that agrees with the per-day mid-run gate's ``day_source × 1.5``. Floor
    enforced so a tiny test trip still reserves the OST/final workspace.
    """

    _GB = 1_073_741_824

    def test_returns_floor_for_empty_source(self, tmp_path):
        from gomoveshift.pipeline.preflight import estimate_peak_cache_gb

        est = estimate_peak_cache_gb([], floor_gb=30.0)
        assert est == 30.0

    def test_scales_with_source_size(self, tmp_path):
        from gomoveshift.pipeline.preflight import estimate_peak_cache_gb

        big = tmp_path / "big.mp4"
        big.touch()  # size comes from the stat mock below, never from disk
        # Mock stat to simulate a 100 GB source
        with patch.object(Path, "stat") as mock_stat:
            mock_stat.return_value = type(
                "S", (), {"st_size": 100 * self._GB, "st_mode": 0o100644}
            )()
            with patch.object(Path, "is_file", return_value=True):
                est = estimate_peak_cache_gb([big], safety_factor=0.5, floor_gb=0.0)
        # 100 GB × 0.5 = 50 GB
        assert 49.0 < est < 51.0

    def test_floor_applied_when_source_small(self, tmp_path):
        from gomoveshift.pipeline.preflight import estimate_peak_cache_gb

        small = tmp_path / "small.mp4"
        small.write_bytes(b"x" * 1024)  # 1 KB

        est = estimate_peak_cache_gb([small], safety_factor=0.5, floor_gb=30.0)
        assert est == 30.0  # floor kicks in

    def test_skips_missing_files(self, tmp_path):
        from gomoveshift.pipeline.preflight import estimate_peak_cache_gb

        ghost = tmp_path / "does_not_exist.mp4"
        est = estimate_peak_cache_gb([ghost], floor_gb=0.0)
        assert est == 0.0

    def test_default_covers_per_day_peak(self, tmp_path):
        """The default factor must size the TRUE peak (≥ the per-day gate's
        ``day_source × 1.5``), so preflight catches a shortfall the old
        ``× 1.2`` under-estimated. Worst case (single-day trip): the whole
        source IS the busiest day, so peak == total × 1.5."""
        from gomoveshift.pipeline.preflight import estimate_peak_cache_gb

        big = tmp_path / "big.mp4"
        big.write_bytes(b"x")
        with (
            patch.object(
                Path,
                "stat",
                return_value=type("S", (), {"st_size": 123 * self._GB, "st_mode": 0o100644})(),
            ),
            patch.object(Path, "is_file", return_value=True),
        ):
            est = estimate_peak_cache_gb([big], floor_gb=0.0)
        # 123 GB source × 1.5 = 184.5 GB — covers the per-day ``123 × 1.5``
        # spike that can fire a mid-run low-space failure.
        assert 184.0 < est < 185.0
        # The old ``× 1.2`` factor asks for only 147.6 GB, less than the true
        # 184.5 GB peak.
        assert est > 123 * 1.2

    def test_completed_source_bytes_subtracted_on_resume(self, tmp_path):
        """A resume with completed-day masters subtracts those days' raw
        source bytes BEFORE the safety factor, so the gate no longer demands
        the full ``source × 1.5`` for work already done."""
        from gomoveshift.pipeline.preflight import estimate_peak_cache_gb

        big = tmp_path / "big.mp4"
        big.write_bytes(b"x")
        with (
            patch.object(
                Path,
                "stat",
                return_value=type("S", (), {"st_size": 100 * self._GB, "st_mode": 0o100644})(),
            ),
            patch.object(Path, "is_file", return_value=True),
        ):
            # 60 GB of the 100 GB source is already-encoded days → 40 GB remains.
            est = estimate_peak_cache_gb([big], floor_gb=0.0, completed_source_bytes=60 * self._GB)
        # (100 − 60) GB × 1.5 = 60 GB, not the fresh 150 GB.
        assert 59.0 < est < 61.0

    def test_completed_source_bytes_clamped_not_negative(self, tmp_path):
        """An over-count (stale record summing to more than the live source)
        clamps remaining at 0 → the floor applies, never a negative estimate."""
        from gomoveshift.pipeline.preflight import estimate_peak_cache_gb

        big = tmp_path / "big.mp4"
        big.write_bytes(b"x")
        with (
            patch.object(
                Path,
                "stat",
                return_value=type("S", (), {"st_size": 50 * self._GB, "st_mode": 0o100644})(),
            ),
            patch.object(Path, "is_file", return_value=True),
        ):
            est = estimate_peak_cache_gb(
                [big], floor_gb=30.0, completed_source_bytes=999 * self._GB
            )
        assert est == 30.0  # clamped to 0 remaining → floor


class TestPreflightCacheSpaceCheck:
    """run_preflight must verify the cache/staging volume has enough free
    space for the estimated peak usage — not a static threshold. Catches
    the failure mode where free space passes a static check but the estimated
    peak does not fit (fails mid-encode)."""

    _GB = 1_073_741_824

    def _make_sources(self, src: Path, total_gb: int, n: int = 3):
        """Create n fake source files summing to roughly total_gb."""
        src.mkdir(exist_ok=True)
        per_file = (total_gb * self._GB) // n
        for i in range(n):
            f = src / f"CLIP{i:04d}.MP4"
            # Write a tiny stub — the test mocks os.stat to report big sizes
            f.write_bytes(b"x")
        return sorted(src.glob("CLIP*.MP4"))

    def test_aborts_when_cache_too_small_for_estimate(self, tmp_path):
        from gomoveshift.pipeline.preflight import run_preflight

        src = tmp_path / "src"
        self._make_sources(src, total_gb=200)
        out = tmp_path / "out"
        staging = tmp_path / "staging"

        # Simulate: 200 GB source → ~100 GB peak. Staging has only 40 GB free.
        # Output has plenty.
        def fake_disk_usage(path):
            path_str = str(path)
            if "staging" in path_str or path_str == str(staging):
                return type("U", (), {"free": 40 * self._GB})()
            return type("U", (), {"free": 500 * self._GB})()

        def fake_stat(self_path, *, follow_symlinks=True):
            if str(self_path).endswith(".MP4"):
                return type(
                    "S",
                    (),
                    {"st_size": 200 * self._GB // 3, "st_mode": 0o100644},
                )()
            from os import stat as os_stat

            return os_stat(str(self_path))

        with (
            patch("shutil.disk_usage", side_effect=fake_disk_usage),
            patch.object(Path, "stat", fake_stat),
        ):
            with pytest.raises(RuntimeError, match="(?i)peak|cache|intermediate"):
                run_preflight(
                    source_dir=src,
                    output_dir=out,
                    staging_dir=staging,
                    require_tools=None,
                )

    def test_passes_when_cache_has_headroom(self, tmp_path):
        from gomoveshift.pipeline.preflight import run_preflight

        src = tmp_path / "src"
        self._make_sources(src, total_gb=50)
        out = tmp_path / "out"
        staging = tmp_path / "staging"

        def fake_disk_usage(path):
            return type("U", (), {"free": 500 * self._GB})()  # plenty everywhere

        def fake_stat(self_path, *, follow_symlinks=True):
            if str(self_path).endswith(".MP4"):
                return type(
                    "S",
                    (),
                    {"st_size": 50 * self._GB // 3, "st_mode": 0o100644},
                )()
            from os import stat as os_stat

            return os_stat(str(self_path))

        with (
            patch("shutil.disk_usage", side_effect=fake_disk_usage),
            patch.object(Path, "stat", fake_stat),
        ):
            # Should not raise
            run_preflight(
                source_dir=src,
                output_dir=out,
                staging_dir=staging,
                require_tools=None,
            )

    def test_error_message_includes_estimate_and_free(self, tmp_path):
        """Error message must tell operator how much is needed + free +
        suggest GMS_INTERMEDIATE_DIR as the escape hatch."""
        from gomoveshift.pipeline.preflight import run_preflight

        src = tmp_path / "src"
        self._make_sources(src, total_gb=200)
        out = tmp_path / "out"
        staging = tmp_path / "staging"

        def fake_disk_usage(path):
            if "staging" in str(path):
                return type("U", (), {"free": 20 * self._GB})()
            return type("U", (), {"free": 500 * self._GB})()

        def fake_stat(self_path, *, follow_symlinks=True):
            if str(self_path).endswith(".MP4"):
                return type(
                    "S",
                    (),
                    {"st_size": 200 * self._GB // 3, "st_mode": 0o100644},
                )()
            from os import stat as os_stat

            return os_stat(str(self_path))

        with (
            patch("shutil.disk_usage", side_effect=fake_disk_usage),
            patch.object(Path, "stat", fake_stat),
        ):
            with pytest.raises(RuntimeError) as exc_info:
                run_preflight(
                    source_dir=src,
                    output_dir=out,
                    staging_dir=staging,
                    require_tools=None,
                )
        msg = str(exc_info.value)
        assert "GB" in msg
        assert "GMS_INTERMEDIATE_DIR" in msg


class TestPreflightDriveCredentials:
    """``check_drive_credentials``
    surfaces missing SOPS-encrypted credential files at preflight time so
    operators learn about the misconfiguration in seconds instead of after
    several hours of encode work."""

    def test_missing_credentials_raises(self, tmp_path, monkeypatch):
        """When ``GMS_DRIVE_CREDENTIALS_ENC`` points to a nonexistent path,
        ``check_drive_credentials`` raises ``RuntimeError`` with a message
        naming the env var + the resolved path."""
        from gomoveshift.pipeline.preflight import check_drive_credentials

        bogus = tmp_path / "does_not_exist.enc"
        # Point all four env vars at non-existent files so we know exactly
        # which one trips the check first (any of them is sufficient).
        monkeypatch.setenv("GMS_DRIVE_CREDENTIALS_ENC", str(bogus))
        monkeypatch.setenv("GMS_DRIVE_TOKEN_ENC", str(bogus))
        monkeypatch.setenv("GMS_AGE_KEY_FILE", str(bogus))
        monkeypatch.setenv("GMS_SOPS_CONFIG", str(bogus))

        with pytest.raises(RuntimeError, match="(?i)credential"):
            check_drive_credentials()

    def test_missing_token_raises(self, tmp_path, monkeypatch):
        """Each env var is checked independently — token-only miss raises."""
        from gomoveshift.pipeline.preflight import check_drive_credentials

        # Make creds + age + sops point to real files; only token is bogus.
        real_creds = tmp_path / "creds.enc"
        real_creds.write_text("")
        real_age = tmp_path / "age.key"
        real_age.write_text("")
        real_sops = tmp_path / ".sops.yaml"
        real_sops.write_text("")
        bogus_token = tmp_path / "does_not_exist_token.enc"

        monkeypatch.setenv("GMS_DRIVE_CREDENTIALS_ENC", str(real_creds))
        monkeypatch.setenv("GMS_DRIVE_TOKEN_ENC", str(bogus_token))
        monkeypatch.setenv("GMS_AGE_KEY_FILE", str(real_age))
        monkeypatch.setenv("GMS_SOPS_CONFIG", str(real_sops))

        with pytest.raises(RuntimeError, match="GMS_DRIVE_TOKEN_ENC"):
            check_drive_credentials()

    def test_empty_string_disables_check(self, tmp_path, monkeypatch):
        """An empty env value skips the existence check for that variable (the
        uploader still falls back to the default file); it does NOT raise even
        if the defaulted paths don't exist."""
        from gomoveshift.pipeline.preflight import check_drive_credentials

        # All four env vars set to "" → every check is skipped → no raise.
        monkeypatch.setenv("GMS_DRIVE_CREDENTIALS_ENC", "")
        monkeypatch.setenv("GMS_DRIVE_TOKEN_ENC", "")
        monkeypatch.setenv("GMS_AGE_KEY_FILE", "")
        monkeypatch.setenv("GMS_SOPS_CONFIG", "")

        # Should not raise.
        check_drive_credentials()

    def test_tilde_override_is_expanded(self, tmp_path, monkeypatch):
        """``~`` in an override is expanded and surrounding spaces are stripped."""
        from gomoveshift.pipeline.preflight import check_drive_credentials

        monkeypatch.setenv("HOME", str(tmp_path))
        for env_var, name in (
            ("GMS_DRIVE_CREDENTIALS_ENC", "creds.enc"),
            ("GMS_DRIVE_TOKEN_ENC", "token.enc"),
            ("GMS_AGE_KEY_FILE", "age.key"),
            ("GMS_SOPS_CONFIG", ".sops.yaml"),
        ):
            (tmp_path / name).write_text("")
            value = f"~/{name}"
            if name == "age.key":
                value = f"  {value}  "
            monkeypatch.setenv(env_var, value)

        check_drive_credentials()

    def test_whitespace_only_override_falls_back_to_default(self, tmp_path, monkeypatch):
        """A whitespace-only override is unset: the config-home default is checked."""
        from gomoveshift.paths import gms_secret_path
        from gomoveshift.pipeline.preflight import check_drive_credentials

        monkeypatch.setenv("GMS_CONFIG_HOME", str(tmp_path))
        names = (
            ("GMS_DRIVE_CREDENTIALS_ENC", "drive_credentials.json.enc"),
            ("GMS_DRIVE_TOKEN_ENC", "drive_token.json.enc"),
            ("GMS_AGE_KEY_FILE", "age.key"),
            ("GMS_SOPS_CONFIG", ".sops.yaml"),
        )
        for env_var, _ in names:
            monkeypatch.setenv(env_var, "   ")

        with pytest.raises(RuntimeError) as exc_info:
            check_drive_credentials()
        for env_var, filename in names:
            assert str(gms_secret_path(env_var, filename)) in str(exc_info.value)

    def test_all_present_passes(self, tmp_path, monkeypatch):
        """When every env-resolved path exists, the check passes silently."""
        from gomoveshift.pipeline.preflight import check_drive_credentials

        for env_var, name in (
            ("GMS_DRIVE_CREDENTIALS_ENC", "creds.enc"),
            ("GMS_DRIVE_TOKEN_ENC", "token.enc"),
            ("GMS_AGE_KEY_FILE", "age.key"),
            ("GMS_SOPS_CONFIG", ".sops.yaml"),
        ):
            p = tmp_path / name
            p.write_text("")
            monkeypatch.setenv(env_var, str(p))

        # Should not raise.
        check_drive_credentials()

    def test_run_preflight_check_credentials_flag_default_false(self, tmp_path, monkeypatch):
        """Default ``check_credentials=False`` keeps existing test entry
        points working without touching the env vars."""
        from gomoveshift.pipeline.preflight import run_preflight

        # Point creds at bogus paths so the check WOULD fail if it ran.
        monkeypatch.setenv("GMS_DRIVE_CREDENTIALS_ENC", "/nonexistent/creds.enc")

        src = tmp_path / "source"
        src.mkdir()
        f = src / "CLIP0001.MP4"
        f.touch()
        os.truncate(f, 200 * 1024 * 1024)  # sparse: 200 MB st_size, no disk
        out = tmp_path / "output"

        usage = type("Usage", (), {"free": 200 * 1_073_741_824})()
        with patch("shutil.disk_usage", return_value=usage):
            # check_credentials=False (default) → no raise.
            run_preflight(source_dir=src, output_dir=out)

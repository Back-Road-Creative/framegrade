"""Tests for ``gomoveshift enqueue-plan`` (cli/handlers/enqueue_plan.py).

The properties under test: shorts spool automatically through the automated
pre-spool gates (no human answer required), and a plan a human
DID confirm is refused once it changes after that confirmation.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from gomoveshift.cli.handlers.enqueue_plan import (
    GateNotApprovedError,
    assert_gate_cd_approved,
    cmd_enqueue_plan,
)
from gomoveshift.pipeline.drive_phases.phase_d import _serialize_publish_plan
from gomoveshift.publishing.models import (
    MediaFingerprint,
    PlatformPayloads,
    PublishPlan,
    RouteSelection,
    ScheduleWindow,
    ShortPublishRecord,
    YouTubePayload,
)
from gomoveshift.pipeline.encoding_constants import PAYLOAD_LIMITS
from gomoveshift.publishing.queue import cancel_delivery, enqueue, open_queue_db, transition

SCHEDULED = datetime(2026, 9, 1, 19, 0, tzinfo=timezone.utc)

# A real file on disk whose fingerprint fields genuinely match, so the
# happy-path tests below exercise the same enqueue-time media verification
# a real run goes through rather than being exempted from it.
_MEDIA_PATH = Path(tempfile.mkdtemp(prefix="gms-enqueue-plan-test-media-")) / "short.mp4"
_MEDIA_PATH.write_bytes(b"gms enqueue-plan test media fixture bytes")
_REAL_MEDIA = MediaFingerprint(
    str(_MEDIA_PATH),
    _MEDIA_PATH.stat().st_size,
    2.0,
    1080,
    1920,
    "h264",
    hashlib.sha256(_MEDIA_PATH.read_bytes()).hexdigest(),
)


def _record(short_index=3, selected=("youtube",), media=None, payloads=None):
    return ShortPublishRecord(
        project_name="proj",
        lf_index=0,
        short_index=short_index,
        media=media or _REAL_MEDIA,
        schedule=ScheduleWindow(SCHEDULED, "America/Chicago"),
        recording_date="2026-08-01",
        sampled_regions=("a",),
        parent_lf_video_id="vid",
        parent_lf_publish_at=SCHEDULED,
        payloads=payloads or PlatformPayloads(youtube=YouTubePayload("t", "d")),
        consent_hash="c",
        reviewed_plan_hash="HASH1",
        routes=RouteSelection(selected=tuple(selected)),
    )


def _sidecar(out_dir, *records):
    path = out_dir / "proj_publish_plan.json"
    path.write_text(_serialize_publish_plan(PublishPlan(shorts=tuple(records))), encoding="utf-8")
    return path


def _state(**over):
    fields = {
        "regular_lf_paths": [],
        "gate_cd_confirmed": True,
        "gate_cd_confirmation_hash": None,
    }
    fields.update(over)
    return SimpleNamespace(**fields)


def _config(tmp_path, ceiling=None):
    seo_kwargs = {"storage_path": str(tmp_path)}
    if ceiling is not None:
        seo_kwargs["rolling_week_ceiling_per_platform"] = ceiling
    return SimpleNamespace(seo=SimpleNamespace(**seo_kwargs))


def _args(state_path, **over):
    return SimpleNamespace(
        **{
            "state_path": str(state_path),
            "apply": False,
            "output_dir": None,
            "shadow": False,
            **over,
        }
    )


def _approved_state(out_dir):
    """A state whose recorded hash matches the sidecar actually on disk."""
    from gomoveshift.pipeline.review_gate import compute_gate_cd_hash

    state = _state()
    state.gate_cd_confirmation_hash = compute_gate_cd_hash(state=state, output_dir=out_dir)
    return state


def _approved_state_for(out_dir):
    """Like ``_approved_state`` but also carries ``output_base`` — mimics a
    real ``PipelineState`` where Phase D recorded the project directory."""
    from gomoveshift.pipeline.review_gate import compute_gate_cd_hash

    state = _state(output_base=str(out_dir))
    state.gate_cd_confirmation_hash = compute_gate_cd_hash(state=state, output_dir=out_dir)
    return state


# --- the gate property ------------------------------------------------------


def test_unreviewed_run_is_not_refused(tmp_path):
    """Shorts publish automatically, with no human review gate. A --no-review run never records a human Gate C/D
    answer; that absence must not strand its shorts. The automated
    pre-spool gates in cmd_enqueue_plan decide instead."""
    _sidecar(tmp_path, _record())
    assert_gate_cd_approved(_state(gate_cd_confirmed=False), tmp_path)  # must not raise
    assert_gate_cd_approved(_state(gate_cd_confirmed=None), tmp_path)  # must not raise


def test_confirmed_without_a_hash_is_refused(tmp_path):
    """A confirmation not tied to content proves nothing about what would
    actually be published."""
    _sidecar(tmp_path, _record())
    with pytest.raises(GateNotApprovedError) as exc:
        assert_gate_cd_approved(_state(gate_cd_confirmation_hash=None), tmp_path)
    assert "no confirmation hash" in str(exc.value)


def test_approved_and_unchanged_passes(tmp_path):
    _sidecar(tmp_path, _record())
    assert_gate_cd_approved(_approved_state(tmp_path), tmp_path)  # must not raise


def test_plan_edited_after_approval_is_refused(tmp_path):
    """The core protection: approve, then change the plan, and the approval
    no longer applies to what would be published."""
    _sidecar(tmp_path, _record(short_index=3))
    state = _approved_state(tmp_path)
    _sidecar(tmp_path, _record(short_index=3), _record(short_index=4))  # plan changed
    with pytest.raises(GateNotApprovedError) as exc:
        assert_gate_cd_approved(state, tmp_path)
    assert "CHANGED since Gate C/D was confirmed" in str(exc.value)


# --- command behaviour ------------------------------------------------------


def test_dry_run_writes_nothing_to_the_queue(tmp_path, capsys):
    out = tmp_path / "out"
    out.mkdir()
    _sidecar(out, _record())
    state_path = out / "pipeline_state.json"
    state_path.write_text("{}", encoding="utf-8")

    with (
        mock.patch(
            "gomoveshift.cli.handlers.enqueue_plan._load_state",
            return_value=_approved_state(out),
        ),
        mock.patch("gomoveshift.cli.handlers.enqueue_plan.enqueue_from_plan_sidecar") as enq,
    ):
        rc = cmd_enqueue_plan(_args(state_path), _config(tmp_path))

    assert rc == 0
    enq.assert_not_called()
    assert "DRY RUN" in capsys.readouterr().out


def test_apply_enqueues_and_is_idempotent(tmp_path, capsys):
    out = tmp_path / "out"
    out.mkdir()
    _sidecar(out, _record(selected=("youtube", "tiktok")))
    state_path = out / "pipeline_state.json"
    state_path.write_text("{}", encoding="utf-8")

    with mock.patch(
        "gomoveshift.cli.handlers.enqueue_plan._load_state",
        return_value=_approved_state(out),
    ):
        assert cmd_enqueue_plan(_args(state_path, apply=True), _config(tmp_path)) == 0
        first = capsys.readouterr().out
        assert cmd_enqueue_plan(_args(state_path, apply=True), _config(tmp_path)) == 0
        second = capsys.readouterr().out

    assert "enqueued=2 already_present=0" in first
    assert "enqueued=0 already_present=2" in second

    conn = open_queue_db(tmp_path)
    rows = sorted(r["platform"] for r in conn.execute("SELECT platform FROM deliveries"))
    conn.close()
    assert rows == ["tiktok", "youtube"]


def test_unreviewed_run_enqueues_through_the_automated_gates(tmp_path, capsys):
    out = tmp_path / "out"
    out.mkdir()
    _sidecar(out, _record())
    state_path = out / "pipeline_state.json"
    state_path.write_text("{}", encoding="utf-8")

    with mock.patch(
        "gomoveshift.cli.handlers.enqueue_plan._load_state",
        return_value=_state(gate_cd_confirmed=None),
    ):
        rc = cmd_enqueue_plan(_args(state_path, apply=True), _config(tmp_path))

    assert rc == 0
    assert "automated gates" in capsys.readouterr().out
    conn = open_queue_db(tmp_path)
    count = conn.execute("SELECT COUNT(*) FROM deliveries").fetchone()[0]
    conn.close()
    assert count == 1


def test_no_sidecar_is_success_not_an_error(tmp_path, capsys):
    """A run with every route on metricool writes no sidecar at all — the
    normal state today."""
    out = tmp_path / "out"
    out.mkdir()
    state_path = out / "pipeline_state.json"
    state_path.write_text("{}", encoding="utf-8")

    with mock.patch(
        "gomoveshift.cli.handlers.enqueue_plan._load_state", return_value=_approved_state(out)
    ):
        rc = cmd_enqueue_plan(_args(state_path, apply=True), _config(tmp_path))

    assert rc == 0
    assert "nothing to enqueue" in capsys.readouterr().out


def test_unreadable_state_is_reported_not_crashed(tmp_path):
    state_path = tmp_path / "missing.json"
    rc = cmd_enqueue_plan(_args(state_path), _config(tmp_path))
    assert rc == 1


def test_apply_spools_live_ammunition_by_default(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    _sidecar(out, _record(selected=("youtube",)))
    state_path = out / "pipeline_state.json"
    state_path.write_text("{}", encoding="utf-8")

    with mock.patch(
        "gomoveshift.cli.handlers.enqueue_plan._load_state",
        return_value=_approved_state(out),
    ):
        assert cmd_enqueue_plan(_args(state_path, apply=True), _config(tmp_path)) == 0

    conn = open_queue_db(tmp_path)
    rows = [r["shadow"] for r in conn.execute("SELECT shadow FROM deliveries")]
    conn.close()
    assert rows == [0]


def test_shadow_apply_spools_rows_no_drain_can_publish(tmp_path, capsys):
    # A shadow run: prove the whole chain without leaving anything the
    # cron could fire. Without this the operator's only protection was
    # remembering not to install the crontab.
    out = tmp_path / "out"
    out.mkdir()
    _sidecar(out, _record(selected=("youtube",)))
    state_path = out / "pipeline_state.json"
    state_path.write_text("{}", encoding="utf-8")

    with mock.patch(
        "gomoveshift.cli.handlers.enqueue_plan._load_state",
        return_value=_approved_state(out),
    ):
        rc = cmd_enqueue_plan(_args(state_path, apply=True, shadow=True), _config(tmp_path))

    assert rc == 0
    assert "SHADOW" in capsys.readouterr().out
    conn = open_queue_db(tmp_path)
    rows = [r["shadow"] for r in conn.execute("SELECT shadow FROM deliveries")]
    conn.close()
    assert rows == [1]


def test_shadow_without_apply_still_writes_nothing(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    _sidecar(out, _record(selected=("youtube",)))
    state_path = out / "pipeline_state.json"
    state_path.write_text("{}", encoding="utf-8")

    with mock.patch(
        "gomoveshift.cli.handlers.enqueue_plan._load_state",
        return_value=_approved_state(out),
    ):
        assert cmd_enqueue_plan(_args(state_path, shadow=True), _config(tmp_path)) == 0

    conn = open_queue_db(tmp_path)
    count = conn.execute("SELECT COUNT(*) AS n FROM deliveries").fetchone()["n"]
    conn.close()
    assert count == 0


# --- default output-dir resolution ---------------------------


def test_default_output_dir_resolves_from_state_output_base(tmp_path, capsys):
    """Phase D writes sidecars into the project dir (``state.output_base``),
    not next to the run's ``pipeline_state.json`` — which can live under a
    relocated ``_stages`` dir entirely. The default must follow the state
    field, not the state file's own location."""
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    _sidecar(project_dir, _record())
    stages_dir = tmp_path / "_stages"
    stages_dir.mkdir()
    state_path = stages_dir / "pipeline_state.json"
    state_path.write_text("{}", encoding="utf-8")

    state = _approved_state_for(project_dir)
    with mock.patch("gomoveshift.cli.handlers.enqueue_plan._load_state", return_value=state):
        rc = cmd_enqueue_plan(_args(state_path), _config(tmp_path))

    assert rc == 0
    assert "Sidecars: 1" in capsys.readouterr().out


def test_default_output_dir_falls_back_from_stages_dir(tmp_path, capsys):
    """A legacy state with no ``output_base`` field: the ``_stages``-parent
    heuristic still finds the sidecar one directory up."""
    project_dir = tmp_path
    _sidecar(project_dir, _record())
    stages_dir = project_dir / "_stages"
    stages_dir.mkdir()
    state_path = stages_dir / "pipeline_state.json"
    state_path.write_text("{}", encoding="utf-8")

    state = _approved_state(project_dir)
    with mock.patch("gomoveshift.cli.handlers.enqueue_plan._load_state", return_value=state):
        rc = cmd_enqueue_plan(_args(state_path), _config(tmp_path))

    assert rc == 0
    assert "Sidecars: 1" in capsys.readouterr().out


def test_explicit_output_dir_overrides_state_output_base(tmp_path, capsys):
    """An explicit ``--output-dir`` always wins, even over a state-recorded
    ``output_base`` pointing somewhere else."""
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    _sidecar(real_dir, _record())
    wrong_dir = tmp_path / "wrong"
    wrong_dir.mkdir()
    state_path = tmp_path / "pipeline_state.json"
    state_path.write_text("{}", encoding="utf-8")

    state = _approved_state_for(real_dir)
    state.output_base = str(wrong_dir)
    with mock.patch("gomoveshift.cli.handlers.enqueue_plan._load_state", return_value=state):
        rc = cmd_enqueue_plan(_args(state_path, output_dir=str(real_dir)), _config(tmp_path))

    assert rc == 0
    assert "Sidecars: 1" in capsys.readouterr().out


def test_no_sidecar_message_names_directory_and_hint(tmp_path, capsys):
    """The all-metricool no-op is legitimate (rc 0) but the message must
    name the directory searched and how to point elsewhere."""
    out = tmp_path / "out"
    out.mkdir()
    state_path = tmp_path / "pipeline_state.json"
    state_path.write_text("{}", encoding="utf-8")

    state = _approved_state_for(out)
    with mock.patch("gomoveshift.cli.handlers.enqueue_plan._load_state", return_value=state):
        rc = cmd_enqueue_plan(_args(state_path), _config(tmp_path))

    assert rc == 0
    msg = capsys.readouterr().out
    assert str(out) in msg
    assert "--output-dir" in msg


# --- media verification -------------------------------------------


def test_apply_refuses_when_media_missing(tmp_path, capsys):
    """A delivery can outlive its own media (unmounted drive, deleted
    short) — enqueue-plan must refuse it at the terminal, not spool it for
    a cron to burn days later."""
    out = tmp_path / "out"
    out.mkdir()
    missing = tmp_path / "gone.mp4"
    bad_media = MediaFingerprint(str(missing), 1, 2.0, 1080, 1920, "h264", "b" * 64)
    _sidecar(out, _record(media=bad_media))
    state_path = out / "pipeline_state.json"
    state_path.write_text("{}", encoding="utf-8")

    with mock.patch(
        "gomoveshift.cli.handlers.enqueue_plan._load_state",
        return_value=_approved_state(out),
    ):
        rc = cmd_enqueue_plan(_args(state_path, apply=True), _config(tmp_path))

    assert rc == 1
    assert "gone.mp4" in capsys.readouterr().out
    conn = open_queue_db(tmp_path)
    count = conn.execute("SELECT COUNT(*) AS n FROM deliveries").fetchone()["n"]
    conn.close()
    assert count == 0


def test_apply_refuses_when_media_fingerprint_mismatch(tmp_path, capsys):
    """The file exists but no longer matches what Gate C/D reviewed — a
    swapped or truncated file is just as unpublishable as a missing one."""
    out = tmp_path / "out"
    out.mkdir()
    real_path = tmp_path / "swapped.mp4"
    real_path.write_bytes(b"different bytes than what was reviewed")
    stale_media = MediaFingerprint(
        str(real_path), real_path.stat().st_size, 2.0, 1080, 1920, "h264", "f" * 64
    )
    _sidecar(out, _record(media=stale_media))
    state_path = out / "pipeline_state.json"
    state_path.write_text("{}", encoding="utf-8")

    with mock.patch(
        "gomoveshift.cli.handlers.enqueue_plan._load_state",
        return_value=_approved_state(out),
    ):
        rc = cmd_enqueue_plan(_args(state_path, apply=True), _config(tmp_path))

    assert rc == 1
    assert "sha256 mismatch" in capsys.readouterr().out
    conn = open_queue_db(tmp_path)
    count = conn.execute("SELECT COUNT(*) AS n FROM deliveries").fetchone()["n"]
    conn.close()
    assert count == 0


# --- payload limits ----------------------------------------------------


def test_apply_refuses_when_a_payload_field_exceeds_its_limit(tmp_path, capsys):
    """A short whose YouTube title exceeds the platform's documented limit
    must never reach the queue -- refused at enqueue, not a 400 from the
    platform after everything else already passed."""
    out = tmp_path / "out"
    out.mkdir()
    limit = PAYLOAD_LIMITS["youtube"].title_max_len
    over_limit = PlatformPayloads(youtube=YouTubePayload(title="x" * (limit + 1), description="d"))
    _sidecar(out, _record(payloads=over_limit))
    state_path = out / "pipeline_state.json"
    state_path.write_text("{}", encoding="utf-8")

    with mock.patch(
        "gomoveshift.cli.handlers.enqueue_plan._load_state",
        return_value=_approved_state(out),
    ):
        rc = cmd_enqueue_plan(_args(state_path, apply=True), _config(tmp_path))

    out_text = capsys.readouterr().out
    assert rc == 1
    assert "youtube title" in out_text
    assert str(limit) in out_text
    conn = open_queue_db(tmp_path)
    count = conn.execute("SELECT COUNT(*) AS n FROM deliveries").fetchone()["n"]
    conn.close()
    assert count == 0


def test_apply_spools_payload_at_the_limit(tmp_path):
    """The boundary is inclusive: exactly at the limit is publishable."""
    out = tmp_path / "out"
    out.mkdir()
    limit = PAYLOAD_LIMITS["youtube"].title_max_len
    at_limit = PlatformPayloads(youtube=YouTubePayload(title="x" * limit, description="d"))
    _sidecar(out, _record(payloads=at_limit))
    state_path = out / "pipeline_state.json"
    state_path.write_text("{}", encoding="utf-8")

    with mock.patch(
        "gomoveshift.cli.handlers.enqueue_plan._load_state",
        return_value=_approved_state(out),
    ):
        rc = cmd_enqueue_plan(_args(state_path, apply=True), _config(tmp_path))

    assert rc == 0
    conn = open_queue_db(tmp_path)
    count = conn.execute("SELECT COUNT(*) AS n FROM deliveries").fetchone()["n"]
    conn.close()
    assert count == 1


# --- volume cap (--max-shorts) ---------------------------------------------
#
# A scale-back is a VOLUME decision, not a content edit. Trimming the
# reviewed sidecar on disk to publish fewer shorts flips the Gate C/D
# content hash and is refused -- correctly, because an edited plan is
# indistinguishable from a tampered one. The cap therefore applies in
# memory, AFTER the hash of the untouched approved plan verifies: every
# spooled short is one a human reviewed, there are just fewer of them.


def test_max_shorts_spools_only_the_lowest_indexes(tmp_path, capsys):
    out = tmp_path / "out"
    out.mkdir()
    _sidecar(out, _record(short_index=1), _record(short_index=2), _record(short_index=3))
    state_path = out / "pipeline_state.json"
    state_path.write_text("{}", encoding="utf-8")

    with mock.patch(
        "gomoveshift.cli.handlers.enqueue_plan._load_state",
        return_value=_approved_state(out),
    ):
        rc = cmd_enqueue_plan(_args(state_path, apply=True, max_shorts=2), _config(tmp_path))

    assert rc == 0
    conn = open_queue_db(tmp_path)
    rows = sorted(r["short_index"] for r in conn.execute("SELECT short_index FROM deliveries"))
    conn.close()
    assert rows == [1, 2]
    assert "deferred=1" in capsys.readouterr().out


def test_max_shorts_leaves_the_approved_plan_untouched_on_disk(tmp_path):
    """The gate must still pass on a later run: capping never rewrites the
    reviewed artefact."""
    out = tmp_path / "out"
    out.mkdir()
    path = _sidecar(out, _record(short_index=1), _record(short_index=2))
    before = path.read_bytes()
    state_path = out / "pipeline_state.json"
    state_path.write_text("{}", encoding="utf-8")
    state = _approved_state(out)

    with mock.patch("gomoveshift.cli.handlers.enqueue_plan._load_state", return_value=state):
        assert cmd_enqueue_plan(_args(state_path, apply=True, max_shorts=1), _config(tmp_path)) == 0

    assert path.read_bytes() == before
    assert_gate_cd_approved(state, out)  # must not raise


def test_max_shorts_does_not_verify_media_of_deferred_shorts(tmp_path):
    """A short past the cap is never published, so its media is not a
    reason to refuse the whole run."""
    out = tmp_path / "out"
    out.mkdir()
    gone = MediaFingerprint(str(out / "absent.mp4"), 10, 2.0, 1080, 1920, "h264", "0" * 64)
    _sidecar(out, _record(short_index=1), _record(short_index=2, media=gone))
    state_path = out / "pipeline_state.json"
    state_path.write_text("{}", encoding="utf-8")

    with mock.patch(
        "gomoveshift.cli.handlers.enqueue_plan._load_state",
        return_value=_approved_state(out),
    ):
        assert cmd_enqueue_plan(_args(state_path, apply=True, max_shorts=1), _config(tmp_path)) == 0
        # ...and without the cap the same missing media IS a refusal.
        assert cmd_enqueue_plan(_args(state_path, apply=True), _config(tmp_path)) == 1


def test_max_shorts_above_the_plan_size_is_a_no_op(tmp_path, capsys):
    out = tmp_path / "out"
    out.mkdir()
    _sidecar(out, _record(short_index=1))
    state_path = out / "pipeline_state.json"
    state_path.write_text("{}", encoding="utf-8")

    with mock.patch(
        "gomoveshift.cli.handlers.enqueue_plan._load_state",
        return_value=_approved_state(out),
    ):
        assert (
            cmd_enqueue_plan(_args(state_path, apply=True, max_shorts=99), _config(tmp_path)) == 0
        )

    assert "deferred" not in capsys.readouterr().out


# --- rolling 7-day per-platform ceiling --------------------------------
#
# The cross-drive case a per-drive gate cannot see: a second region's plan
# enqueuing into the same rolling week as a first would double a platform's
# weekly volume. This is the only place the cross-drive total is knowable --
# so it is counted here, against the queue, not per-drive.


def test_rolling_ceiling_refuses_whole_plan_when_platform_would_exceed(tmp_path, capsys):
    out = tmp_path / "out"
    out.mkdir()
    _sidecar(out, _record(short_index=5, selected=("youtube",)))
    state_path = out / "pipeline_state.json"
    state_path.write_text("{}", encoding="utf-8")

    conn = open_queue_db(tmp_path)
    for i in range(2):
        enqueue(conn, "other-proj", i, "youtube", f"hash{i}", scheduled_at=SCHEDULED.isoformat())
    conn.close()

    with (
        mock.patch(
            "gomoveshift.cli.handlers.enqueue_plan._load_state",
            return_value=_approved_state(out),
        ),
        mock.patch(
            "gomoveshift.cli.handlers.enqueue_plan.current_brand",
            return_value="no-such-brand",
        ),
    ):
        rc = cmd_enqueue_plan(_args(state_path, apply=True), _config(tmp_path, ceiling=2))

    assert rc == 1
    out_text = capsys.readouterr().out
    assert "youtube" in out_text
    assert "ceiling=2" in out_text
    assert "proj#5" in out_text

    conn = open_queue_db(tmp_path)
    count = conn.execute(
        "SELECT COUNT(*) AS n FROM deliveries WHERE project_name='proj'"
    ).fetchone()["n"]
    conn.close()
    assert count == 0


def test_rolling_ceiling_allows_when_under(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    _sidecar(out, _record(short_index=6, selected=("youtube",)))
    state_path = out / "pipeline_state.json"
    state_path.write_text("{}", encoding="utf-8")

    conn = open_queue_db(tmp_path)
    enqueue(conn, "other-proj", 0, "youtube", "hash0", scheduled_at=SCHEDULED.isoformat())
    conn.close()

    with (
        mock.patch(
            "gomoveshift.cli.handlers.enqueue_plan._load_state",
            return_value=_approved_state(out),
        ),
        mock.patch(
            "gomoveshift.cli.handlers.enqueue_plan.current_brand",
            return_value="no-such-brand",
        ),
    ):
        rc = cmd_enqueue_plan(_args(state_path, apply=True), _config(tmp_path, ceiling=2))

    assert rc == 0
    conn = open_queue_db(tmp_path)
    count = conn.execute(
        "SELECT COUNT(*) AS n FROM deliveries WHERE project_name='proj'"
    ).fetchone()["n"]
    conn.close()
    assert count == 1


def test_rolling_ceiling_default_calibrated_to_three_drives_of_seven(tmp_path):
    """The documented cadence: 3 long-forms/week
    (Ireland Sun, USA Tue, USA Fri) x 7 shorts/long-form
    (``DriveConfig.min_shorts_per_lf``) x 1 delivery/platform/short = 21 per
    platform per rolling week -- the whole-account steady state, not a
    per-region figure. This run's own drive (7 shorts on youtube) landing on
    top of two prior drives already in the queue (14) totals exactly 21 and
    must PASS: the gate must never refuse the documented happy path. A
    fourth drive's plan on top of that full 21 must be refused."""
    out = tmp_path / "out"
    out.mkdir()
    # This run IS the third drive of the week: one long-form's 7 shorts.
    _sidecar(out, *(_record(short_index=i, selected=("youtube",)) for i in range(7)))
    state_path = out / "pipeline_state.json"
    state_path.write_text("{}", encoding="utf-8")

    conn = open_queue_db(tmp_path)
    # Two prior drives already in the queue this week: 14 deliveries.
    for i in range(14):
        enqueue(conn, "other-proj", i, "youtube", f"h{i}", scheduled_at=SCHEDULED.isoformat())
    conn.close()

    with (
        mock.patch(
            "gomoveshift.cli.handlers.enqueue_plan._load_state",
            return_value=_approved_state(out),
        ),
        mock.patch(
            "gomoveshift.cli.handlers.enqueue_plan.current_brand",
            return_value="no-such-brand",
        ),
    ):
        rc = cmd_enqueue_plan(_args(state_path, apply=True), _config(tmp_path))

    assert rc == 0
    conn = open_queue_db(tmp_path)
    count = conn.execute(
        "SELECT COUNT(*) AS n FROM deliveries WHERE project_name='proj'"
    ).fetchone()["n"]
    conn.close()
    assert count == 7  # the full third drive spooled -- 14 + 7 = 21, not refused

    # A fourth drive's plan on top of the now-21-deep queue IS refused.
    _sidecar(out, _record(short_index=99, selected=("youtube",)))
    with (
        mock.patch(
            "gomoveshift.cli.handlers.enqueue_plan._load_state",
            return_value=_approved_state(out),
        ),
        mock.patch(
            "gomoveshift.cli.handlers.enqueue_plan.current_brand",
            return_value="no-such-brand",
        ),
    ):
        rc = cmd_enqueue_plan(_args(state_path, apply=True), _config(tmp_path))

    assert rc == 1


def test_seo_config_rolling_week_ceiling_default_is_twenty_one():
    from gomoveshift.config import SEOConfig

    assert SEOConfig().rolling_week_ceiling_per_platform == 21


# --- tracking store registration at enqueue --------------------------
#
# The publish hook (queue.py's _register_platform_id_best_effort) only ever
# UPDATES a short already in the TrackingStore -- nothing put it there, so
# the store never got any records. Enqueuing is the moment the queue first
# knows every (project_name, short_index) it is about to spool; this is
# where the record has to be created.


def test_apply_registers_every_enqueued_short_in_the_tracking_store(tmp_path, monkeypatch):
    from gomoveshift.tracking import TrackingStore

    monkeypatch.setenv("GMS_OUTPUT_DIR", str(tmp_path / "gms-output"))

    out = tmp_path / "out"
    out.mkdir()
    _sidecar(
        out,
        _record(short_index=1, selected=("youtube",)),
        _record(short_index=2, selected=("youtube", "tiktok")),
    )
    state_path = out / "pipeline_state.json"
    state_path.write_text("{}", encoding="utf-8")

    with mock.patch(
        "gomoveshift.cli.handlers.enqueue_plan._load_state",
        return_value=_approved_state(out),
    ):
        assert cmd_enqueue_plan(_args(state_path, apply=True), _config(tmp_path)) == 0

    record = TrackingStore().load("proj")
    assert sorted(s.index for s in record.shorts) == [1, 2]

    # The publish hook's contract: update_platform_id must find the record
    # by (project_name, short_index) once a delivery actually publishes.
    TrackingStore().update_platform_id("proj", 1, "youtube", "yt123")
    updated = TrackingStore().load("proj")
    short = next(s for s in updated.shorts if s.index == 1)
    assert short.platform_ids["youtube"] == "yt123"


def test_a_shadow_enqueue_registers_nothing_in_tracking(tmp_path, monkeypatch):
    """A shadow delivery can never publish, so a tracking record for it would
    be a short the analytics store counts forever and never sees posted."""
    from gomoveshift.tracking import TrackingStore

    monkeypatch.setenv("GMS_OUTPUT_DIR", str(tmp_path / "gms-output"))
    out = tmp_path / "out"
    out.mkdir()
    _sidecar(out, _record(short_index=1, selected=("youtube",)))
    state_path = out / "pipeline_state.json"
    state_path.write_text("{}", encoding="utf-8")

    with mock.patch(
        "gomoveshift.cli.handlers.enqueue_plan._load_state",
        return_value=_approved_state(out),
    ):
        assert cmd_enqueue_plan(_args(state_path, apply=True, shadow=True), _config(tmp_path)) == 0

    assert not TrackingStore().exists("proj")


def test_reenqueuing_the_same_plan_does_not_duplicate_tracking_records(tmp_path, monkeypatch):
    from gomoveshift.tracking import TrackingStore

    monkeypatch.setenv("GMS_OUTPUT_DIR", str(tmp_path / "gms-output"))

    out = tmp_path / "out"
    out.mkdir()
    _sidecar(out, _record(short_index=1, selected=("youtube",)))
    state_path = out / "pipeline_state.json"
    state_path.write_text("{}", encoding="utf-8")

    with mock.patch(
        "gomoveshift.cli.handlers.enqueue_plan._load_state",
        return_value=_approved_state(out),
    ):
        assert cmd_enqueue_plan(_args(state_path, apply=True), _config(tmp_path)) == 0
        assert cmd_enqueue_plan(_args(state_path, apply=True), _config(tmp_path)) == 0

    record = TrackingStore().load("proj")
    assert [s.index for s in record.shorts] == [1]


def test_tracking_registration_failure_does_not_block_enqueue(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("GMS_OUTPUT_DIR", str(tmp_path / "gms-output"))

    out = tmp_path / "out"
    out.mkdir()
    _sidecar(out, _record(short_index=1, selected=("youtube",)))
    state_path = out / "pipeline_state.json"
    state_path.write_text("{}", encoding="utf-8")

    with (
        mock.patch(
            "gomoveshift.cli.handlers.enqueue_plan._load_state",
            return_value=_approved_state(out),
        ),
        mock.patch(
            "gomoveshift.tracking.TrackingStore.register_shorts_batch",
            side_effect=RuntimeError("disk exploded"),
        ),
        caplog.at_level("WARNING"),
    ):
        rc = cmd_enqueue_plan(_args(state_path, apply=True), _config(tmp_path))

    assert rc == 0
    conn = open_queue_db(tmp_path)
    count = conn.execute("SELECT COUNT(*) AS n FROM deliveries").fetchone()["n"]
    conn.close()
    assert count == 1
    assert any("disk exploded" in r.message for r in caplog.records)


def test_a_cancelled_delivery_frees_its_cadence_slot_but_a_published_one_does_not(tmp_path):
    """The ceiling counts slots a rolling week has SPENT. A published delivery
    spent one and still counts; a cancelled delivery will never reach a
    platform, so holding its slot would silently shrink the week's capacity."""
    out = tmp_path / "out"
    out.mkdir()
    _sidecar(out, _record(short_index=5, selected=("youtube",)))
    state_path = out / "pipeline_state.json"
    state_path.write_text("{}", encoding="utf-8")

    conn = open_queue_db(tmp_path)
    kept = enqueue(conn, "other-proj", 0, "youtube", "hash0", scheduled_at=SCHEDULED.isoformat())
    freed = enqueue(conn, "other-proj", 1, "youtube", "hash1", scheduled_at=SCHEDULED.isoformat())
    transition(conn, kept, "queued", "published", published_at_source="platform_confirmed")
    assert cancel_delivery(conn, freed) is True
    conn.close()

    with mock.patch(
        "gomoveshift.cli.handlers.enqueue_plan._load_state",
        return_value=_approved_state(out),
    ):
        rc = cmd_enqueue_plan(_args(state_path, apply=True), _config(tmp_path, ceiling=2))

    # published(1) + this plan's 1 == the ceiling; the cancelled row is not counted.
    assert rc == 0

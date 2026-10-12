"""Regression test: ReleaseSchedule.save() must not truncate the existing
release_schedule.yaml when the write fails partway.

The original implementation did ``with open(path, "w") as f: yaml.dump(...)``
— opening the destination path in "w" mode truncates it immediately, before
any bytes are written. A crash or full disk mid-dump then leaves a truncated
(but often still-parseable) file with fewer slots, and nothing downstream
notices. Fix: route through ``_atomic_write.atomic_write_yaml``, which dumps
to a sibling temp file and ``os.replace``s it in only on success.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gomoveshift.pipeline.scheduler import ReleaseSchedule


def _make_schedule() -> ReleaseSchedule:
    return ReleaseSchedule(
        project_name="testproj",
        location="A to B",
        long_form_publish="2026-04-13T11:00:00-04:00",
        window_start="2026-04-13T12:00:00-04:00",
        window_end="2026-04-15T11:00:00-04:00",
        num_shorts=2,
        num_slots=2,
        slots=[],
    )


def test_save_failure_leaves_existing_file_intact(tmp_path, monkeypatch):
    path = tmp_path / "release_schedule.yaml"
    original_content = "project_name: original\nlocation: intact\n" + ("x" * 200) + "\n"
    path.write_text(original_content)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("simulated crash mid-write")

    monkeypatch.setattr("gomoveshift.pipeline._atomic_write.yaml.dump", _boom)

    schedule = _make_schedule()
    with pytest.raises(RuntimeError):
        schedule.save(path)

    assert path.read_text() == original_content, (
        "save() truncated the pre-existing schedule file instead of leaving it "
        "intact on a failed write — the write must be atomic (temp file + "
        "os.replace), not open(path, 'w') directly."
    )

"""Phase D hard-fails when an LF thumbnail cannot be generated.

Day-first drive is the only supported production mode, so Phase D is where a
missing long-form thumbnail reaches YouTube; the three generation-failure
paths used to warn and report success anyway.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gomoveshift.pipeline.drive_phases.phase_d import _generate_lf_thumbnails

from conftest import fake_generate_lf_thumbnail  # type: ignore[import-not-found]
from test_phase_d_lf_outputs import (  # type: ignore[import-not-found]
    _build_state,
    _install_fake_youtube_uploader,
    _run_phase_d,
)

pytestmark = pytest.mark.real_lf_thumbnail_generation

GEN_PATH = "gomoveshift.pipeline.drive_phases.thumbnail_generator.generate_lf_thumbnail"


def _generate(tmp_path):
    _generate_lf_thumbnails(
        drive_plan=SimpleNamespace(regular_lfs=[SimpleNamespace(index=1, total_duration=600.0)]),
        regular_lf_paths=["/v/lf1.mp4"],
        metadata_per_lf={1: {"title": "T1"}},
        state=SimpleNamespace(per_lf_ost_text={1: "Kerry Coast Drive"}),
        output_base=tmp_path,
    )


@pytest.mark.parametrize("gen", [lambda **kw: None, OSError("ffmpeg exploded")])
def test_generation_failure_raises(tmp_path, gen):
    with patch(GEN_PATH, side_effect=gen):
        with pytest.raises(RuntimeError, match="thumbnail"):
            _generate(tmp_path)


def test_canonical_promotion_failure_raises(tmp_path):
    with patch(GEN_PATH, side_effect=fake_generate_lf_thumbnail):
        with patch("shutil.copyfile", side_effect=OSError("read-only fs")):
            with pytest.raises(RuntimeError, match="thumbnail"):
                _generate(tmp_path)


def test_failed_generation_blocks_the_youtube_upload(tmp_path, monkeypatch):
    """The point of the gate: nothing ships once generation fails."""
    yt = _install_fake_youtube_uploader(monkeypatch)
    state = _build_state(tmp_path, num_lfs=1, youtube_upload=True, skip_upload=False)

    with patch(GEN_PATH, side_effect=lambda **kw: None):
        with pytest.raises(RuntimeError, match="thumbnail"):
            _run_phase_d(state, num_lfs=1)

    assert yt.upload_video.call_count == 0
    assert yt.upload_if_needed.call_count == 0
    assert not (state.get("youtube_uploads") or {})

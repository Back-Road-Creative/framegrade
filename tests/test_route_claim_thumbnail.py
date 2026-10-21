"""D8b T1/T2: the LF thumbnail burns the route-checked label, never a stale one.

``enforce_route_claims`` may broaden an LF's label in place; the thumbnail
must then carry the broadened text, byte-equal to ``per_lf_ost_text``, and
never the meta title or the original over-claiming label. This composes the
real entry (``_enrich_per_lf_gps``), the real flatten and the real thumbnail
step; only the image writer is faked.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from gomoveshift.pipeline.drive_phases.phase_bc_day_first import flatten_ost_text
from gomoveshift.pipeline.drive_phases.phase_d import _generate_lf_thumbnails

from test_retained_gps_claims import _enrich, _lf  # type: ignore[import-not-found]

GEN_PATH = "gomoveshift.pipeline.drive_phases.thumbnail_generator.generate_lf_thumbnail"


def _thumbnail_title(tmp_path, ost_text, meta_title):
    seen = {}

    def _gen(*, video_path, title, subtitle, output_dir, duration):
        seen["title"] = title
        output_dir.mkdir(parents=True, exist_ok=True)
        out = output_dir / "v.jpg"
        out.write_bytes(b"jpg")
        return out

    with patch(GEN_PATH, side_effect=_gen):
        _generate_lf_thumbnails(
            drive_plan=SimpleNamespace(regular_lfs=[SimpleNamespace(index=1, total_duration=1.0)]),
            regular_lf_paths=["/v/lf1.mp4"],
            metadata_per_lf={1: {"title": meta_title}},
            state=SimpleNamespace(per_lf_ost_text=ost_text),
            output_base=tmp_path,
        )
    return seen["title"]


def test_t2_broadened_label_reaches_the_thumbnail(tmp_path):
    # Retained Harrow + Pellmere footage cannot carry "Harrow" alone.
    out = _enrich([_lf(1, (0.0, 280.0), (320.0, 580.0))], {1: "Harrow, Westmark"})
    ost_text = flatten_ost_text(out.per_lf_ost_text_by_date)
    assert ost_text == {1: "Ashby County, Westmark"}
    title = _thumbnail_title(tmp_path, ost_text, meta_title="Harrow, Westmark (Part 1 of 2)")
    assert title == "Ashby County, Westmark"


def test_t1_thumbnail_is_byte_equal_to_the_checked_text_not_the_meta_title(tmp_path):
    out = _enrich([_lf(1, (0.0, 280.0))], {1: "Harrow, Westmark"})
    ost_text = flatten_ost_text(out.per_lf_ost_text_by_date)
    title = _thumbnail_title(tmp_path, ost_text, meta_title="Somewhere Else, Westmark")
    assert title == ost_text[1] == "Harrow, Westmark"


def test_flatten_drops_empty_labels_and_keys_by_int():
    assert flatten_ost_text({"d1": {"1": "A", 2: ""}, "d2": {3: "C"}}) == {1: "A", 3: "C"}

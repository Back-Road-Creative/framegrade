"""astro_select.select_frames is a pure decision function -- no I/O, no decoding,
no orchestrator wiring. These tests build synthetic FrameRecords by hand."""

from pathlib import Path

from gomoveshift.pipeline.photos.astro_select import (
    REASON_BELOW_KEEP_THRESHOLD,
    REASON_CLOUD,
    REASON_DEW,
    REASON_EXPOSURE_OUTLIER,
    REASON_UNMEASURED_QUALITY,
    REASON_UNUSABLE,
    FrameRecord,
    SelectionPolicy,
    select_frames,
)


def _rec(name, *, exposure_s=15.0, usable=True, cloud=False, dew=False, **quality):
    return FrameRecord(
        path=Path(name),
        exposure_s=exposure_s,
        usable=usable,
        cloud=cloud,
        dew=dew,
        star_count=quality.get("star_count"),
        background_median=quality.get("background_median"),
        fwhm_px=quality.get("fwhm_px"),
        ellipticity=quality.get("ellipticity"),
    )


def _full(name, *, fwhm_px, ellipticity, background_median, star_count, **kw):
    return _rec(
        name,
        fwhm_px=fwhm_px,
        ellipticity=ellipticity,
        background_median=background_median,
        star_count=star_count,
        **kw,
    )


def test_odd_exposure_frame_dropped_rest_survive():
    records = [
        _full(f"f{i}.nef", fwhm_px=2.5, ellipticity=0.1, background_median=100.0, star_count=50)
        for i in range(4)
    ]
    records.append(
        _full(
            "DSC_0038.nef",
            exposure_s=5.0,
            fwhm_px=2.5,
            ellipticity=0.1,
            background_median=100.0,
            star_count=50,
        )
    )
    result = select_frames(records, SelectionPolicy())
    by_path = {sel.path.name: sel for sel in result.selections}
    assert by_path["DSC_0038.nef"].selected is False
    assert by_path["DSC_0038.nef"].reason == REASON_EXPOSURE_OUTLIER
    for i in range(4):
        assert by_path[f"f{i}.nef"].selected is True
    assert result.kept_all_below_floor is True


def test_unusable_cloud_dew_dropped_with_reasons():
    records = [
        _full("good.nef", fwhm_px=2.0, ellipticity=0.05, background_median=90.0, star_count=80),
        _full(
            "unusable.nef",
            usable=False,
            fwhm_px=2.0,
            ellipticity=0.05,
            background_median=90.0,
            star_count=80,
        ),
        _full(
            "cloud.nef",
            cloud=True,
            fwhm_px=2.0,
            ellipticity=0.05,
            background_median=90.0,
            star_count=80,
        ),
        _full(
            "dew.nef",
            dew=True,
            fwhm_px=2.0,
            ellipticity=0.05,
            background_median=90.0,
            star_count=80,
        ),
    ]
    result = select_frames(records, SelectionPolicy())
    by_path = {sel.path.name: sel for sel in result.selections}
    assert by_path["good.nef"].selected is True
    assert by_path["unusable.nef"].reason == REASON_UNUSABLE
    assert by_path["cloud.nef"].reason == REASON_CLOUD
    assert by_path["dew.nef"].reason == REASON_DEW
    for name in ("unusable.nef", "cloud.nef", "dew.nef"):
        assert by_path[name].selected is False
        assert by_path[name].score is None


def test_keep_fraction_is_honoured():
    records = [
        _full(
            f"f{i}.nef",
            fwhm_px=1.0 + i * 0.1,
            ellipticity=0.05,
            background_median=90.0,
            star_count=100 - i,
        )
        for i in range(10)
    ]
    policy = SelectionPolicy(keep_fraction=0.5, min_frames=2)
    result = select_frames(records, policy)
    assert len(result.selected) == 5
    assert result.floor_applied is False
    # best (lowest fwhm, most stars) frames are f0..f4
    assert set(p.name for p in result.selected) == {f"f{i}.nef" for i in range(5)}


def test_min_frames_floor_overrides_fraction():
    records = [
        _full(
            f"f{i}.nef",
            fwhm_px=1.0 + i * 0.1,
            ellipticity=0.05,
            background_median=90.0,
            star_count=100 - i,
        )
        for i in range(10)
    ]
    policy = SelectionPolicy(keep_fraction=0.1, min_frames=8)
    result = select_frames(records, policy)
    assert len(result.selected) == 8
    assert result.floor_applied is True


def test_run_smaller_than_floor_keeps_everything():
    records = [
        _full(f"f{i}.nef", fwhm_px=1.0 + i, ellipticity=0.05, background_median=90.0, star_count=50)
        for i in range(5)
    ]
    policy = SelectionPolicy(keep_fraction=0.7, min_frames=100)
    result = select_frames(records, policy)
    assert len(result.selected) == 5
    assert result.kept_all_below_floor is True
    for sel in result.selections:
        assert sel.selected is True


def test_ranking_is_by_quality_not_input_order():
    # The sharpest, roundest, darkest-sky, most-starred frames are placed LAST in the
    # input list; selection must still pick them, not the first N by position.
    records = [
        _full(f"noisy{i}.nef", fwhm_px=6.0, ellipticity=0.6, background_median=500.0, star_count=5)
        for i in range(6)
    ]
    records += [
        _full(f"best{i}.nef", fwhm_px=1.5, ellipticity=0.02, background_median=50.0, star_count=200)
        for i in range(4)
    ]
    policy = SelectionPolicy(keep_fraction=0.4, min_frames=1)
    result = select_frames(records, policy)
    assert set(p.name for p in result.selected) == {f"best{i}.nef" for i in range(4)}
    noisy_selections = [sel for sel in result.selections if sel.path.name.startswith("noisy")]
    assert all(sel.reason == REASON_BELOW_KEEP_THRESHOLD for sel in noisy_selections)
    assert all(sel.score is not None for sel in noisy_selections)  # scored, just not kept


def test_unmeasured_metric_uses_only_available_terms_not_zero():
    # frame1 has every quality field measured; frame2 is missing fwhm_px only -- its
    # score must be the mean of its OTHER three terms, not diluted by a synthetic 0.
    records = [
        _full("frame1.nef", fwhm_px=2.0, ellipticity=0.05, background_median=90.0, star_count=100),
        _rec("frame2.nef", ellipticity=0.05, background_median=90.0, star_count=100, fwhm_px=None),
    ]
    policy = SelectionPolicy(keep_fraction=1.0, min_frames=1)
    result = select_frames(records, policy)
    by_path = {sel.path.name: sel for sel in result.selections}
    assert by_path["frame1.nef"].selected is True
    assert by_path["frame2.nef"].selected is True
    # Identical on the three shared terms -> identical score, since the missing fwhm
    # term is excluded rather than counted as a worst-possible 0.
    assert by_path["frame1.nef"].score == by_path["frame2.nef"].score


def test_frame_with_no_measured_quality_dropped_not_scored_as_worst():
    records = [
        _full("good.nef", fwhm_px=2.0, ellipticity=0.05, background_median=90.0, star_count=100),
        _rec("blank.nef"),  # every quality field left unmeasured
    ]
    result = select_frames(records, SelectionPolicy())
    by_path = {sel.path.name: sel for sel in result.selections}
    assert by_path["blank.nef"].selected is False
    assert by_path["blank.nef"].reason == REASON_UNMEASURED_QUALITY
    assert by_path["blank.nef"].score is None
    assert by_path["good.nef"].selected is True


def test_empty_input_returns_empty_result():
    result = select_frames([], SelectionPolicy())
    assert result.selections == ()
    assert result.selected == ()
    assert result.kept_all_below_floor is False

"""CLI contract for combining reusable astro linear artifacts."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gomoveshift.cli.handlers import photos as photos_handler
from gomoveshift.cli.parser import _build_parser
from gomoveshift.pipeline.photos import astro_combine, astro_combine_output
from gomoveshift.pipeline.photos.astro_linear import write_linear_artifact
from gomoveshift.pipeline.photos.astro_stack import StackResult


def test_astro_combine_parser_requires_output_and_accepts_reference():
    args = _build_parser().parse_args(
        "photos astro-combine a.linear b.linear --out night.tif --reference b.linear".split()
    )

    assert args.photos_command == "astro-combine"
    assert args.paths == ["a.linear", "b.linear"]
    assert args.out == "night.tif"
    assert args.reference == "b.linear"
    assert args.scope == "auto"


def test_astro_combine_parser_accepts_an_explicit_scope():
    args = _build_parser().parse_args(
        "photos astro-combine a.linear b.linear --out night.tif --scope full".split()
    )

    assert args.scope == "full"


def test_astro_combine_parser_rejects_an_unknown_scope():
    with pytest.raises(SystemExit):
        _build_parser().parse_args(
            "photos astro-combine a.linear b.linear --out night.tif --scope bogus".split()
        )


def test_astro_combine_routes_inputs_and_reports_material_thin_coverage(
    tmp_path, monkeypatch, capsys
):
    result = SimpleNamespace(coverage=SimpleNamespace(below_floor_fraction=0.125, floor=100.0))
    seen = []

    def combine(paths, *, reference, scope):
        seen.append(("combine", paths, reference, scope))
        return result

    def publish(value, out):
        seen.append(("publish", value, out))
        return SimpleNamespace(
            image=Path(out),
            coverage=Path(out).with_suffix(".coverage.npy"),
            sidecar=Path(out).with_suffix(".deep.json"),
        )

    monkeypatch.setattr(astro_combine, "deep_combine", combine)
    monkeypatch.setattr(astro_combine_output, "publish_deep_product", publish)
    out = tmp_path / "night.tif"
    args = _build_parser().parse_args(
        ["photos", "astro-combine", "a.linear", "b.linear", "--out", str(out)]
    )

    assert photos_handler.cmd_photos(args, None) == 0
    assert seen == [
        ("combine", [Path("a.linear"), Path("b.linear")], None, "auto"),
        ("publish", result, out),
    ]
    captured = capsys.readouterr()
    assert "WARNING" in captured.err and "12.50%" in captured.err
    assert str(out.with_suffix(".coverage.npy")) in captured.out


def test_astro_combine_passes_an_explicit_scope_through_to_deep_combine(tmp_path, monkeypatch):
    result = SimpleNamespace(coverage=SimpleNamespace(below_floor_fraction=0.0, floor=100.0))
    seen = []

    monkeypatch.setattr(
        astro_combine,
        "deep_combine",
        lambda paths, *, reference, scope: seen.append(scope) or result,
    )
    monkeypatch.setattr(
        astro_combine_output,
        "publish_deep_product",
        lambda value, out: SimpleNamespace(
            image=Path(out),
            coverage=Path(out).with_suffix(".coverage.npy"),
            sidecar=Path(out).with_suffix(".deep.json"),
        ),
    )
    out = tmp_path / "night.tif"
    args = _build_parser().parse_args(
        ["photos", "astro-combine", "a.linear", "b.linear", "--out", str(out), "--scope", "full"]
    )

    assert photos_handler.cmd_photos(args, None) == 0
    assert seen == ["full"]


def test_astro_combine_reports_validation_errors(monkeypatch, capsys):
    monkeypatch.setattr(
        astro_combine,
        "deep_combine",
        lambda *args, **kwargs: (_ for _ in ()).throw(astro_combine.AstroCombineError("bad set")),
    )
    args = _build_parser().parse_args(
        "photos astro-combine a.linear b.linear --out night.tif".split()
    )

    assert photos_handler.cmd_photos(args, None) == 1
    assert "bad set" in capsys.readouterr().err


def test_astro_combine_end_to_end_fixture_writes_all_products(tmp_path, monkeypatch):
    artifacts = []
    shape = (40, 40, 3)
    sky = np.full(shape, 0.2, dtype=np.float32)
    sky[5:12, 5:12] += 0.3
    for index in range(2):
        path = tmp_path / f"chunk-{index}.linear"
        count = np.full(shape[:2], 10, dtype=np.uint16)
        write_linear_artifact(
            path,
            StackResult(
                rgb=sky,
                n_frames=10,
                mean_rejected_fraction=0.0,
                sky_sum=sky * count[..., None],
                sky_count=count,
                ground=np.full(shape, 0.1, dtype=np.float32),
            ),
            reference_raw=f"RAW-{index}-mid.NEF",
            capture_identity={
                "camera": "NIKON Z 6",
                "raw_shape": [40, 40],
                "cfa_pattern": [0, 1, 3, 2],
                "iso": 6400.0,
                "exposure_s": 15.0,
            },
            mask_provenance={"mode": "automatic", "resolved_boundary_row": 20},
            source_frames=[f"RAW-{index}-{frame}.NEF" for frame in range(10)],
            registration_provenance=[{"frame": f"RAW-{index}-0.NEF"}],
            photometry_provenance=[{"frame": f"RAW-{index}-0.NEF", "scale": 1.0}],
            algorithm_version="linear-hash-a",
        )
        artifacts.append(path)
    monkeypatch.setattr(
        astro_combine,
        "fit_artifact_pair",
        lambda *args, **kwargs: np.eye(2, 3, dtype=np.float64),
    )
    out = tmp_path / "night.tif"
    args = _build_parser().parse_args(
        ["photos", "astro-combine", *(str(path) for path in artifacts), "--out", str(out)]
    )

    assert photos_handler.cmd_photos(args, None) == 0
    assert out.is_file()
    assert out.with_suffix(".coverage.npy").is_file()
    assert out.with_suffix(".deep.json").is_file()

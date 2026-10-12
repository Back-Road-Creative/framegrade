"""``photos astro-inspect``: parser surface plus a real end-to-end CLI run
against a small synthetic TIFF -- no mocks, the pipeline pieces here are cheap
at this size. READ-ONLY: must never modify the master it inspects."""

import numpy as np
import pytest
import tifffile

from gomoveshift.cli.handlers import photos as photos_handler
from gomoveshift.cli.parser import _build_parser


def _parse(argv):
    return _build_parser().parse_args(argv.split())


def _run(argv):
    return photos_handler.cmd_photos(_parse(argv), None)


def _write_synthetic_master(tmp_path):
    rng = np.random.default_rng(7)
    rgb16 = (rng.random((240, 240, 3)) * 30000).astype(np.uint16)
    tif_path = tmp_path / "night_master.tif"
    tifffile.imwrite(tif_path, rgb16)
    return tif_path


def test_astro_inspect_parser_surface():
    args = _parse("photos astro-inspect master.tif --out sheet.jpg")
    assert args.master == "master.tif"
    assert args.out == "sheet.jpg"
    with pytest.raises(SystemExit):
        _parse("photos astro-inspect master.tif")  # --out is required


def test_astro_inspect_writes_contact_sheet_and_leaves_the_master_byte_identical(tmp_path):
    tif_path = _write_synthetic_master(tmp_path)
    before = tif_path.read_bytes()
    out_path = tmp_path / "sheet.jpg"

    rc = _run(f"photos astro-inspect {tif_path} --out {out_path}")

    assert rc == 0
    assert out_path.exists() and out_path.stat().st_size > 0
    assert tif_path.read_bytes() == before, "astro-inspect must never modify the master"


def test_astro_inspect_reports_the_read_failure_and_never_constructs_a_sheet(tmp_path, capsys):
    bogus = tmp_path / "not_a_tiff.tif"
    bogus.write_bytes(b"nope")
    out_path = tmp_path / "sheet.jpg"

    rc = _run(f"photos astro-inspect {bogus} --out {out_path}")

    assert rc == 1
    assert not out_path.exists()
    assert "astro-inspect" in capsys.readouterr().err

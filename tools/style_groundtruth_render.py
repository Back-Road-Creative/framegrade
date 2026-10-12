"""Corpus manifest + blind judging grids for the photo style ground-truth harness.

Two SUBCOMMANDS, not two flags: ``manifest`` is a cheap pure walk, ``render``
decodes and grades every capture in it. Pinning the corpus once, as JSON, means
a re-render (or a sibling tool) judges the SAME capture set, not whatever is on
the card that day.

By default ``render`` passes NO style list to ``develop_variants``, so the
shipped auto-selection (``select_styles`` -> ``_distinct_styled``) picks the
variants: that choice is the thing under measurement. ``--styles a,b,...``
renders that EXPLICIT ladder instead, shuffled under a ``wide-``-prefixed seed
so its panel ordering never collides with the auto run's. Panels are captioned
``A``/``B``/``C``... only, the mapping is shuffled with ``random.Random(seed)``,
and the style name goes to the sidecar JSON — never into the image, never into a
panel filename. A judge cannot read the machine's own ranking off the picture it
is being asked to rank. Each render.json capture also carries the ``scene``
profile the selector actually scored and the manifest's ``exif_dt``, so the
sidecar alone feeds the scorer. See ``docs/style-groundtruth.md``.
"""

from __future__ import annotations

import argparse
import json
import random
import string
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = 1
RAW_SUFFIXES = frozenset({".nef", ".dng", ".cr2", ".arw"})
JPEG_SUFFIXES = frozenset({".jpg", ".jpeg"})
#: Rendered pipeline OUTPUT and LUT tables — corpus neighbours, never sources.
EXCLUDED_DIRS = frozenset({"gms_test", "luts"})
#: Long-edge bound per panel — a readable grade, and a 3-panel grid fits a judge's budget.
PANEL_LONG_EDGE = 1024


def _exif_dt(path: Path) -> str | None:
    """Capture timestamp as ISO-8601, or ``None``. Uses the pipeline's own Exif
    sub-IFD reader (Pillow, then the TIFF walk a ``.NEF`` needs) — the public
    readers there return a date only."""
    from gomoveshift.pipeline.photos.exif_gps import _DATETIME_ORIGINAL, _exif_sub_ifd

    try:
        return datetime.strptime(
            str(_exif_sub_ifd(path).get(_DATETIME_ORIGINAL) or "").strip(), "%Y:%m:%d %H:%M:%S"
        ).isoformat()
    except ValueError:
        return None


def build_manifest(corpus: Path) -> dict[str, Any]:
    """Every capture under ``corpus``, deduped by stem and sorted by id. One
    capture ships as several files (``shot_0020.JPG`` + ``.NEF`` + ``.dng``), so a
    file count over-reports it ~2x. The walk is sorted, so the raw kept and the
    order written are the same on every machine."""
    captures: dict[str, dict[str, Any]] = {}
    for path in sorted(corpus.rglob("*")):
        if not path.is_file() or set(path.relative_to(corpus).parts[:-1]) & EXCLUDED_DIRS:
            continue
        suffix = path.suffix.lower()
        kind = "raw" if suffix in RAW_SUFFIXES else "jpeg" if suffix in JPEG_SUFFIXES else None
        if kind is None:  # .cube, sidecars, anything that is not a photo
            continue
        entry = captures.setdefault(path.stem, {"id": path.stem, "raw": None, "jpeg": None})
        if entry[kind] is None:
            entry[kind] = str(path.resolve())
    for entry in captures.values():
        entry["exif_dt"] = _exif_dt(Path(entry["raw"] or entry["jpeg"]))
    return {
        "schema": SCHEMA,
        "corpus_root": str(corpus.resolve()),
        "captures": [captures[stem] for stem in sorted(captures)],
    }


def blind_panels(
    capture_id: str, variants: list[tuple[str, np.ndarray]]
) -> list[tuple[str, str, np.ndarray]]:
    """``(label, style, frame)`` per offered variant, in shuffled label order.
    The seed is the capture id (``wide-``-prefixed by an explicit-ladder run)
    and nothing else: same corpus, same panels, on any host and any day — a
    clock-seeded shuffle would make the harness unre-runnable, the one thing a
    ground-truth set cannot be."""
    order = list(range(len(variants)))
    random.Random(capture_id).shuffle(order)  # noqa: S311 - reproducibility, not secrecy
    return [
        (string.ascii_uppercase[label], variants[pick][0], variants[pick][1])
        for label, pick in enumerate(order)
    ]


def _bound(frame: np.ndarray, long_edge: int) -> np.ndarray:
    from PIL import Image

    img = Image.fromarray(np.ascontiguousarray(frame), mode="RGB")
    if max(img.size) > long_edge:
        scale = long_edge / max(img.size)
        size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
        img = img.resize(size, Image.Resampling.BOX)
    return np.asarray(img)


def blind_grid(panels: list[tuple[str, str, np.ndarray]]) -> np.ndarray:
    """The panels side by side, captioned by LABEL only — composed by the pipeline's
    own ``compose_contact_sheet``, which is simply never handed the style name:
    blindness by construction, not by a scrubbing step somebody can forget."""
    from gomoveshift.pipeline.photos import develop

    labelled = [(label, _bound(frame, PANEL_LONG_EDGE)) for label, _, frame in panels]
    return develop.compose_contact_sheet(labelled)


def render_capture(
    capture: dict[str, Any], out_dir: Path, styles: list[str] | None = None
) -> dict[str, Any]:
    """Render one capture's variants — the shipped auto-selection, or ``styles``
    as an explicit ladder — plus its panels, its grid, and the scene profile."""
    from gomoveshift.pipeline.photos.develop import decode, develop_variants, export_jpeg
    from gomoveshift.pipeline.photos.exif_gps import extract_exif_capture_metadata
    from gomoveshift.pipeline.photos.scene import compute_scene_profile

    source, capture_id = capture["raw"] or capture["jpeg"], capture["id"]
    # Decode ONCE and profile exactly as ``plan_variants`` does in production —
    # the decoded frame plus the file's own EXIF — so the ``scene`` written here
    # EQUALS the profile the selector scores; ``develop_variants`` accepts the
    # already-decoded photo, so the corpus is never demosaiced twice.
    decoded = decode(source)
    scene = compute_scene_profile(decoded.rgb, extract_exif_capture_metadata(decoded.path))
    seed = f"wide-{capture_id}" if styles else capture_id
    panels = blind_panels(seed, [(n, p.rgb) for n, p in develop_variants(decoded, styles=styles)])
    records = []
    for label, style, frame in panels:  # panel filenames carry the LABEL, never the style
        out = export_jpeg(frame, out_dir / capture_id / f"{capture_id}_{label}.jpg")
        records.append({"label": label, "style": style, "file": str(out)})
    grid = export_jpeg(blind_grid(panels), out_dir / f"{capture_id}_grid.jpg")
    return {
        "id": capture_id,
        "source": str(source),
        "exif_dt": capture.get("exif_dt"),
        "scene": asdict(scene),
        "grid": str(grid),
        "panels": records,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    walk = sub.add_parser("manifest", help="pin the corpus as JSON")
    walk.add_argument("--corpus", type=Path, required=True, help="photo corpus directory")
    walk.add_argument("--out", type=Path, required=True, help="manifest JSON to write")
    render = sub.add_parser("render", help="render blind judging grids from a manifest")
    render.add_argument("--manifest", type=Path, required=True)
    render.add_argument("--out", type=Path, required=True, help="directory for grids + panels")
    render.add_argument(
        "--styles",
        help="comma-separated EXPLICIT style ladder; default: the shipped auto-selection",
    )
    args = parser.parse_args(argv)
    if args.command == "manifest":
        manifest = build_manifest(args.corpus)
        _write_json(args.out, manifest)
        print(f"{len(manifest['captures'])} captures -> {args.out}")
        return 0
    manifest = json.loads(args.manifest.read_text())
    styles = [s.strip() for s in args.styles.split(",")] if args.styles else None
    rendered = [render_capture(c, args.out, styles) for c in manifest["captures"]]
    _write_json(args.out / "render.json", {"schema": SCHEMA, "captures": rendered})
    print(f"{len(rendered)} grids -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

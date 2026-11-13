"""``gomoveshift queue edit-facebook-caption``: rewrite a published Reel's description.

Dry run unless ``--apply``. The new text must pass the strictest place check that needs no
route evidence, ``content_gates.placeless_label_claims_nothing`` (claim-free words only).
``--apply`` edits the live post, verifies the read-back, then records the text in the
publish-plan sidecar. Never changes the queue row, never deletes anything.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from ...pipeline.validation.content_gates import (
    placeless_label_claimed_words,
    placeless_label_claims_nothing,
    placeless_label_problem,
)
from ...publishing.plan_io import load_publish_plan
from ...publishing.queue import open_queue_db


def _adapter(config):
    from ...publishing.facebook_adapter import FacebookReelsAdapter
    from ...publishing.meta_auth import load_meta_credentials

    return FacebookReelsAdapter(load_meta_credentials(Path(config.seo.credentials_dir)))


def _refuse(msg: str) -> int:
    print(f"REFUSED: {msg}", file=sys.stderr)
    return 1


def _gate_problem(text: str):
    if placeless_label_claims_nothing(text):
        return None
    words = ", ".join(repr(w) for w in placeless_label_claimed_words(text))
    return "; ".join(x for x in (placeless_label_problem(text), f"claim words: {words}") if x)


def _write_sidecar(sidecar: Path, raw: dict) -> None:
    tmp = sidecar.with_name(sidecar.name + ".fbcaption-tmp")
    tmp.write_text(json.dumps(raw, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    try:
        load_publish_plan(tmp)  # must still parse before it replaces the old
    except Exception:
        tmp.unlink()
        raise
    os.replace(tmp, sidecar)


def cmd_edit_facebook_caption(args, config) -> int:
    conn = open_queue_db(config.seo.storage_path)
    try:
        row = conn.execute("SELECT * FROM deliveries WHERE id=?", (args.id,)).fetchone()
    finally:
        conn.close()
    if row is None:
        return _refuse(f"no delivery {args.id}")
    if row["platform"] != "facebook":
        return _refuse(f"delivery {args.id} is platform {row['platform']!r}, not facebook")
    if row["status"] != "published":
        return _refuse(f"delivery {args.id} is {row['status']!r}, not published")
    upload_id = row["platform_upload_id"]
    if not upload_id:
        return _refuse(f"delivery {args.id} has no platform upload id")
    problem = _gate_problem(args.text)
    if problem:
        return _refuse(f"delivery {args.id}: new text is not claim-free ({problem})")
    sidecar = Path(row["sidecar_path"] or "")
    if not sidecar.is_file():
        return _refuse(f"publish-plan sidecar not readable: {sidecar}")
    raw = json.loads(sidecar.read_text(encoding="utf-8"))
    short = next(
        (
            s
            for s in raw.get("shorts") or []
            if s.get("project_name") == row["project_name"]
            and s.get("short_index") == row["short_index"]
        ),
        None,
    )
    fb = ((short or {}).get("payloads") or {}).get("facebook")
    if not fb:
        return _refuse(f"{sidecar}: no facebook payload for the short")
    print(
        f"delivery {args.id} upload {upload_id}\n"
        f"current caption: {fb.get('description')!r}\n"
        f"new text: {args.text!r}\n"
        "place gate: pass (claim-free)"
    )
    if not args.apply:
        print("dry run: nothing written; re-run with --apply to edit the live post")
        return 0
    live = _adapter(config).update_description(upload_id, args.text)
    if live.get("description") != args.text:
        return _refuse(
            f"read-back of {upload_id} is {live.get('description')!r}, not the new text; "
            "sidecar left unchanged"
        )
    fb["description"] = args.text
    _write_sidecar(sidecar, raw)
    print(f"wrote: delivery {args.id} live caption and sidecar now read {args.text!r}")
    return 0

#!/usr/bin/env python3
"""Blind LLM judge producing ground-truth labels for the photo-style harness.

One photograph, graded three ways side by side, captioned only A/B/C. The judge never sees
a style name: the prompt speaks in panel letters, this module maps them back, and
``build_prompt`` raises rather than emit a prompt naming an offered style. Input is the
render manifest (``schema: 1``), read through two stable per-capture keys besides
``id``/``exif_dt`` — ``grid`` (the blind A|B|C image) and ``panels``, which the render tool
emits as ``[{"label": "A", "style": "natural", "file": ...}, ...]`` and ``panels_of`` also
accepts as a plain ``{"A": "natural"}`` map; output (``schema: 1``) is ``{schema, judge, passes, captures: [{id, exif_dt,
offered, best, acceptable, wrong, consensus, subject_tags, source}]}``, identified by
``id`` + ``exif_dt`` only, no hashes.

``best`` and ``wrong`` each need a strict majority of the three passes, and ``consensus``
records the tally (``1of1`` for a human row). Without a majority the row is still written —
``consensus: "none"``, ``best: null`` — and must then be excluded from agreement
denominators downstream rather than silently dropped; whatever is offered and neither is
``acceptable``. An unparseable reply is retried once and then that pass is dropped, as is
any tag outside the closed vocabulary. ``--human`` replays the same blind grids to a person, writing ``source: "human"`` rows that override the LLM row for an id, and calls no CLI.

``--manifest`` and ``--out`` are required (no default paths). The scorer gives override
priority only to rows whose ``source`` is exactly ``"human"``; a labels file written by an
older version of this tool with a different ``source`` value for reviewed rows must have
that field rewritten to ``"human"`` or those rows lose their priority.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import signal
import subprocess
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

Cap = Mapping[str, Any]
Row = dict[str, Any]
SCHEMA = 1
SUBJECT_TAGS = ("sky", "water", "foliage", "road", "structure", "snow", "interior", "person")
# A model alias, never a dated API model id: the alias follows the logged-in CLI
# session, whereas a dated id is a separately billed artefact that rots.
DEFAULT_MODEL = "sonnet"
_KEY_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
# STDIN prompt (never argv, so it stays out of ``ps``); own process group so a timeout reaps
# the CLI's node/MCP descendants instead of orphaning them.
_POPEN_KW: dict[str, Any] = dict.fromkeys(("stdin", "stdout", "stderr"), subprocess.PIPE)
_POPEN_KW.update(text=True, start_new_session=True)
_PROMPT = """Read the image file at {grid}. It shows one photograph {count} times side by side, captioned {letters}; the frame is identical in every panel and only the colour grade differs — exposure, contrast, white balance, saturation.
Pick the panel whose grade best suits this photograph, list any panel whose grade is actively wrong for it, and tag what the photograph shows.
Reply with ONLY this JSON object, no prose and no code fence:
{{"best": "{first}", "wrong": [], "subject_tags": []}}
"best" is one of {letters}; "wrong" holds zero or more of the others; "subject_tags" holds zero or more of exactly these words: {tags}.
"""


def panels_of(capture: Cap) -> dict[str, str]:
    """The label -> style map. The render tool emits ``panels`` as a LIST of
    ``{"label", "style", "file"}`` records, because it carries each panel's own file, so
    that is the shape on the wire; a plain ``{"A": "natural"}`` mapping is also accepted.
    Reading the shape in ONE place is what stops the two tools disagreeing about it."""
    panels = capture["panels"]
    if isinstance(panels, Mapping):
        return {str(k): str(v) for k, v in panels.items()}
    return {str(p["label"]): str(p["style"]) for p in panels}


def build_prompt(capture: Cap) -> str:
    """Render the blind prompt, refusing to emit an offered style name."""
    panels = panels_of(capture)
    letters = sorted(panels)
    tags, joined = ", ".join(SUBJECT_TAGS), "/".join(letters)
    prompt = _PROMPT.format(
        grid=capture["grid"], count=len(letters), letters=joined, first=letters[0], tags=tags
    )
    leaked = sorted({s for s in panels.values() if str(s).lower() in prompt.lower()})
    if leaked:
        raise ValueError(f"prompt would name style(s) {leaked}; the judge must stay blind")
    return prompt


def call_judge(prompt: str, model: str = DEFAULT_MODEL, timeout: float = 300.0) -> str:
    """One ``claude -p`` pass, retried once after a backoff. The API-key vars are scrubbed:
    a stray one silently flips the CLI from logged-in subscription auth to API-key billing."""
    exe = shutil.which("claude")
    if exe is None:
        raise RuntimeError("claude CLI not on PATH — install Claude Code, or label with --human")
    argv = [exe, "-p", "--model", model, "--output-format", "text"]
    argv += ["--permission-mode", "default", "--allowedTools", "Read"]
    env = {k: v for k, v in os.environ.items() if k not in _KEY_VARS}
    last: Exception = RuntimeError("claude -p never ran")
    for attempt in range(2):
        proc = subprocess.Popen(argv, env=env, **_POPEN_KW)  # noqa: S603
        try:
            stdout, stderr = proc.communicate(input=prompt, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            with contextlib.suppress(Exception):
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait(timeout=5)
            last = exc
        else:
            if proc.returncode == 0:
                return stdout.strip()
            last = RuntimeError(f"claude -p exit {proc.returncode}: {stderr[:300]}")
        if attempt == 0:
            time.sleep(2)
    raise last


def _norm(value: Any, case: Callable[[str], str]) -> list[str]:
    items = value if isinstance(value, list) else []
    return list(dict.fromkeys(case(str(x).strip()) for x in items))


def parse_verdict(text: str, letters: Sequence[str]) -> tuple[str, list[str], list[str]] | None:
    """``(best, wrong, tags)`` in panel letters, or None when the reply will not parse."""
    start, end = text.find("{"), text.rfind("}")
    try:
        data = json.loads(text[start : end + 1]) if 0 <= start < end else {}
    except ValueError:
        return None
    best = str(data.get("best", "")).strip().upper() if isinstance(data, dict) else ""
    if best not in letters:
        return None
    wrong = [w for w in _norm(data.get("wrong"), str.upper) if w in letters and w != best]
    return best, wrong, [t for t in _norm(data.get("subject_tags"), str.lower) if t in SUBJECT_TAGS]


def _majority(votes: Iterable[str], total: int) -> set[str]:
    return {k for k, n in Counter(votes).items() if n * 2 > total}


def _row(cap: Cap, offered: list[str], best: str | None, wrong: list[str], **kw: Any) -> Row:
    row: Row = {"id": cap["id"], "exif_dt": cap.get("exif_dt"), "offered": offered, "best": best}
    row["acceptable"] = [s for s in offered if s != best and s not in wrong]
    row.update(wrong=wrong, consensus=kw["consensus"], subject_tags=kw["tags"], source=kw["source"])
    return row


def judge_capture(capture: Cap, *, runner: Callable[[str], str], passes: int = 3) -> Row:
    """Run ``passes`` blind passes and majority-vote them into one label row."""
    panels = panels_of(capture)
    letters = sorted(panels)
    prompt = build_prompt(capture)
    verdicts = []
    for _ in range(passes):
        for _retry in range(2):
            verdict = parse_verdict(runner(prompt), letters)
            if verdict is not None:
                verdicts.append(verdict)
                break
    letter, votes = (Counter(b for b, _w, _t in verdicts).most_common(1) or [("", 0)])[0]
    best = panels[letter] if votes * 2 > passes else None
    bad = _majority((w for _b, ws, _t in verdicts for w in ws), passes)
    hot = _majority((t for _b, _w, ts in verdicts for t in ts), passes)
    wrong = [panels[x] for x in letters if x in bad and panels[x] != best]
    tally_text = f"{votes}of{passes}" if best else "none"
    tags = [t for t in SUBJECT_TAGS if t in hot]
    offered = [panels[x] for x in letters]
    return _row(capture, offered, best, wrong, consensus=tally_text, tags=tags, source="llm")


def judge_capture_human(capture: Cap, ask: Callable[[str], str] = input) -> Row:
    """Replay one blind grid to a person; no CLI involved."""
    panels = panels_of(capture)
    letters = sorted(panels)
    print(f"{capture['id']}: open {capture['grid']} — panels {'/'.join(letters)}")
    best = ask(f"best panel [{'/'.join(letters)}]: ").strip().upper()
    if best not in letters:
        raise SystemExit(f"{best!r} is not one of {letters}")
    marked = {w.strip().upper() for w in ask("wrong panels (comma, blank=none): ").split(",")}
    wrong = [panels[x] for x in letters if x in marked and x != best]
    offered = [panels[x] for x in letters]
    return _row(capture, offered, panels[best], wrong, consensus="1of1", tags=[], source="human")


def merge_rows(llm_rows: Iterable[Cap], human_rows: Iterable[Cap]) -> list[Row]:
    """Overlay human rows on LLM rows by capture id; a human row wins."""
    merged = {str(r["id"]): dict(r) for r in llm_rows}
    for row in human_rows:
        tags = row.get("subject_tags") or merged.get(str(row["id"]), {}).get("subject_tags", [])
        merged[str(row["id"])] = {**row, "subject_tags": tags}
    return list(merged.values())


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Blind LLM judge: label the offered photo grades A/B/C, style names hidden.",
        epilog="Reads captures[].grid and captures[].panels from the render manifest.",
    )
    ap.add_argument("--manifest", type=Path, required=True, help="render manifest JSON")
    ap.add_argument("--out", type=Path, required=True, help="labels JSON to write")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="judge model for claude -p")
    ap.add_argument("--passes", type=int, default=3, help="independent passes per capture")
    ap.add_argument("--human", action="store_true", help="label by hand; never calls the CLI")
    args = ap.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("schema") != SCHEMA:
        raise SystemExit(f"manifest schema {manifest.get('schema')!r} != {SCHEMA}")
    captures = sorted(manifest["captures"], key=lambda c: str(c["id"]))
    if args.human:
        prior = json.loads(args.out.read_text()) if args.out.exists() else {}
        rows = merge_rows(prior.get("captures", []), [judge_capture_human(c) for c in captures])
    else:
        run = partial(call_judge, model=args.model)
        rows = [judge_capture(c, runner=run, passes=args.passes) for c in captures]
    who, npass = ("human", 1) if args.human else (args.model, args.passes)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": SCHEMA, "judge": who, "passes": npass, "captures": rows}
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    stuck = sum(1 for r in rows if r["consensus"] == "none")
    print(f"wrote {len(rows)} rows, {stuck} without consensus → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

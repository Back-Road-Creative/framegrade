"""Blind-judge tests for ``tools/style_groundtruth_judge.py``; the subprocess is always
mocked, so the suite stays hermetic (CI has neither the claude CLI nor a logged-in session)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools import style_groundtruth_judge as judge  # noqa: E402

PANELS = {"A": "natural", "B": "moody", "C": "vivid"}
CAPTURE = {"id": "DSC_0020", "exif_dt": "2020-01-01T12:00", "grid": "/g/0020.jpg", "panels": PANELS}
# The shape the RENDER tool actually writes into render.json — a list of records, not a map.
RENDERED = {
    **CAPTURE,
    "panels": [
        {"label": lt, "style": st, "file": f"/g/0020_{lt}.jpg"} for lt, st in PANELS.items()
    ],
}
STYLES = ("natural", "vivid", "moody", "golden", "mono", "soft", "matte", "daylight", "shade")
SEEN: dict = {}


class Proc:  # stands in for the claude subprocess; ``boom`` makes the call time out
    returncode, pid, boom = 0, 99, False

    def communicate(self, input=None, timeout=None):
        if self.boom:
            raise subprocess.TimeoutExpired(cmd="claude", timeout=timeout)
        SEEN["stdin"] = input
        return "{}", ""

    def wait(self, timeout=None):
        return 0


def _reply(best, wrong=(), tags=("water",)):
    return json.dumps({"best": best, "wrong": list(wrong), "subject_tags": list(tags)})


def _runner(replies):
    return lambda _prompt, _it=iter(replies): next(_it)


def test_prompt_names_panels_only_never_a_style():
    prompt = judge.build_prompt(CAPTURE)
    assert CAPTURE["grid"] in prompt and all(letter in prompt for letter in PANELS)
    assert not [name for name in STYLES if name in prompt.lower()]
    with pytest.raises(ValueError):
        judge.build_prompt({**CAPTURE, "panels": {**PANELS, "A": "photograph"}})


def test_prompt_count_and_letters_follow_the_panels_for_a_six_panel_wide_round():
    """The A–F wide round offers six grades of one photograph; the prompt must state the
    real panel count instead of a hardcoded 'three', and the blindness check still holds."""
    wide_panels = dict(zip("ABCDEF", ("natural", "moody", "vivid", "golden", "mono", "matte")))
    wide = {**CAPTURE, "panels": wide_panels}
    prompt = judge.build_prompt(wide)
    assert "6 times" in prompt and "three" not in prompt.lower()
    assert "A/B/C/D/E/F" in prompt
    with pytest.raises(ValueError):
        judge.build_prompt({**wide, "panels": {**wide_panels, "F": "photograph"}})


def test_the_render_tools_own_panel_records_are_judged_the_same_as_a_plain_map():
    """The two tools are separate modules joined only by this JSON, so the judge must read
    the shape the renderer WRITES. Pinning the record list here is what makes them compose."""
    assert judge.panels_of(RENDERED) == PANELS
    assert judge.build_prompt(RENDERED) == judge.build_prompt(CAPTURE)
    row = judge.judge_capture(RENDERED, runner=_runner([_reply("B")] * 3))
    assert (row["best"], row["offered"]) == ("moody", ["natural", "moody", "vivid"])


def test_unanimous_pass_maps_letters_back_to_styles_and_rejects_unknown_tags():
    row = judge.judge_capture(CAPTURE, runner=_runner([_reply("B", tags=["water", "dragon"])] * 3))
    assert (row["best"], row["offered"]) == ("moody", ["natural", "moody", "vivid"])
    assert (row["consensus"], row["subject_tags"]) == ("3of3", ["water"])
    assert (row["id"], row["exif_dt"], row["source"]) == ("DSC_0020", CAPTURE["exif_dt"], "llm")


def test_consensus_records_a_majority_a_split_and_an_unreadable_answer():
    split = [_reply("B", wrong=["C"]), _reply("B", wrong=["C"]), _reply("A")]
    row = judge.judge_capture(CAPTURE, runner=_runner(split))
    assert (row["best"], row["consensus"]) == ("moody", "2of3")
    assert (row["wrong"], row["acceptable"]) == (["vivid"], ["natural"])
    three = judge.judge_capture(CAPTURE, runner=_runner([_reply("A"), _reply("B"), _reply("C")]))
    assert (three["best"], three["consensus"]) == (None, "none")
    calls = []
    unreadable = judge.judge_capture(CAPTURE, runner=lambda p: calls.append(p) or "sorry, I cannot")
    assert (len(calls), unreadable["consensus"]) == (6, "none")  # one reparse retry per pass


def test_call_judge_scrubs_both_key_vars_pipes_stdin_and_reaps_on_timeout(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-reach-the-cli")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "must-not-reach-the-cli")
    monkeypatch.setattr(judge.shutil, "which", lambda _name: "/usr/bin/claude")
    monkeypatch.setattr(judge.subprocess, "Popen", lambda a, **k: SEEN.update(k, argv=a) or Proc())
    assert judge.call_judge("PROMPT", model="m") == "{}"
    assert (SEEN["stdin"], SEEN["start_new_session"]) == ("PROMPT", True)
    assert "PROMPT" not in " ".join(SEEN["argv"]) and SEEN["argv"][:2] == ["/usr/bin/claude", "-p"]
    assert not [v for v in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN") if v in SEEN["env"]]
    assert SEEN["argv"][-2:] == ["--allowedTools", "Read"]
    killed = []
    monkeypatch.setattr(Proc, "boom", True)
    monkeypatch.setattr(judge.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(judge.os, "killpg", lambda pgid, _sig: killed.append(pgid))
    monkeypatch.setattr(judge.time, "sleep", lambda _s: None)
    with pytest.raises(subprocess.TimeoutExpired):
        judge.call_judge("p", model="m", timeout=0.01)
    assert killed == [99, 99]  # both attempts reap the group, not just the child


def test_human_row_needs_no_llm_and_overrides_the_llm_row(monkeypatch):
    monkeypatch.setattr(judge.shutil, "which", lambda _name: None)
    answers = iter(["b", "C"])
    row = judge.judge_capture_human(CAPTURE, ask=lambda _q: next(answers))
    assert (row["best"], row["wrong"], row["source"]) == ("moody", ["vivid"], "human")
    assert (row["acceptable"], row["consensus"]) == (["natural"], "1of1")
    llm = judge.judge_capture(CAPTURE, runner=_runner([_reply("A")] * 3))
    merged = judge.merge_rows([llm], [row])
    assert [(r["best"], r["source"]) for r in merged] == [("moody", "human")]
    assert merged[0]["subject_tags"] == ["water"]

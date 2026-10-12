"""Operator-facing remediation text must name knobs that actually exist.

An error or degradation message is read at the worst possible moment: the
run has stopped (or silently lost signal) and the operator is deciding
what to do next. Naming a CLI flag that argparse will reject sends them
down a dead end. Two such phantom flags shipped in these modules
(``--weather-hwaccel``, ``--chunk-max-sec``); neither was ever wired into
``gomoveshift/cli/parser.py``. The real knobs are config fields
(``config.drive.weather_hwaccel``, ``config.drive.weather_chunk_max_sec``).

The last test here is a class guard: every ``--flag`` token in these two
modules must be a real GMS flag, or an explicitly listed external-tool
flag (ffmpeg build options, ``wsl.exe`` options).
"""

from __future__ import annotations

import argparse
import inspect
import logging
import re
import subprocess
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.cli import parser as parser_mod
from gomoveshift.pipeline import blur_gate as blur_gate_mod
from gomoveshift.pipeline import preflight as preflight_mod
from gomoveshift.pipeline import weather as weather_mod
from gomoveshift.pipeline.ost_overrides import _OVERRIDE_FILENAME
from gomoveshift.pipeline.preflight import check_weather_decode_path
from gomoveshift.pipeline.weather import WeatherDetector

FILENAME_RE = re.compile(r"\bost_overrides\.[a-zA-Z0-9]+\b")

FLAG_RE = re.compile(r"--[a-zA-Z][a-zA-Z0-9-]*")

# A flag named only to disclaim it ("...has no --skip-blur flag; that flag
# exists only on `gomoveshift drive`") is not a phantom recommendation — the
# operator is being told the opposite. Exempt tokens matching this pattern
# from the per-subcommand check below so it catches an affirmative "use
# --flag" recommendation without also tripping on a correct denial.
NEGATED_FLAG_RE = re.compile(r"\bno (--[a-zA-Z][a-zA-Z0-9-]*) flag")


def _source_of(mod: ModuleType) -> str:
    path = mod.__file__
    assert path is not None, f"{mod.__name__} has no source file"
    return Path(path).read_text()


# Flags belonging to OTHER programs that these messages legitimately quote.
EXTERNAL_TOOL_FLAGS = {
    "--enable-cuda-llvm",  # ffmpeg ./configure
    "--enable-nvdec",  # ffmpeg ./configure
    "--shutdown",  # wsl.exe
    "--manage",  # wsl.exe
    "--set-sparse",  # wsl.exe
    "--no-deps",  # pip install
}


@pytest.fixture(autouse=True)
def _reset_capability_cache():
    weather_mod.reset_ffmpeg_capability_cache()
    yield
    weather_mod.reset_ffmpeg_capability_cache()


def _decode_gate_message() -> str:
    """Force the CUDA-decode preflight failure and return its text."""
    weather_mod._FFMPEG_CAPABILITY_CACHE["hwaccels"] = set()
    weather_mod._FFMPEG_CAPABILITY_CACHE["decoders"] = set()
    weather_mod._FFMPEG_CAPABILITY_CACHE["probed"] = True

    def fake_run(cmd, **kwargs):
        if cmd and cmd[0] == "ffprobe":
            return MagicMock(stdout="hevc\n", returncode=0)
        return MagicMock(stdout="", stderr="", returncode=0)

    with patch("subprocess.run", side_effect=fake_run):
        with pytest.raises(RuntimeError) as exc:
            check_weather_decode_path([Path("/fake.mp4")], "auto")
    return str(exc.value)


def _degraded_brightness_message(caplog) -> str:
    """Force the below-min-ratio brightness degradation and return its text."""
    detector = WeatherDetector(chunk_max_sec=100.0, min_success_ratio=0.5)
    calls = [0]

    def fake_run(cmd, **kwargs):
        calls[0] += 1
        if calls[0] >= 2:
            raise subprocess.TimeoutExpired("ffmpeg", 30)
        mock = MagicMock()
        mock.stderr = "[Parsed_signalstats_1] YAVG:90.0"
        mock.returncode = 0
        return mock

    with caplog.at_level(logging.ERROR, logger="gomoveshift.pipeline.weather"):
        with patch("subprocess.run", side_effect=fake_run):
            assert detector._get_avg_brightness_ffmpeg(Path("/fake/v.mp4"), 0.0, 300.0) is None
    degraded = [r.message for r in caplog.records if "degraded" in r.message]
    assert degraded, "expected the degraded-brightness ERROR log"
    return degraded[0]


class TestRemediationNamesRealKnobs:
    def test_decode_gate_points_at_config_key_not_phantom_flag(self):
        msg = _decode_gate_message()
        assert "--weather-hwaccel" not in msg, (
            "preflight offers a CLI flag argparse does not accept"
        )
        assert "weather_hwaccel" in msg

    def test_degraded_brightness_points_at_config_key_not_phantom_flag(self, caplog):
        msg = _degraded_brightness_message(caplog)
        assert "--chunk-max-sec" not in msg, "weather offers a CLI flag argparse does not accept"
        assert "weather_chunk_max_sec" in msg


class TestNoPhantomFlagsInTheseModules:
    """Class guard: no ``--flag`` in these two modules may be invented."""

    def test_every_flag_token_is_real(self):
        known = set(FLAG_RE.findall(_source_of(parser_mod))) | EXTERNAL_TOOL_FLAGS
        assert "--output-dir" in known, "flag extraction from parser.py found nothing usable"

        phantom: dict[str, set[str]] = {}
        for mod in (preflight_mod, weather_mod, blur_gate_mod):
            bad = {f for f in FLAG_RE.findall(_source_of(mod)) if f not in known}
            if bad:
                phantom[mod.__name__] = bad
        assert not phantom, f"messages name flags that do not exist: {phantom}"


class TestBlurGateFunctionsMatchTheirCallingSubcommand:
    """The module-wide guard above resolves flags GLOBALLY (any ``--flag`` in
    ``parser.py``, from any subcommand), so it cannot catch a flag that is
    real somewhere but is not registered on the subcommand that actually
    calls the function emitting the message. ``blur_gate.py`` is imported
    from both the `pipeline` and `drive` code paths, but individual
    functions in it are each reached from only one of the two — verified by
    direct caller grep:

      * ``check_mask_ready`` / ``check_applied`` are called only from
        ``pipeline/stages/single_pass_encode.py`` (the `pipeline` subcommand).
      * ``require_master_blurred`` is called only from
        ``pipeline/day_master_encoder.py`` (the `drive` subcommand).

    The two paths differ in which flags exist: ``--skip-blur``
    is registered only on the `drive` subparser (and the `photos` gen/pub/ba
    subcommands), not on `pipeline`, so a `pipeline`-reached message may
    never affirmatively recommend it.
    """

    FUNCTION_SUBCOMMAND = {
        "check_mask_ready": "pipeline",
        "check_applied": "pipeline",
        "require_master_blurred": "drive",
    }

    @staticmethod
    def _flags_for_subcommand(name: str) -> set[str]:
        top = parser_mod._build_parser()
        sub_action = next(a for a in top._actions if isinstance(a, argparse._SubParsersAction))
        cmd_parser = sub_action.choices[name]
        return {
            opt
            for action in cmd_parser._actions
            for opt in action.option_strings
            if opt.startswith("--")
        }

    def test_flags_named_are_registered_on_the_calling_subcommand(self):
        phantom: dict[str, set[str]] = {}
        for func_name, subcommand in self.FUNCTION_SUBCOMMAND.items():
            func = getattr(blur_gate_mod, func_name)
            src = inspect.getsource(func)
            known = self._flags_for_subcommand(subcommand) | EXTERNAL_TOOL_FLAGS
            disclaimed = set(NEGATED_FLAG_RE.findall(src))
            bad = {f for f in FLAG_RE.findall(src) if f not in known} - disclaimed
            if bad:
                phantom[func_name] = bad
        assert not phantom, (
            "blur_gate function(s) recommend flags not registered on the "
            f"subcommand that actually calls them: {phantom}"
        )


class TestNoPhantomConfigFilenames:
    """A message naming a ``*.json``/``*.yaml`` config file must name one the
    code actually reads. ``ost_overrides.py`` is the single source of truth
    for the OST override filename (``load_overrides`` reads only
    ``_OVERRIDE_FILENAME``); a message naming any other spelling (e.g. the
    ``.yaml`` variant that briefly shipped in ``preflight.py``) sends the
    operator to author a file the loader silently ignores."""

    def test_ost_override_filename_tokens_match_the_loader(self):
        tokens: set[str] = set()
        for mod in (preflight_mod, blur_gate_mod):
            tokens |= set(FILENAME_RE.findall(_source_of(mod)))
        assert tokens, "expected at least one ost_overrides.* reference in these modules"
        assert tokens == {_OVERRIDE_FILENAME}, (
            f"message(s) name {tokens - {_OVERRIDE_FILENAME}} but "
            f"ost_overrides.load_overrides only reads {_OVERRIDE_FILENAME!r}"
        )

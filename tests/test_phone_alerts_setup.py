"""``tools/phone_alerts_setup.py`` -- one stdlib-only command that makes the ntfy topic
the pipeline's phone pushes go to. Tmp HOME, no TTY, no network (``urlopen`` is replaced)."""

from __future__ import annotations

import builtins
import json
import re
import stat

import pytest

from tools import phone_alerts_setup as setup

_TOPIC_RE = re.compile(r"^gms-[A-Za-z0-9_-]{24}$")


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("HOME", str(h))
    monkeypatch.delenv("GMS_CONFIG_HOME", raising=False)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)

    def _no_prompt(*_a, **_k):
        raise AssertionError("the setup tool must never prompt (no TTY)")

    monkeypatch.setattr(builtins, "input", _no_prompt)
    return h


def _url_file(home):
    return home / ".config" / "gomoveshift" / "notify_url"


def _topic(home):
    text = _url_file(home).read_text()
    assert text.startswith("https://ntfy.sh/") and text.endswith("\n")
    return text[len("https://ntfy.sh/") :].strip()


def test_no_args_creates_a_private_topic_file_and_prints_the_phone_steps(home, capsys):
    assert setup.main([]) == 0

    path = _url_file(home)
    topic = _topic(home)
    assert _TOPIC_RE.match(topic)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    out = capsys.readouterr().out
    assert topic in out  # printed only to the terminal of whoever ran it
    assert "ntfy" in out and "--test" in out


def test_an_existing_looser_config_dir_is_tightened_to_0700(home):
    parent = _url_file(home).parent
    parent.mkdir(parents=True, mode=0o755)
    parent.chmod(0o755)

    assert setup.main([]) == 0

    assert stat.S_IMODE(parent.stat().st_mode) == 0o700


def test_a_url_file_with_no_topic_is_not_treated_as_a_topic(home, monkeypatch, capsys):
    path = _url_file(home)
    path.parent.mkdir(parents=True)
    path.write_text("https://ntfy.sh/\n")
    posted: list = []
    monkeypatch.setattr(setup.urllib.request, "urlopen", lambda *a, **k: posted.append(a))

    assert setup.main(["--test"]) == 1

    assert posted == []
    assert path.read_text() == "https://ntfy.sh/\n"


def test_second_run_says_it_exists_and_never_overwrites(home, capsys):
    setup.main([])
    before = _url_file(home).read_text()
    capsys.readouterr()

    assert setup.main([]) == 0

    assert _url_file(home).read_text() == before
    out = capsys.readouterr().out
    assert "already" in out.lower() and "--test" in out


def test_rotate_replaces_the_topic_and_keeps_the_file_private(home):
    setup.main([])
    first = _topic(home)

    assert setup.main(["--rotate"]) == 0

    second = _topic(home)
    assert _TOPIC_RE.match(second) and second != first
    assert stat.S_IMODE(_url_file(home).stat().st_mode) == 0o600


def test_test_flag_posts_exactly_one_push_and_never_echoes_the_url(home, monkeypatch, capsys):
    setup.main([])
    topic = _topic(home)
    capsys.readouterr()
    sent: list = []
    monkeypatch.setattr(setup.urllib.request, "urlopen", lambda req, timeout=None: sent.append(req))

    assert setup.main(["--test"]) == 0

    (req,) = sent
    assert req.full_url == "https://ntfy.sh/"
    assert json.loads(req.data)["topic"] == topic
    out = capsys.readouterr().out
    assert "OK" in out
    assert topic not in out and "ntfy.sh" not in out


def test_test_flag_reports_failure_without_echoing_the_url(home, monkeypatch, capsys):
    setup.main([])
    topic = _topic(home)
    capsys.readouterr()

    def _boom(req, timeout=None):
        raise OSError(f"unreachable {req.full_url} {topic}")

    monkeypatch.setattr(setup.urllib.request, "urlopen", _boom)

    assert setup.main(["--test"]) == 1

    out = capsys.readouterr()
    assert "failed" in (out.out + out.err).lower()
    assert topic not in out.out + out.err


def test_test_flag_without_a_topic_file_fails_and_posts_nothing(home, monkeypatch, capsys):
    sent: list = []
    monkeypatch.setattr(setup.urllib.request, "urlopen", lambda *a, **k: sent.append(a))

    assert setup.main(["--test"]) == 1

    assert sent == []
    assert not _url_file(home).exists()


def test_default_url_file_is_under_the_home_config_dir(home):
    assert setup._path() == str(_url_file(home))


def test_gms_config_home_moves_the_url_file(home, monkeypatch, tmp_path):
    custom = tmp_path / "custom-config"
    monkeypatch.setenv("GMS_CONFIG_HOME", str(custom))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))  # GMS_CONFIG_HOME wins
    assert setup._path() == str(custom / "notify_url")
    assert setup.main([]) == 0
    assert (custom / "notify_url").is_file()
    assert not _url_file(home).exists()


def test_xdg_config_home_moves_the_url_file(home, monkeypatch, tmp_path):
    xdg = tmp_path / "xdg"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    assert setup._path() == str(xdg / "gomoveshift" / "notify_url")
    assert setup.main([]) == 0
    assert (xdg / "gomoveshift" / "notify_url").is_file()
    assert not _url_file(home).exists()


@pytest.mark.parametrize("variable", ["GMS_CONFIG_HOME", "XDG_CONFIG_HOME"])
def test_console_reads_the_file_the_setup_tool_wrote(home, monkeypatch, tmp_path, variable):
    """The tool and ``console.notify`` must agree on the notify-url file wherever the
    config home is moved: a topic written where the console never looks is a silent no-op."""
    from gomoveshift.console import notify

    monkeypatch.delenv("GMS_CONSOLE_NOTIFY_URL", raising=False)
    monkeypatch.setenv(variable, str(tmp_path / "moved-config"))
    assert setup.main([]) == 0

    written = open(setup._path(), encoding="utf-8").readline().strip()
    assert written.startswith("https://ntfy.sh/")
    assert notify._notify_url() == written

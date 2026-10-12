"""``gomoveshift metrics report``: wires the previously call-site-less
``PerformanceReporter`` into the CLI. Report-only, always exits 0 -- a
missing project or empty store is the normal case, not a fault."""

from __future__ import annotations

from types import SimpleNamespace

from gomoveshift.cli.handlers.metrics import cmd_metrics


def _args(**kwargs):
    base = dict(metrics_command="report", project=None, all=False, output=None)
    base.update(kwargs)
    return SimpleNamespace(**base)


def test_project_report_prints_reporter_output(monkeypatch, capsys):
    calls = {}

    class _FakeReporter:
        def __init__(self):
            self.store = SimpleNamespace(exists=lambda project: True)

        def generate_project_report(self, project):
            calls["project"] = project
            return "# fake project report"

    monkeypatch.setattr("gomoveshift.tracking.report.PerformanceReporter", lambda: _FakeReporter())

    rc = cmd_metrics(_args(project="demo"), config=SimpleNamespace())

    assert rc == 0
    assert calls["project"] == "demo"
    assert "# fake project report" in capsys.readouterr().out


def test_all_calls_cross_project_report(monkeypatch, capsys):
    calls = {}

    class _FakeReporter:
        def generate_cross_project_report(self):
            calls["called"] = True
            return "# fake cross-project report"

    monkeypatch.setattr("gomoveshift.tracking.report.PerformanceReporter", lambda: _FakeReporter())

    rc = cmd_metrics(_args(all=True), config=SimpleNamespace())

    assert rc == 0
    assert calls["called"] is True
    assert "# fake cross-project report" in capsys.readouterr().out


def test_output_writes_a_file_for_a_project_report(tmp_path, monkeypatch, capsys):
    out_path = tmp_path / "report.md"

    class _FakeReporter:
        def __init__(self):
            self.store = SimpleNamespace(exists=lambda project: True)

        def save_report(self, project, output_path=None):
            output_path.write_text("# saved report", encoding="utf-8")
            return output_path

    monkeypatch.setattr("gomoveshift.tracking.report.PerformanceReporter", lambda: _FakeReporter())

    rc = cmd_metrics(_args(project="demo", output=str(out_path)), config=SimpleNamespace())

    assert rc == 0
    assert out_path.read_text(encoding="utf-8") == "# saved report"
    assert str(out_path) in capsys.readouterr().out


def test_missing_project_prints_one_line_and_exits_zero(monkeypatch, capsys):
    class _FakeReporter:
        def __init__(self):
            self.store = SimpleNamespace(exists=lambda project: False)

    monkeypatch.setattr("gomoveshift.tracking.report.PerformanceReporter", lambda: _FakeReporter())

    rc = cmd_metrics(_args(project="ghost"), config=SimpleNamespace())

    assert rc == 0
    out = capsys.readouterr().out
    assert "no tracking record" in out.lower()


def test_analytics_probe_ok_exits_0(monkeypatch, capsys):
    from gomoveshift.tracking.collector import AnalyticsStatus

    monkeypatch.setattr(
        "gomoveshift.tracking.collector.YouTubeCollector.probe_analytics",
        lambda self: AnalyticsStatus("ok"),
    )
    rc = cmd_metrics(_args(metrics_command="analytics-probe"), config=SimpleNamespace())
    assert rc == 0
    assert capsys.readouterr().out.strip() == "ANALYTICS_OK"


def test_analytics_probe_scope_missing_exits_2(monkeypatch, capsys):
    from gomoveshift.tracking.collector import AnalyticsStatus

    monkeypatch.setattr(
        "gomoveshift.tracking.collector.YouTubeCollector.probe_analytics",
        lambda self: AnalyticsStatus("scope_missing", "x"),
    )
    rc = cmd_metrics(_args(metrics_command="analytics-probe"), config=SimpleNamespace())
    assert rc == 2
    assert "ANALYTICS_SCOPE_MISSING scope=yt-analytics.readonly" in capsys.readouterr().out

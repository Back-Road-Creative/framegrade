"""The SEO & Performance tab shows the performance.

The tab has been called "SEO & Performance" since it was built and has never
shown a single performance number -- only the place-name corrections ledger
and the denylist. The data has existed the whole time: `seo_manager`'s event
store holds every PERFORMANCE_UPDATED event `youtube_sync` writes, and
`AnalyticsTracker` already aggregates it.

What this pins is not "a table appears". It is that a number arrives on the
page WITH the four things that make it readable, which the tracker
carries:

* **what it measures** (`ctr_source`) -- a card click rate is not a thumbnail
  click-through rate, and the page must not let a reader assume it is;
* **its denominator** (`ctr_basis`) -- recomputed from counts we hold, or a
  rate a platform reported;
* **how fresh it is** (`last_updated`);
* **why it is missing**, in words, when it is. An unmeasured rate renders as a
  sentence, never as `0` or `0.0%` -- that is the same lie the tracker
  avoids, and a template is free to reintroduce it.

The reader degrades rather than 500s, twice over: `seo_manager` is an optional
dependency of this service, and a fresh install has no event log at all.
Neither is an error page -- the same shape `schedule_view` uses.
"""

from __future__ import annotations

import builtins
import json
from types import SimpleNamespace

from fastapi.testclient import TestClient

from gomoveshift.config import Config, SEOConfig
from gomoveshift.console import insights
from gomoveshift.console.app import create_app


def _config(tmp_path, site_id="gomoveshift"):
    """Mirrors the real `SEOConfig`: `events_jsonl` is a property there, and the
    reader must use it rather than rebuild the path -- see the test below that
    pins the two against each other."""
    return SimpleNamespace(
        seo=SimpleNamespace(
            storage_path=str(tmp_path),
            site_id=site_id,
            events_jsonl=f"{tmp_path}/{site_id}/events.jsonl",
        )
    )


def _point_at(tmp_path, monkeypatch, site_id="gomoveshift"):
    monkeypatch.setattr(insights.Config, "load", staticmethod(lambda: _config(tmp_path, site_id)))


def _event(content_id, site_id="gomoveshift", **data):
    """One PERFORMANCE_UPDATED event, written the way the store stores it."""
    return {
        "event_id": f"e-{content_id}-{data.get('views', 0)}",
        "event_type": "performance_updated",
        "site_id": site_id,
        "content_id": content_id,
        "timestamp": data.pop("timestamp", "2026-09-10T12:00:00+00:00"),
        "data": data,
    }


def _log(tmp_path, *events, site_id="gomoveshift"):
    """A `<storage_path>/<site_id>/events.jsonl`, as `SEOConfig.events_jsonl`
    resolves it."""
    directory = tmp_path / site_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )


def test_a_measured_rate_arrives_with_what_it_measured(tmp_path, monkeypatch):
    _point_at(tmp_path, monkeypatch)
    _log(
        tmp_path,
        _event("vid-a", views=900, ctr=0.021, ctr_source="youtube_card_click_rate"),
    )

    view = insights.performance_view()

    assert view["hint"] == ""
    row = next(r for r in view["rows"] if r["content_id"] == "vid-a")
    assert row["ctr"] == 0.021
    assert row["ctr_source"] == "youtube_card_click_rate"
    assert row["ctr_basis"] == "reported_rate"
    assert row["last_updated"] == "2026-09-10T12:00:00+00:00"


def test_a_recomputed_rate_is_labelled_as_such(tmp_path, monkeypatch):
    _point_at(tmp_path, monkeypatch)
    _log(tmp_path, _event("vid-a", views=900, clicks=40, impressions=1000))

    row = insights.performance_view()["rows"][0]

    assert row["ctr"] == 0.04
    assert row["ctr_basis"] == "clicks_over_impressions"


def test_an_unmeasured_rate_stays_none_through_the_reader(tmp_path, monkeypatch):
    """The tracker's rule, restated at the console boundary: the reader must not
    helpfully substitute a 0 on the way to the template."""
    _point_at(tmp_path, monkeypatch)
    _log(tmp_path, _event("vid-a", views=900))

    row = insights.performance_view()["rows"][0]

    assert row["ctr"] is None
    assert row["ctr_basis"] is None
    assert row["ctr_source"] == "unknown"


def test_the_busiest_video_is_first(tmp_path, monkeypatch):
    _point_at(tmp_path, monkeypatch)
    _log(
        tmp_path,
        _event("quiet", views=10),
        _event("busy", views=5000),
    )

    rows = insights.performance_view()["rows"]

    assert [row["content_id"] for row in rows] == ["busy", "quiet"]


def test_a_configured_site_id_reads_its_own_ledger_not_gomoveshift(tmp_path, monkeypatch):
    """Waypost is multi-tenant: a console configured for a different brand
    must read that brand's own ledger, not GoMoveShift's -- see
    `SEOConfig.site_id`."""
    _point_at(tmp_path, monkeypatch, site_id="examplebrand")
    _log(
        tmp_path,
        _event("ex-vid", site_id="examplebrand", views=42),
        site_id="examplebrand",
    )

    view = insights.performance_view()

    assert view["hint"] == ""
    assert [row["content_id"] for row in view["rows"]] == ["ex-vid"]


def test_no_event_log_is_an_empty_report_with_a_reason(tmp_path, monkeypatch):
    """A fresh install. Not an error page, and not a silent empty table that
    reads as "every video has zero views"."""
    _point_at(tmp_path, monkeypatch)

    view = insights.performance_view()

    assert view["rows"] == []
    assert "no performance data" in view["hint"].lower()


def test_viewing_the_tab_creates_nothing(tmp_path, monkeypatch):
    """`SEOEventStore.__init__` mkdirs its storage path, so a reader that
    constructs one unconditionally writes to disk just by being looked at."""
    _point_at(tmp_path, monkeypatch)

    insights.performance_view()

    assert list(tmp_path.iterdir()) == []


def test_the_optional_dependency_being_absent_is_a_hint_not_a_crash(tmp_path, monkeypatch):
    """`seo_manager` is an optional dependency of this service -- every other
    caller imports it lazily inside the function for exactly this reason."""
    _point_at(tmp_path, monkeypatch)
    _log(tmp_path, _event("vid-a", views=900))
    real_import = builtins.__import__

    def _no_seo_manager(name, *args, **kwargs):
        if name.startswith("seo_manager"):
            raise ImportError("No module named 'seo_manager'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_seo_manager)

    view = insights.performance_view()

    assert view["rows"] == []
    assert "seo_manager" in view["hint"]


def test_a_corrupt_event_log_is_a_hint_not_a_500(tmp_path, monkeypatch):
    _point_at(tmp_path, monkeypatch)
    directory = tmp_path / "gomoveshift"
    directory.mkdir(parents=True)
    (directory / "events.jsonl").write_text("{not json at all\n", encoding="utf-8")

    view = insights.performance_view()

    assert view["rows"] == []
    assert view["hint"]


def test_the_tab_shows_the_rate_and_says_what_it_is_a_rate_of(tmp_path, monkeypatch):
    monkeypatch.setattr(insights.Config, "load", staticmethod(lambda: _config(tmp_path)))
    _log(
        tmp_path,
        _event("vid-a", views=900, ctr=0.021, ctr_source="youtube_card_click_rate"),
    )

    body = TestClient(create_app()).get("/seo").text

    assert "vid-a" in body
    # The identity has to be ON the page. A bare "2.1%" invites the reader to
    # assume it is the thumbnail CTR that YouTube Studio shows, which this
    # service cannot fetch at all.
    assert "card click rate" in body.lower()
    assert "2.1" in body


def test_an_unmeasured_rate_reads_as_words_on_the_page(tmp_path, monkeypatch):
    """The template is the last place a `None` can become a `0`."""
    monkeypatch.setattr(insights.Config, "load", staticmethod(lambda: _config(tmp_path)))
    _log(tmp_path, _event("vid-a", views=900))

    body = TestClient(create_app()).get("/seo").text

    assert "not measured" in body.lower()
    # Cell-anchored: the base template's own CSS carries `100%`, so a bare
    # substring check would pass or fail for reasons that have nothing to do
    # with this row.
    assert ">0%<" not in body
    assert ">0.0%<" not in body


def test_the_corrections_ledger_still_works_beside_it(tmp_path, monkeypatch):
    """The performance half is additive: the tab's existing editor must not
    have been displaced, and a performance failure must not hide it."""
    monkeypatch.setattr(insights.Config, "load", staticmethod(lambda: _config(tmp_path)))

    body = TestClient(create_app()).get("/seo").text

    assert 'action="/seo/corrections"' in body
    assert "Place-name corrections" in body


def test_the_population_and_cap_are_passed_through(tmp_path, monkeypatch):
    """The tracker computes how many videos had performance data and
    what cap was applied to reach the ten shown. The console must carry both
    through rather than re-deriving the population from ``len(rows)``, which
    would silently state the cap as if it were the population."""
    _point_at(tmp_path, monkeypatch)
    _log(
        tmp_path,
        _event("vid-a", views=900, clicks=40, impressions=1000),
        _event("vid-b", views=10),
    )

    view = insights.performance_view()

    assert view["total_content_with_performance"] == 2
    assert view["top_performers_cap"] == 10


def test_site_wide_totals_are_passed_through_with_their_basis(tmp_path, monkeypatch):
    """The pooled ``performance_metrics`` the tracker already computes across
    every event -- not just the top ten -- must reach the page. Previously
    ``performance_view`` read only ``top_performers`` and discarded the rest
    of the dict."""
    _point_at(tmp_path, monkeypatch)
    _log(
        tmp_path,
        _event("vid-a", views=900, clicks=40, impressions=1000),
    )

    view = insights.performance_view()

    assert view["totals"]["total_views"] == 900
    assert view["totals"]["total_clicks"] == 40
    assert view["totals"]["total_impressions"] == 1000
    assert view["totals"]["ctr"] == 0.04
    assert view["totals"]["ctr_basis_label"] == "computed from counted clicks and impressions"


def test_an_unmeasured_pooled_rate_is_none_not_zero(tmp_path, monkeypatch):
    _point_at(tmp_path, monkeypatch)
    _log(tmp_path, _event("vid-a", views=900))

    view = insights.performance_view()

    assert view["totals"]["ctr"] is None
    assert view["totals"]["ctr_basis_label"] == ""


def test_the_empty_hint_paths_still_carry_the_shape_the_template_needs(tmp_path, monkeypatch):
    """Every ``hint``-returning early path must still return a dict the
    template can render without a KeyError -- the ``_EMPTY``-style discipline
    extends to the two new keys."""
    _point_at(tmp_path, monkeypatch)

    view = insights.performance_view()

    assert view["rows"] == []
    assert view["total_content_with_performance"] is None
    assert view["top_performers_cap"] == insights.TOP_N
    assert view["totals"] is None


def test_the_page_states_the_population_and_shows_totals(tmp_path, monkeypatch):
    monkeypatch.setattr(insights.Config, "load", staticmethod(lambda: _config(tmp_path)))
    _log(
        tmp_path,
        _event("vid-a", views=900, clicks=40, impressions=1000),
        _event("vid-b", views=10),
    )

    body = " ".join(TestClient(create_app()).get("/seo").text.split())

    assert "10 highest-view videos out of 2" in body
    assert "not evidence about that video" in body.lower()


def test_the_console_reads_the_file_the_pipeline_writes(tmp_path):
    """The reader takes the path from `SEOConfig.events_jsonl` instead of
    rebuilding `<storage_path>/gomoveshift/events.jsonl` itself. Two copies of
    that layout is the drift that ends with a tab confidently showing an empty
    table while the pipeline fills a different file."""
    config = Config(seo=SEOConfig(storage_path=str(tmp_path)))

    assert str(insights.events_path(config)) == config.seo.events_jsonl


# A "what we do not know" trust panel. Every line must be derived from
# state `performance_view` already computed -- never static prose that could
# go stale independently of the numbers beside it.


def test_an_unmeasured_row_rate_is_named_in_the_unknowns_list(tmp_path, monkeypatch):
    _point_at(tmp_path, monkeypatch)
    _log(tmp_path, _event("vid-a", views=900))

    view = insights.performance_view()

    assert any("vid-a" in note and "not measured" in note for note in view["unknowns"])
    # The row itself must still show None, never 0 -- the badge is additive,
    # not a replacement for the existing never-invent-a-0 contract.
    assert view["rows"][0]["ctr"] is None


def test_a_row_with_no_declared_source_is_named_in_the_unknowns_list(tmp_path, monkeypatch):
    _point_at(tmp_path, monkeypatch)
    _log(tmp_path, _event("vid-a", views=900, ctr=0.05))

    view = insights.performance_view()

    assert any("vid-a" in note and "source" in note for note in view["unknowns"])


def test_a_row_with_a_declared_source_is_not_in_the_unknowns_list(tmp_path, monkeypatch):
    _point_at(tmp_path, monkeypatch)
    _log(
        tmp_path,
        _event("vid-a", views=900, ctr=0.021, ctr_source="youtube_card_click_rate"),
    )

    view = insights.performance_view()

    assert not any("vid-a" in note for note in view["unknowns"])


def test_an_unmeasured_pooled_rate_is_named_in_the_unknowns_list(tmp_path, monkeypatch):
    _point_at(tmp_path, monkeypatch)
    _log(tmp_path, _event("vid-a", views=900))

    view = insights.performance_view()

    assert any("channel-wide" in note.lower() for note in view["unknowns"])


def test_nothing_unknown_when_every_row_is_fully_measured(tmp_path, monkeypatch):
    """Impressions make the site-wide rate real (the pooled rate never folds
    in an explicit `ctr`), and a declared `ctr_source` alongside it
    means the row's provenance is known too."""
    _point_at(tmp_path, monkeypatch)
    _log(
        tmp_path,
        _event(
            "vid-a",
            views=900,
            clicks=40,
            impressions=1000,
            ctr=0.04,
            ctr_source="youtube_card_click_rate",
        ),
    )

    view = insights.performance_view()

    assert view["unknowns"] == []
    assert "nothing" in view["unknowns_badge"].lower()


def test_the_badge_counts_the_unknowns(tmp_path, monkeypatch):
    _point_at(tmp_path, monkeypatch)
    _log(tmp_path, _event("vid-a", views=900))

    view = insights.performance_view()

    assert str(len(view["unknowns"])) in view["unknowns_badge"]


def test_an_early_return_path_still_carries_the_unknowns_shape(tmp_path, monkeypatch):
    """A fresh install (no event log) must not KeyError the template -- same
    discipline as the other keys in `_EMPTY`."""
    _point_at(tmp_path, monkeypatch)

    view = insights.performance_view()

    assert view["rows"] == []
    assert view["unknowns"] != []
    assert view["unknowns_badge"]


def test_the_badge_and_panel_render_on_the_page(tmp_path, monkeypatch):
    monkeypatch.setattr(insights.Config, "load", staticmethod(lambda: _config(tmp_path)))
    _log(tmp_path, _event("vid-a", views=900))

    body = TestClient(create_app()).get("/seo").text

    assert "what we do not know" in body.lower()
    assert "vid-a" in body
    assert "not measured" in body.lower()


def test_the_nothing_unknown_state_renders_on_the_page(tmp_path, monkeypatch):
    monkeypatch.setattr(insights.Config, "load", staticmethod(lambda: _config(tmp_path)))
    _log(
        tmp_path,
        _event("vid-a", views=900, clicks=40, impressions=1000),
    )

    body = TestClient(create_app()).get("/seo").text

    assert "nothing" in body.lower()


# Data freshness -- first_updated per row, the observation window it
# implies, and the page-level data_as_of/oldest_sample derived from the rows
# actually shown. No staleness threshold or verdict is introduced here (that
# call belongs to the operator, same as the daemon's evidence-age module) -- only "how
# old" and "how long observed", stated in the reader's own
# never-invent-a-0 vocabulary.


def test_a_row_carries_its_first_updated_and_observation_window():
    row = insights._row(
        {
            "content_id": "vid-a",
            "views": 900,
            "first_updated": "2026-09-01T00:00:00+00:00",
            "last_updated": "2026-09-04T00:00:00+00:00",
        }
    )

    assert row["first_updated"] == "2026-09-01T00:00:00+00:00"
    assert row["observation_window_days"] == 3


def test_a_row_with_only_one_timestamp_has_no_observation_window():
    """Never a fabricated 0 -- a missing timestamp means the window is
    unknown, not that the video was observed for zero days."""
    only_last = insights._row(
        {"content_id": "vid-a", "views": 900, "last_updated": "2026-09-04T00:00:00+00:00"}
    )
    only_first = insights._row(
        {"content_id": "vid-a", "views": 900, "first_updated": "2026-09-01T00:00:00+00:00"}
    )

    assert only_last["first_updated"] is None
    assert only_last["observation_window_days"] is None
    assert only_first["observation_window_days"] is None


def test_the_page_states_when_the_data_was_last_updated(tmp_path, monkeypatch):
    monkeypatch.setattr(insights.Config, "load", staticmethod(lambda: _config(tmp_path)))
    _log(
        tmp_path,
        _event("vid-a", views=900, timestamp="2026-09-01T00:00:00+00:00"),
        _event("vid-b", views=10, timestamp="2026-09-20T00:00:00+00:00"),
    )

    view = insights.performance_view()
    body = TestClient(create_app()).get("/seo").text

    assert view["data_as_of"] == "2026-09-20T00:00:00+00:00"
    assert "2026-09-20T00:00:00+00:00" in body


def test_the_page_states_the_oldest_sample_in_the_table(tmp_path, monkeypatch):
    monkeypatch.setattr(insights.Config, "load", staticmethod(lambda: _config(tmp_path)))
    _log(
        tmp_path,
        _event("vid-a", views=900, timestamp="2026-09-01T00:00:00+00:00"),
        _event("vid-b", views=10, timestamp="2026-09-20T00:00:00+00:00"),
    )

    view = insights.performance_view()
    body = TestClient(create_app()).get("/seo").text

    assert view["oldest_sample"] == "2026-09-01T00:00:00+00:00"
    assert "2026-09-01T00:00:00+00:00" in body


def test_no_rows_means_no_freshness_banner_claim(tmp_path, monkeypatch):
    """A fresh install: no row carries a timestamp, so the page must not
    claim a data-as-of date or an oldest sample it does not have."""
    _point_at(tmp_path, monkeypatch)

    view = insights.performance_view()
    assert view["data_as_of"] is None
    assert view["oldest_sample"] is None

    body = TestClient(create_app()).get("/seo").text
    assert "data as of" not in body.lower()

"""Gate A must surface the actually-burned OST text.

Previously ``ReviewGate.gate_a`` printed only the
drive-wide ``--location`` even when the operator passed
``--per-lf-location`` overrides, training operators to confirm a value
that the burn cascade ignores. The OST priority cascade matches the
actual burn path in phase_bc / ost_applier:

  1. ``state["per_lf_overrides"]`` — CLI ``--per-lf-location``
  2. ``state["episode_locations"]`` — GPS-centroid reverse-geocoded
  3. drive-wide ``--location`` — Super LF / non-LF fallback only

The gate display must follow the same cascade.
"""

from __future__ import annotations

from pathlib import Path

from gomoveshift.pipeline.review_gate import ReviewGate

from _state_helpers import make_state


def _state(**extra):
    return make_state(
        episode_files=["/tmp/ep_01.mp4"],
        episode_dates=["2025-09-14"],
        file_durations=[1800.0],
        gps_per_episode=[None],
        **extra,
    )


class TestGateAPerLfOverrideDisplay:
    def test_per_lf_overrides_shown_when_set(self, capsys):
        gate = ReviewGate(interactive=False)
        gate.gate_a(
            state=_state(
                per_lf_overrides={1: "Dingle to Conor Pass, Kerry, Ireland"},
            ),
            output_dir=Path("/tmp"),
            location="Kerry, Ireland",
        )
        out = capsys.readouterr().out
        assert "Per-LF OST overrides" in out
        assert "Dingle to Conor Pass, Kerry, Ireland" in out
        assert "BURNED" in out
        # Drive-wide still printed, but labeled as fallback-only
        assert "Kerry, Ireland" in out
        assert "fallback" in out.lower()

    def test_per_lf_takes_priority_over_episode_locations(self, capsys):
        gate = ReviewGate(interactive=False)
        gate.gate_a(
            state=_state(
                per_lf_overrides={1: "Custom Route"},
                episode_locations=["Geocoded Episode Location"],
            ),
            output_dir=Path("/tmp"),
            location="Drive Wide, Country",
        )
        out = capsys.readouterr().out
        assert "Custom Route" in out
        # Episode locations branch must not print under per-LF priority
        assert "Per-episode OST" not in out

    def test_falls_back_to_episode_locations_when_no_per_lf(self, capsys):
        gate = ReviewGate(interactive=False)
        gate.gate_a(
            state=_state(episode_locations=["Geocoded Loc"]),
            output_dir=Path("/tmp"),
            location="Drive Wide",
        )
        out = capsys.readouterr().out
        assert "Per-episode OST" in out
        assert "Geocoded Loc" in out

    def test_falls_back_to_drive_wide_when_neither_set(self, capsys):
        gate = ReviewGate(interactive=False)
        gate.gate_a(
            state=_state(),
            output_dir=Path("/tmp"),
            location="Drive Wide Only",
        )
        out = capsys.readouterr().out
        assert "Drive Wide Only" in out

    def test_drive_wide_hidden_when_per_lf_covers_all_lfs(self, capsys):
        """When per-LF overrides cover every skeleton LF, drive-wide
        is dead data — show "not used" friendly note instead of the label.
        """
        gate = ReviewGate(interactive=False)
        gate.gate_a(
            state=_state(
                per_lf_overrides={1: "Dingle to Conor Pass, Kerry, Ireland"},
            ),
            output_dir=Path("/tmp"),
            location="Kerry, Ireland",
            skeleton_lf_indices=[1],
        )
        out = capsys.readouterr().out
        assert "Per-LF OST overrides" in out
        assert "Dingle to Conor Pass, Kerry, Ireland" in out
        # Drive-wide label MUST NOT appear in the gate output when fully covered
        assert "Kerry, Ireland" not in out.replace("Dingle to Conor Pass, Kerry, Ireland", "")
        # Friendly "not used" note replaces it
        assert "not used" in out.lower()
        assert "every LF has a per-LF override" in out

    def test_drive_wide_shown_when_per_lf_partial(self, capsys):
        """When per-LF overrides leave at least one skeleton LF
        uncovered, drive-wide remains as the fallback for that LF.
        """
        gate = ReviewGate(interactive=False)
        gate.gate_a(
            state=_state(
                per_lf_overrides={1: "Route A, Kerry, Ireland"},
            ),
            output_dir=Path("/tmp"),
            location="Kerry, Ireland",
            skeleton_lf_indices=[1, 2],
        )
        out = capsys.readouterr().out
        assert "Per-LF OST overrides" in out
        assert "Route A, Kerry, Ireland" in out
        # Drive-wide line shows because LF 2 is uncovered
        assert "Drive-wide --location" in out
        assert "fallback" in out.lower()
        assert "not used" not in out.lower()

    def test_multiple_per_lf_overrides_listed_sorted(self, capsys):
        gate = ReviewGate(interactive=False)
        gate.gate_a(
            state=_state(
                per_lf_overrides={
                    2: "Route B, Kerry, Ireland",
                    1: "Route A, Kerry, Ireland",
                },
            ),
            output_dir=Path("/tmp"),
            location="Kerry, Ireland",
        )
        out = capsys.readouterr().out
        a_pos = out.find("Route A")
        b_pos = out.find("Route B")
        assert a_pos != -1 and b_pos != -1
        assert a_pos < b_pos  # sorted by LF index

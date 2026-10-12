"""Stop-detection config consistency.

`stop_detection.*` is authoritative for the pipeline's actual stop removal;
`dead_spots.*` drives only the standalone `analyze` preview. They must stay in
lockstep or the preview misrepresents what the pipeline cuts. These tests pin
that invariant on the shipped config and verify the load-time drift warning.
"""

import logging
from pathlib import Path

import yaml

from gomoveshift.config import Config

REPO_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"


def test_shipped_config_keeps_stop_knobs_in_sync():
    """The shipped sample config.yaml must keep the two blocks equal — tuning one
    without the other is the silent-drift footgun this guards against."""
    cfg = Config.load(REPO_CONFIG)
    assert cfg.dead_spots.motion_threshold == cfg.stop_detection.motion_threshold
    assert cfg.dead_spots.min_duration_seconds == int(cfg.stop_detection.min_stop_duration)


def test_load_warns_on_operator_drift(tmp_path, caplog):
    """An operator file that diverges the two blocks logs a loud warning."""
    p = tmp_path / "config.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "dead_spots": {"motion_threshold": 0.02, "min_duration_seconds": 30},
                "stop_detection": {"motion_threshold": 0.008, "min_stop_duration": 5.0},
            }
        )
    )
    with caplog.at_level(logging.WARNING, logger="gomoveshift.config"):
        Config.load(p)
    assert "diverge between config blocks" in caplog.text
    assert "motion_threshold" in caplog.text and "min duration" in caplog.text


def test_in_sync_operator_config_does_not_warn(tmp_path, caplog):
    p = tmp_path / "config.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "dead_spots": {"motion_threshold": 0.008, "min_duration_seconds": 5},
                "stop_detection": {"motion_threshold": 0.008, "min_stop_duration": 5.0},
            }
        )
    )
    with caplog.at_level(logging.WARNING, logger="gomoveshift.config"):
        Config.load(p)
    assert "diverge between config blocks" not in caplog.text


def test_bare_default_config_does_not_warn(caplog):
    """Programmatic Config() (defaults legitimately differ) must NOT warn —
    the check only runs in load() for operator-provided files."""
    with caplog.at_level(logging.WARNING, logger="gomoveshift.config"):
        Config()
    assert "diverge between config blocks" not in caplog.text

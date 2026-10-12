#!/usr/bin/env python3
"""The secret scan fails on a planted credential and on a report it cannot trust.

Planted values are assembled at run time from documentation placeholders, so
this file holds no credential-shaped string, and no assertion message prints
one. Run directly: ``python3 ci/tests/test_secret_scan.py``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import secret_scan

PLUGINS = [{"name": "AWSKeyDetector"}]


def repo(files: dict[str, str]) -> Path:
    root = Path(tempfile.mkdtemp(prefix="secret-scan-"))
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    for args in (["init", "-q"], ["add", "-A"]):
        subprocess.run(["git", *args], cwd=root, check=True)
    return root


def placeholder_key() -> str:
    """The vendor's published documentation example, split so it is not a literal."""
    return "".join(["AK", "IA", "IOSFODNN", "7EXAMPLE"])


class Report(unittest.TestCase):
    def test_clean_report_has_no_findings(self):
        self.assertEqual(
            secret_scan.findings({"plugins_used": PLUGINS, "results": {}}), []
        )

    def test_finding_is_reported_without_its_text(self):
        hit = {"type": "AWS Access Key", "line_number": 3, "hashed_secret": "0" * 40}
        got = secret_scan.findings(
            {"plugins_used": PLUGINS, "results": {"a.py": [hit]}}
        )
        self.assertEqual(got, ["a.py:3: AWS Access Key"])

    def test_untrustworthy_reports_are_refused(self):
        bad = [
            [],
            {},
            {"results": {}},
            {"plugins_used": [], "results": {}},
            {"plugins_used": PLUGINS},
            {"plugins_used": PLUGINS, "results": []},
            {"plugins_used": PLUGINS, "results": {"a.py": []}},
            {"plugins_used": PLUGINS, "results": {"a.py": [{"type": "x"}]}},
        ]
        for report in bad:
            with (
                self.subTest(report=report),
                self.assertRaises(secret_scan.MalformedReport),
            ):
                secret_scan.findings(report)


class Scan(unittest.TestCase):
    def test_clean_tree_passes(self):
        self.assertEqual(secret_scan.scan(repo({"a.py": "x = 1\n"})), [])

    def test_placeholder_key_in_source_fails(self):
        got = secret_scan.scan(repo({"pkg/a.py": f'KEY = "{placeholder_key()}"\n'}))
        self.assertEqual([line.split(":")[0] for line in got], ["pkg/a.py"])
        self.assertFalse(any(placeholder_key() in line for line in got))

    def test_placeholder_key_in_a_lock_file_fails(self):
        text = f"somepkg==1.0\n# token {placeholder_key()}\n"
        got = secret_scan.scan(repo({"requirements.lock": text}))
        self.assertEqual([line.split(":")[0] for line in got], ["requirements.lock"])

    def test_a_directory_with_no_tracked_files_is_not_a_pass(self):
        with self.assertRaises(secret_scan.MalformedReport):
            secret_scan.scan(Path(tempfile.mkdtemp(prefix="secret-scan-")))

    def test_missing_scanner_is_an_error_not_a_pass(self):
        old = os.environ.get("DETECT_SECRETS")
        os.environ["DETECT_SECRETS"] = "/nonexistent/detect-secrets"
        try:
            with self.assertRaises(OSError):
                secret_scan.scan(repo({"a.py": "x = 1\n"}))
        finally:
            if old is None:
                del os.environ["DETECT_SECRETS"]
            else:
                os.environ["DETECT_SECRETS"] = old


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""The import smoke fails on a newly broken module and on a listed one that imports.

Run directly: ``python3 ci/tests/test_import_smoke.py``.
"""

from __future__ import annotations

import io
import sys
import tempfile
import textwrap
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import import_smoke

# One package per test run, named so it cannot collide with a real import.
FILES = {
    "smokepkg/__init__.py": "",
    "smokepkg/ok.py": "VALUE = 1\n",
    "smokepkg/broken.py": "import a_module_that_does_not_exist\n",
    "smokepkg/sub/__init__.py": "",
    "smokepkg/sub/leaf.py": "from smokepkg import broken\n",
    "smokepkg/__main__.py": "raise SystemExit('a __main__ is never imported')\n",
}


class Smoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp(prefix="import-smoke-"))
        for name, text in FILES.items():
            (cls.root / name).parent.mkdir(parents=True, exist_ok=True)
            (cls.root / name).write_text(textwrap.dedent(text))
        sys.path.insert(0, str(cls.root))

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(str(cls.root))
        for name in [m for m in sys.modules if m.split(".")[0] == "smokepkg"]:
            del sys.modules[name]

    def run_smoke(self, expected: set[str]) -> tuple[int, list[str]]:
        out = io.StringIO()
        with redirect_stdout(out):
            rc = import_smoke.check("smokepkg", expected)
        return rc, out.getvalue().splitlines()

    def test_every_module_is_listed_except_main(self):
        self.assertEqual(
            import_smoke.modules("smokepkg"),
            [
                "smokepkg",
                "smokepkg.broken",
                "smokepkg.ok",
                "smokepkg.sub",
                "smokepkg.sub.leaf",
            ],
        )

    def test_listed_failures_pass(self):
        rc, lines = self.run_smoke({"smokepkg.broken", "smokepkg.sub.leaf"})
        self.assertEqual(rc, 0, lines)
        self.assertEqual(lines[-1], "IMPORT_SMOKE ok modules=5 expected_failures=2")

    def test_an_unlisted_failure_fails_with_its_error(self):
        rc, lines = self.run_smoke({"smokepkg.broken"})
        self.assertEqual(rc, 1)
        self.assertIn(
            "IMPORT_SMOKE unexpected=smokepkg.sub.leaf "
            "ModuleNotFoundError: No module named 'a_module_that_does_not_exist'",
            lines,
        )

    def test_a_listed_module_that_now_imports_fails_so_the_list_shrinks(self):
        rc, lines = self.run_smoke(
            {"smokepkg.broken", "smokepkg.sub.leaf", "smokepkg.ok"}
        )
        self.assertEqual(rc, 1)
        self.assertIn("IMPORT_SMOKE fixed=smokepkg.ok (delete it from the list)", lines)

    def test_a_listed_name_that_is_not_a_module_fails(self):
        rc, lines = self.run_smoke(
            {"smokepkg.broken", "smokepkg.sub.leaf", "smokepkg.gone"}
        )
        self.assertEqual(rc, 1)
        self.assertIn("IMPORT_SMOKE unknown=smokepkg.gone (no such module)", lines)

    def test_expected_file_ignores_comments_and_blank_lines(self):
        path = self.root / "expected.txt"
        path.write_text("# why\n\nsmokepkg.broken  # fcntl\nsmokepkg.sub.leaf\n")
        self.assertEqual(
            import_smoke.read_expected(path), {"smokepkg.broken", "smokepkg.sub.leaf"}
        )


if __name__ == "__main__":
    unittest.main()

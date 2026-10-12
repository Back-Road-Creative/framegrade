"""No test may patch ``__new__`` on a class it does not own.

CPython keeps a class's ``tp_new`` slot pointed at the Python-level wrapper
after ``__new__`` is set and then removed again, which is exactly what
``monkeypatch.setattr(cls, "__new__", ...)`` and ``patch.object(cls,
"__new__")`` do on teardown. From then on ``cls(key=value)`` raises
``TypeError: object.__new__() takes exactly one argument`` for every later
test in the same process. Patch the name the code under test looks up
instead (the module attribute).
"""

import ast
from pathlib import Path

TESTS_DIR = Path(__file__).parent


def _dunder_new_patches(path: Path) -> list[int]:
    """Return the lines where a call names ``__new__`` as its patch target."""
    lines = []
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.Call):
            continue
        for arg in [*node.args, *(kw.value for kw in node.keywords)]:
            if not (isinstance(arg, ast.Constant) and isinstance(arg.value, str)):
                continue
            if arg.value == "__new__" or arg.value.endswith(".__new__"):
                lines.append(node.lineno)
    return lines


def test_no_test_patches_dunder_new():
    offenders = {
        str(path.relative_to(TESTS_DIR)): lines
        for path in sorted(TESTS_DIR.rglob("*.py"))
        if path != Path(__file__) and (lines := _dunder_new_patches(path))
    }
    assert offenders == {}, f"__new__ patched by name (leaks past the test): {offenders}"

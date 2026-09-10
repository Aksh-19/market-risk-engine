"""
Smoke test.

Purpose: prove that (a) the package installs and imports correctly, and
(b) pytest + CI are wired up correctly, BEFORE we write any real logic.
This is intentionally trivial — every future test file replaces the
"trivial" part, never the "prove the pipes work" part.
"""

from risk_engine import __version__


def test_version_is_a_string():
    assert isinstance(__version__, str)
    assert __version__ != ""

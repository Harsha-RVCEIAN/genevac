"""Smoke test to verify test harness and package discovery in Phase 1."""

import genevac


def test_package_metadata():
    """Verify package version is defined and importable."""
    assert genevac.__version__ == "0.1.0"

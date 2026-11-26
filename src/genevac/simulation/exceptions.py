"""Custom exceptions for the GenEvac-Sentinel simulation engine."""

from __future__ import annotations


class NoRouteError(ValueError):
    """Raised when an assigned target exit is topologically unreachable within the building network."""
    pass

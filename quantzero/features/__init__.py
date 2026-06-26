"""Importing this package registers all built-in feature groups.

Import order is the feature-vector column order. Keep related groups together.
"""

from quantzero.feature import Feature, all_features

# Side-effecting imports: each module's @register calls populate the global registry.
from quantzero.features import flow, price, rangepos, session, volatility, volume  # noqa: F401


def default_features() -> list[type[Feature]]:
    """All registered built-in feature groups."""
    return all_features()


__all__ = ["default_features"]

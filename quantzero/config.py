"""Environment-backed configuration. Loads ``.env`` (gitignored) at import."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class AlpacaConfig:
    key_id: str
    secret_key: str
    data_feed: str  # "sip" or "iex"


def alpaca_config() -> AlpacaConfig:
    """Read Alpaca credentials. Raises KeyError if a required var is missing."""
    return AlpacaConfig(
        key_id=os.environ["ALPACA_KEY_ID"],
        secret_key=os.environ["ALPACA_SECRET_KEY"],
        data_feed=os.environ.get("ALPACA_DATA_FEED", "sip").lower(),
    )


def store_root() -> str:
    return os.environ.get("QZ_STORE_ROOT", "./store")


def raw_root() -> str:
    """Raw market-data store — kept separate from the feature store on purpose."""
    return os.environ.get("QZ_RAW_ROOT", "./raw_store")


def metrics_port() -> int:
    return int(os.environ.get("QZ_METRICS_PORT", "0"))

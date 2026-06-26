"""Time helpers in US Eastern (the trading calendar timezone).

All functions take an integer UTC epoch in nanoseconds. None of them read wall-clock
time, so they behave identically for live and historical events.
"""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
_NS_PER_SECOND = 1_000_000_000

# Regular-trading-hours open/close in minutes since ET midnight.
RTH_OPEN_MINUTE = 9 * 60 + 30  # 09:30
RTH_CLOSE_MINUTE = 16 * 60  # 16:00


def ns_to_et(ts_ns: int) -> dt.datetime:
    """Convert a UTC epoch (ns) to an aware ET datetime."""
    seconds = ts_ns / _NS_PER_SECOND
    return dt.datetime.fromtimestamp(seconds, tz=ET)


def et_session_date(ts_ns: int) -> int:
    """ET calendar date as a proleptic-Gregorian ordinal (cheap to compare/store)."""
    return ns_to_et(ts_ns).date().toordinal()


def et_minute_of_day(ts_ns: int) -> int:
    """Minutes since ET midnight, in [0, 1440)."""
    moment = ns_to_et(ts_ns)
    return moment.hour * 60 + moment.minute


def minutes_since_open(ts_ns: int) -> int:
    """Minutes since the 09:30 ET regular open (negative before the open)."""
    return et_minute_of_day(ts_ns) - RTH_OPEN_MINUTE


def minutes_to_close(ts_ns: int) -> int:
    """Minutes until the 16:00 ET regular close (negative after the close)."""
    return RTH_CLOSE_MINUTE - et_minute_of_day(ts_ns)


def date_to_ns_range(day: dt.date) -> tuple[int, int]:
    """[start, end) UTC epoch-ns bounds covering one ET calendar day."""
    start_et = dt.datetime(day.year, day.month, day.day, tzinfo=ET)
    end_et = start_et + dt.timedelta(days=1)
    start_ns = int(start_et.timestamp()) * _NS_PER_SECOND
    end_ns = int(end_et.timestamp()) * _NS_PER_SECOND
    return start_ns, end_ns


def to_ns(moment: dt.datetime) -> int:
    """Aware datetime -> UTC epoch ns."""
    return int(moment.timestamp() * _NS_PER_SECOND)

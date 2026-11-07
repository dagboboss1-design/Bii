"""Economic-calendar ingestion + causal surprise features.

SCHEMA ASSUMPTION (UNVERIFIED): biquote/Forex-Factory-style JSON. Key names are matched
tolerantly; confirm against a real payload before trusting. null/absent = NOT PUBLISHED = NaN,
never 0.
"""
from __future__ import annotations
import re
from typing import Callable
import numpy as np
import pandas as pd

_KEYS = {"event_id": ("eventId", "id"), "time": ("dateTime", "date", "time", "timestamp"),
         "currency": ("currency", "country"), "name": ("name", "event", "title"),
         "importance": ("importance", "impact"), "actual": ("actual",), "forecast": ("forecast",),
         "previous": ("previous",), "revised_previous": ("revisedPrevious", "revised_previous")}
_IMP = {"high": 3, "medium": 2, "low": 1}
_MULT = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12, "%": 1.0}


def to_num(v) -> float:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    if isinstance(v, (int, float)):
        return float(v)
    m = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*([KMBT%]?)\s*", str(v).replace(",", ""))
    return float(m.group(1)) * _MULT.get(m.group(2), 1.0) if m else np.nan


def _pick(row: dict, key: str):
    for k in _KEYS[key]:
        if k in row:
            return row[k]
    return None


def parse_events(payload: list[dict]) -> pd.DataFrame:
    rows = []
    for r in payload:
        imp = _pick(r, "importance")
        rows.append({
            "event_id": _pick(r, "event_id"), "time_utc": pd.to_datetime(_pick(r, "time"), utc=True),
            "currency": _pick(r, "currency"), "name": _pick(r, "name"),
            "importance": _IMP.get(str(imp).lower(), imp if isinstance(imp, int) else 0),
            **{k: to_num(_pick(r, k)) for k in ("actual", "forecast", "previous", "revised_previous")}})
    df = pd.DataFrame(rows)
    if len(df):
        df["published"] = df["actual"].notna()
    return df.sort_values("time_utc").reset_index(drop=True) if len(df) else df


def fetch_upcoming(get: Callable[[str, dict], list], importance: str = "high",
                   countries: str = "US,EU,GB,JP,AU,CA") -> pd.DataFrame:
    """get(url, params) -> list[dict]. Endpoint per your spec; verify live."""
    return parse_events(get("https://biquote.io/api/calendar/upcoming",
                            {"importance": importance, "countries": countries}))


def add_surprise(ev: pd.DataFrame, min_history: int = 8) -> pd.DataFrame:
    """surprise_z = (actual - forecast) / std of PAST surprises of the same event name."""
    ev = ev.sort_values("time_utc").copy()
    ev["raw_surprise"] = ev["actual"] - ev["forecast"]          # NaN if either missing
    past_std = ev.groupby("name")["raw_surprise"].transform(
        lambda s: s.shift(1).expanding(min_periods=min_history).std())
    ev["surprise_z"] = ev["raw_surprise"] / past_std.replace(0, np.nan)
    return ev


def calendar_features(bar_index: pd.DatetimeIndex, ev: pd.DataFrame, currency: str,
                      min_importance: int = 3, bar: str = "1h",
                      release_lag: str = "1min") -> pd.DataFrame:
    """Per-bar calendar features, knowable at bar CLOSE time.
    - last_surprise_z / mins_since_event: only events whose actual was published by close time
    - mins_to_next_event: SCHEDULED time of next event (schedules are known in advance)"""
    close = pd.DataFrame({"close_time": bar_index + pd.Timedelta(bar)}, index=bar_index)
    e = ev[(ev["currency"] == currency) & (ev["importance"] >= min_importance)]
    past = e[e["actual"].notna()].assign(avail=lambda d: d["time_utc"] + pd.Timedelta(release_lag))
    past = past[["avail", "time_utc", "surprise_z"]].sort_values("avail")
    nxt = e[["time_utc"]].rename(columns={"time_utc": "next_time"}).sort_values("next_time")
    out = pd.merge_asof(close.reset_index(), past, left_on="close_time", right_on="avail", direction="backward")
    out = pd.merge_asof(out, nxt, left_on="close_time", right_on="next_time", direction="forward")
    res = pd.DataFrame({
        "last_surprise_z": out["surprise_z"].to_numpy(),
        "mins_since_event": ((out["close_time"] - out["time_utc"]).dt.total_seconds() / 60).to_numpy(),
        "mins_to_next_event": ((out["next_time"] - out["close_time"]).dt.total_seconds() / 60).to_numpy(),
    }, index=bar_index)
    return res

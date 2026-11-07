"""Data-quality module: UTC normalisation, duplicates, weekend/gap handling, spread estimation.

Expected raw columns: time, bid_o,bid_h,bid_l,bid_c, ask_o,ask_h,ask_l,ask_c, volume
Output index: UTC DatetimeIndex = bar OPEN time. Columns: o,h,l,c (mid), bid_c, ask_c,
spread_pips, volume.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from fxsig.costs import pip_size

REQUIRED = ["time", "bid_o", "bid_h", "bid_l", "bid_c",
            "ask_o", "ask_h", "ask_l", "ask_c", "volume"]


def normalize_utc(s: pd.Series, assume_tz: str | None = None) -> pd.DatetimeIndex:
    """Naive timestamps are rejected unless the source timezone is declared explicitly."""
    s = pd.to_datetime(s)
    if s.dt.tz is None:
        if assume_tz is None:
            raise ValueError("Naive timestamps: pass assume_tz explicitly (never guess).")
        s = s.dt.tz_localize(assume_tz)
    return pd.DatetimeIndex(s.dt.tz_convert("UTC"))


def market_closed(idx: pd.DatetimeIndex) -> np.ndarray:
    """FX weekend: Fri 22:00 UTC -> Sun 22:00 UTC (approximation; DST shifts it by 1h)."""
    wd, hr = idx.weekday, idx.hour
    return np.asarray((wd == 5) | ((wd == 4) & (hr >= 22)) | ((wd == 6) & (hr < 22)))


def detect_gaps(idx: pd.DatetimeIndex, freq: str = "1h") -> pd.DataFrame:
    """Runs of missing bars that fall inside market hours."""
    cols = ["start", "end", "n_bars"]
    if len(idx) == 0:
        return pd.DataFrame(columns=cols)
    full = pd.date_range(idx.min(), idx.max(), freq=freq)
    full = full[~market_closed(full)]
    missing = full.difference(idx)
    if len(missing) == 0:
        return pd.DataFrame(columns=cols)
    step = pd.Timedelta(freq)
    rows, start, prev = [], missing[0], missing[0]
    for t in missing[1:]:
        if t - prev != step:
            rows.append((start, prev, int((prev - start) / step) + 1))
            start = t
        prev = t
    rows.append((start, prev, int((prev - start) / step) + 1))
    return pd.DataFrame(rows, columns=cols)


@dataclass
class QualityReport:
    pair: str
    n_raw: int
    n_duplicates: int
    n_crossed: int
    n_weekend_dropped: int
    n_missing_bars: int
    n_gap_runs: int
    median_spread_pips: float
    max_missing_ratio: float = 0.01

    @property
    def ok(self) -> bool:
        total = max(self.n_raw, 1)
        return self.n_crossed == 0 and (self.n_missing_bars / total) <= self.max_missing_ratio


def clean_candles(raw: pd.DataFrame, pair: str, freq: str = "1h",
                  assume_tz: str | None = None) -> tuple[pd.DataFrame, QualityReport, pd.DataFrame]:
    missing_cols = [c for c in REQUIRED if c not in raw.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    df = raw.copy()
    df.index = normalize_utc(df["time"], assume_tz)
    df = df.drop(columns=["time"]).sort_index(kind="stable")  # stable: "keep last" below must mean last-arrived

    n_raw = len(df)
    dup = df.index.duplicated(keep="last")
    n_dup = int(dup.sum())
    df = df[~dup]

    wk = market_closed(df.index)
    n_wk = int(wk.sum())
    df = df[~wk]

    crossed = (df["ask_c"] < df["bid_c"]).to_numpy()
    n_crossed = int(crossed.sum())
    df = df[~crossed]  # crossed quotes are corrupt; drop and count them

    gaps = detect_gaps(df.index, freq)
    out = pd.DataFrame({
        "o": (df["bid_o"] + df["ask_o"]) / 2, "h": (df["bid_h"] + df["ask_h"]) / 2,
        "l": (df["bid_l"] + df["ask_l"]) / 2, "c": (df["bid_c"] + df["ask_c"]) / 2,
        "bid_c": df["bid_c"], "ask_c": df["ask_c"],
        "spread_pips": (df["ask_c"] - df["bid_c"]) / pip_size(pair),
        "volume": df["volume"],
    })
    out.index.name = "time"
    rep = QualityReport(pair, n_raw, n_dup, n_crossed, n_wk,
                        int(gaps["n_bars"].sum()) if len(gaps) else 0, len(gaps),
                        float(out["spread_pips"].median()) if len(out) else float("nan"))
    return out, rep, gaps


def rolling_spread_estimate(spread_pips: pd.Series, window: int = 24) -> pd.Series:
    """Causal spread estimate: median of PAST bars only (shifted so bar t never sees itself)."""
    return spread_pips.shift(1).rolling(window, min_periods=1).median()

"""Leakage guard. A bar stamped t (open time) is only KNOWN at t + bar_duration."""
from __future__ import annotations
from typing import Callable
import numpy as np
import pandas as pd


class LeakageError(AssertionError):
    pass


def as_of(df: pd.DataFrame, ts: pd.Timestamp, bar: str = "1h") -> pd.DataFrame:
    """Return only bars fully closed at or before `ts`."""
    return df[df.index + pd.Timedelta(bar) <= ts]


def assert_causal(feature_fn: Callable[[pd.DataFrame], pd.Series], df: pd.DataFrame,
                  probes: int = 25, seed: int = 0, atol: float = 1e-9) -> None:
    """Truncation-invariance test: feature at bar t computed on the full history must equal
    the value computed from data available at t only. Any use of future rows breaks this."""
    full = feature_fn(df)
    rng = np.random.default_rng(seed)
    pos = rng.choice(np.arange(50, len(df) - 1), size=min(probes, len(df) - 51), replace=False)
    for i in pos:
        t = df.index[i]
        trunc = feature_fn(df.iloc[: i + 1])
        a, b = full.loc[t], trunc.loc[t]
        # NaN-vs-value mismatch is a leak too (e.g. shift(-1) is NaN on the truncated tail)
        mismatch = (np.isnan(a) != np.isnan(b)) or (not np.isnan(a) and abs(a - b) > atol)
        if mismatch:
            raise LeakageError(f"Feature uses future data at {t}: full={a} truncated={b}")

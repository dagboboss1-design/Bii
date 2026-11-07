"""Technical feature registry.

CONVENTION: row t = information through the CLOSE of bar t (known at t + 1h).
A signal built from row t may only be executed at the OPEN of bar t+1 or later.
Every feature here is registered with a formula, lookback and rationale, and every one is
run through assert_causal() in the tests.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Feature:
    name: str
    fn: Callable[[pd.DataFrame], pd.Series]
    lookback: int
    formula: str
    rationale: str


def rsi(c: pd.Series, n: int = 14) -> pd.Series:
    d = c.diff()
    au = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    ad = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + au / ad)


def true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["c"].shift(1)
    return pd.concat([df["h"] - df["l"], (df["h"] - pc).abs(), (df["l"] - pc).abs()], axis=1).max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up, dn = df["h"].diff(), -df["l"].diff()
    plus = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    a = atr(df, n)
    pdi = 100 * plus.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / a
    mdi = 100 * minus.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi)
    return dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def _macd(c, f=12, s=26, g=9):
    line = c.ewm(span=f, adjust=False, min_periods=s).mean() - c.ewm(span=s, adjust=False, min_periods=s).mean()
    sig = line.ewm(span=g, adjust=False, min_periods=g).mean()
    return line, sig


def _boll(c, n=20, k=2.0):
    m, sd = c.rolling(n).mean(), c.rolling(n).std(ddof=0)
    return m, m + k * sd, m - k * sd


def _lr(d):  # 1-bar log return
    return np.log(d["c"]).diff()


FEATURES: list[Feature] = []


def _reg(name, fn, lookback, formula, rationale):
    FEATURES.append(Feature(name, fn, lookback, formula, rationale))


for _n in (1, 4, 24, 120):
    _reg(f"ret_{_n}", (lambda d, n=_n: np.log(d["c"]).diff(n)), _n,
         f"ln(C_t / C_(t-{_n}))", "Momentum / reversal at multiple horizons (1h, 4h, 1d, 1w).")
_reg("ema_dist_50", lambda d: d["c"] / d["c"].ewm(span=50, adjust=False, min_periods=50).mean() - 1, 50,
     "C/EMA50 - 1", "Distance from medium trend; stretch measure.")
_reg("ema_dist_200", lambda d: d["c"] / d["c"].ewm(span=200, adjust=False, min_periods=200).mean() - 1, 200,
     "C/EMA200 - 1", "Long-trend side (above/below 200 EMA).")
_reg("sma_dist_20", lambda d: d["c"] / d["c"].rolling(20).mean() - 1, 20, "C/SMA20 - 1", "Short mean-reversion stretch.")
_reg("rsi_14", lambda d: rsi(d["c"], 14), 14, "Wilder RSI(14)", "Overbought/oversold exhaustion.")
_reg("macd_line", lambda d: _macd(d["c"])[0], 26, "EMA12 - EMA26", "Trend momentum.")
_reg("macd_hist", lambda d: (lambda l, s: l - s)(*_macd(d["c"])), 35, "MACD - EMA9(MACD)",
     "Momentum acceleration / exhaustion.")
_reg("atr_14", lambda d: atr(d, 14), 14, "Wilder EMA(TR, 14)", "Volatility scale for SL/TP and sizing.")
_reg("atr_pct_14", lambda d: atr(d, 14) / d["c"], 14, "ATR14 / C", "Scale-free volatility; comparable across pairs.")
_reg("boll_pctb", lambda d: (lambda m, u, l: (d["c"] - l) / (u - l))(*_boll(d["c"])), 20,
     "(C - lower)/(upper - lower), 20,2", "Where price sits in the band; touch = <=0 or >=1.")
_reg("boll_width", lambda d: (lambda m, u, l: (u - l) / m)(*_boll(d["c"])), 20, "(upper-lower)/mid",
     "Volatility squeeze / expansion.")
_reg("adx_14", lambda d: adx(d, 14), 28, "Wilder ADX(14)", "Trend strength regardless of direction.")
_reg("rvol_24", lambda d: _lr(d).rolling(24).std(), 25, "std of 1h log returns over 24 bars", "Realised volatility.")
_reg("skew_50", lambda d: _lr(d).rolling(50).skew(), 51, "rolling skew of 1h returns, 50", "Tail asymmetry.")
_reg("kurt_50", lambda d: _lr(d).rolling(50).kurt(), 51, "rolling excess kurtosis, 50", "Fat-tail / jump risk.")
_reg("hour_sin", lambda d: pd.Series(np.sin(2 * np.pi * d.index.hour / 24), index=d.index), 0,
     "sin(2*pi*hour_utc/24)", "Session seasonality (known at bar open, not leaky).")
_reg("hour_cos", lambda d: pd.Series(np.cos(2 * np.pi * d.index.hour / 24), index=d.index), 0,
     "cos(2*pi*hour_utc/24)", "Session seasonality.")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({f.name: f.fn(df) for f in FEATURES}, index=df.index)


def feature_dictionary() -> pd.DataFrame:
    return pd.DataFrame([(f.name, f.formula, f.lookback, f.rationale) for f in FEATURES],
                        columns=["name", "formula", "lookback_bars", "rationale"])

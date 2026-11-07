"""Regime filter: {trending, ranging, high_vol, low_vol} from ADX, ATR percentile, Hurst.

Precedence (documented, because it changes results): high_vol > low_vol > trending > ranging.
ALL thresholds below are tunable = overfitting surface. Evaluate parameter stability, never a
single optimum (Phase 6 harness).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from fxsig.features.technical import adx, atr

CODES = {"trending": 0, "ranging": 1, "high_vol": 2, "low_vol": 3}


@dataclass(frozen=True)
class RegimeConfig:
    adx_trend: float = 25.0
    hurst_trend: float = 0.52
    atr_hi: float = 0.80
    atr_lo: float = 0.20
    pct_window: int = 500
    hurst_window: int = 100
    hurst_lags: tuple = tuple(range(2, 21))


def hurst_vr(close: pd.Series, window: int = 100, lags=range(2, 21)) -> pd.Series:
    """Rolling Hurst via scaling of lagged-difference std: std(logP_t - logP_(t-k)) ~ k^H.
    Random walk ~0.5, persistent fluctuations >0.5, mean-reverting <0.5. Causal (trailing windows).
    LIMITATION: std removes the mean, so a pure steady drift is NOT detected here - that is
    ADX's job. Hence 'trending' requires BOTH ADX and Hurst to agree."""
    lp = np.log(close)
    lags = list(lags)
    y = np.column_stack([np.log(lp.diff(k).rolling(window).std()).to_numpy() for k in lags])
    x = np.log(np.array(lags, dtype=float))
    xc = x - x.mean()
    return pd.Series((y * xc).sum(axis=1) / (xc ** 2).sum(), index=close.index)


def classify(df: pd.DataFrame, cfg: RegimeConfig = RegimeConfig()) -> pd.DataFrame:
    a = adx(df)
    ap = atr(df).rolling(cfg.pct_window).rank(pct=True)  # percentile within TRAILING window
    h = hurst_vr(df["c"], cfg.hurst_window, cfg.hurst_lags)
    ready = a.notna() & ap.notna() & h.notna()
    reg = np.select(
        [~ready, ap >= cfg.atr_hi, ap <= cfg.atr_lo, (a >= cfg.adx_trend) & (h >= cfg.hurst_trend)],
        ["unknown", "high_vol", "low_vol", "trending"], default="ranging")
    out = pd.DataFrame({"adx": a, "atr_pct": ap, "hurst": h, "regime": reg}, index=df.index)
    out["regime_code"] = out["regime"].map(CODES).astype(float)  # NaN for unknown
    return out


ALLOWED = {"mean_reversion": {"ranging"}, "trend": {"trending"}}


@dataclass
class VetoLog:
    records: list = field(default_factory=list)

    def check(self, strategy: str, regime: str, ts, pair: str, oversold_low_vol: bool = False) -> bool:
        ok = regime in ALLOWED[strategy] or (strategy == "mean_reversion" and regime == "low_vol" and oversold_low_vol)
        if not ok:
            self.records.append({"ts": ts, "pair": pair, "strategy": strategy, "regime": regime,
                                 "reason": "regime_veto"})  # also unknown/high_vol -> veto
        return ok

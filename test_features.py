import numpy as np
import pandas as pd
import pytest
from fxsig.data.quality import clean_candles
from fxsig.features.as_of import assert_causal
from fxsig.features.technical import FEATURES, build_features, feature_dictionary, rsi, atr, adx
from fxsig.features.regime import classify, hurst_vr, VetoLog, CODES, RegimeConfig
from tests.conftest import make_raw


def _df(days=90, seed=1):
    return clean_candles(make_raw(n_days=days, seed=seed), "EUR_USD")[0]


def test_every_registered_feature_is_causal():
    df = _df()
    for f in FEATURES:
        assert_causal(f.fn, df, probes=15)


def test_feature_dictionary_complete():
    d = feature_dictionary()
    assert len(d) == len(FEATURES) and d["rationale"].str.len().min() > 5


def test_rsi_extremes():
    up = pd.Series(np.linspace(1, 2, 100))
    assert rsi(up).iloc[-1] == 100
    assert rsi(-up).iloc[-1] == 0


def test_atr_constant_range():
    idx = pd.date_range("2024-01-01", periods=60, freq="1h", tz="UTC")
    df = pd.DataFrame({"o": 1.0, "h": 1.001, "l": 0.999, "c": 1.0}, index=idx)
    assert abs(atr(df).iloc[-1] - 0.002) < 1e-9


def test_adx_trend_beats_noise():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=400, freq="1h", tz="UTC")
    def mk(c):
        return pd.DataFrame({"o": c, "h": c + 0.0005, "l": c - 0.0005, "c": c}, index=idx)
    trend = mk(1.1 + np.arange(400) * 0.0004 + rng.normal(0, 0.0001, 400))
    noise = mk(1.1 + rng.normal(0, 0.0004, 400))
    assert adx(trend).iloc[-1] > 25 > adx(noise).iloc[-1]


def test_hurst_orders_regimes():
    rng = np.random.default_rng(3)
    n = 3000
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    e = rng.normal(0, 0.001, n)
    rw = np.cumsum(e)                                    # random walk
    ou = np.zeros(n); mom = np.zeros(n)
    for i in range(1, n):
        ou[i] = 0.9 * ou[i - 1] + e[i]                   # mean-reverting level
        mom[i] = 0.6 * mom[i - 1] + e[i]                 # persistent returns ...
    mom = np.cumsum(mom)                                 # ... integrated into a trending path
    h = {k: hurst_vr(pd.Series(np.exp(v), index=idx), 500).dropna().mean()
         for k, v in (("rw", rw), ("ou", ou), ("mom", mom))}
    assert 0.40 < h["rw"] < 0.60
    assert h["ou"] < h["rw"] - 0.1 < h["rw"] < h["mom"] - 0.1


def test_regime_causal_and_valid_labels():
    df = _df(days=120)
    r = classify(df, RegimeConfig(pct_window=200))
    assert set(r["regime"].unique()) <= set(CODES) | {"unknown"}
    assert (r["regime"].iloc[:20] == "unknown").all()
    assert_causal(lambda d: classify(d, RegimeConfig(pct_window=200))["regime_code"], df, probes=15)


def test_regime_veto_logging():
    log = VetoLog()
    assert log.check("mean_reversion", "ranging", "t", "EUR_USD")
    assert not log.check("mean_reversion", "trending", "t", "EUR_USD")
    assert not log.check("mean_reversion", "low_vol", "t", "EUR_USD")
    assert log.check("mean_reversion", "low_vol", "t", "EUR_USD", oversold_low_vol=True)
    assert not log.check("trend", "unknown", "t", "EUR_USD")
    assert [r["regime"] for r in log.records] == ["trending", "low_vol", "unknown"]

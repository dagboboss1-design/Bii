import pytest
from fxsig.data.quality import clean_candles
from fxsig.features.as_of import as_of, assert_causal, LeakageError
import pandas as pd


def _df(raw):
    return clean_candles(raw, "EUR_USD")[0]


def test_as_of_excludes_forming_bar(raw):
    df = _df(raw)
    t = pd.Timestamp("2024-01-03 10:30", tz="UTC")
    visible = as_of(df, t)
    assert visible.index.max() == pd.Timestamp("2024-01-03 09:00", tz="UTC")  # 10:00 bar still forming


def test_causal_feature_passes(raw):
    df = _df(raw)
    assert_causal(lambda d: d["c"].rolling(14).mean(), df)
    assert_causal(lambda d: d["c"].pct_change(5), df)


def test_future_shift_is_caught(raw):
    df = _df(raw)
    with pytest.raises(LeakageError):
        assert_causal(lambda d: d["c"].shift(-1), df)


def test_centered_window_is_caught(raw):
    df = _df(raw)
    with pytest.raises(LeakageError):
        assert_causal(lambda d: d["c"].rolling(11, center=True).mean(), df)


def test_full_sample_normalisation_is_caught(raw):
    df = _df(raw)
    with pytest.raises(LeakageError):  # classic z-score using whole-sample mean/std
        assert_causal(lambda d: (d["c"] - d["c"].mean()) / d["c"].std(), df)

import pandas as pd
import pytest
from fxsig.data.quality import clean_candles, detect_gaps, normalize_utc, market_closed, rolling_spread_estimate
from tests.conftest import make_raw


def test_weekend_rows_dropped(raw):
    out, rep, _ = clean_candles(raw, "EUR_USD")
    assert rep.n_weekend_dropped > 0
    assert not market_closed(out.index).any()


def test_duplicates_removed_keep_last(raw):
    dup = pd.concat([raw, raw.iloc[[100]].assign(bid_c=9.0, ask_c=9.1)])
    out, rep, _ = clean_candles(dup, "EUR_USD")
    assert rep.n_duplicates == 1
    assert out.index.is_unique
    assert out.loc[pd.Timestamp(raw["time"].iloc[100]), "bid_c"] == 9.0


def test_gap_detected_inside_market_hours(raw):
    holed = raw.drop(index=range(30, 35))  # Tue 06:00-10:00 gone, market open
    out, rep, gaps = clean_candles(holed, "EUR_USD")
    assert rep.n_gap_runs == 1 and int(gaps["n_bars"].iloc[0]) == 5


def test_weekend_is_not_a_gap(raw):
    _, rep, gaps = clean_candles(raw, "EUR_USD")
    assert rep.n_missing_bars == 0 and len(gaps) == 0


def test_crossed_quotes_dropped(raw):
    bad = raw.copy()
    bad.loc[40, ["bid_c", "ask_c"]] = [1.2, 1.1]
    out, rep, _ = clean_candles(bad, "EUR_USD")
    assert rep.n_crossed == 1 and not rep.ok


def test_naive_timestamps_rejected():
    with pytest.raises(ValueError):
        normalize_utc(pd.Series(pd.to_datetime(["2024-01-02 10:00"])))


def test_timezone_normalised_to_utc():
    s = pd.Series(pd.to_datetime(["2024-01-02 10:00"])).dt.tz_localize("America/New_York")
    assert str(normalize_utc(s)[0]) == "2024-01-02 15:00:00+00:00"


def test_spread_in_pips_and_jpy_pip_size(raw):
    out, _, _ = clean_candles(raw, "EUR_USD")
    assert abs(out["spread_pips"].iloc[0] - 0.8) < 1e-6
    out_j, _, _ = clean_candles(raw, "USD_JPY")
    assert abs(out_j["spread_pips"].iloc[0] - 0.008) < 1e-6  # same raw prices, 0.01 pip


def test_spread_estimate_is_causal(raw):
    out, _, _ = clean_candles(raw, "EUR_USD")
    s = out["spread_pips"].copy()
    s.iloc[200] = 50.0  # spike must NOT influence its own estimate
    est = rolling_spread_estimate(s)
    assert est.iloc[200] < 5 and est.iloc[201] < 5  # median robust, and bar 200 unseen at 200

import numpy as np
import pandas as pd
from fxsig.costs import CostModel
from fxsig.data.calendar import parse_events, add_surprise, calendar_features, to_num


def _payload(n=12, nfuture=1):
    ev = []
    for i in range(n):
        t = pd.Timestamp("2024-01-05 13:30", tz="UTC") + pd.Timedelta(days=30 * i)
        pub = i < n - nfuture
        ev.append({"eventId": f"e{i}", "dateTime": t.isoformat(), "currency": "USD",
                   "name": "NFP", "importance": "high",
                   "actual": f"{180 + 10 * (i % 4)}K" if pub else None,
                   "forecast": "180K", "previous": "175K", "revisedPrevious": None})
    return ev


def test_null_is_nan_not_zero():
    ev = parse_events(_payload())
    last = ev.iloc[-1]
    assert np.isnan(last["actual"]) and not last["published"]
    assert np.isnan(last["revised_previous"])


def test_unit_suffix_parsing():
    assert to_num("254K") == 254000 and to_num("2.5%") == 2.5 and to_num(None) != 0 and np.isnan(to_num("n/a"))


def test_surprise_uses_only_past_and_needs_history():
    ev = add_surprise(parse_events(_payload()))
    assert ev["surprise_z"].iloc[:8].isna().all()           # not enough history yet
    base = ev["surprise_z"].iloc[9]
    p2 = _payload(); p2[10]["actual"] = "999K"             # change a LATER event
    ev2 = add_surprise(parse_events(p2))
    assert ev2["surprise_z"].iloc[9] == base or (np.isnan(base) and np.isnan(ev2["surprise_z"].iloc[9]))


def test_unpublished_event_has_no_surprise():
    ev = add_surprise(parse_events(_payload()))
    assert np.isnan(ev["surprise_z"].iloc[-1])


def test_calendar_features_causal_around_release():
    ev = add_surprise(parse_events(_payload()))
    t_rel = ev.iloc[9]["time_utc"]                       # 13:30 UTC
    idx = pd.date_range(t_rel.floor("h") - pd.Timedelta("3h"), periods=6, freq="1h")
    f = calendar_features(idx, ev, "USD")
    pre = f.loc[t_rel.floor("h") - pd.Timedelta("1h")]   # bar 12:00 closes 13:00, BEFORE the print
    post = f.loc[t_rel.floor("h")]                       # bar 13:00 closes 14:00, after the print
    assert pre["mins_to_next_event"] == 30
    assert pre["last_surprise_z"] == ev.iloc[8]["surprise_z"]      # still the OLD event
    assert post["mins_since_event"] == 30
    assert post["last_surprise_z"] == ev.iloc[9]["surprise_z"] != pre["last_surprise_z"]


def test_pair_cost_override(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("default: {slippage_pips: 0.2, commission_per_lot_roundturn_usd: 3.5, news_spread_multiplier: 3}\n"
                 "spread_floor_pips: {EUR_USD: 0.8}\npairs:\n  EUR_USD: {slippage_pips: 0.5}\n")
    cm = CostModel.from_yaml(p)
    assert abs(cm.round_trip_pips("EUR_USD") - (0.8 + 1.0 + 0.35)) < 1e-9

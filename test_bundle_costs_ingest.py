import pytest
from fxsig.costs import CostModel
from fxsig.data.bundle import DataBundle
from fxsig.data.quality import clean_candles
from fxsig.data.ingest_oanda import parse_candles, fetch_candles


def test_snapshot_id_deterministic_and_sensitive(raw):
    a = clean_candles(raw, "EUR_USD")[0]
    b1, b2 = DataBundle({"EUR_USD": a}), DataBundle({"EUR_USD": a.copy()})
    assert b1.snapshot_id == b2.snapshot_id
    c = a.copy(); c.iloc[5, 0] += 1e-5
    assert DataBundle({"EUR_USD": c}).snapshot_id != b1.snapshot_id


def test_parquet_roundtrip_and_tamper_detection(raw, tmp_path):
    pytest.importorskip("pyarrow")
    b = DataBundle({"EUR_USD": clean_candles(raw, "EUR_USD")[0]})
    path = b.save(tmp_path)
    assert DataBundle.load(path).snapshot_id == b.snapshot_id


def test_cost_model_round_trip():
    cm = CostModel.from_yaml("config/costs.yaml")
    # 0.8 spread + 2*0.2 slippage + 3.5/10 commission = 1.55 pips
    assert abs(cm.round_trip_pips("EUR/USD") - 1.55) < 1e-9
    assert abs(cm.round_trip_pips("EUR_USD", news=True) - (2.4 + 0.4 + 0.35)) < 1e-9
    assert cm.round_trip_pips("EUR_USD", spread_pips=2.0) > cm.round_trip_pips("EUR_USD")


def _c(t, complete=True):
    q = {"o": "1.1", "h": "1.2", "l": "1.0", "c": "1.1"}
    return {"time": t, "complete": complete, "volume": 10, "bid": q, "ask": q}


def test_forming_candle_never_ingested():
    df = parse_candles({"candles": [_c("2024-01-02T10:00:00Z"), _c("2024-01-02T11:00:00Z", False)]})
    assert len(df) == 1


def test_fetch_uses_bid_ask_and_stops():
    calls = []
    def fake_get(url, headers, params):
        calls.append(params)
        return {"candles": [_c("2024-01-02T10:00:00Z")]}
    df = fetch_candles("EUR_USD", "2024-01-01T00:00:00Z", "tok", fake_get)
    assert calls[0]["price"] == "BA" and len(df) == 1 and len(calls) == 1

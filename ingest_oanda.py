"""OANDA v20 candle ingestion (bid+ask). HTTP is injected so it is testable offline.
LIVE ORDER ROUTING IS NOT IN THIS MODULE."""
from __future__ import annotations
from typing import Callable
import pandas as pd

BASE = "https://api-fxpractice.oanda.com"  # practice by default


def parse_candles(payload: dict) -> pd.DataFrame:
    rows = []
    for c in payload.get("candles", []):
        if not c.get("complete", False):
            continue  # never ingest the still-forming bar (look-ahead risk)
        rows.append({
            "time": c["time"], "volume": c["volume"],
            **{f"bid_{k}": float(c["bid"][k]) for k in "ohlc"},
            **{f"ask_{k}": float(c["ask"][k]) for k in "ohlc"},
        })
    return pd.DataFrame(rows)


def fetch_candles(instrument: str, start: str, token: str,
                  get: Callable[[str, dict, dict], dict],
                  granularity: str = "H1", max_pages: int = 100) -> pd.DataFrame:
    """instrument like 'EUR_USD'. `get(url, headers, params) -> json dict`."""
    headers, frames, cursor = {"Authorization": f"Bearer {token}"}, [], start
    for _ in range(max_pages):
        payload = get(f"{BASE}/v3/instruments/{instrument}/candles", headers,
                      {"price": "BA", "granularity": granularity, "from": cursor, "count": 5000})
        df = parse_candles(payload)
        if df.empty:
            break
        frames.append(df)
        if len(payload["candles"]) < 5000:
            break
        cursor = df["time"].iloc[-1]
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return out.drop_duplicates("time", keep="last") if len(out) else out

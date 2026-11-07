import numpy as np
import pandas as pd
import pytest


def make_raw(n_days=21, pair="EUR_USD", seed=1, start="2024-01-01 00:00"):
    """Synthetic H1 bid/ask candles, UTC, including weekend rows (to be dropped)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n_days * 24, freq="1h", tz="UTC")
    mid = 1.10 + np.cumsum(rng.normal(0, 0.0004, len(idx)))
    half = 0.00004
    df = pd.DataFrame({"time": idx})
    for k, off in (("o", 0.0), ("h", 0.0003), ("l", -0.0003), ("c", 0.0)):
        df[f"bid_{k}"] = mid + off - half
        df[f"ask_{k}"] = mid + off + half
    df["volume"] = rng.integers(100, 1000, len(idx))
    return df


@pytest.fixture
def raw():
    return make_raw()

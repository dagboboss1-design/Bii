"""Cost model. Every backtest/paper/live P&L must go through this - no gross results."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml

PIP_VALUE_PER_LOT_USD = 10.0  # approx for USD-quoted pairs; replace with exact conversion in the risk module


def pip_size(pair: str) -> float:
    return 0.01 if pair.replace("/", "_").endswith("JPY") else 0.0001


@dataclass(frozen=True)
class CostModel:
    slippage_pips: float
    commission_per_lot: float
    news_spread_multiplier: float
    spread_floor: dict
    pair_overrides: dict = None  # per-pair {slippage_pips, commission_per_lot_roundturn_usd}

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CostModel":
        cfg = yaml.safe_load(Path(path).read_text())
        d = cfg["default"]
        return cls(d["slippage_pips"], d["commission_per_lot_roundturn_usd"],
                   d["news_spread_multiplier"], cfg.get("spread_floor_pips", {}), cfg.get("pairs") or {})

    def round_trip_pips(self, pair: str, spread_pips: float | None = None,
                        news: bool = False) -> float:
        """Spread is paid once per round trip; slippage on entry and exit; commission in pips."""
        key = pair.replace("/", "_")
        spread = spread_pips if spread_pips is not None else self.spread_floor[key]
        if news:
            spread *= self.news_spread_multiplier
        ov = (self.pair_overrides or {}).get(key, {})
        slip = ov.get("slippage_pips", self.slippage_pips)
        comm = ov.get("commission_per_lot_roundturn_usd", self.commission_per_lot)
        return spread + 2 * slip + comm / PIP_VALUE_PER_LOT_USD

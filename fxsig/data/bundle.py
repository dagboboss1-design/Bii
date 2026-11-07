"""DataBundle: immutable, content-hashed snapshot so every backtest is reproducible."""
from __future__ import annotations
import hashlib, json
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd

SCHEMA_VERSION = "1"


def _hash_frames(frames: dict[str, pd.DataFrame]) -> str:
    h = hashlib.sha256(SCHEMA_VERSION.encode())
    for name in sorted(frames):
        h.update(name.encode())
        h.update(pd.util.hash_pandas_object(frames[name], index=True).values.tobytes())
    return h.hexdigest()[:16]


@dataclass(frozen=True)
class DataBundle:
    candles: dict[str, pd.DataFrame]                       # pair -> cleaned H1 frame
    calendar: pd.DataFrame | None = None                   # economic events (UTC)
    meta: dict = field(default_factory=dict)

    @property
    def snapshot_id(self) -> str:
        frames = dict(self.candles)
        if self.calendar is not None:
            frames["__calendar__"] = self.calendar
        return _hash_frames(frames)

    def save(self, root: str | Path) -> Path:
        d = Path(root) / self.snapshot_id
        d.mkdir(parents=True, exist_ok=True)
        for pair, df in self.candles.items():
            df.to_parquet(d / f"candles_{pair}.parquet")
        if self.calendar is not None:
            self.calendar.to_parquet(d / "calendar.parquet")
        (d / "manifest.json").write_text(json.dumps(
            {"snapshot_id": self.snapshot_id, "schema": SCHEMA_VERSION,
             "pairs": sorted(self.candles), "meta": self.meta}, indent=2))
        return d

    @classmethod
    def load(cls, path: str | Path) -> "DataBundle":
        p = Path(path)
        man = json.loads((p / "manifest.json").read_text())
        candles = {pr: pd.read_parquet(p / f"candles_{pr}.parquet") for pr in man["pairs"]}
        cal_path = p / "calendar.parquet"
        cal = pd.read_parquet(cal_path) if cal_path.exists() else None
        b = cls(candles, cal, man["meta"])
        if b.snapshot_id != man["snapshot_id"]:
            raise ValueError("Snapshot hash mismatch: data changed or corrupted.")
        return b

# fxsig - Phase 1: Data layer

Run: `pip install -e ".[dev]" && pytest -q`

- `fxsig/data/quality.py`   UTC normalisation, dedupe, weekend/gap handling, spread (pips)
- `fxsig/data/bundle.py`    DataBundle with content-hash snapshot_id + Parquet save/load
- `fxsig/data/ingest_oanda.py` OANDA v20 bid/ask H1 candles (practice endpoint, complete bars only)
- `fxsig/features/as_of.py` as_of() guard + assert_causal() truncation-invariance leakage test
- `fxsig/costs.py` + `config/costs.yaml` cost model (ASSUMED numbers - replace with your broker's)

## Known limitations
- Weekend window is fixed at Fri 22:00 - Sun 22:00 UTC (no DST handling, no holiday calendar yet).
- Calendar/news ingestion, tick data and COT are not built yet.
- Parquet round-trip test skips if pyarrow is missing.
- PIP_VALUE_PER_LOT_USD=10 is only right for USD-quoted pairs; the risk module will fix this.

## Phase 2
- `fxsig/features/technical.py` feature registry; `FEATURE_DICTIONARY.md` documents all of them
- `fxsig/features/regime.py` regime classifier + VetoLog
- `fxsig/data/calendar.py` calendar parsing, surprise z-score, per-bar calendar features
- Fill `pairs:` in `config/costs.yaml` with your broker's real costs

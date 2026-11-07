# Phase 2 notes: features, regime, calendar

## Built
- 19 technical features (registry with formula/lookback/rationale), each tested with assert_causal().
- Regime filter (ADX + trailing ATR percentile + Hurst) with veto log.
- Calendar parser (null != 0), causal surprise z-score, per-bar calendar features.
- Per-pair cost overrides in config/costs.yaml (`pairs:` section, to be filled with real broker numbers).

## Overfitting surfaces to watch
- Regime thresholds (adx_trend 25, hurst_trend 0.52, atr_hi 0.8, atr_lo 0.2, windows) are all tunable.
  Validate with stability surfaces, do not pick one optimum.
- Regime precedence order changes results; it is fixed and documented, do not tune it.

## Known limitations
- Hurst estimator ignores pure drift (ADX covers it); Hurst on 100 bars is noisy.
- Calendar JSON schema is an UNVERIFIED assumption; confirm against a real biquote payload.
- Unit suffix (K/M/B) is stripped consistently for actual and forecast; forecast revisions are not tracked,
  so historical surprise may differ from what was visible in real time.
- Not built yet: microstructure, cross-asset, COT, retail sentiment, FinBERT (deliberately after baseline).
- Tests ran via a stand-in runner (no network); run `pytest -q` locally to confirm.

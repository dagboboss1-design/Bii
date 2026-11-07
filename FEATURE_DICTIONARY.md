# Feature dictionary (auto-generated)

Row t = information through the CLOSE of bar t. Execute at open of t+1 or later.

| name | formula | lookback (bars) | rationale |
|---|---|---|---|
| ret_1 | ln(C_t / C_(t-1)) | 1 | Momentum / reversal at multiple horizons (1h, 4h, 1d, 1w). |
| ret_4 | ln(C_t / C_(t-4)) | 4 | Momentum / reversal at multiple horizons (1h, 4h, 1d, 1w). |
| ret_24 | ln(C_t / C_(t-24)) | 24 | Momentum / reversal at multiple horizons (1h, 4h, 1d, 1w). |
| ret_120 | ln(C_t / C_(t-120)) | 120 | Momentum / reversal at multiple horizons (1h, 4h, 1d, 1w). |
| ema_dist_50 | C/EMA50 - 1 | 50 | Distance from medium trend; stretch measure. |
| ema_dist_200 | C/EMA200 - 1 | 200 | Long-trend side (above/below 200 EMA). |
| sma_dist_20 | C/SMA20 - 1 | 20 | Short mean-reversion stretch. |
| rsi_14 | Wilder RSI(14) | 14 | Overbought/oversold exhaustion. |
| macd_line | EMA12 - EMA26 | 26 | Trend momentum. |
| macd_hist | MACD - EMA9(MACD) | 35 | Momentum acceleration / exhaustion. |
| atr_14 | Wilder EMA(TR, 14) | 14 | Volatility scale for SL/TP and sizing. |
| atr_pct_14 | ATR14 / C | 14 | Scale-free volatility; comparable across pairs. |
| boll_pctb | (C - lower)/(upper - lower), 20,2 | 20 | Where price sits in the band; touch = <=0 or >=1. |
| boll_width | (upper-lower)/mid | 20 | Volatility squeeze / expansion. |
| adx_14 | Wilder ADX(14) | 28 | Trend strength regardless of direction. |
| rvol_24 | std of 1h log returns over 24 bars | 25 | Realised volatility. |
| skew_50 | rolling skew of 1h returns, 50 | 51 | Tail asymmetry. |
| kurt_50 | rolling excess kurtosis, 50 | 51 | Fat-tail / jump risk. |
| hour_sin | sin(2*pi*hour_utc/24) | 0 | Session seasonality (known at bar open, not leaky). |
| hour_cos | cos(2*pi*hour_utc/24) | 0 | Session seasonality. |

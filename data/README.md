# Data and execution notes

The Yahoo Finance chart snapshots for SPY, QQQ and TLT were retrieved on 23 October
2026. Analysis uses 3 January 2006–31 December 2025, with 5,031 observations per
fund and 15,093 in total. Unused later source rows are excluded before calculations.
`manifest.json` records snapshot hashes; `SHA256.json` verifies all frozen inputs.
`universe.csv` records the three funds' identities and issuer references.

Source endpoint: `https://query1.finance.yahoo.com/v8/finance/chart/{symbol}`.
The frozen JSON includes OHLC, adjusted close, volume and corporate actions.
The notebook runs offline; a refreshed provider snapshot may revise historical
adjustments and need not reproduce these values.

Dates use America/New_York and XNYS exchange sessions. There are no internal missing
sessions, duplicates, zero volumes or invalid OHLC inequalities in the study period.
No market prices are interpolated, forward-filled or median-imputed.

Adjusted opening is `Open * AdjustedClose / Close`. The vendor's split/distribution
convention is retained without applying splits twice. Adjusted openings approximate
total-return exposure, rather than exact fills or a cash-dividend ledger. These are
revised historical prices, not a verified point-in-time vendor archive.

Only six SPY indicators and QQQ/TLT daily returns are computed. Eligibility requires
twenty complete feature rows and actual SPY prices from t−40 through t+6. The first
eligible origin is 2 March 2006. Fitting uses float32 inputs and targets on every
fifth session, with 797 training observations through 2021. Evaluation has 997 daily
SPY origins with overlapping five-session outcomes. The historical mean is fitted
on each stage's training labels and stays fixed within that stage.

Signals assume completed daily bars before the next opening. The 16:15 New York
forecast timestamp is a processing convention, not a verified historical vendor
publication time. Saved forecasts were reconstructed retrospectively, not archived
contemporaneously. No final-period outcome is used to fit the forecasting model.

Trading is anchored to the first eligible Friday origin, 7 January 2022, then every
fifth exchange session. Execution is at the following opening. The ledger spans
10 January 2022–29 December 2025, with 995 daily opening-to-opening returns per
strategy. Fees are 0.0005 times one-way position change, charged before the interval
return; unchanged positions incur no fee. No terminal liquidation is forced.

Cash uses Federal Reserve H.15 DGS3MO from
[FRED](https://fred.stlouisfed.org/series/DGS3MO). Observation dates are delayed
three calendar days; only previously available rates are carried over non-release
days. Percentage yields are divided by 100 and accrued using actual calendar
days/365 simple interest. This is a historical yield proxy, not an exact Treasury
investment return or a guaranteed broker rate.

CAGR uses 252 sessions/year. Sharpe uses daily returns above the cash proxy and
sample standard deviation, annualized by √252. Drawdown includes initial wealth
and daily opening marks; intraday or closing extremes can be missed. Taxes and
capacity constraints are not modelled. The data are included for reproducibility,
without a claim to waive provider terms or redistribution restrictions.

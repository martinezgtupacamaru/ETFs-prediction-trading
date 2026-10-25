"""Eight SPY/QQQ/TLT predictors, paired HAC inference and SPY/cash accounting."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import exchange_calendars as xc
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent
FEATURES = ['SPY_r1','SPY_r2','SPY_r3','SPY_mom5','SPY_mom20','SPY_vol20','QQQ_r1','TLT_r1']
MODELS = ['Mean','XGBoost']
HAC_LAGS = 20


def verify_inputs():
    for path, expected in json.loads((ROOT/'data/SHA256.json').read_text()).items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == expected, path


def load_prices():
    spec = json.loads((ROOT/'model_spec.json').read_text())
    calendar = xc.get_calendar('XNYS', start='2005-01-01', end='2027-01-01')
    index = calendar.sessions_in_range(spec['start'], spec['end']).tz_localize(None)
    frames = {}
    for symbol in spec['universe']:
        raw = json.loads((ROOT/'data'/f'{symbol}.json').read_text())['chart']['result'][0]
        dates = pd.to_datetime(raw['timestamp'], unit='s', utc=True)
        dates = dates.tz_convert('America/New_York').tz_localize(None).normalize()
        d = pd.DataFrame(raw['indicators']['quote'][0], index=dates)
        d['adjclose'] = raw['indicators']['adjclose'][0]['adjclose']
        assert not d.index.has_duplicates
        d = d.reindex(index)
        d['adjopen'] = d.open*d.adjclose/d.close
        frames[symbol] = d
    return frames


def make_samples(frames):
    """Only the eight predictors; preserve float32 fitting and sample eligibility."""
    d = frames['SPY']
    r = np.log(d.adjclose).diff()
    f = pd.DataFrame(dict(SPY_r1=r, SPY_r2=r.shift(1), SPY_r3=r.shift(2),
                         SPY_mom5=r.rolling(5).sum(), SPY_mom20=r.rolling(20).sum(),
                         SPY_vol20=r.rolling(20).std(ddof=1),
                         QQQ_r1=np.log(frames['QQQ'].adjclose).diff(),
                         TLT_r1=np.log(frames['TLT'].adjclose).diff()), index=d.index)
    y = 100*np.log(d.adjopen.shift(-6)/d.adjopen.shift(-1))
    history = f.notna().all(axis=1).rolling(20).sum().eq(20)
    complete = (d[['adjclose','adjopen']].notna().all(axis=1).astype(float)
                .rolling(47).min().shift(-6).eq(1))
    ok = history & complete & y.notna()
    s = f.loc[ok].astype('float32').copy()
    s['actual'] = y.loc[ok].astype('float32')
    s['origin'] = s.index
    dates = pd.Series(d.index, index=d.index)
    s['entry'] = dates.shift(-1).loc[ok]
    s['exit'] = dates.shift(-6).loc[ok]
    return s.reset_index(drop=True)


def split_samples(samples, train_end, start, end):
    grid = pd.DatetimeIndex(sorted(samples.origin.unique()))[::5]
    train = samples.origin.le(train_end) & samples.exit.lt(start) & samples.origin.isin(grid)
    evaluation = samples.origin.ge(start) & samples.origin.le(end) & samples.exit.le(end)
    return train, evaluation


def metrics(y, p, base):
    # Stored errors are decimal log returns; multiply RMSE/MAE by 100 for display.
    y, p, base = np.asarray(y)/100, np.asarray(p)/100, np.asarray(base)/100
    return dict(n=len(y), RMSE=float(np.sqrt(np.mean((y-p)**2))),
                MAE=float(np.mean(abs(y-p))),
                OOS_R2=float(1-np.sum((y-p)**2)/np.sum((y-base)**2)),
                directional_accuracy=float(np.mean((y>0)==(p>0))))


def newey_west(difference, alternative):
    """One-sided mean inference, Bartlett HAC20 and n/(n−1) correction."""
    if alternative not in ['less','greater']:
        raise ValueError('Alternative must be less or greater')
    series = difference.sort_index()
    dates = pd.DatetimeIndex(series.index)
    calendar = xc.get_calendar('XNYS', start='2005-01-01', end='2027-01-01')
    expected = calendar.sessions_in_range(dates.min(), dates.max()).tz_localize(None)
    if not dates.as_unit('ns').equals(expected.as_unit('ns')):
        raise ValueError('Observations must cover consecutive unique trading sessions')
    values = series.to_numpy(dtype=float)
    n = len(values)
    if not np.isfinite(values).all() or n <= HAC_LAGS+1:
        raise ValueError('Need finite observations and enough sessions for HAC20')
    mean = float(values.mean())
    u = values-mean
    meat = u@u
    for k in range(1, HAC_LAGS+1):
        meat += 2*(1-k/(HAC_LAGS+1))*(u[k:]@u[:-k])
    se = float(np.sqrt(max(0., meat)/(n*(n-1))))
    z = mean/se if se else np.nan
    less = alternative == 'less'
    return dict(n=n, mean_difference=mean, hac_se=se, hac_lags=HAC_LAGS,
                **{'upper_95' if less else 'lower_95': mean+(1 if less else -1)*norm.ppf(.95)*se},
                p_one_sided=float(norm.cdf(z) if less else norm.sf(z)))


def cash_returns(dates):
    rates = pd.read_csv(ROOT/'data/cash_csv.csv', index_col=0, parse_dates=True).iloc[:,0]
    rates = pd.to_numeric(rates, errors='coerce').dropna()
    rates.index += pd.Timedelta(days=3)
    rates = rates.reindex(rates.index.union(dates)).sort_index().ffill().reindex(dates)/100
    assert rates.notna().all()
    days = pd.Series(dates,index=dates).shift(-1)-pd.Series(dates,index=dates)
    return (rates*days.dt.days/365).iloc[:-1].to_numpy()


def trade(prices, forecasts):
    """Binary SPY/cash, five-session schedule, next-opening execution and 5 bp fees."""
    p = forecasts.pivot(index='origin',columns='model',values='prediction').sort_index()
    first = p.index[p.index.day_of_week==4][0]
    positions = np.arange(prices.index.get_loc(first),prices.index.get_loc(p.index[-1])+1,5)
    positions = [i for i in positions if prices.index[i] in p.index and i+6<len(prices)]
    start,end = positions[0]+1,positions[-1]+6
    dates = prices.index[start:end+1]
    cash = cash_returns(dates)
    asset = prices.adjopen.iloc[start+1:end+1].to_numpy()/prices.adjopen.iloc[start:end].to_numpy()-1
    momentum = np.log(prices.adjclose).diff().rolling(20).sum()
    signals = {i+1:i for i in positions}
    rows = []
    for strategy in ['BuyHold','Momentum20','Forecast']:
        wealth,weight = 1.,0.
        for j,i in enumerate(range(start,end)):
            turnover = 0.
            origin = None
            if i==start or (i in signals and strategy!='BuyHold'):
                origin = signals.get(i,positions[0])
                date = prices.index[origin]
                target = (1. if strategy=='BuyHold' else float(momentum.iloc[origin]>0)
                          if strategy=='Momentum20' else float(p.loc[date,'XGBoost']>0))
                turnover = abs(target-weight)
                weight = target
            fee = .0005*turnover
            gross = weight*asset[j]+(1-weight)*cash[j]
            net = (1-fee)*(1+gross)-1
            wealth *= 1+net
            rows.append(dict(date=dates[j+1],entry_date=dates[j],strategy=strategy,
                             origin=prices.index[origin] if origin is not None else pd.NaT,
                             weight=weight,asset_return=asset[j],cash_return=cash[j],gross_return=gross,
                             net_return=net,turnover=turnover,cost=fee,wealth=wealth))
            weight = weight*(1+asset[j])/(1+gross)
    return pd.DataFrame(rows)


def trading_metrics(g):
    r = g.net_return.to_numpy()
    excess = r-g.cash_return.to_numpy()
    nav = np.r_[1.,np.cumprod(1+r)]
    return dict(annual_return=float(nav[-1]**(252/len(r))-1),
                excess_sharpe=float(excess.mean()/excess.std(ddof=1)*np.sqrt(252)),
                max_drawdown=float((nav/np.maximum.accumulate(nav)-1).min()))


def trading_hac(ledger):
    """Paired daily XGBoost minus buy-and-hold, annualized arithmetic mean gap."""
    paired = ledger.loc[ledger.strategy.isin(['Forecast','BuyHold']),['date','strategy','net_return']].copy()
    paired['date'] = pd.to_datetime(paired.date)
    if paired.duplicated(['date','strategy']).any():
        raise ValueError('Duplicate strategy/date observations')
    wide = paired.pivot(index='date',columns='strategy',values='net_return').sort_index()
    result = newey_west(wide.Forecast-wide.BuyHold,'greater')
    return dict(n=result['n'],first_return_date=str(wide.index.min().date()),
                last_return_date=str(wide.index.max().date()),hac_lags=HAC_LAGS,
                mean_daily_difference=result['mean_difference'],daily_hac_se=result['hac_se'],
                annualized_mean_difference=252*result['mean_difference'],
                annualized_hac_se=252*result['hac_se'],annualized_lower_95=252*result['lower_95'],
                p_one_sided=result['p_one_sided'],significant_5pct=result['p_one_sided']<.05,
                status='Retrospective exploratory; hypotheses and methods selected after outcomes were available')

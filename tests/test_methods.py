from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import norm
from xgboost import XGBRegressor

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from methods import *


def test_inputs_and_eight_feature_scope():
    verify_inputs()
    spec=json.loads((ROOT/'model_spec.json').read_text())
    assert spec['universe']==['SPY','QQQ','TLT']
    assert spec['features']==FEATURES and len(FEATURES)==8
    assert set(p.stem for p in (ROOT/'data').glob('*.json'))=={'SPY','QQQ','TLT','manifest','SHA256'}
    frames=load_prices()
    assert sum(d.adjclose.notna().sum() for d in frames.values())==15093
    for d in frames.values():
        assert not d.index.has_duplicates
        assert d[['adjclose','adjopen']].notna().all().all()
    for stage in ['cv1','cv2','cv3','validation','test']:
        f=pd.read_csv(ROOT/'results'/f'{stage}_forecasts.csv')
        assert set(f.model)==set(MODELS)
        assert f.groupby('origin').model.nunique().eq(2).all()


def test_features_target_purge_and_future_invariance():
    frames=load_prices();s=make_samples(frames)
    assert list(s.columns[:8])==FEATURES
    date=pd.Timestamp('2020-01-02')
    row=s.loc[s.origin.eq(date)].iloc[0]
    d=frames['SPY'];i=d.index.get_loc(date);returns=np.log(d.adjclose).diff()
    expected=[returns.iloc[i],returns.iloc[i-1],returns.iloc[i-2],
              returns.iloc[i-4:i+1].sum(),returns.iloc[i-19:i+1].sum(),returns.iloc[i-19:i+1].std(ddof=1),
              np.log(frames['QQQ'].adjclose).diff().iloc[i],np.log(frames['TLT'].adjclose).diff().iloc[i]]
    np.testing.assert_array_equal(row[FEATURES].to_numpy(dtype='float32'),np.asarray(expected,dtype='float32'))
    assert row.actual==np.float32(100*np.log(d.adjopen.iloc[i+6]/d.adjopen.iloc[i+1]))
    changed={name:frame.copy() for name,frame in frames.items()}
    for frame in changed.values():frame.loc['2020-02-01':,'adjclose']*=2
    after=make_samples(changed).loc[lambda x:x.origin.eq(date)]
    np.testing.assert_array_equal(row[FEATURES].to_numpy(dtype=float),after[FEATURES].iloc[0].to_numpy())
    changed['SPY'].loc[date,'adjclose']=np.nan
    assert not make_samples(changed).origin.eq(date).any()
    tr,ev=split_samples(s,'2021-12-31','2022-01-01','2025-12-31')
    assert tr.sum()==797 and ev.sum()==997
    assert s.loc[tr,'exit'].max()<s.loc[ev,'origin'].min()
    assert np.all(np.diff(d.index.get_indexer(s.loc[tr,'origin']))==5)


def test_all_fixed_configuration_fits_reproduce_saved_forecasts():
    spec=json.loads((ROOT/'model_spec.json').read_text());s=make_samples(load_prices())
    stages=[(f'cv{i}',*dates) for i,dates in enumerate(spec['development_folds'],1)]
    stages += [('validation',*spec['validation']),('test',*spec['test'])]
    for stage,train_end,start,end in stages:
        tr,ev=split_samples(s,train_end,start,end);a=s.loc[tr];b=s.loc[ev].sort_values('origin')
        fit=XGBRegressor(**spec['xgboost']).fit(a[FEATURES].to_numpy(dtype='float32'),a.actual.to_numpy(dtype='float32'))
        f=pd.read_csv(ROOT/'results'/f'{stage}_forecasts.csv',parse_dates=['origin'],float_precision='round_trip')
        w=f.pivot(index='origin',columns='model',values='prediction').sort_index()
        np.testing.assert_array_equal(w.index.to_numpy(),b.origin.to_numpy())
        np.testing.assert_array_equal(fit.predict(b[FEATURES].to_numpy(dtype='float32')),w.XGBoost)
        np.testing.assert_array_equal(np.repeat(a.actual.mean(),len(b)),w.Mean)


def test_forecast_metrics_and_saved_loss_differences():
    f=pd.read_csv(ROOT/'results/test_forecasts.csv',parse_dates=['origin'],float_precision='round_trip')
    w=f.pivot(index='origin',columns='model',values='prediction').sort_index()
    y=f.loc[f.model.eq('XGBoost')].set_index('origin').actual.reindex(w.index)
    saved=pd.read_csv(ROOT/'results/forecast_metrics.csv',index_col='model',float_precision='round_trip')
    for model in MODELS:
        q=metrics(y,w[model],w.Mean)
        for key,value in q.items():np.testing.assert_allclose(value,saved.loc[model,key],rtol=1e-12,atol=1e-16)
    loss=pd.read_csv(ROOT/'results/loss_differences.csv',index_col=0,parse_dates=True,float_precision='round_trip')
    np.testing.assert_allclose(loss.XGBoost,(y-w.XGBoost)**2/10000-(y-w.Mean)**2/10000,rtol=1e-10,atol=1e-18)


def test_realized_trading_accounting_execution_and_metrics():
    frames=load_prices();f=pd.read_csv(ROOT/'results/test_forecasts.csv',parse_dates=['origin','entry','exit'],float_precision='round_trip')
    ledger=trade(frames['SPY'],f)
    saved=pd.read_csv(ROOT/'results/trading_ledger.csv',parse_dates=['date','entry_date','origin'],float_precision='round_trip')
    saved=saved.astype({c:'datetime64[ns]' for c in ['date','entry_date','origin']})
    pd.testing.assert_frame_equal(ledger,saved,check_dtype=False,check_exact=False,rtol=1e-12,atol=1e-14)
    expected={'BuyHold':(.12104166362342639,.484470798824668,-.25188350311541596),
              'Momentum20':(.12970828715945304,.7622781539996938,-.10053335231224081),
              'Forecast':(.1767872139154696,.8058065754332251,-.1727841318875416)}
    for name,g in ledger.groupby('strategy'):
        wealth=1.
        for row in g.itertuples():
            wealth*=(1-.0005*row.turnover)*(1+row.weight*row.asset_return+(1-row.weight)*row.cash_return)
            np.testing.assert_allclose(wealth,row.wealth,rtol=1e-12)
        assert len(g)==995 and g.weight.isin([0,1]).all()
        assert g.iloc[0].cost==.0005
        assert (g.loc[g.origin.notna(),'origin']<g.loc[g.origin.notna(),'entry_date']).all()
        assert g.loc[g.turnover.eq(0),'cost'].eq(0).all()
        np.testing.assert_allclose(list(trading_metrics(g).values()),expected[name],atol=1e-12)


def test_one_sided_hac_forecasting_and_trading():
    def matrix_se(values):
        x=np.asarray(values);n=len(x);u=x-x.mean()
        distance=np.abs(np.arange(n)[:,None]-np.arange(n)[None,:])
        kernel=np.maximum(0,1-distance/21)
        return np.sqrt((u@kernel@u)/(n*(n-1)))
    loss=pd.read_csv(ROOT/'results/loss_differences.csv',index_col=0,parse_dates=True,float_precision='round_trip').XGBoost
    result=newey_west(loss,'less');se=matrix_se(loss)
    np.testing.assert_allclose(result['hac_se'],se,rtol=1e-12)
    np.testing.assert_allclose(result['upper_95'],loss.mean()+norm.ppf(.95)*se,rtol=1e-12)
    np.testing.assert_allclose(result['p_one_sided'],.6434149813810807,rtol=1e-12)
    ledger=pd.read_csv(ROOT/'results/trading_ledger.csv',float_precision='round_trip')
    result=trading_hac(ledger)
    wide=ledger.pivot(index='date',columns='strategy',values='net_return').sort_index()
    difference=wide.Forecast-wide.BuyHold;se=matrix_se(difference)
    np.testing.assert_allclose(result['daily_hac_se'],se,rtol=1e-12)
    np.testing.assert_allclose(result['annualized_lower_95'],252*(difference.mean()-norm.ppf(.95)*se),rtol=1e-12)
    np.testing.assert_allclose(result['p_one_sided'],.13221918331865,rtol=1e-12)
    assert not result['significant_5pct'] and result['annualized_lower_95']<0

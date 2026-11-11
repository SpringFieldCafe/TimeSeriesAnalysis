"""课程考查：严格时序回测、月度长期预测和小时油温多步预测。

使用本机 Anaconda base。所有输入、输出路径相对于本脚本。
每个历史起点只使用当时已知的信息；最终检验不参与配置选择。
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, "2")
from pathlib import Path
import json
import hashlib
import platform
import time
import warnings
import argparse
import shutil

import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm
import statsmodels
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose, STL, MSTL
from statsmodels.tsa.stattools import acf, adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn import __version__ as sklearn_version
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
RESULT = ROOT / "results"
DATA = ROOT / "data"
SEED = 20261029
ALPHA = .05
FIT_LOG: list[dict] = []
MONTH_ORIGINS = [96, 108, 120, 132]
OIL_ORIGINS = [pd.Timestamp(f"2017-{m:02d}-01") + pd.offsets.MonthEnd(0)
               + pd.Timedelta(hours=23) for m in range(1, 12)]
OIL_TEST_ORIGINS = pd.date_range("2017-12-31 23:00", periods=8, freq="7D")
LAGS = [1, 2, 3, 24, 25, 48, 168]
HGB = dict(max_iter=100, max_leaf_nodes=15, learning_rate=.06,
           min_samples_leaf=35, l2_regularization=10., early_stopping=False,
           random_state=SEED)


def plain(x):
    if isinstance(x, dict): return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)): return [plain(v) for v in x]
    if isinstance(x, (np.ndarray, pd.Series, pd.Index)): return plain(x.tolist())
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating,)): return float(x) if np.isfinite(x) else None
    if isinstance(x, (np.bool_,)): return bool(x)
    if isinstance(x, pd.Timestamp): return x.isoformat()
    return x


def save_json(obj, name):
    (RESULT / name).write_text(json.dumps(plain(obj), ensure_ascii=False, indent=2,
                                         allow_nan=False), encoding="utf-8", newline='\n')


def csv(df, name):
    df.to_csv(RESULT / name, index=False, encoding="utf-8-sig", float_format="%.10g",lineterminator='\n')


def score(y, pred, lo, hi):
    y, pred, lo, hi = map(np.asarray, (y, pred, lo, hi))
    error = pred-y
    interval_score = hi-lo + 2/ALPHA * (np.maximum(lo-y, 0)+np.maximum(y-hi, 0))
    return dict(rmse=float(np.sqrt(np.mean(error**2))), mae=float(np.mean(abs(error))),
                bias=float(np.mean(error)), coverage=float(np.mean((y >= lo)&(y <= hi))),
                width=float(np.mean(hi-lo)), interval_score=float(np.mean(interval_score)))


def records(actual, fc, model, phase, origin):
    return pd.DataFrame(dict(time=actual.index, actual=actual.to_numpy(),
        forecast=fc[0], lower_95=fc[1], upper_95=fc[2], model=model,
        phase=phase, origin=origin, horizon=np.arange(1, len(actual)+1)))


def summary(predictions):
    rows=[]
    for model, g in predictions.groupby("model", sort=False):
        s=score(g.actual, g.forecast, g.lower_95, g.upper_95)
        fold_rmses=g.groupby("origin").apply(
            lambda z: np.sqrt(np.mean((z.forecast-z.actual)**2)), include_groups=False)
        rows.append(dict(model=model, **s, folds=g.origin.nunique(),
                         fold_rmse_std=float(fold_rmses.std(ddof=0))))
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)


def choose(table):
    # 预先固定：RMSE在最佳值1%内，优先区间得分，再看跨起点稳定性。
    near=table[table.rmse <= table.rmse.min()*1.01]
    return str(near.sort_values(["interval_score", "fold_rmse_std", "mae"]).iloc[0].model)


def diagnostics(y, name, lags):
    y=np.asarray(y, float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adf=adfuller(y, maxlag=min(24, max(1, len(y)//10)), autolag="AIC")
        kp=kpss(y, regression="c", nlags="auto")
    lb=acorr_ljungbox(y, lags=lags, return_df=True).reset_index(names="lag")
    lb.insert(0, "series", name)
    return dict(series=name,n=len(y),adf_stat=adf[0],adf_p=adf[1],adf_lag=adf[2],
                kpss_stat=kp[0],kpss_p_boundary=kp[1],kpss_lags=kp[2],
                ljung_box=lb.to_dict("records"))


def prepare():
    DATA.mkdir(exist_ok=True); RESULT.mkdir(exist_ok=True)
    if not (ROOT/'文件1.xlsx').exists() or not (ROOT/'文件2.csv').exists():
        monthly=pd.read_csv(DATA/'monthly.csv',parse_dates=['time']).set_index('time').asfreq('MS')
        train=pd.read_csv(DATA/'oil_training.csv',parse_dates=['time'],float_precision='round_trip').set_index('time').asfreq('h')
        assert len(monthly)==188 and len(train)==14592 and train.index[-1]==pd.Timestamp('2018-02-28 23:00')
        print('Using committed pure-data snapshots; original metadata-bearing files are absent.',flush=True)
        return monthly.x,train
    # 首次运行即保护原件；后续重复运行不覆盖最初备份与校验基准。
    for folder in ['qa','backup','report']:(ROOT/folder).mkdir(exist_ok=True)
    originals=[p for p in ROOT.iterdir() if p.is_file() and not p.name.startswith('~$') and p.suffix in ['.doc','.xlsx','.csv']]
    initial={}
    for p in originals:
        initial[p.name]=dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        dest=ROOT/'backup'/p.name
        if not dest.exists():shutil.copy2(p,dest)
    baseline=ROOT/'qa/source_hashes.json'
    if not baseline.exists():baseline.write_text(json.dumps(initial,ensure_ascii=False,indent=2),encoding='utf-8')
    monthly=pd.read_excel(ROOT / "文件1.xlsx", sheet_name=0)
    monthly.columns=["time", "x"]
    monthly.time=pd.to_datetime(monthly.time)
    monthly=monthly.set_index("time").asfreq("MS")
    assert len(monthly)==188 and monthly.x.notna().all() and monthly.index.is_unique
    oil=pd.read_csv(ROOT / "文件2.csv", parse_dates=["date"]).set_index("date")
    oil.index.name="time"
    train=oil.loc[:"2018-02-28 23:00:00"].asfreq("h")
    assert len(train)==14592 and train.notna().all().all() and train.index.is_unique
    # 后续仅这一周作为最终检验；2018-03-08以后的数据不使用。
    test=oil.loc["2018-03-01":"2018-03-07 23:00:00", ["OT"]]
    assert len(test)==168
    monthly.to_csv(DATA/"monthly.csv", encoding="utf-8-sig",lineterminator='\n')
    train.to_csv(DATA/"oil_training.csv", encoding="utf-8-sig",lineterminator='\n')
    test.to_csv(DATA/"oil_final_test.csv", encoding="utf-8-sig",lineterminator='\n')
    audit={p.name:dict(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
           for p in [ROOT/"文件1.xlsx", ROOT/"文件2.csv"]}
    save_json(audit,"data_sources.json")
    return monthly.x, train


def decomposition(monthly, oil):
    for kind in ["additive", "multiplicative"]:
        d=seasonal_decompose(monthly, model=kind, period=12)
        csv(pd.DataFrame(dict(time=monthly.index, observed=monthly, trend=d.trend,
                             seasonal=d.seasonal, remainder=d.resid)),f"monthly_{kind}_components.csv")
    rate=monthly/monthly.index.days_in_month
    for name, y in [("total",monthly),("daily_rate",rate)]:
        d=STL(y, period=12, seasonal=13, robust=True).fit()
        csv(pd.DataFrame(dict(time=y.index,observed=y,trend=d.trend,seasonal=d.seasonal,
                             remainder=d.resid)),f"monthly_stl_{name}.csv")
    d=MSTL(oil.OT, periods=(24,168), windows=(15,15), iterate=2,
           stl_kwargs=dict(robust=True,seasonal_jump=2,trend_jump=10,low_pass_jump=10)).fit()
    out=pd.DataFrame(dict(time=oil.index, observed=oil.OT, trend=d.trend,
        daily=d.seasonal.iloc[:,0], weekly=d.seasonal.iloc[:,1], remainder=d.resid))
    csv(out,"oil_mstl_components.csv")
    save_json([diagnostics(monthly,"monthly",[12,24]),
               diagnostics(np.diff(np.log(monthly)),"monthly_log_diff",[12,24]),
               diagnostics(oil.OT.to_numpy()[::24],"oil_daily_sample",[7,14]),
               diagnostics(np.diff(oil.OT),"oil_diff",[24,168])],"diagnostics.json")


def month_specs():
    specs=[dict(name="SeasonalNaive12",family="naive",scale="total")]
    for scale in ["total","daily_rate"]:
        for trend in ["level","linear","damped"]:
            specs.append(dict(name=f"ETS_{scale}_{trend}",family="ets",scale=scale,trend=trend))
        for p,d,q in [(0,1,1),(1,0,0),(1,0,1)]:
            specs.append(dict(name=f"SARIMA_{scale}_{p}{d}{q}_011",family="sarima",
                              scale=scale,order=[p,d,q],seasonal_order=[0,1,1,12]))
    return specs


def monthly_forecast(y, spec, horizon, phase):
    t0=time.perf_counter()
    dates=pd.date_range(y.index[-1]+pd.offsets.MonthBegin(1),periods=horizon,freq="MS")
    logy=np.log(y/(y.index.days_in_month if spec['scale']=="daily_rate" else 1))
    fit=None
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        if spec["family"]=="naive":
            mu=np.resize(logy.to_numpy()[-12:],horizon)
            variance=np.mean((logy.to_numpy()[12:]-logy.to_numpy()[:-12])**2)
            var=variance*(np.arange(horizon)//12+1)
            params={"innovation_variance":variance}
            conv=True
        elif spec["family"]=="ets":
            trend=spec["trend"]
            model=ETSModel(logy,error="add",trend=None if trend=="level" else "add",
                           damped_trend=trend=="damped",seasonal="add",seasonal_periods=12)
            fit=model.fit(disp=False,maxiter=600)
            pred=fit.get_prediction(start=len(y),end=len(y)+horizon-1,method="exact")
            mu=np.asarray(pred.predicted_mean); var=np.asarray(pred.var_pred_mean)
            params=dict(zip(fit.param_names,fit.params)); conv=bool(fit.mle_retvals.get("converged",True))
        else:
            fit=SARIMAX(logy,order=tuple(spec['order']),seasonal_order=tuple(spec['seasonal_order']),
                        trend="n",enforce_stationarity=True,enforce_invertibility=True).fit(disp=False,maxiter=180)
            pred=fit.get_forecast(horizon)
            mu=np.asarray(pred.predicted_mean);var=np.asarray(pred.var_pred_mean)
            params=dict(zip(fit.param_names,fit.params));conv=bool(fit.mle_retvals.get("converged",False))
    factor=dates.days_in_month.to_numpy() if spec['scale']=="daily_rate" else 1
    sd=np.sqrt(np.maximum(var,0))
    fc=(np.exp(mu+var/2)*factor,np.exp(mu-norm.ppf(.975)*sd)*factor,
        np.exp(mu+norm.ppf(.975)*sd)*factor)
    assert all(np.isfinite(a).all() for a in fc)
    FIT_LOG.append(dict(task="monthly",model=spec['name'],phase=phase,train_n=len(y),
        train_end=y.index[-1],max_label_time=y.index[-1],spec=spec,params=params,
        fit_succeeded=conv,optimizer_converged=conv if spec['family']!='naive' else None,
        seconds=time.perf_counter()-t0,warnings=list(dict.fromkeys(str(w.message) for w in ws))))
    return fc,fit,conv


def monthly_analysis(y):
    specs=month_specs(); lookup={s['name']:s for s in specs}; predictions=[]; failures=[]
    for spec in specs:
        one=[]
        try:
            for origin in MONTH_ORIGINS:
                fc,_,converged=monthly_forecast(y.iloc[:origin],spec,24,"validation")
                if not converged: raise RuntimeError("optimizer did not converge")
                one.append(records(y.iloc[origin:origin+24],fc,spec['name'],"validation",y.index[origin-1]))
            predictions.extend(one)
        except Exception as e:
            failures.append(dict(model=spec['name'],error=str(e)))
        print("monthly validation",spec['name'],"ok" if len(one)==len(MONTH_ORIGINS) else "excluded",flush=True)
    val=pd.concat(predictions,ignore_index=True); tab=summary(val); winner=choose(tab)
    csv(val,"monthly_validation_predictions.csv");csv(tab,"monthly_model_comparison.csv")
    # 冻结选择，然后查看最后24个月与长跨度压力检验。
    save_json(dict(selected=winner,rule="RMSE within 1%; interval score, fold stability, MAE",
                   specs=specs,validation_origins=MONTH_ORIGINS,failures=failures),"monthly_selection.json")
    test=[];future=[];stress=[]
    keep=list(dict.fromkeys([winner,"SeasonalNaive12",*tab.model.head(3)]))
    final_fit=None
    for name in keep:
        spec=lookup[name]
        fc,_,_=monthly_forecast(y.iloc[:-24],spec,24,"independent_test")
        test.append(records(y.iloc[-24:],fc,name,"independent_test",y.index[-25]))
        fc,fit,_=monthly_forecast(y,spec,96,"final")
        if name==winner:final_fit=fit
        future.append(pd.DataFrame(dict(time=pd.date_range("1995-09-01",periods=96,freq="MS"),
            horizon=np.arange(1,97),model=name,forecast=fc[0],lower_95=fc[1],upper_95=fc[2])))
        if name in [winner,"SeasonalNaive12"]:
            for origin in [72,84]:
                fc,_,_=monthly_forecast(y.iloc[:origin],spec,96,"long_horizon_stress")
                stress.append(records(y.iloc[origin:origin+96],fc,name,"long_horizon_stress",y.index[origin-1]))
    test=pd.concat(test,ignore_index=True);future=pd.concat(future,ignore_index=True)
    csv(test,"monthly_test_predictions.csv");csv(summary(test),"monthly_test_scores.csv")
    csv(future,"monthly_candidate_forecasts.csv")
    csv(future[future.model==winner],"monthly_forecast.csv")
    csv(pd.concat(stress,ignore_index=True),"monthly_long_stress.csv")
    if final_fit is not None:
        logfit=np.asarray(final_fit.fittedvalues)
        fac=y.index.days_in_month if lookup[winner]['scale']=="daily_rate" else 1
        fitted_median=np.asarray(np.exp(logfit)*fac)
        fitted_median[:13]=np.nan
        csv(pd.DataFrame(dict(time=y.index,actual=y,fitted_median=fitted_median,
                             innovation=np.asarray(final_fit.resid),diagnostic_available=np.arange(len(y))>=13)),"monthly_fitted.csv")
    return dict(selected=winner,validation=tab.to_dict("records"),test=summary(test).to_dict("records"),
                first_forecast=future[future.model==winner].iloc[0].to_dict(),
                last_forecast=future[future.model==winner].iloc[-1].to_dict(),failures=failures)


def fourier(index,weekly):
    t=(pd.DatetimeIndex(index)-pd.Timestamp("2016-07-01"))/pd.Timedelta(hours=1)
    cols=[np.ones(len(index))]
    for period, kmax in [(24,2)]+([(168,2)] if weekly else []):
        for k in range(1,kmax+1):
            cols.extend([np.sin(2*np.pi*k*np.asarray(t)/period),np.cos(2*np.pi*k*np.asarray(t)/period)])
    return np.column_stack(cols)


def oil_specs():
    specs=[dict(name=f"Naive{p}",family="naive",period=p,window=None) for p in [1,24,168]]
    for window in [90,180,None]:
        label=str(window) if window else "all"
        for weekly in [False,True]:
            specs.append(dict(name=f"ARX_{label}_{'dayweek' if weekly else 'day'}",
                              family="arx",window=window,weekly=weekly))
        for loads in [False,True]:
            specs.append(dict(name=f"HGB_{label}_{'loads' if loads else 'OT'}",
                              family="hgb",window=window,loads=loads))
    return specs


def base_features(frame,origins,loads):
    y=frame.OT.to_numpy(); origins=np.asarray(origins); cols=[]; names=[]
    for lag in [0,1,2,6,12,23,47,167]:
        cols.append(y[origins-lag]);names.append(f"OT_lag{lag}")
    for w in [24,168,720]:
        roll=frame.OT.rolling(w)
        for kind in ["mean","std","min","max"]:
            cols.append(getattr(roll,kind)().to_numpy()[origins]);names.append(f"OT_{kind}{w}")
    if loads:
        for c in ["HUFL","HULL","MUFL","MULL","LUFL","LULL"]:
            for lag in [0,23,167]:
                cols.append(frame[c].to_numpy()[origins-lag]);names.append(f"{c}_lag{lag}")
            for w in [24,168]:
                cols.append(frame[c].rolling(w).mean().to_numpy()[origins]);names.append(f"{c}_mean{w}")
    return np.column_stack(cols).astype(np.float32),names


def horizon_features(base,origins,dates):
    h=np.arange(1,169)
    target_dates=pd.DatetimeIndex(np.repeat(np.asarray(dates),168))+pd.to_timedelta(np.tile(h,len(base)),unit="h")
    hour=target_dates.hour.to_numpy(); weekday=target_dates.dayofweek.to_numpy()
    extra=np.column_stack([np.tile(h/168,len(base)),np.sin(2*np.pi*hour/24),
        np.cos(2*np.pi*hour/24),np.sin(2*np.pi*(weekday+hour/24)/7),
        np.cos(2*np.pi*(weekday+hour/24)/7)])
    return np.column_stack([np.repeat(base,168,axis=0),extra]).astype(np.float32)


def oil_forecast(frame,spec,phase):
    started=time.perf_counter();y=frame.OT.to_numpy();n=len(y)
    dates=pd.date_range(frame.index[-1]+pd.Timedelta(hours=1),periods=168,freq="h")
    start=max(0,n-spec['window']*24) if spec['window'] else 0
    params={}; pnames=[]; max_label=frame.index[-1]
    if spec['family']=="naive":
        p=spec['period'];pred=np.resize(y[-p:],168)
        variance=np.mean((y[p:]-y[:-p])**2)
        sd=np.sqrt(variance*(np.arange(168)//p+1));lo=pred-1.96*sd;hi=pred+1.96*sd
        params=dict(period=p,innovation_variance=variance)
    elif spec['family']=="arx":
        rows=np.arange(max(start,168),n)
        X=np.column_stack([fourier(frame.index[rows],spec['weekly']),*[y[rows-l] for l in LAGS]])
        beta=np.linalg.lstsq(X,y[rows],rcond=None)[0]
        errors=y[rows]-X@beta
        variance=float(errors@errors/(len(errors)-X.shape[1]))
        history=list(y);pred=[]
        for i,date in enumerate(dates):
            xx=np.r_[fourier(pd.DatetimeIndex([date]),spec['weekly'])[0],
                     [history[len(history)-lag] for lag in LAGS]]
            v=float(xx@beta);history.append(v);pred.append(v)
        pred=np.asarray(pred)
        phi=dict(zip(LAGS,beta[-len(LAGS):]));psi=np.zeros(168);psi[0]=1
        for i in range(1,168):psi[i]=sum(v*psi[i-l] for l,v in phi.items() if l<=i)
        sd=np.sqrt(variance*np.cumsum(psi**2));lo=pred-1.96*sd;hi=pred+1.96*sd
        params=dict(coefficients=beta,ar_coefficients=phi,innovation_variance=variance,rows=len(rows))
    else:
        # 日末为训练起点；每个起点直接预测h=1..168，所有标签均在本折截止前。
        origins=np.arange(max(start+720,720)-1,n-168,24)
        assert len(origins)>15
        base,names=base_features(frame,origins,spec['loads'])
        X=horizon_features(base,origins,frame.index[origins])
        target=(y[origins[:,None]+np.arange(1,169)]-y[origins,None]).ravel()
        b,_=base_features(frame,[n-1],spec['loads'])
        future=horizon_features(b,[n-1],frame.index[[-1]])
        outputs=[]
        for loss,q in [("squared_error",None),("quantile",.025),("quantile",.975)]:
            model=HistGradientBoostingRegressor(loss=loss,quantile=q,**HGB)
            model.fit(X,target)
            outputs.append(model.predict(future)+y[-1])
        pred=outputs[0];lo=np.minimum(outputs[1],outputs[2]);hi=np.maximum(outputs[1],outputs[2])
        params=dict(**HGB,rows=len(target),origins=len(origins),features=names+
                    ["horizon","hour_sin","hour_cos","week_sin","week_cos"],
                    loss="squared_error + quantile .025/.975",target="OT(t+h)-OT(t)",
                    origin_step=24,first_origin=frame.index[origins[0]],
                    last_origin=frame.index[origins[-1]],quantile_crossings=int(np.sum(outputs[1]>outputs[2])))
        max_label=frame.index[origins[-1]+168]
    assert np.isfinite(np.r_[pred,lo,hi]).all()
    FIT_LOG.append(dict(task="oil",model=spec['name'],phase=phase,train_n=n-start,
        train_start=frame.index[start],train_end=frame.index[-1],max_label_time=max_label,
        spec=spec,params=params,seconds=time.perf_counter()-started,fit_succeeded=True,
        optimizer_converged=None,fit_method={'naive':'closed_form','arx':'least_squares',
        'hgb':'fixed_100_iterations'}[spec['family']],warnings=[]))
    return pred,lo,hi


def oil_analysis(frame):
    specs=oil_specs();lookup={s['name']:s for s in specs};predictions=[]
    for spec in specs:
        for origin in OIL_ORIGINS:
            train=frame.loc[:origin];actual=frame.loc[origin+pd.Timedelta(hours=1):].OT.iloc[:168]
            fc=oil_forecast(train,spec,"validation")
            predictions.append(records(actual,fc,spec['name'],"validation",origin))
        print("oil validation",spec['name'],"completed",flush=True)
        csv(pd.DataFrame(FIT_LOG).drop(columns=['params','spec']),"fit_progress.csv")
        save_json(FIT_LOG,"fit_log.json")
    val=pd.concat(predictions,ignore_index=True);tab=summary(val);winner=choose(tab)
    csv(val,"oil_validation_predictions.csv");csv(tab,"oil_model_comparison.csv")
    # 每个窗口均有OT/负荷配对，直接比较不混淆模型变化和信息增加。
    ar_best=choose(tab[tab.model.str.startswith("ARX")])
    ml_best=choose(tab[tab.model.str.startswith("HGB")])
    paired=ml_best.replace("_loads","_OT") if ml_best.endswith("_loads") else ml_best[:-3]+"_loads"
    frozen=dict(selected=winner,best_statistical=ar_best,best_machine_learning=ml_best,
                matched_machine_learning=paired,specs=specs,
                rule="RMSE within 1%; interval score, fold stability, MAE",
                validation_origins=OIL_ORIGINS,independent_test_origins=OIL_TEST_ORIGINS.tolist())
    save_json(frozen,"oil_selection_frozen.json")
    test=[]
    keep=list(dict.fromkeys([winner,ar_best,ml_best,paired,"Naive1","Naive168"]))
    for name in keep:
        for origin in OIL_TEST_ORIGINS:
            fc=oil_forecast(frame.loc[:origin],lookup[name],"independent_test")
            actual=frame.loc[origin+pd.Timedelta(hours=1):].OT.iloc[:168]
            test.append(records(actual,fc,name,"independent_test",origin))
        print("oil independent test",name,"completed",flush=True)
    test=pd.concat(test,ignore_index=True)
    csv(test,"oil_test_predictions.csv");csv(summary(test),"oil_test_scores.csv")
    forecasts=[]
    for name in keep:
        fc=oil_forecast(frame,lookup[name],"final")
        forecasts.append(pd.DataFrame(dict(time=pd.date_range("2018-03-01",periods=168,freq="h"),
            horizon=np.arange(1,169),model=name,forecast=fc[0],lower_95=fc[1],upper_95=fc[2])))
    forecasts=pd.concat(forecasts,ignore_index=True)
    csv(forecasts,"oil_candidate_forecasts.csv")
    csv(forecasts[forecasts.model==winner],"oil_forecast.csv")
    # 首次在模型冻结和所有预测完成后读取最终一周的观测，用于一次性核验。
    held=pd.read_csv(DATA/"oil_final_test.csv",parse_dates=['time']).set_index('time').OT
    final=[]
    for name,g in forecasts.groupby('model',sort=False):
        final.append(records(held,(g.forecast.to_numpy(),g.lower_95.to_numpy(),g.upper_95.to_numpy()),
                             name,"final_test",frame.index[-1]))
    final=pd.concat(final,ignore_index=True)
    csv(final,"oil_final_test_predictions.csv");csv(summary(final),"oil_final_test_scores.csv")
    daily=[]
    for phase,pred in [('validation',val),('independent_test',test),('final_test',final)]:
        for (name,day),g in pred.assign(day=(pred.horizon-1)//24+1).groupby(['model','day']):
            daily.append(dict(phase=phase,model=name,day=int(day),**score(g.actual,g.forecast,g.lower_95,g.upper_95)))
    csv(pd.DataFrame(daily),"oil_scores_by_day.csv")
    return dict(**frozen,validation=tab.to_dict('records'),test=summary(test).to_dict('records'),
                final_test=summary(final).to_dict('records'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--task',choices=['all','monthly','oil'],default='all')
    args=parser.parse_args();t=time.perf_counter();monthly,oil=prepare()
    config=dict(seed=SEED,alpha=ALPHA,monthly_validation_origins=MONTH_ORIGINS,
        monthly_horizon=96,monthly_test_horizon=24,oil_horizon=168,oil_validation_origins=OIL_ORIGINS,
        oil_test_origins=OIL_TEST_ORIGINS.tolist(),oil_cutoff="2018-02-28 23:00:00",
        hgb=HGB,oil_lags=LAGS,threads=2,selection="within best validation RMSE*1.01, then interval score, stability, MAE",
        decomposition=dict(monthly_classical_period=12,monthly_stl=dict(period=12,seasonal=13,robust=True),
            oil_mstl=dict(periods=[24,168],windows=[15,15],iterate=2,
                          robust=True,seasonal_jump=2,trend_jump=10,low_pass_jump=10)),
        versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
                      scipy=scipy.__version__,statsmodels=statsmodels.__version__,sklearn=sklearn_version),
        intervals="pointwise nominal 95%; conditional on selected model; no guaranteed joint coverage")
    save_json(config,"configuration.json")
    with threadpool_limits(limits=2):
        if args.task in ['all','monthly']:
            monthly_result=monthly_analysis(monthly);save_json(monthly_result,"monthly_results.json")
        if args.task in ['all','oil']:
            oil_result=oil_analysis(oil);save_json(oil_result,"oil_results.json")
        if args.task=='all':
            print('decomposition and diagnostics',flush=True);decomposition(monthly,oil)
    save_json(FIT_LOG,"fit_log.json")
    csv(pd.DataFrame(FIT_LOG).drop(columns=['params','spec']),"fit_progress.csv")
    save_json(dict(seconds=time.perf_counter()-t,fit_count=len(FIT_LOG)),"runtime.json")
    print('DONE',round(time.perf_counter()-t,1),'seconds',flush=True)


if __name__=='__main__':main()

"""主模型冻结后的月度季节结构敏感性分析；不改动冻结选择或主预测。

本项由最终拟合的残差诊断触发，因此明确作为补充研究，不能称全新独立验证。
"""
import time
import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from threadpoolctl import threadpool_limits
import analysis as a


def main():
    t=time.perf_counter()
    y=pd.read_csv(a.DATA/'monthly.csv',parse_dates=['time']).set_index('time').x.asfreq('MS')
    specs=[dict(name=f'Sensitivity_SARIMA_{p}01_111',family='sarima',scale='total',
                order=[p,0,1],seasonal_order=[1,1,1,12]) for p in [1,3]]
    predictions=[];diags=[];failed=[]
    with threadpool_limits(limits=2):
        for s in specs:
            one=[]
            for origin in a.MONTH_ORIGINS:
                fc,fit,conv=a.monthly_forecast(y.iloc[:origin],s,24,'supplementary_validation')
                if not conv:failed.append(dict(model=s['name'],origin=origin));break
                one.append(a.records(y.iloc[origin:origin+24],fc,s['name'],'supplementary_validation',y.index[origin-1]))
            if len(one)==len(a.MONTH_ORIGINS):predictions.extend(one)
            _,fit,conv=a.monthly_forecast(y,s,96,'supplementary_full_fit')
            e=np.asarray(fit.resid)[13:]
            lb=acorr_ljungbox(e,lags=[12,24],model_df=s['order'][0]+s['order'][2]+2,return_df=True)
            for lag,z in lb.iterrows():diags.append(dict(model=s['name'],lag=lag,lb_stat=z.lb_stat,lb_pvalue=z.lb_pvalue,converged=conv))
            print(s['name'],'done',flush=True)
    if predictions:
        pred=pd.concat(predictions,ignore_index=True);a.csv(pred,'monthly_sensitivity_predictions.csv')
        a.csv(a.summary(pred),'monthly_sensitivity_scores.csv')
    a.csv(pd.DataFrame(diags),'monthly_sensitivity_diagnostics.csv')
    a.save_json(dict(status='supplementary only; original selection and forecasts remain frozen',
                     trigger='remaining annual correlation in full-fit residuals',specs=specs,
                     failures=failed,seconds=time.perf_counter()-t),'sensitivity_protocol.json')
    import json
    original=json.loads((a.RESULT/'fit_log.json').read_text(encoding='utf-8'))
    original=[v for v in original if not v['phase'].startswith('supplementary')]
    a.save_json(original+a.FIT_LOG,'fit_log.json')


if __name__=='__main__':main()

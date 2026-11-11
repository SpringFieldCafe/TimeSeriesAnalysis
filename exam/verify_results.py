"""独立核对原件、时间边界、预测行数、区间和已发表指标，不改变模型。"""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd

R=Path(__file__).resolve().parent


def main():
    checks={}
    baseline=R/'qa/source_hashes.json'
    if baseline.exists():
        originals=json.loads(baseline.read_text(encoding='utf-8'))
        checks['originals_unchanged']=all(hashlib.sha256((R/k).read_bytes()).hexdigest()==v['sha256'] for k,v in originals.items())
        checks['backups_identical']=all((R/k).read_bytes()==(R/'backup'/k).read_bytes() for k in originals)
    else:
        # Git交付不含本地原件，缺少原件时不可声称已检查原件保护。
        checks['originals_unchanged']=None
        checks['backups_identical']=None
    for task,n,start,end,freq in [('monthly',96,'1995-09-01','2003-08-01','MS'),
                                 ('oil',168,'2018-03-01 00:00','2018-03-07 23:00','h')]:
        f=pd.read_csv(R/'results'/f'{task}_forecast.csv',parse_dates=['time'])
        checks[task+'_dates']=len(f)==n and f.time.tolist()==pd.date_range(start,end,freq=freq).tolist()
        checks[task+'_finite']=bool(np.isfinite(f[['forecast','lower_95','upper_95']].to_numpy()).all())
        checks[task+'_ordered_intervals']=bool(f.lower_95.le(f.upper_95).all())
        for phase in ['validation','test']+(['final_test'] if task=='oil' else []):
            pred=pd.read_csv(R/'results'/f'{task}_{phase}_predictions.csv')
            name=f'{task}_model_comparison.csv' if phase=='validation' else f'{task}_{phase}_scores.csv'
            table=pd.read_csv(R/'results'/name).set_index('model')
            good=True
            for model,g in pred.groupby('model'):
                e=g.forecast-g.actual
                values=dict(rmse=np.sqrt(np.mean(e*e)),mae=np.mean(np.abs(e)),bias=np.mean(e),
                    coverage=np.mean((g.actual>=g.lower_95)&(g.actual<=g.upper_95)),
                    width=np.mean(g.upper_95-g.lower_95),
                    interval_score=np.mean(g.upper_95-g.lower_95+40*(np.maximum(g.lower_95-g.actual,0)+np.maximum(g.actual-g.upper_95,0))))
                good &= all(np.isclose(v,table.loc[model,k],rtol=1e-7,atol=1e-7) for k,v in values.items())
            checks[f'{task}_{phase}_metrics_recomputed']=bool(good)
            expected=24 if task=='monthly' else 168
            checks[f'{task}_{phase}_complete_horizons']=bool(pred.groupby(['model','origin']).size().eq(expected).all())
    log=json.loads((R/'results/fit_log.json').read_text(encoding='utf-8'))
    checks['all_labels_available_at_fit']=all(pd.Timestamp(z['max_label_time'])<=pd.Timestamp(z['train_end']) for z in log)
    checks['oil_no_fit_after_february']=all(pd.Timestamp(z['train_end'])<=pd.Timestamp('2018-02-28 23:00') for z in log if z['task']=='oil')
    checks['all_fit_procedures_succeeded']=all(z['fit_succeeded'] for z in log)
    checks['all_numerical_optimizers_converged']=all(z['optimizer_converged'] is not False for z in log)
    oil=pd.read_csv(R/'data/oil_training.csv',parse_dates=['time'])
    checks['oil_training_shape']=len(oil)==14592 and oil.time.iloc[-1]==pd.Timestamp('2018-02-28 23:00')
    for task in ['monthly','oil']:
        name='monthly_selection.json' if task=='monthly' else 'oil_selection_frozen.json'
        selected=json.loads((R/'results'/name).read_text(encoding='utf-8'))['selected']
        tab=pd.read_csv(R/'results'/f'{task}_model_comparison.csv')
        near=tab[tab.rmse<=tab.rmse.min()*1.01]
        recomputed=near.sort_values(['interval_score','fold_rmse_std','mae']).iloc[0].model
        checks[task+'_selection_recomputed']=selected==recomputed
    checks['all_passed']=all(v for v in checks.values() if v is not None)
    (R/'results/verification.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8',newline='\n')
    files=[]
    for folder in ['data','results','figures']:
        files += [p for p in (R/folder).rglob('*') if p.is_file() and p.name!='artifact_manifest.csv']
    files += list(R.glob('*.py'))+list(R.glob('*.md'))+[R/'requirements.txt',R/'.gitignore',R/'.gitattributes']
    rows=[dict(path=p.relative_to(R).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(set(files))]
    pd.DataFrame(rows).to_csv(R/'results/artifact_manifest.csv',index=False,encoding='utf-8-sig',lineterminator='\n')
    print(json.dumps(checks,ensure_ascii=False,indent=2))
    assert checks['all_passed']


if __name__=='__main__':main()

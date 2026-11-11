"""从已保存的结果绘制报告图；不重新拟合或改变模型选择。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
from scipy import stats
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox
import analysis as a

R=a.ROOT;F=R/'figures';F.mkdir(exist_ok=True)
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
    'axes.unicode_minus':False,'font.size':9,'axes.titlesize':10,'axes.labelsize':9,
    'legend.fontsize':7.5,'xtick.labelsize':8,'ytick.labelsize':8,
    'savefig.dpi':220,'axes.spines.top':False,'axes.spines.right':False})
BLUE='#275D8C';ORANGE='#C77832';GREEN='#47866A';GRAY='#7A8188'


def read(name):return pd.read_csv(R/'results'/name)
def js(name):return json.loads((R/'results'/name).read_text(encoding='utf-8'))
def dates(df):df['time']=pd.to_datetime(df.time);return df
def save(fig,name):
    # 以12.4厘米插入Word时，保证图内最小字体仍有可读尺寸。
    w,h=fig.get_size_inches();fig.set_size_inches(6.3,h*6.3/w)
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_fontsize()<9:text.set_fontsize(9)
    for ax in fig.axes:
        ax.grid(alpha=.18);ax.set_axisbelow(True)
        if isinstance(ax.xaxis.get_major_locator(),mdates.AutoDateLocator):
            loc=mdates.AutoDateLocator(minticks=3,maxticks=5)
            ax.xaxis.set_major_locator(loc);ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    fig.tight_layout(pad=1.2);fig.savefig(F/(name+'.png'),bbox_inches='tight',facecolor='white');plt.close(fig)


def alias(s):
    s=s.replace('SeasonalNaive12','去年同月基准').replace('Naive168','上周重复').replace('Naive24','昨日重复').replace('Naive1','最后值基准')
    s=s.replace('daily_rate','日历校正').replace('total','原总量')
    s=s.replace('_level','·无趋势').replace('_linear','·线性趋势').replace('_damped','·阻尼趋势')
    s=s.replace('ARX_','统计_').replace('HGB_','树模型_').replace('_all','_全部历史')
    s=s.replace('_dayweek','_日周周期').replace('_day','_日周期').replace('_loads','_加负荷').replace('_OT','_仅油温')
    return s


def main():
    m=pd.read_csv(R/'data/monthly.csv',parse_dates=['time']).set_index('time').x
    o=pd.read_csv(R/'data/oil_training.csv',parse_dates=['time']).set_index('time')
    mr=js('monthly_results.json');orr=js('oil_results.json')
    mt=read('monthly_model_comparison.csv');ot=read('oil_model_comparison.csv')
    fig,axs=plt.subplots(2,1,figsize=(7.2,5.2))
    axs[0].plot(m.index,m,color=BLUE,lw=.9);axs[0].set(title='月度屠宰量：水平变化与季节起伏',ylabel='给定月度总量')
    axs[0].annotate('1980年3月低值\n保留原始观测',xy=(m.index[2],m.iloc[2]),xytext=(m.index[22],53000),
                    arrowprops=dict(arrowstyle='->',color=GRAY),fontsize=8)
    for year,g in m.groupby(m.index.year):
        if len(g)==12:axs[1].plot(g.index.month,g.to_numpy(),alpha=.45,lw=.8)
    axs[1].set(title='完整年度按月份对齐（1980—1994年）',xlabel='月份',ylabel='给定月度总量',xticks=range(1,13))
    save(fig,'monthly_overview')
    d=dates(read('monthly_additive_components.csv'))
    fig,axs=plt.subplots(4,1,figsize=(7.2,6),sharex=True)
    for ax,c,title in zip(axs,['observed','trend','seasonal','remainder'],['原始观测','趋势与低频变化','固定月份季节项','不规则余项']):
        ax.plot(d.time,d[c],lw=.85,color=BLUE);ax.set_ylabel(title,fontsize=8)
    axs[0].set_title('经典加法分解：首尾各6个月不估计中心趋势')
    save(fig,'monthly_decomposition')
    total=dates(read('monthly_stl_total.csv'));daily=dates(read('monthly_stl_daily_rate.csv'))
    fig,axs=plt.subplots(1,2,figsize=(7.2,3))
    for y,label,color in [(m,'月总量',BLUE),(m/m.index.days_in_month,'按日历天数校正',ORANGE)]:
        y=y[y.index.year<1995];v=y.groupby(y.index.month).mean()
        axs[0].plot(v.index,v/v.mean()*100,'o-',ms=3,label=label,color=color)
    axs[0].set(title='校正前后的月份平均轮廓',xlabel='月份',ylabel='各自均值=100',xticks=[1,3,5,7,9,11]);axs[0].legend()
    sub=mt[mt.model.str.contains('ETS_.*level|SARIMA_.*101',regex=True)]
    axs[1].barh([alias(v) for v in sub.model],sub.rmse,color=[ORANGE if 'daily_rate' in v else BLUE for v in sub.model])
    axs[1].set(title='相同模型结构的校正对照',xlabel='24个月历史回测RMSE（原总量尺度）')
    save(fig,'monthly_calendar')
    fig,ax=plt.subplots(figsize=(7.2,4.2));sub=mt.sort_values('rmse',ascending=False)
    ax.barh([alias(v) for v in sub.model],sub.rmse,color=[ORANGE if v==mr['selected'] else BLUE for v in sub.model])
    ax.set(title='月度候选模型比较：4个历史起点，每次预测24个月',xlabel='RMSE，越小越好')
    save(fig,'monthly_comparison')
    fc=dates(read('monthly_forecast.csv'));candidates=dates(read('monthly_candidate_forecasts.csv'))
    fig,axs=plt.subplots(2,1,figsize=(7.2,5.1))
    axs[0].plot(m.index,m,color=GRAY,lw=.7,label='历史观测');axs[0].plot(fc.time,fc.forecast,color=BLUE,lw=.9,label='冻结模型预测均值')
    axs[0].fill_between(fc.time,fc.lower_95,fc.upper_95,color=BLUE,alpha=.15,label='逐期名义95%预测区间')
    axs[0].set(title='未来96个月预测：1995年9月—2003年8月',ylabel='月度总量');axs[0].legend(loc='upper left')
    axs[1].plot(fc.time,fc.forecast,color=BLUE,lw=1,label=alias(mr['selected']))
    for name,g in candidates.groupby('model'):
        if name==mr['selected']:continue
        axs[1].plot(g.time,g.forecast,lw=.7,alpha=.65,label=alias(name))
    axs[1].set(title='候选模型的长期点预测对照',ylabel='月度总量');axs[1].legend(ncol=1,fontsize=6.5)
    save(fig,'monthly_forecast')
    mf=dates(read('monthly_fitted.csv'));res=mf.innovation.to_numpy()[13:]
    fig,axs=plt.subplots(2,2,figsize=(7.2,4.6))
    axs[0,0].plot(mf.time,mf.actual,color=GRAY,lw=.7,label='观测');axs[0,0].plot(mf.time,mf.fitted_median,color=BLUE,lw=.7,label='条件中位数拟合');axs[0,0].legend()
    axs[0,0].set_title('拟合与观测（不代表样本外精度）')
    axs[0,1].plot(mf.time.iloc[13:],res,color=BLUE,lw=.7);axs[0,1].axhline(0,color=GRAY,lw=.6);axs[0,1].set_title('剔除初始化期后的对数创新')
    ac=acf(res,nlags=24,fft=True);axs[1,0].stem(range(1,25),ac[1:],basefmt=' ');axs[1,0].axhline(1.96/np.sqrt(len(res)),ls='--',lw=.6,color=ORANGE);axs[1,0].axhline(-1.96/np.sqrt(len(res)),ls='--',lw=.6,color=ORANGE);axs[1,0].set_title('创新自相关（逐阶参考线）')
    stats.probplot(res,dist='norm',plot=axs[1,1]);axs[1,1].set(title='创新正态Q-Q图',xlabel='正态理论分位数',ylabel='样本分位数')
    save(fig,'monthly_diagnostics')
    lb=acorr_ljungbox(res,lags=[12,24],model_df=3,return_df=True).reset_index(names='lag')
    a.csv(lb,'monthly_final_residual_ljung_box.csv')
    fig,axs=plt.subplots(2,1,figsize=(7.2,4.5))
    mean=o.OT.resample('D').mean();axs[0].plot(mean.index,mean,color=BLUE,lw=.8)
    axs[0].set(title='题定油温范围：2016年7月—2018年2月（日均展示）',ylabel='OT给定尺度')
    tail=o.OT.iloc[-24*14:];axs[1].plot(tail.index,tail,color=BLUE,lw=.8)
    axs[1].set(title='末两周逐小时变化',ylabel='OT给定尺度')
    save(fig,'oil_overview')
    d=dates(read('oil_mstl_components.csv'))
    fig,axs=plt.subplots(4,1,figsize=(7.2,5.8))
    axs[0].plot(d.time.iloc[::12],d.trend.iloc[::12],color=BLUE,lw=.8);axs[0].set_ylabel('趋势与低频变化')
    tail=d[d.time>='2018-02-15']
    for ax,c,title in zip(axs[1:],['daily','weekly','remainder'],['日周期成分','周周期成分','余项']):
        ax.plot(tail.time,tail[c],color=BLUE,lw=.7);ax.set_ylabel(title)
    axs[0].set_title('多周期分解：趋势展示全样本，其余展示末两周')
    save(fig,'oil_decomposition')
    fig,axs=plt.subplots(1,2,figsize=(7.2,2.8))
    for ax,y,title in [(axs[0],o.OT,'原序列自相关'),(axs[1],o.OT.diff().dropna(),'一阶变化量自相关')]:
        c=acf(y,nlags=336,fft=True);ax.plot(range(1,337),c[1:],color=BLUE,lw=.7)
        for p in [24,168]:ax.axvline(p,color=ORANGE,lw=.7,ls='--')
        ax.set(title=title,xlabel='滞后小时',ylabel='ACF')
    save(fig,'oil_correlations')
    fig,ax=plt.subplots(figsize=(7.2,4.9));sub=ot.sort_values('rmse',ascending=False)
    ax.barh([alias(v) for v in sub.model],sub.rmse,color=[ORANGE if v==orr['selected'] else BLUE for v in sub.model])
    ax.set(title='油温候选模型比较：11个起点，每次完整预测168小时',xlabel='RMSE，越小越好')
    save(fig,'oil_comparison')
    fig,axs=plt.subplots(1,2,figsize=(7.2,3.1))
    for suffix,label,color in [('dayweek','统计模型（日周）',BLUE),('OT','树模型（仅OT）',ORANGE),('loads','树模型（加负荷）',GREEN)]:
        vals=[]
        for w in ['90','180','all']:
            name=f"{'ARX' if suffix=='dayweek' else 'HGB'}_{w}_{suffix}"
            vals.append(float(ot.set_index('model').loc[name,'rmse']))
        axs[0].plot(['90天','180天','全部历史'],vals,'o-',ms=3,label=label,color=color)
    axs[0].set(title='历史窗口敏感性',ylabel='验证集RMSE');axs[0].legend(fontsize=6.5)
    corr=o.drop(columns='OT').corr();im=axs[1].imshow(corr,vmin=-1,vmax=1,cmap='RdBu_r')
    axs[1].set(xticks=range(6),yticks=range(6),xticklabels=corr.columns,yticklabels=corr.columns,title='负荷变量之间的相关性')
    axs[1].tick_params(axis='x',rotation=60);fig.colorbar(im,ax=axs[1],fraction=.045)
    save(fig,'oil_windows_loads')
    final=dates(read('oil_final_test_predictions.csv'));g=final[final.model==orr['selected']]
    fig,axs=plt.subplots(2,1,figsize=(7.2,4.9),sharex=True)
    axs[0].fill_between(g.time,g.lower_95,g.upper_95,color=BLUE,alpha=.15,label='逐小时名义95%预测区间')
    axs[0].plot(g.time,g.actual,color=GRAY,lw=.9,label='冻结后核验真实值');axs[0].plot(g.time,g.forecast,color=BLUE,lw=.9,label='冻结模型预测')
    axs[0].set(title='最终一周核验：2018年3月1日—7日',ylabel='OT给定尺度');axs[0].legend(ncol=1)
    for name in list(dict.fromkeys([orr['selected'],orr['best_statistical'],orr['best_machine_learning'],'Naive1'])):
        z=final[final.model==name];axs[1].plot(z.time,z.forecast-z.actual,lw=.8,label=alias(name))
    axs[1].axhline(0,color=GRAY,lw=.6);axs[1].set(title='同一预测起点的误差对照',ylabel='预测－真实值');axs[1].legend(fontsize=6.5)
    save(fig,'oil_forecast')
    byday=read('oil_scores_by_day.csv');fig,axs=plt.subplots(1,2,figsize=(7.2,3))
    for ax,phase,title in [(axs[0],'validation','历史验证：第1—7天'),(axs[1],'independent_test','独立历史检验：第1—7天')]:
        for name in list(dict.fromkeys([orr['selected'],orr['best_statistical'],orr['best_machine_learning'],'Naive1'])):
            z=byday[(byday.phase==phase)&(byday.model==name)];ax.plot(z.day,z.rmse,'o-',ms=3,label=alias(name))
        ax.set(title=title,xlabel='预测到第几天',ylabel='RMSE');ax.legend(fontsize=6)
    save(fig,'oil_horizon')
    fig,axs=plt.subplots(1,2,figsize=(7.2,3))
    for ax,tab,title in [(axs[0],mt,'月度验证区间'),(axs[1],ot,'油温验证区间')]:
        scale=10000 if title.startswith('月') else 1
        ax.scatter(tab.width/scale,tab.coverage*100,color=BLUE,s=20)
        chosen=mr['selected'] if title.startswith('月') else orr['selected']
        best=tab[tab.model==chosen].iloc[0];ax.scatter([best.width/scale],[best.coverage*100],color=ORANGE,s=45,label='最终选定配置')
        ax.axhline(95,color=GRAY,ls='--',lw=.7,label='名义95%');ax.set(title=title,xlabel='平均区间宽度（万）' if scale>1 else '平均区间宽度（OT尺度）',ylabel='实际覆盖比例（%）');ax.legend(fontsize=6.5)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    save(fig,'interval_quality')
    # 保存统计对照模型的创新，机器学习采用样本外误差诊断而非套用ARMA自由度。
    fitlog=js('fit_log.json');fit=next(v for v in fitlog if v['model']==orr['best_statistical'] and v['phase']=='final')
    spec=fit['spec'];start=max(0,len(o)-spec['window']*24) if spec['window'] else 0
    idx=np.arange(max(start,168),len(o));y=o.OT.to_numpy();X=np.column_stack([a.fourier(o.index[idx],spec['weekly']),*[y[idx-l] for l in a.LAGS]])
    err=y[idx]-X@np.asarray(fit['params']['coefficients'])
    a.csv(pd.DataFrame(dict(time=o.index[idx],innovation=err)),'oil_statistical_innovations.csv')
    lb=acorr_ljungbox(err,lags=[24,168],model_df=len(a.LAGS),return_df=True).reset_index(names='lag');a.csv(lb,'oil_statistical_residual_ljung_box.csv')
    fig,axs=plt.subplots(1,2,figsize=(7.2,2.8));axs[0].plot(o.index[idx][::6],err[::6],lw=.5,color=BLUE);axs[0].set(title='统计对照的一步创新（每6点展示）',ylabel='创新')
    c=acf(err,nlags=168,fft=True);axs[1].plot(range(1,169),c[1:],color=BLUE,lw=.7);axs[1].axhline(1.96/np.sqrt(len(err)),color=ORANGE,ls='--',lw=.6);axs[1].axhline(-1.96/np.sqrt(len(err)),color=ORANGE,ls='--',lw=.6);axs[1].set(title='创新自相关：检验剩余结构',xlabel='滞后小时')
    save(fig,'oil_statistical_diagnostics')
    print('Figures:',len(list(F.glob('*.png'))),flush=True)


if __name__=='__main__':main()

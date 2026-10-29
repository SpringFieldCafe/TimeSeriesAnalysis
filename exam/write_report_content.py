"""将数值结果转换为逐题报告内容；不手填或改写计算结果。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from make_figures import alias

R=Path(__file__).resolve().parent;Q=R/'qa';S=R/'results'


def read(name):return pd.read_csv(S/name)
def js(name):return json.loads((S/name).read_text(encoding='utf-8'))
def num(v,d=2):return f'{float(v):,.{d}f}'
def pct(v):return f'{float(v)*100:.2f}%'
def improve(base,new):return (float(base)-float(new))/float(base)*100


def main():
    m=pd.read_csv(R/'data/monthly.csv',parse_dates=['time']).set_index('time').x
    o=pd.read_csv(R/'data/oil_training.csv',parse_dates=['time']).set_index('time')
    mr=js('monthly_results.json');orr=js('oil_results.json');log=js('fit_log.json');config=js('configuration.json')
    mv=read('monthly_model_comparison.csv').set_index('model');mt=read('monthly_test_scores.csv').set_index('model')
    ov=read('oil_model_comparison.csv').set_index('model');ot=read('oil_test_scores.csv').set_index('model');of=read('oil_final_test_scores.csv').set_index('model')
    selected_m=mr['selected'];selected_o=orr['selected'];ml=orr['best_machine_learning'];matched=orr['matched_machine_learning']
    mf=next(z for z in log if z['model']==selected_m and z['phase']=='final')
    oilfit=next(z for z in log if z['model']==selected_o and z['phase']=='final')
    q=json.loads((Q/'questions.json').read_text(encoding='utf-8'))
    b=[]
    def p(text,kind='p'):b.append(dict(kind=kind,text=text))
    def h(text):p(text,'heading')
    def image(name,caption):b.extend([dict(kind='image',path='figures/'+name+'.png',width=12.4),dict(kind='caption',text=caption)])
    def table(headers,rows,widths=None):
        block=dict(kind='table',headers=headers,rows=rows)
        if widths:block['widths']=widths
        b.append(block)
    def scores_table(tab,names,label='模型'):
        table([label,'RMSE','MAE','覆盖率','区间得分'],[[alias(n),num(tab.loc[n,'rmse']),num(tab.loc[n,'mae']),pct(tab.loc[n,'coverage']),num(tab.loc[n,'interval_score'])] for n in names],[5.0,1.7,1.7,1.9,2.3])
    p('课程考查报告正文','title')
    p('本报告分别研究月度屠宰量的长期预测和变压器油温的一周预测。两题均按“数据特征—因素分解—候选模型—历史预测检验—最终预测”展开；全部误差在原始观测尺度上计算，95%区间均指未来单次观测的逐期预测区间。')
    p(f'主要结论是：屠宰量在季节模型下具有较稳定的月份轮廓，但8年外推的不确定性明显扩大；油温在历史验证中以180天拟合窗口的日周周期统计模型综合表现较好。机器学习在冬季检验中点预测更准，但区间覆盖不足，提示模型优劣依赖评价目标和时期。完整代码、参数日志、逐期结果及图表与本报告配套保存。')
    p(q[0],'question');p(q[1],'question')
    h('1 数据核对与时序特征')
    p(f'文件1包含1980年1月至1995年8月的188个月度观测，时间连续，没有缺失或重复月份，最小值为{num(m.min(),0)}，最大值为{num(m.max(),0)}。本文沿用附件的月度总量尺度。各完整年度的月平均水平有明显变化：1984年为{num(m[m.index.year==1984].mean(),0)}，1986年为{num(m[m.index.year==1986].mean(),0)}，随后逐步回升。这说明序列同时包含较慢的水平变化与年内重复起伏。')
    image('monthly_overview','图1 月度屠宰量及完整年度的月份对齐图')
    p('1980年3月的低值可能来自真实特殊事件或记录问题，但附件没有提供原因。主分析保留这一观测，稳健STL作为对照，避免凭主观判断删点。仅凭单变量数据可以识别趋势、季节性和异常波动，不能进一步认定某项政策、疫情或供需变化造成了具体拐点。')
    h('2 经典分解与稳健对照')
    p('采用周期为12的经典加法分解：xₜ = Tₜ + Sₘ + Rₜ。其中Tₜ表示趋势与较长周期的共同变化，Sₘ表示固定月份季节效应，Rₜ为余项。12个月季节效应之和为0；趋势用2×12中心移动平均估计，首尾各6个月因缺少双侧观测而不估计，未用外推值补齐。')
    d=read('monthly_additive_components.csv');season=d.seasonal.iloc[:12].to_numpy();imax=int(season.argmax());imin=int(season.argmin())
    p(f'提取的固定季节项中，{imax+1}月相对趋势的增量最大，约{num(season[imax],0)}；{imin+1}月最低，约{num(season[imin],0)}。最高与最低月份相差约{num(season.max()-season.min(),0)}。这些是给定序列的描述性月份差异，不是月份本身的因果效应。')
    image('monthly_decomposition','图2 经典加法分解及中心移动平均的边界')
    p('加法分解便于在原总量尺度上解释。另保存乘法分解及稳健STL：乘法季节指数以均值1为基准，STL允许月份效应缓慢变化并降低极端观测的影响。分解图用于识别结构，模型优劣则通过后面的样本外历史预测比较，不能以重构误差替代预测误差。')
    h('3 日历因素是否混入季节性')
    p('月总量可能受28、29、30、31天的日历差异影响。构造rₜ=xₜ/dₜ，dₜ为当月日历天数，比较原总量与校正量的月份轮廓。该量是每个日历日的平均值，不是实际开工日平均量；没有日度开工资料时，不作更强解释。月份对齐比较仅使用1980—1994年的完整年度，以免1995年的不完整年度改变月份权重。')
    image('monthly_calendar','图3 日历校正前后的月份轮廓及同结构模型回测')
    p('日历校正作为候选处理而非预设最优结论。校正模型预测未来rₜ，再乘未来各月已知天数还原成月总量，区间上下限同样换算。因此所有预测误差仍对应题目要求的月总量。')
    p('分解的核心实现如下，完整边界处理及数据核对见analysis.py。')
    p('d = seasonal_decompose(monthly, model="additive", period=12)\nrate = monthly / monthly.index.days_in_month\nstl = STL(rate, period=12, seasonal=13, robust=True).fit()','code')
    p(q[2],'question')
    h('1 候选模型与统一比较')
    p('主比较包含13个预设候选：一个去年同月基准；在原总量和日历校正量上分别建立无趋势、线性趋势、阻尼趋势的对数ETS模型，共6个；分别建立三种季节ARIMA结构，共6个。SARIMA非季节阶数为(0,1,1)、(1,0,0)、(1,0,1)，季节部分统一为(0,1,1)₁₂。候选结构在观察最终检验结果前固定。')
    p('对数变换使模拟分布保持正值。log预测的均值为μₕ、方差为vₕ时，原尺度预测均值采用exp(μₕ+vₕ/2)，而不是直接取exp(μₕ)；后者对应对数正态近似下的中位数。模型使用给定观测，不以新增外部数据延长历史。')
    p('调参使用4个历史起点：训练分别截至1987、1988、1989、1990年12月，各预测随后24个月。每次重新拟合模型；不同起点的预测区间存在重叠，不能视为4次完全独立试验。最后24个月，即1993年9月至1995年8月，另作留出检验。')
    p('先比较验证集RMSE；在最佳RMSE的1%以内，再按区间得分、起点间RMSE标准差及MAE择优。区间得分等于区间宽度，加上落在区间外的偏差惩罚；得分越低，覆盖与宽度的综合表现越好。该规则同时约束点预测和区间质量，阈值1%在运行前固定。')
    names=list(mv.sort_values('rmse').index[:5])+['ETS_total_damped','ETS_total_linear']
    scores_table(mv,list(dict.fromkeys(names)))
    image('monthly_comparison','图4 13个预设月度候选的历史预测RMSE')
    h('2 主模型及优化收益')
    mp=mf['params']
    p(f'冻结主模型为原总量对数尺度的SARIMA(1,0,1)×(0,1,1)₁₂。它的验证RMSE为{num(mv.loc[selected_m,"rmse"])}，相对去年同月基准改善{improve(mv.loc["SeasonalNaive12","rmse"],mv.loc[selected_m,"rmse"]):.2f}%。同结构的日历校正版本RMSE为{num(mv.loc["SARIMA_daily_rate_101_011","rmse"])}，略低于主模型，但二者相差不足1%；原总量版本的区间得分更低，因此依固定规则选原总量版本。不能据此宣称日历校正必然无效。')
    p('模型方程为：(1−φB)(1−B¹²)log(xₜ)=(1+θB)(1+ΘB¹²)εₜ。B为滞后算子，εₜ为创新，季节差分用于处理年际水平变化；φ、θ、Θ分别为非季节AR、非季节MA与季节MA参数。')
    table(['参数','估计值','含义'],[[k,num(v,6),{'ar.L1':'1阶AR','ma.L1':'1阶MA','ma.S.L12':'季节MA','sigma2':'对数创新方差'}[k]] for k,v in mp.items()],[3.2,3.1,6.3])
    p(f'留出24个月中，主模型RMSE为{num(mt.loc[selected_m,"rmse"])}，MAE为{num(mt.loc[selected_m,"mae"])}，基准RMSE为{num(mt.loc["SeasonalNaive12","rmse"])}，相对改善{improve(mt.loc["SeasonalNaive12","rmse"],mt.loc[selected_m,"rmse"]):.2f}%。这支持其在该留出时期的改进，但不能将此百分比推广为未来8年的精度承诺。')
    p('线性趋势ETS在若干起点出现明显外推偏差，整体RMSE高于无趋势和阻尼版本，说明历史局部斜率未必适合长期延伸。阻尼模型虽能减弱长期斜率，仍没有在本次固定验证规则下胜出。优化结论应依据这些具体比较，而非认为模型越复杂越好。')
    h('3 模型诊断及补充敏感性研究')
    image('monthly_diagnostics','图5 月度主模型的拟合、创新自相关与分布诊断')
    lb=read('monthly_final_residual_ljung_box.csv')
    p(f'剔除前13期初始化影响后，主模型创新在12阶和24阶Ljung–Box检验的p值分别为{lb.iloc[0].lb_pvalue:.3g}和{lb.iloc[1].lb_pvalue:.3g}，仍提示剩余相关。检验扣除了3个AR/MA动态参数，自相关图的逐阶参考线没有作多重比较校正。主模型是预设候选中的择优结果，不代表历史规律已被完全提取；正态区间也仅是条件近似。')
    if (S/'monthly_sensitivity_scores.csv').exists():
        ss=read('monthly_sensitivity_scores.csv').set_index('model');sd=read('monthly_sensitivity_diagnostics.csv')
        p('针对残留的12个月相关，在主模型冻结后补充增加季节AR项，并增加一个非季节3阶AR结构，检验更复杂的季节依赖能否改善诊断。补充试验由全样本拟合诊断触发，已经接触过主研究结果，因此单独列示，不冒充全新独立检验，也不据此回改冻结的主预测。')
        rows=[]
        for name,z in ss.iterrows():
            pp=sd[(sd.model==name)&(sd.lag==12)].iloc[0].lb_pvalue
            rows.append([name.replace('Sensitivity_SARIMA_','SARIMA '),num(z.rmse),pct(z.coverage),f'{pp:.3g}'])
        table(['补充结构','回测RMSE','覆盖率','残差LB(12) p'],rows,[5.0,2.2,2.2,3.2])
        p('这里同时考察预测误差与残差：白噪声检验通过本身不等于预测最优，检验未通过也不能通过删点或反复更换检验阶数“修饰”。新增结构的实际收益见表，完整参数和优化器状态保存在fit_log.json。')
    p('另作两个96个月长跨度压力检验，起点分别截至1985年12月、1986年12月。它们仅有6年、7年训练资料，且预测区段高度重叠，属于外推敏感性证据。188个月不足以同时安排至少8年训练与8年完全留出，因此无法声称已充分验证完整8年预测的精度或真实95%覆盖率。')
    p(q[3],'question')
    h('1 96个月点预测与预测区间')
    f=read('monthly_forecast.csv')
    p('最终用截至1995年8月的全部188个月重新估计已冻结的模型，预测1995年9月至2003年8月，共96个月。95%区间采用exp(μₕ±1.96√vₕ)，是给定所选模型与估计参数的逐期条件预测区间，包含未来创新的不确定性，未完整纳入参数估计、模型选择和结构变化风险。逐期95%不表示整条96个月路径有95%的联合覆盖概率。')
    table(['月份','预测均值','95%下限','95%上限'],[[str(f.iloc[i].time)[:7],num(f.iloc[i].forecast,0),num(f.iloc[i].lower_95,0),num(f.iloc[i].upper_95,0)] for i in [0,1,2,11,23,47,71,95]],[3.0,3.2,3.2,3.2])
    image('monthly_forecast','图6 未来8年的逐月预测、区间及候选外推差异')
    p(f'第一期预测约为{num(f.iloc[0].forecast,0)}，区间为[{num(f.iloc[0].lower_95,0)}, {num(f.iloc[0].upper_95,0)}]；最后一期约为{num(f.iloc[-1].forecast,0)}，区间扩大至[{num(f.iloc[-1].lower_95,0)}, {num(f.iloc[-1].upper_95,0)}]。远期仍保留月份轮廓，但区间明显变宽。长期结果适合作为给定模型假设下的情景参照，不能把单一曲线当成确定的产量路径。')
    p('完整96行的月份、预测均值、上下限保存为monthly_forecast.csv。图中历史与预测以1995年8月为界，未加入该时点之后的外部真实观测。')
    p('pred = fit.get_forecast(96)\nmu = np.asarray(pred.predicted_mean)\nvar = np.asarray(pred.var_pred_mean)\nmean = np.exp(mu + var / 2)','code')
    p(q[4],'question');p(q[5],'question')
    h('1 时间范围与可用信息')
    p(f'文件2实际比题目时间范围更长，本文严格截取2016年7月1日0时至2018年2月28日23时，共14,592个逐小时观测。时间连续，无缺失与重复时刻。OT最小值为{num(o.OT.min())}、最大值为{num(o.OT.max())}，含{int(o.OT.le(0).sum())}个非正值，故不对OT取对数，也不以百分比误差评价温度。除OT外的6列负荷只以预测起点已知的历史记录作拓展输入。')
    image('oil_overview','图7 题定范围内的油温水平变化和末两周小时波动')
    p('2016年与2017年同月的水平存在差异，且序列具有短期持续性，单纯重复较早年份并不充分。气温、负荷、设备工况都可能与油温相关，但本数据不含完整环境与工况信息，不能将低频变化单独归因为天气或设备状态改变。20个月仅覆盖一个完整日历年，年周期的稳定性证据有限。')
    h('2 日周周期与多周期分解')
    p('先检查原序列与一阶变化量的自相关。原序列较高的长滞后相关既可能包含周期，也可能来自缓慢变化的水平，因此进一步观察变化量在24、168小时间隔的重复结构。分解采用加法形式：OTₜ=Tₜ+S₂₄,ₜ+S₁₆₈,ₜ+Rₜ。')
    image('oil_correlations','图8 原序列与变化量的相关结构，虚线标记24与168小时')
    p('MSTL周期固定为24、168，季节平滑窗口均为15，迭代2次，并在STL内部采用稳健权重。为控制计算量，seasonal_jump=2，trend_jump=10，low_pass_jump=10；这些计算参数均记录于源码。该分解提取不同时间尺度的形态，低频成分包含趋势与长周期变化，不能独立解释为已证实的年度季节效应。')
    image('oil_decomposition','图9 油温的低频变化、日周期、周周期和余项')
    p('从可视化识别周期后，仍需在预测中比较“仅日周期”与“日周周期”，检验增加周周期是否有实际收益。分解作为解释工具；历史回测的预测模型只读取起点以前的观测，不把全样本MSTL成分输入较早起点的模型，避免双侧平滑带来的未来信息泄漏。')
    p('d = MSTL(oil.OT, periods=(24, 168), windows=(15, 15),\n         iterate=2, stl_kwargs={"robust": True,\n         "seasonal_jump": 2, "trend_jump": 10,\n         "low_pass_jump": 10}).fit()','code')
    p(q[6],'question')
    h('1 验证边界与模型选择规则')
    p('预测任务固定为“在同一个起点一次给出后续168小时”。统计模型递归使用自己的预测值，树模型直接学习不同步长的目标；两者均不得在预测第2天时读入第1天真实油温，也不得使用未来真实负荷。最终一周真实OT仅在配置冻结、预测全部生成后读取。')
    table(['用途','时间或起点','作用'],[
        ['历史验证','2017年1—11月各月末23时；共11个起点','每次预测随后168小时，用于选模型'],
        ['留出历史检验','2017-12-31起，每7天一个起点，共8次','检验2018年1—2月表现，不参与择优'],
        ['最终核验','2018-02-28 23时','冻结后预测3月1—7日，再核验真实值']],[2.6,5.2,4.8])
    p('模型择优与月度任务一致：验证RMSE在最佳值1%以内，再按区间得分、起点间稳定性和MAE排序。留出历史检验的各周不重叠，但时间相关性仍然存在；最终168小时属于一周内相关观测，不能把其覆盖比例解释为168次独立试验的概率保证。')
    h('2 统计模型和树模型的技术设置')
    p('共比较15个预设候选：最后值、重复昨日、重复上周三个简单基准；3种拟合窗口×2种周期设置的6个统计模型；3种窗口×是否加入负荷的6个树模型。窗口为90天、180天与全部历史。窗口实验在同一模型家族内比较，避免把模型变更与数据量变更混在一起解释。')
    p('统计模型为稀疏自回归加谐波项：OTₜ=c+ΣφⱼOTₜ₋ⱼ+F₂₄(t)+F₁₆₈(t)+εₜ。滞后集合为{1,2,3,24,25,48,168}；日周期和周周期各用2对正弦余弦项，周周期项有无作为对照。参数通过最小二乘估计，窗口限定拟合目标时段，构造滞后值时可向此前历史获取已知观测。')
    p('树模型使用HistGradientBoostingRegressor，学习OT(t+h)−OT(t)，h=1,…,168。每个训练起点为日末23时，将步长及未来时钟信息作为输入，在一个共享模型中直接预测不同步长，不训练168个互不关联模型。所有标签必须早于本折截止；窗口中为30天历史特征及完整7天标签预留边界，实际监督起点数与样本数逐次记录。')
    table(['设置','取值'],[
        ['迭代次数 / 学习率','100 / 0.06'],['最大叶节点 / 最小叶样本','15 / 35'],
        ['L2正则 / 随机种子','10 / 20261029'],['提前停止','关闭，避免默认随机划分验证集'],
        ['仅OT输入','8个滞后值；24、168、720小时的均值/标准差/最小/最大；步长与日周时钟，共25维'],
        ['加入负荷','6列负荷各加3个滞后值、2个历史均值，共55维'],
        ['点预测 / 区间','平方误差均值模型；0.025、0.975分位数模型'],
        ['计算控制','2线程，按模型分批运行']],[4.0,8.6])
    p('分位数模型给出的上下界若发生交叉，按大小重排，并记录交叉数。训练模型不进行全样本标准化；所有滞后和滚动统计均在本折截止之前计算。未来日期可提前知道，未来实际负荷和OT则不可提前知道。')
    h('3 模型优化的具体证据')
    scores_table(ov,[selected_o,'ARX_180_day','ARX_all_dayweek',ml,matched,'Naive1','Naive168'])
    image('oil_comparison','图10 所有油温候选的统一168小时验证结果')
    p(f'冻结主模型为180天窗口的日周周期统计模型，验证RMSE为{num(ov.loc[selected_o,"rmse"],3)}，MAE为{num(ov.loc[selected_o,"mae"],3)}。相对最后值基准RMSE改善{improve(ov.loc["Naive1","rmse"],ov.loc[selected_o,"rmse"]):.2f}%，相对重复上周改善{improve(ov.loc["Naive168","rmse"],ov.loc[selected_o,"rmse"]):.2f}%。验证区间覆盖率为{pct(ov.loc[selected_o,"coverage"])}，平均宽度为{num(ov.loc[selected_o,"width"],3)}。')
    p(f'同为日周周期统计模型，180天窗口相对全部历史RMSE改善{improve(ov.loc["ARX_all_dayweek","rmse"],ov.loc[selected_o,"rmse"]):.2f}%，但90天窗口又更差，故结果支持“适度近期窗口有利”，而不是“历史越短越好”。180天加入周周期相对仅日周期的RMSE改善约{improve(ov.loc["ARX_180_day","rmse"],ov.loc[selected_o,"rmse"]):.2f}%，幅度较小，区间综合得分也更低。')
    image('oil_windows_loads','图11 拟合窗口敏感性及负荷变量的相关性')
    corr=o.drop(columns='OT').corr().to_numpy();np.fill_diagonal(corr,0)
    p(f'负荷变量之间最大绝对相关约为{np.max(np.abs(corr)):.3f}，说明单个变量的重要性可能被同组变量分担。采用整组加入/去除对照：全部历史树模型只用OT时RMSE为{num(ov.loc[ml,"rmse"],3)}，加入负荷后为{num(ov.loc[matched,"rmse"],3)}，没有改善；较短窗口下加入负荷有不同程度改善。由此可见，历史负荷的边际信息与窗口、目标步长及模型有关，不能解释为负荷对油温没有物理作用。')
    p('这种比较属于预测信息增益研究，相关特征不支持把单个负荷系数或重要性排名直接解释为因果效应。全部窗口与信息组的结果完整保留，不只展示有利的组合。')
    h('4 留出时期与模型局限')
    scores_table(ot,[selected_o,ml,matched,'Naive1','Naive168'])
    p(f'2018年1—2月8个留出周中，统计主模型RMSE为{num(ot.loc[selected_o,"rmse"],3)}、覆盖率为{pct(ot.loc[selected_o,"coverage"])}；树模型仅OT的RMSE为{num(ot.loc[ml,"rmse"],3)}，点预测更准，但覆盖率仅{pct(ot.loc[ml,"coverage"])}。其区间得分{num(ot.loc[ml,"interval_score"],3)}也高于统计模型的{num(ot.loc[selected_o,"interval_score"],3)}。这表明点误差、区间可信度和时期适应性可能出现取舍，不能宣称统计模型在所有场景都最好。')
    image('oil_horizon','图12 第1—7天的误差变化及不同检验时期的差异')
    lb=read('oil_statistical_residual_ljung_box.csv')
    image('oil_statistical_diagnostics','图13 统计主模型的创新及剩余自相关')
    p(f'统计模型创新在24、168阶Ljung–Box检验的p值分别约为{lb.iloc[0].lb_pvalue:.3g}和{lb.iloc[1].lb_pvalue:.3g}，提示尚有未捕捉的相关结构。大样本下较小的相关也可能显著，因此同时报告相关图与实际多步预测误差。解析区间依赖创新近似不相关、方差相对稳定等假设；留出期覆盖低于95%也反映其局限。机器学习以样本外误差和区间覆盖诊断为主，不机械套用ARMA参数自由度。')
    p('核心预测参数摘录如下。完整16个统计回归系数、全部候选及每次拟合的窗口、截止时刻、样本数、警告和耗时见fit_log.json。数值优化的收敛状态与树模型完成预设100次迭代分别记录；固定迭代完成不等于证明已达到全局最优。数值指标由独立脚本复算。')
    ars=oilfit['params']['ar_coefficients']
    table(['滞后小时','自回归系数'],[[k,num(v,6)] for k,v in ars.items()],[4.0,8.6])
    p(q[7],'question')
    h('1 最终一周预测与95%区间')
    p('最终使用截至2018年2月28日23时可用的数据，按已冻结的180天窗口重新估计参数；2018年3月1日0时至3月7日23时共预测168个小时。后续时刻的滞后OT使用模型此前生成的预测，未在过程中更新真实值。区间按稀疏AR的冲击响应累计未来创新方差，以预测均值±1.96倍标准差构造。')
    p('若ψⱼ为AR冲击响应，则vₕ=σ²Σⱼ₌₀…ₕ₋₁ψⱼ²；ψ₀=1，ψⱼ由已估计AR系数递推。谐波项的未来值由日历确定。该区间固定回归参数，未完整包含参数估计与模型选择不确定性；它是名义逐期95%区间，真实覆盖通过历史检验评价，不承诺整周路径同时覆盖95%。')
    f=read('oil_forecast.csv')
    table(['预测时刻','预测均值','95%下限','95%上限'],[[str(f.iloc[i].time)[:16],num(f.iloc[i].forecast,3),num(f.iloc[i].lower_95,3),num(f.iloc[i].upper_95,3)] for i in [0,23,47,71,95,119,143,167]],[4.4,2.7,2.7,2.8])
    image('oil_forecast','图14 冻结预测与最终168小时真实值的核验')
    h('2 一次性核验与区间质量')
    scores_table(of,[selected_o,ml,matched,'Naive1','Naive168'])
    p(f'冻结后的最终一周中，主模型RMSE为{num(of.loc[selected_o,"rmse"],3)}，MAE为{num(of.loc[selected_o,"mae"],3)}，平均偏差为{num(of.loc[selected_o,"bias"],3)}，存在整体偏高。实际有{int(round(of.loc[selected_o,"coverage"]*168))}/168个值落在区间内，覆盖比例为{pct(of.loc[selected_o,"coverage"])}。树模型仅OT的RMSE为{num(of.loc[ml,"rmse"],3)}，该周点预测更好，但区间覆盖为{pct(of.loc[ml,"coverage"])}；统计主模型的区间得分较低。最终一周未用于回改已冻结的选择，也不能据单周表现认定长期最优模型。')
    image('interval_quality','图15 覆盖比例与平均宽度的共同评价，虚线为名义95%')
    p('月度留出样本覆盖较高但区间偏宽，油温树模型区间较窄却漏掉较多真实值，说明“覆盖越高越好”或“区间越窄越好”都不充分。应同时阅读覆盖比例、宽度、区间得分和检验时期；相关观测数量也不等同于独立验证次数。')
    p('完整168行预测保存为oil_forecast.csv，最终核验的真实值与各模型预测保存为oil_final_test_predictions.csv。模型预测过程本身不依赖这些未来观测。')
    h('计算复现与参考资料')
    p(f'使用已有Anaconda base环境：Python {config["versions"]["python"]}，NumPy {config["versions"]["numpy"]}，pandas {config["versions"]["pandas"]}，statsmodels {config["versions"]["statsmodels"]}，scikit-learn {config["versions"]["sklearn"]}。随机种子20261029，计算限制2线程。原始试题和数据保留，分析从附件提取的副本数据运行，原件校验值与全部输出清单可复核。')
    p('python exam/analysis.py\npython exam/supplementary_analysis.py\npython exam/make_figures.py\npython exam/verify_results.py','code')
    p('本报告的改进集中于日历因素辨析、长期外推假设、近期历史窗口、负荷信息增益及区间可靠性评价。各项结果均来自对应对照；没有把成熟算法表述为新发明，也没有用最终检验结果反向调参。')
    p('[1] Hyndman R J, Athanasopoulos G. Forecasting: Principles and Practice, 3rd ed. OTexts. https://otexts.com/fpp3/ （分解、日历调整、ETS、动态回归与时序交叉验证章节）')
    p('[2] Bandara K, Hyndman R J, Bergmeir C. MSTL: A Seasonal-Trend Decomposition Algorithm for Time Series with Multiple Seasonal Patterns. https://arxiv.org/abs/2107.13462')
    p('[3] scikit-learn 1.7文档. HistGradientBoostingRegressor；Prediction Intervals for Gradient Boosting Regression. https://scikit-learn.org/1.7/')
    p('[4] statsmodels文档. SARIMAX、ETSModel、STL与MSTL方法说明. https://www.statsmodels.org/')
    p('[5] 课程所附《文件1.xlsx》《文件2.csv》及《湖北大学课程考查试题纸（时间序列分析）-2026.doc》。')
    first_reference=next(i for i,z in enumerate(b) if z.get('text','').startswith('[1] '))
    b.insert(first_reference,dict(kind='heading',text='参考文献'))
    for block in b:
        if block.get('text','').startswith(tuple(f'[{i}] ' for i in range(1,6))):
            block['kind']='reference_last' if block['text'].startswith('[5] ') else 'reference'
        if block['kind']!='p':continue
        text=block['text']
        if text.startswith('采用周期为12的经典加法分解'):block['text']+=' 方法依据见文献[1]。'
        elif text.startswith('MSTL周期固定'):block['text']+=' 方法依据见文献[2]。'
        elif text.startswith('树模型使用HistGradientBoostingRegressor'):block['text']+=' 实现依据见文献[3]。'
        elif text.startswith('对数变换使模拟分布保持正值'):block['text']+=' 状态空间预测实现见文献[4]。'
    (Q/'report_blocks.json').write_text(json.dumps(b,ensure_ascii=False,indent=2),encoding='utf-8')
    rows=[]
    for fit in [mf,oilfit]:
        params=fit['params']
        for k,v in params.items():
            if isinstance(v,dict):
                rows.extend(dict(task=fit['task'],model=fit['model'],parameter=k+'.'+str(kk),value=vv) for kk,vv in v.items())
            elif not isinstance(v,list):rows.append(dict(task=fit['task'],model=fit['model'],parameter=k,value=v))
        if 'coefficients' in params:
            names=['const','day_sin1','day_cos1','day_sin2','day_cos2','week_sin1','week_cos1','week_sin2','week_cos2']+[f'AR_lag{k}' for k in [1,2,3,24,25,48,168]]
            rows.extend(dict(task=fit['task'],model=fit['model'],parameter=k,value=v) for k,v in zip(names,params['coefficients']))
    pd.DataFrame(rows).to_csv(S/'final_model_parameters.csv',index=False,encoding='utf-8-sig',lineterminator='\n')
    print('Blocks',len(b),'Chinese/text characters',sum(len(z.get('text','')) for z in b),'figures',sum(z['kind']=='image' for z in b))


if __name__=='__main__':main()

from pathlib import Path
import re,json
ROOT=Path(__file__).resolve().parent


def _skill_root():
    # 兼容不同机器的 skill 安装位置
    for cand in (Path.home() / '.claude' / 'skills' / 'lieflat-charts',
                 Path.home() / '.agents' / 'skills' / 'lieflat-charts'):
        if cand.exists():
            return cand
    return Path.home() / '.agents' / 'skills' / 'lieflat-charts'
SKILL = _skill_root()
t= (SKILL/'templates/reports/report-05.zh.html').read_text(encoding='utf-8')
# Copy the entire R05 template, replace demo body/data, keep original style/Mono roles.
start=t.index('<div class="sheet">'); end=t.index('</body>')
css='''<style>
@page{size:A4;margin:0}body{padding:0;display:block;background:var(--bg)}
.sheet{width:760px;max-width:760px;margin:auto}.page{width:760px;height:1122px;padding:68px 58px 64px;position:relative;page-break-after:always;overflow:hidden}.page:last-child{page-break-after:auto}
.bodycontent{font-family:SimSun,'Songti SC',serif;font-size:14px;line-height:1.5;letter-spacing:0;color:var(--ink)}
p{margin:0;text-indent:2em}.storyline{font-family:SimSun,serif;font-size:22px;line-height:1.4;margin:0 0 20px;max-width:none}.subhead{font-family:SimSun,serif;font-size:16px;margin:18px 0 8px;font-weight:bold}
h1{font-family:SimSun,serif;font-size:34px;margin-top:150px;max-width:none}.lede{font-family:SimSun,serif;font-size:14px;line-height:1.5;margin-top:25px;max-width:none;color:var(--ink)}
.foot{position:absolute;bottom:28px;left:58px;right:58px;margin:0;padding-top:8px;font-size:10px;color:#4A4944}.top{font-size:10px;border-bottom:1px solid var(--ink);padding-bottom:7px;margin-bottom:26px;font-family:SimSun,serif;color:#4A4944}
figure{margin:16px 0 8px}figure img{display:block;width:100%;filter:none}figcaption{font-family:SimSun,serif;font-size:12px;text-align:center;line-height:1.5;margin-top:6px;color:var(--ink)}
.tabletitle{font-family:SimSun,serif;font-size:11px;line-height:1.5;text-align:center;margin:14px 0 5px}table{width:100%;border-collapse:collapse;font-family:SimSun,serif;font-size:11px;line-height:1.5;color:var(--ink)}th{border-top:1px solid var(--ink);border-bottom:1px solid var(--ink);padding:6px 4px}td{border-bottom:1px solid var(--grid);padding:5px 4px;text-align:center}.note{font-family:SimSun,serif;font-size:12px;line-height:1.5;text-indent:0;margin-top:5px}.eq{font-size:15px;text-align:center;text-indent:0;margin:12px 0;line-height:1.5}.covermeta{text-align:center;margin-top:50px;font-size:17px;font-family:SimSun,serif;line-height:2.2}.cover h1{text-align:center}.cover .lede{text-align:center;text-indent:0}.figure-small img{max-height:225px;object-fit:contain}
@media print{body{background:var(--bg)}.sheet{zoom:1}.pop,.fade{animation:none}.draw{animation:none;stroke-dashoffset:0}} 
.hlbox{border:1.5px solid var(--ink);padding:10px 14px;margin:14px 0;font-family:SimSun,serif}
.hlbox .hlt{font-weight:bold;font-size:13px;margin-bottom:6px;text-indent:0}
.hlbox ol{margin:0;padding-left:20px;font-size:12px;line-height:1.6}
.hlbox li{text-indent:0;margin-bottom:3px}
.arch{border:1.5px solid var(--ink);margin:14px 0;font-family:SimSun,serif}
.arch .at{font-weight:bold;font-size:12.5px;text-align:center;padding:7px 10px;border-bottom:1px solid var(--ink)}
.arch .ar{display:flex;border-bottom:1px solid var(--grid);font-size:12px;line-height:1.45}
.arch .ar:last-child{border-bottom:none}
.arch .al{flex:0 0 56px;font-weight:bold;padding:7px 4px;border-right:1px solid var(--ink);text-align:center;display:flex;align-items:center;justify-content:center}
.arch .ad{flex:1;padding:7px 10px}
</style>'''
def tab(title,headers,rows):
 return '<div class="tabletitle">'+title+'</div><table><thead><tr>'+''.join('<th>'+x+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'
def fig(file,caption):return '<figure class="figure-small"><img src="../'+file+'"><figcaption>'+caption+'</figcaption></figure>'
def p(s):return '<p>'+s+'</p>'
def sh(s):return '<div class="subhead">'+s+'</div>'
def hlbox(title,items):return '<div class="hlbox"><div class="hlt">'+title+'</div><ol>'+''.join('<li>'+x+'</li>' for x in items)+'</ol></div>'
def arch(title,layers):
    rows=''.join('<div class="ar"><div class="al">'+l+'</div><div class="ad">'+d+'</div></div>' for l,d in layers)
    return '<div class="arch"><div class="at">'+title+'</div>'+rows+'</div>'
pages=[]
pages.append(('摘要与研究背景',p('历史量价记录是量化选股的基础，但一日收益噪声高、市场结构随时间变化，单看预测相关性难以衡量信号的实际用途。本项目（AlphaPulse）围绕赛题五提供的全市场 A 股日线数据，建立以未来一日收益为训练目标、以横截面排序为交付形式的完整分析流程。')+p('方法采用 36 个量价、形态与位置因子，逐日截面标准化、5 折 Purged 时序交叉验证和 LightGBM 稳健回归，并对个股预测序列做因果 EMA 平滑。最终选择 span=20，五折早停树数的中位数为 153。验证期竞赛口径 Rank IC 为 0.0543、Top10% 年化超额为 0.2185、平均预测换手率为 0.1177，官方复合分为 0.3520。')+p('测试期提交包含 1,599,600 条记录，竞赛口径预测换手率为 0.2395。测试期收益与 IC 将在标签公开后验证。报告并列展示可交易口径，解释满面板占位行对低换手的影响，避免把复合分提高等同于收益预测能力提升。')+hlbox('AlphaPulse 三大核心亮点',['严格防泄露：5 折 Purged CV + 6 日 embargo + 60 日内层早停隔离，滚动因子只用截至当日收盘的信息；','双轨评估：竞赛口径与可交易口径并行，揭示满面板无行情行对低换手率的真实归因；','因果平滑：因果 EMA20 过滤一日高频噪声，在评分、换手与一日信号时效之间权衡。'])+sh('1　研究背景与问题定义')+p('赛题要求对每个股票与交易日组合预测未来一天收益。我们将它转化为“当天收盘后，哪些股票在下一日更有可能取得相对较高收益”的横截面排序问题。研究分别回答数据是否可信、模型是否使用未来信息、信号在不同时间段是否稳定，以及组合换手是否可接受四个问题。')+p('官方评分为 IC 均值的 40%、年化超额的 30% 和低换手奖励的 30%。因此不能只最大化拟合精度，也不能只追求静态排名。我们的目标是在排序信号与时间稳定性之间作明确、可复核的取舍。')+ '<p class="note">关键词：原始量价数据；截面排序；Purged CV；LightGBM；EMA 平滑。</p>'))
pages.append(('数据介绍与描述性分析',sh('2.1　数据规模与字段')+p('训练集覆盖 2018-01-02 至 2024-12-31，共 1,699 个交易日；测试集覆盖 2025-01-02 至 2026-06-08，共 344 个交易日。每天都包含全部 4,650 个股票代码，属于满面板结构。价格为后复权开高低收，成交量以股计、成交额以元计，另有涨跌停标志。')+tab('表1　官方数据规模',['数据集','记录数','交易日','股票数'],[['训练集','7,900,350','1,699','4,650'],['测试集','1,599,600','344','4,650']])+sh('2.2　缺失与占位行')+p('训练标签约 14.45% 缺失。有效标签的中位数为 0，标准差约 2.99%，1% 与 99% 分位数约为 -8.35% 与 +10.00%，说明一日收益具有尖峰厚尾特征。年度均值从 2018 年的 -0.126% 到 2024 年的 +0.032%，年度标准差在 2024 年升至约 3.49%，市场结构逐年变化。涨停与跌停行分别约占 1.45% 和 0.59%，因此训练筛选和官方评价口径需要分开说明。缺失标签行不参与模型拟合，但保留它们的时间位置用于滚动计算。无行情行包含未上市、已退市和区间内停牌；它们大多是未上市或已退市的占位行。vol、amount 缺失不能当作零成交量解释。')+tab('表2　无行情行及其在 Top10% 中的影响',['指标','验证段','测试期'],[['无行情行占全部记录','12.16%','2.80%'],['竞赛口径 Top10% 中日均占比','72.15%','25.54%'],['无行情行内：上市前','95.4%','74.6%'],['无行情行内：退市后','3.1%','17.6%'],['无行情行内：区间内停牌','1.5%','7.8%']])+p('模型将缺失因子按当日截面中位数处理后，会给这些行较高排序。测试期无行情行的截面百分位中位数约 0.977，约 89.5% 处于前 10%。这些近乎不变的占位记录会压低官方预测换手率；收益评价则剔除缺失标签，两个指标的样本口径不同。')+sh('2.3　基础数据核查')+p('本地核查中，amount/vol 的量级与收盘价相符，故未把 vol 误换算为“手”。标签对齐检查支持 y_ret_1d 对应 t→t+1 收益：当天一日收益与上一行标签的秩相关约 0.9993，同一行相关仅约 0.036。这是对齐证据，不替代完整防泄露设计。描述性统计还显示，收益厚尾和年度波动变化会让单一时期的验证分数产生较大不确定性。')))
pages.append(('模型分析：因子与时间切分',sh('3.1　36 因子信息集')+p('因子仅使用当天及之前的官方字段。先在个股完整行情时间轴上计算，再筛选拟合样本，避免先删除标签缺失或涨跌停日期导致滚动窗口变形。短长期收益率显式关闭缺失价格的前值填充，防止跨股票串值或制造人工收益。')+tab('表3　因子组成与信息含义',['类别','数量','代表信息'],[['动量 / 反转','7','1/5/10/20日收益、反转、均线偏离'],['日内形态','4','振幅、区间占比、收盘位置、跳空'],['波动率','4','短长期收益波动、偏度'],['量价协同','6','量能变化、成交额强度、均价偏离'],['流动性 / 规模','4','相对成交量、对数成交额、量波动'],['Beta 代理','1','对昨日截面中位收益的敏感度'],['补充动量 / 形态 / 位置','8','2/60日收益、影线、实体、价格位置'],['截面排名','2','1/20日收益的当日百分位']])+p('全部因子在当日截面做 Z-Score，缺失按当日中位数填充并裁剪到 ±5。截面排名不跨日期。测试期拼接训练尾部 90 个连续交易日，以支持 60 日窗口；拼接只使用测试日前的历史。')+sh('3.2　时序验证和模型训练')+p('采用按交易日划分的五折扩展窗口验证，每个外层验证段之前留 6 个交易日 embargo。每折训练期内部末尾 60 个交易日用于早停，并再次隔离；外层验证不参与树数选择。得到树数后，在该折完整训练区间重拟合用于样本外预测。')+p('模型为 LightGBM 回归，使用 Huber 损失处理收益厚尾；学习率 0.03、叶子数 63、最大深度 6、叶子最少样本 300、L2 正则 5.0。最终以五折树数中位数 153 在全部合格样本上训练。原始回归输出转为每日 0～1 百分位分数，并做 EMA20；最终 pred 为 0～1 截面排序分数，赛制按排序评分。')+p('在训练集最后 1000 日设置两个连续 250 日验证窗口，各模型同因子、同切分、同 EMA20 后处理。树模型及融合综合分 0.3194—0.3223 相当，线性 Ridge 基线 0.2880，差距集中在超额收益与换手而非 IC；融合无跨窗口一致改善，故保留 LightGBM＋EMA20。此为受限窗口比较；两窗口结果接近，样本量不足以支撑严格显著性结论；单因子基线留作后续对照。')+tab('表4　多模型对比（最后 1000 日两窗口等权均分）',['模型','Rank IC','年化超额','换手率','综合分'],[['Ridge（线性基线）','0.0606','0.1064','0.2271','0.2880'],['LightGBM','0.0591','0.1875','0.1937','0.3218'],['XGBoost','0.0594','0.1887','0.2032','0.3194'],['CatBoost','0.0596','0.1864','0.1918','0.3222'],['融合（LGB+XGB）','0.0594','0.1924','0.1977','0.3222'],['融合（LGB+XGB+Cat）','0.0598','0.1893','0.1947','0.3223']])))
pages.append(('模型分析：信号证据',sh('3.3　样本外排序能力')+p('验证段为 2019-03-06 至 2024-12-31，共 1,416 个交易日。未平滑模型 IC 为 0.0620，平滑后为 0.0543；IC 正向天数比例为 69.35%。图1 展示逐日波动和累计信号，说明平均正相关并不意味着每天都有效。')+fig('fig1_ic_trend.png','图1　验证段逐日 Rank IC 及累计 IC（正式运行产物）')+sh('3.4　量能因子主导跨折重要性')+p('图2 使用五折平均 gain，不能与最终 153 棵树模型的 gain 混用。成交量 5 日变化率和 20 日变化率分别为 727、397，其次是均价偏离度 63、相对成交量 36、收盘位置 31。重要性反映模型对各因子的利用程度。')+fig('fig3_feature_importance.png','图2　五折平均 gain 的 Top10 特征重要性')+p('与旧 26 因子模型相比，全验证竞赛复合分提高约 0.0086，其中约 0.0055 来自换手；可交易口径总提升仅约 0.0012。最近 250 日年化超额从 0.1983 降至 0.1906，说明扩展因子的收益端改善有限、增益主要来自换手控制。36 因子与 EMA 跨度在已观察窗口上筛选，验证分数包含选择效应。')))
pages.append(('应用验证：评分与组合',sh('4.1　竞赛与可交易口径分开')+p('竞赛提交对齐官方脚本：IC 和超额只用有标签记录，超额与换手按规则剔除涨停。可交易对照将无行情行 pred 置为 -999，使它们不进入 Top10%，作为业务约束。可交易口径作为业务约束的对照评估。')+tab('表5　全验证段指标',['方案','Rank IC','年化超额','换手率','综合分'],[['未平滑','0.0620','0.2711','0.8734','0.1441'],['竞赛 EMA20','0.0543','0.2185','0.1177','0.3520'],['可交易 EMA20','0.0544','0.2192','0.2984','0.2980']])+p('Top10% 超额按官方定义为逐日平均超额乘 252；该口径为逐日均值年化，交易成本另作敏感性分析。两种口径的收益和 IC 接近，主要差别在无行情行参与换手计算。')+fig('fig2_excess_return.png','图3　验证期 Top10% 组合与全市场基准累计收益（曲线为复利展示，官方超额指标按日均值年化）')+tab('表6　竞赛口径的近期分段',['窗口','Rank IC','年化超额','换手率','综合分'],[['最近250日','0.0556','0.1906','0.2440','0.3062'],['最近500日','0.0605','0.2147','0.2073','0.3264']])+p('末尾 250 日参与选参，其指标为选参口径。测试期竞赛换手率为 0.2395、可交易为 0.2869，两种口径换手率项差约 0.0142。测试期换手率高于全验证段的 0.1177 是预期内的：验证段 Top10% 中日均 72.15% 是无行情占位行，它们近乎不动，把验证段换手压得很低；测试期该比例仅 25.54%，换手更接近真实持仓变化，并非模型退化。全验证段 0.3520 主要受早期无行情行和低换手奖励抬高；最近 250 日为 0.3062；测试期可直接核对预测换手率 0.2395，收益与 IC 待标签公开。')))
pages.append(('应用验证：EMA 的取舍',sh('4.2　先过滤噪声，再变得迟钝')+p('EMA 为因果平滑：s_t=αp_t+(1−α)s_(t−1)，其中 α=2/(span+1)。短跨度先过滤噪声：末尾窗口 IC 从 span=0 的 0.0541 升至 span=5 的 0.0588；更大跨度则逐渐减弱一日信号时效。')+tab('表7　EMA 跨度敏感性（w=0，竞赛口径）',['span','半衰期/日','末尾250日IC','末尾综合分','早期综合分'],[['0','0.0','0.0541','0.136','0.146'],['5','1.7','0.0588','0.226','0.308'],['12','4.1','0.0578','0.274','0.355'],['20','6.9','0.0556','0.306','0.362'],['30','10.4','0.0527','0.325','0.361'],['60','20.8','0.0454','0.347','0.354']])+p('末尾选参窗口偏好 span=60，但早期 1,166 日竞赛综合分在 span=20 更高。全验证段由 20 增至 60，IC 从 0.0543 降至 0.0430、年化超额从 0.2185 降至 0.1716，复合分却仅增加约 0.0005。本方案基于时效性与解释性保留 span=20，而非机械取网格最优。当前测试预测从测试集首日重新初始化 EMA，没有把训练尾部的预测状态带入；这是一个实现边界，可能影响测试初期的排名，后续应以固定留出段比较暖启动与冷启动。')+p('早期区间未参与本次选参，但属于此前查看过的窗口。训练目标（Huber/L2）与评分目标（IC/超额/换手）存在差异，属已知设计取舍；最终选择是对评分、收益时效性和解释性的综合判断。概括而言，更长的 EMA 跨度虽能在评分公式中继续压低换手、抬高复合分，但代价是 IC 从 0.0543 降至 0.0430、年化超额从 0.2185 降至 0.1716，损耗了一日收益信号的敏锐度；选择 span=20 是在竞赛打分、一日信号时效与实盘可解释性之间的客观取舍。')+sh('4.3　成本敏感性与交付核验')+p('官方换手率是 Jaccard 距离 d。等规模组合单日实际换出比例为 r=d/(2−d)。在可交易选参窗口，未平滑时每天替换约 76% 持仓，EMA20 后约 17%。若每次换仓买卖合计费率假设为 0.3%，采用日度 r 平均乘 252 的简单估算，年成本约由 57% 降至 12.5%。0.3% 仅为示意费率，实盘需按券商佣金、印花税与冲击成本重新测算；成本为假设敏感性分析，实盘净收益需在真实费率与冲击成本下复算。')+p('正式提交记录数为 1,599,600，键集合无重复、无缺失且与测试集完全一致；预测有限并位于 0～1。文件按交易日和代码排序，非测试输入原始行序。模型保存真实因子名，结果摘要和三张图来自同一次正式运行。')))
pages.append(('产品思路与总结结论',sh('5　产品思路与业务应用')+p('AlphaPulse 定位为研究员盘后候选池自动化辅助引擎：收盘后先校验字段和行情状态，再计算因子、输出竞赛排序分数，同时生成剔除无行情行的可交易候选池。研究员看到每只股票的分数、主要因子、近 20 日排名变化和数据状态；系统对未上市、退市、停牌、涨跌停分别标记，占位行被排除在候选池外。')+arch('图4　AlphaPulse 盘后自动化投研与风控系统',[('输入层','每日收盘 OHLCV 数据 → 完整面板对齐 → 涨跌停与无行情行标记'),('计算层','36 维因子截面 Z-Score → LightGBM 排序 → 因果 EMA20 平滑'),('风控层','双轨过滤：竞赛口径（全量交付）vs 可交易口径（无行情行置 -999）'),('输出层','Top10% 研究员复核候选池 → 次日 IC / 超额 / 换手 / 漂移监控看板')])+p('一个可执行的日流程是：数据检查通过后生成候选池，研究员从前 10% 中复核流动性和交易限制，记录最终入选与剔除原因，次日回收 Rank IC、Top10% 超额、换手和缺失率四项指标。达到预设阈值才进入观察名单，连续异常则暂停信号。这样 AlphaPulse 把研究员从每日全市场初筛中解放出来，聚焦真正的投资判断；系统处于研究验证阶段，实盘部署、行业中性化与容量测算为后续工程。')+p('上线验证应监控 IC、超额、换手和预测分布漂移，增加实际成交价格、费用、停复牌与涨跌停执行规则，另做前向观察。模型冻结后，版本更新须记录参数、数据时间和产物，防止训练代码与最终提交脱节。')+sh('6　总结与局限')+p('项目形成了完整的量价排序流水线，解决了股票分组、连续窗口、截面归一化、内层早停与最终全量训练的一致性问题。36 因子与 EMA20 在已观察验证框架下得到 0.3520 竞赛复合分，测试期可核对预测换手率 0.2395，收益与 IC 待标签公开，验证分数含因子与参数筛选的选择效应。')+p('后续改进方向：前向观察（独立留出段），以及含涨跌停、停牌与 ST 限制的完整执行回测。')+sh('资料来源与复现依据')+p('数据来源为赛题五提供的训练集与测试集字段；评价依据为赛题公布的 Rank IC、Top 组年化超额收益和预测换手率加权规则。报告中的图表均来自本项目在训练数据上的时序验证结果，未使用外部行情或外部收益标签。')))
body='<div class="sheet"><section class="page cover"><div class="top">2026 FinTechathon · 数据分析赛道 · 赛题五</div><h1>基于原始量价数据的<br><span class="hl">股票收益预测</span></h1><p class="lede">AlphaPulse · 36 因子与因果 EMA 的截面排序系统</p><div class="covermeta">小组名称：LiSquaredLab<br>小组号：245<br>2026 年 10 月</div><div class="foot"><span>项目书 · 赛题五</span><span>245-赛题5</span></div></section>'
for i,(title,c) in enumerate(pages,1):
 body+=f'<section class="page"><div class="top">2026 FinTechathon · 赛题五数据分析报告</div><h2 class="storyline"><span class="hl">{title}</span></h2><div class="bodycontent">{c}</div><div class="foot"><span>36 因子 + EMA20 · 量价排序信号</span><span>{i} / {len(pages)}</span></div></section>'
body+='</div>'
out=t[:start]+body+'</body></html>'
out=out.replace('</head>',css+'</head>')
out=re.sub(r'<link[^>]+>','',out).replace('把回答写在明处 — Lieflat 报告模板 · 反色高亮','基于原始量价数据的股票收益预测')
out=out.replace('LIEFLAT 报告模板 05 · 影响力故事（中文版）','基于原始量价数据的股票收益预测')
out=out.replace('我们将它转化为','本项目将其转化为').replace('我们的目标是在','本方案目标是在').replace('我们主动保留','本方案主动保留')
out=out.replace('测试期竞赛换手率为 0.2395、可交易为 0.2869，两种口径换手率项差约 0.0142。测试期换手率高于全验证段的 0.1177 是预期内的：验证段 Top10% 中日均 72.15% 是无行情占位行，它们近乎不动，把验证段换手压得很低；测试期该比例仅 25.54%，换手更接近真实持仓变化，并非模型退化。','测试期竞赛口径换手率为 0.2395（可交易口径为 0.2869），高于全验证段的 0.1177。验证段 Top10% 中无行情行的日均占比为 72.15%，这些近乎不变的占位记录压低了换手率；测试期该比例降至 25.54%，因此换手率升高不能单独作为模型退化的证据。')
out=out.replace('当前测试预测从测试集首日重新初始化 EMA，没有把训练尾部的预测状态带入；这是一个实现边界，可能影响测试初期的排名，后续应以固定留出段比较暖启动与冷启动。','当前测试预测从测试集首日重新初始化 EMA，没有把训练尾部的预测状态带入；span=20 时初始状态权重按 (19/21)^n 指数衰减，约 20 次更新后剩 13.5%、30 次后剩 5%，因此早期预测受冷启动影响。暖启动（继承训练尾部平滑状态）的测试期换手率为 0.2418，与冷启动 0.2395 相差约 0.002，可交易口径为 0.2911 vs 0.2869；冷启动换手率略低，该实现边界对评分的影响可忽略。')
out=out.replace('若每次换仓买卖合计费率假设为 0.3%，','若每次换仓买卖合计费率假设为 0.3%（示意值，实际需按券商佣金、印花税、过户费和市场冲击成本重新测算），')
out=out.replace('0.3% 仅为示意费率，实盘需按券商佣金、印花税与冲击成本重新测算；','')
out=out.replace('次日回收 Rank IC、Top10% 超额、换手和缺失率四项指标。','次日回收日度 Rank IC、Top10% 超额、预测换手率和无行情行占比四项指标，同时监控预测分布漂移。')
out=out.replace('次日回收 IC、超额、换手、缺失率；漂移则暂停信号','次日回收 IC、超额、换手、无行情行占比；漂移则暂停信号')
out=out.replace('上线验证应监控 IC、超额、换手和预测分布漂移，增加实际成交价格、费用、停复牌与涨跌停执行规则，另做前向观察。','上线验证应增加实际成交价格、费用、停复牌与涨跌停执行规则，另做前向观察。')
(ROOT/'project_book.html').write_text(out,encoding='utf-8')
print('Built HTML from R05; 7 body pages + anonymous cover')

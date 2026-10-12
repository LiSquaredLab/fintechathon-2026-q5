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
.tabletitle{font-family:SimSun,serif;font-size:11px;line-height:1.5;text-align:center;margin:14px 0 5px}table{width:100%;border-collapse:collapse;font-family:SimSun,serif;font-size:11px;line-height:1.5;color:var(--ink)}th{border-top:1px solid var(--ink);border-bottom:1px solid var(--ink);padding:6px 4px}td{border-bottom:1px solid var(--grid);padding:5px 4px;text-align:center}.note{font-family:SimSun,serif;font-size:14px;line-height:1.5;text-indent:2em;margin:0}.eq{font-size:15px;text-align:center;text-indent:0;margin:12px 0;line-height:1.5}.covermeta{text-align:center;margin-top:50px;font-size:17px;font-family:SimSun,serif;line-height:2.2}.cover h1{text-align:center}.cover .lede{text-align:center;text-indent:0}.figure-small img{max-height:225px;object-fit:contain}
@media print{body{background:var(--bg)}.sheet{zoom:1}.pop,.fade{animation:none}.draw{animation:none;stroke-dashoffset:0}} 
.hlbox{border:1.5px solid var(--ink);padding:10px 14px;margin:14px 0;font-family:SimSun,serif}
.hlbox .hlt{font-weight:bold;font-size:14px;margin-bottom:6px;text-indent:0}
.hlbox ol{margin:0;padding-left:22px;font-size:14px;line-height:1.5}
.hlbox li{text-indent:0;margin-bottom:3px}
.arch{border:1.5px solid var(--ink);margin:14px 0;font-family:SimSun,serif}
.arch .at{font-weight:bold;font-size:12.5px;text-align:center;padding:7px 10px;border-bottom:1px solid var(--ink)}
.arch .ar{display:flex;border-bottom:1px solid var(--grid);font-size:12px;line-height:1.45}
.arch .ar:last-child{border-bottom:none}
.arch .al{flex:0 0 56px;font-weight:bold;padding:7px 4px;border-right:1px solid var(--ink);text-align:center;display:flex;align-items:center;justify-content:center}
.arch .ad{flex:1;padding:7px 10px}
/* Tighten the two tables on body page 3 while preserving body type and leading. */
.cover .titleblock{margin-top:148px;text-align:center}
.cover h1{font-family:Arial,'Microsoft YaHei',sans-serif;font-size:68px;font-weight:700;line-height:1.05;letter-spacing:-2px;margin:0;color:var(--ink)}
.cover .title-rule{width:54px;height:3px;background:var(--ink);margin:27px auto 25px}
.cover .subtitle{font-family:SimSun,serif;font-size:27px;line-height:1.55;font-weight:bold;text-indent:0;margin:0}
.cover .lede{font-size:14px;margin-top:18px;color:#4A4944}
.cover .covermeta{margin-top:70px;font-size:16px}
.page:nth-child(4) th{padding:4px 4px}
.page:nth-child(4) td{padding:3px 4px}
</style>'''
def tab(title,headers,rows):
 return '<div class="tabletitle">'+title+'</div><table><thead><tr>'+''.join('<th>'+x+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'
def fig(file,caption):return '<figure class="figure-small"><img src="../'+file+'"><figcaption>'+caption+'</figcaption></figure>'
def p(s):return '<p>'+s+'</p>'
def sh(s):return '<div class="subhead">'+s+'</div>'
def ema_table():
 rows=json.loads((ROOT/'ema_sensitivity.json').read_text(encoding='utf-8'))
 return tab('表7　EMA 跨度敏感性（w=0，竞赛口径）',
            ['span','半衰期/日','末尾250日IC','末尾综合分','早期1166日综合分'],
            [[str(r['span']),f"{r['half_life']:.1f}",f"{r['recent']['ic']:.4f}",
              f"{r['recent']['score']:.3f}",f"{r['early']['score']:.3f}"] for r in rows])
def hlbox(title,items):return '<div class="hlbox"><div class="hlt">'+title+'</div><ol>'+''.join('<li>'+x+'</li>' for x in items)+'</ol></div>'
def arch(title):
    return '<figure class="figure-small"><img src="architecture.png" alt="LightGBM回归树集成与EMA20模型架构"><figcaption>'+title+'<br>实线：预测路径；虚线：训练与树数选择。正式提交使用单一 LightGBM 模型。</figcaption></figure>'
pages=[]
pages.append(('摘要与研究背景',p('历史量价记录是量化选股的基础，但一日收益噪声高、市场结构随时间变化，单看预测相关性难以衡量信号的实际用途。本项目（AlphaPulse）围绕赛题五提供的全市场 A 股日线数据，建立以未来一日收益为训练目标、以横截面排序为交付形式的完整分析流程。')+p('方法以 36 个量价、形态与位置因子为输入，经逐日截面标准化和 5 折 Purged 时序交叉验证，确定单一 LightGBM 稳健回归模型（153 棵树）。回归预测转为每日截面百分位，再经个股因果 EMA20 平滑，形成最终排序分数。验证期竞赛口径 Rank IC 为 0.0543、Top10% 年化超额为 0.2185、平均预测换手率为 0.1177，官方复合分为 0.3520。')+arch('图1　AlphaPulse 模型架构')+p('测试期提交包含 1,599,600 条记录，竞赛口径预测换手率为 0.2395。测试期收益与 IC 尚待验证。报告并列展示可交易口径，解释满面板占位行对低换手的影响，分别衡量排序能力与换手控制的贡献。')+hlbox('AlphaPulse 三大核心亮点',['严格防泄露：5 折 Purged CV + 6 日 embargo + 60 日内层早停窗口，拟合与早停段隔离 6 日，滚动因子只用截至当日收盘的信息；','双轨评估：竞赛口径与可交易口径并行，量化满面板无行情行对低换手率的影响；','因果平滑：因果 EMA20 降低日度排名波动，在评分、换手与一日信号时效之间权衡。'])+sh('1　研究背景与问题定义')+p('赛题要求对每个股票与交易日组合预测未来一天收益。我们将它转化为“当天收盘后，哪些股票在下一日更有可能取得相对较高收益”的横截面排序问题。研究分别回答数据是否可信、模型是否使用未来信息、信号在不同时间段是否稳定，以及组合换手是否可接受四个问题。')+p('官方评分为 IC 均值的 40%、年化超额的 30% 和低换手奖励的 30%。本方案综合评价排序信号、组合收益与时间稳定性，并给出可复核的取舍依据。')+ '<p class="note">关键词：原始量价数据；截面排序；Purged CV；LightGBM；EMA 平滑。</p>'))
pages.append(('数据介绍与描述性分析',sh('2.1　数据规模与字段')+p('训练集覆盖 2018-01-02 至 2024-12-31，共 1,699 个交易日；测试集覆盖 2025-01-02 至 2026-06-08，共 344 个交易日。每天都包含全部 4,650 个股票代码，属于满面板结构。价格为后复权开高低收，成交量以股计、成交额以元计，另有涨跌停标志。')+tab('表1　官方数据规模',['数据集','记录数','交易日','股票数'],[['训练集','7,900,350','1,699','4,650'],['测试集','1,599,600','344','4,650']])+sh('2.2　缺失与占位行')+p('训练标签约 14.45% 缺失。有效标签的中位数为 0，标准差约 2.99%，1% 与 99% 分位数约为 -8.35% 与 +10.00%，说明一日收益具有尖峰厚尾特征。年度均值从 2018 年的 -0.126% 到 2024 年的 +0.032%，年度标准差在 2024 年升至约 3.49%，市场结构逐年变化。涨停与跌停行分别约占 1.45% 和 0.59%，模型拟合剔除缺失标签、涨跌停及不满足基本行情有效性条件的样本；因子计算仍保留完整时间轴，评价则按官方规则保留相应记录。无行情行按收盘价缺失或成交量、成交额缺失及非正值识别，包含未上市、已退市和区间内停牌；它们大多是未上市或已退市的占位行。vol、amount 的缺失值按行情缺失处理，与零成交量区分。')+tab('表2　无行情行及其在 Top10% 中的影响',['指标','验证段','测试期'],[['无行情行占全部记录','12.16%','2.80%'],['竞赛口径 Top10% 中日均占比','72.15%','25.54%'],['无行情行内：上市前','95.4%','74.6%'],['无行情行内：退市后','3.1%','17.6%'],['无行情行内：区间内停牌','1.5%','7.8%']])+p('缺失因子按当日截面中位数填补；预测结果显示，无行情行整体排序较高。测试期无行情行的截面百分位中位数约 0.977，约 89.5% 处于前 10%。这些近乎不变的占位记录会压低官方预测换手率；收益评价则剔除缺失标签，两个指标的样本口径不同。')+sh('2.3　基础数据核查')+p('amount/vol 的量级与收盘价相符，与 vol 以“股”计的字段定义一致。标签对齐检查支持 y_ret_1d 对应 t→t+1 收益：当天一日收益与上一行标签的秩相关约 0.9993，同一行相关仅约 0.036。标签对齐核查与时序隔离共同构成防泄露检查。描述性统计还显示，收益厚尾和年度波动变化会让单一时期的验证分数产生较大不确定性。')))
pages.append(('模型分析：因子与时间切分',sh('3.1　36 因子信息集')+p('因子使用截至当日的官方字段，沿个股完整时间轴计算后再筛选拟合样本，以保持滚动窗口连续。收益率计算显式保留缺失价格产生的缺口。')+tab('表3　因子组成与信息含义',['类别','数量','代表信息'],[['动量 / 反转','7','1/5/10/20日收益、反转、均线偏离'],['日内形态','4','振幅、区间占比、收盘位置、跳空'],['波动率','4','短长期收益波动、偏度'],['量价协同','6','量能变化、成交额强度、均价偏离'],['流动性 / 规模','4','相对成交量、对数成交额、量波动'],['Beta 代理','1','对昨日截面中位收益的敏感度'],['补充动量 / 形态 / 位置','8','2/60日收益、影线、实体、价格位置'],['截面排名','2','1/20日收益的当日百分位']])+p('全部因子在当日截面做 Z-Score，缺失按当日中位数填充并裁剪到 ±5。截面排名按日独立计算。测试期拼接训练尾部 90 个连续交易日，以支持 60 日窗口；拼接只使用测试日前的历史。')+sh('3.2　时序验证和模型训练')+p('采用按交易日划分的五折扩展窗口验证，每个外层验证段之前留 6 个交易日 embargo。每折训练期内部末尾 60 个交易日用于早停，并再次隔离；外层验证不参与树数选择。得到树数后，在该折完整训练区间重拟合用于样本外预测。')+p('模型为 LightGBM 回归，使用 Huber 损失处理收益厚尾；学习率 0.03、叶子数 63、最大深度 6、叶子最少样本 300、L2 正则 5.0。最终以五折树数中位数 153 在全部合格样本上训练。模型以未来一日收益率为回归目标；为适配排序评价，提交截面百分位经因果 EMA20 平滑后的分数，pred 不代表预期收益率的数值大小。')+p('最后 1000 日内设两个连续 250 日窗口，统一因子、切分及 EMA20。相较 LightGBM，Ridge 的 IC 略高，但超额较低、换手较高；XGBoost 换手更高、综合分更低；CatBoost 综合分仅增约 0.0004；融合最多增约 0.0005，且跨窗口改善不一致。微小增益不足以支持更换已核验流程或增加多模型复杂度，最终选择 LightGBM＋EMA20。本表为已查看窗口上的描述性比较，分数与全量五折结果分开解读，增益的统计显著性尚待检验。')+tab('表4　多模型对比（最后 1000 日两窗口等权均分）',['模型','Rank IC','年化超额','换手率','综合分'],[['Ridge（线性基线）','0.0606','0.1064','0.2271','0.2880'],['LightGBM','0.0591','0.1875','0.1937','0.3218'],['XGBoost','0.0594','0.1887','0.2032','0.3194'],['CatBoost','0.0596','0.1864','0.1918','0.3222'],['融合（LGB+XGB）','0.0594','0.1924','0.1977','0.3222'],['融合（LGB+XGB+Cat）','0.0598','0.1893','0.1947','0.3223']])+p('表注：LightGBM 使用 Huber；XGBoost 使用平方误差，CatBoost 使用 RMSE。')))
pages.append(('模型分析：信号证据',sh('3.3　样本外排序能力')+p('验证段为 2019-03-06 至 2024-12-31，共 1,416 个交易日。未平滑模型 IC 为 0.0620，平滑后为 0.0543；IC 正向天数比例为 69.35%。图2 显示信号整体呈正向累积，同时存在日度波动和负相关交易日。')+fig('fig1_ic_trend.png','图2　验证段逐日 Rank IC 及累计 IC（正式运行产物）')+sh('3.4　量能因子主导跨折重要性')+p('图3 按五折平均分裂增益（gain）衡量因子使用程度。成交量 5 日变化率和 20 日变化率对应的 gain 分别为 727、397，其次为均价偏离度 63、相对成交量 36、收盘位置 31。')+fig('fig3_feature_importance.png','图3　五折平均 gain 的 Top10 特征重要性')+p('与旧 26 因子模型相比，全验证竞赛复合分提高约 0.0086，其中约 0.0055 来自换手；可交易口径总提升仅约 0.0012。最近 250 日年化超额从 0.1983 降至 0.1906，说明扩展因子的收益端改善有限、增益主要来自换手控制。36 因子与 EMA 跨度在已观察窗口上筛选，验证分数包含选择效应。')))
pages.append(('应用验证：评分与组合',sh('4.1　竞赛与可交易口径分开')+p('竞赛提交对齐官方脚本：IC 和超额只用有标签记录，超额与换手按规则剔除涨停。“可交易口径”指剔除无行情行的约束对照：将其 pred 置为 -999，使其不进入 Top10%；涨跌停执行、停复牌、费用与冲击成本留待完整执行回测。')+tab('表5　全验证段指标',['方案','Rank IC','年化超额','换手率','综合分'],[['未平滑','0.0620','0.2711','0.8734','0.1441'],['竞赛 EMA20','0.0543','0.2185','0.1177','0.3520'],['可交易 EMA20','0.0544','0.2192','0.2984','0.2980']])+p('Top10% 超额按官方定义为逐日平均超额乘 252；该口径为逐日均值年化，交易成本另作敏感性分析。两种口径的收益和 IC 接近，主要差别在无行情行参与换手计算。')+fig('fig2_excess_return.png','图4　验证期 Top10% 组合与全市场基准累计收益（曲线为复利展示，官方超额指标按日均值年化）')+tab('表6　竞赛口径的近期分段',['窗口','Rank IC','年化超额','换手率','综合分'],[['最近250日','0.0556','0.1906','0.2440','0.3062'],['最近500日','0.0605','0.2147','0.2073','0.3264']])+p('末尾 250 日参与选参，其指标为选参口径。测试期竞赛换手率为 0.2395、可交易为 0.2869，两种口径换手率项差约 0.0142。测试期换手率高于全验证段的 0.1177 是预期内的：验证段 Top10% 中日均 72.15% 是无行情占位行，它们近乎不动，把验证段换手压得很低；测试期该比例仅 25.54%，换手更接近真实持仓变化，并非模型退化。全验证段 0.3520 主要受早期无行情行和低换手奖励抬高；最近 250 日为 0.3062；测试期可直接核对预测换手率 0.2395，收益与 IC 尚待验证。')))
pages.append(('应用验证：EMA 的取舍',sh('4.2　EMA 平滑与信号时效的权衡')+p('EMA 为因果平滑：s_t=αp_t+(1−α)s_(t−1)，其中 α=2/(span+1)。短跨度先过滤噪声：末尾窗口 IC 从 span=0 的 0.0541 升至 span=5 的 0.0588；更大跨度则逐渐减弱一日信号时效。')+ema_table()+p('末尾选参窗口偏好 span=60，但早期 1,166 日竞赛综合分在 span=20 更高。平滑跨度由 20 增至 60 时，全验证段 IC 从 0.0543 降至 0.0430、年化超额从 0.2185 降至 0.1716，复合分却仅增加约 0.0005。本方案综合信号时效与解释性选择 span=20。当前测试预测采用冷启动，EMA 从测试集首日初始化；这是一个实现边界，可能影响测试初期的排名，后续应以固定留出段比较暖启动与冷启动。')+p('早期 1,166 日用于回顾性对照，此前已查看过，独立验证仍需新增留出段。训练目标与评分目标的差异也提示：模型选择应结合 IC、超额收益与换手率综合判断。')+sh('4.3　成本敏感性与交付核验')+p('官方换手率是 Jaccard 距离 d。等规模组合单日实际换出比例为 r=d/(2−d)。在可交易选参窗口，未平滑时每天替换约 76% 持仓，EMA20 后约 17%。若每次换仓买卖合计费率假设为 0.3%，采用日度 r 平均乘 252 的简单估算，年成本约由 57% 降至 12.5%。0.3% 仅为示意费率，实盘需按券商佣金、印花税与冲击成本重新测算；该估算用于成本敏感性分析，实际净收益需结合成交规则与真实费用复算。')+p('正式提交记录数为 1,599,600，键集合无重复、无缺失且与测试集完全一致；预测有限并位于 0～1。文件按交易日和代码排序，以键对齐测试集。模型保存真实因子名，结果摘要和三张图来自同一次正式运行。')))
pages.append(('产品思路与总结结论',sh('5　产品思路与业务应用')+sh('5.1　目标用户与核心痛点')+p('AlphaPulse 面向量化研究团队，拟作为盘后选股辅助工具。全市场 4,650 只股票的每日初筛与复核工作量较大；无行情占位行容易影响候选池组成；一日收益信号波动较大，频繁换仓会增加交易成本。产品围绕这三项问题组织数据处理、排序与人工复核流程。')+sh('5.2　核心产品功能模块')+p('拟设置四个模块：① 数据处理，校验日线字段、对齐完整面板并标记行情状态；② 排序计算，运行冻结的 36 因子、LightGBM 与因果 EMA20，生成最终分数；③ 候选池筛选，分别输出竞赛全量结果及剔除无行情行的约束对照；④ 研究看板，展示前 10% 候选清单、近 20 日排名变化和因子信息。个股因子贡献分析留待后续实现。')+sh('5.3　业务流程与日度复核')+p('T 日盘后，校验数据并完成因子计算、模型预测与平滑，生成候选池。T+1 日盘前，研究员复核流动性、行业分布及交易限制，记录入选、剔除和调整原因。T+1 日盘后，取得收益数据后计算日度 Rank IC、Top10% 超额收益及换手率，并检查预测分布变化，分别保留模型清单与人工调整结果。')+sh('5.4　商业价值与评估依据')+p('试用阶段以团队现有流程为对照，记录筛选时间、人工复核工作量及候选保留比例，效率改善幅度以实测为准。4.3 节可交易约束对照中，EMA20 将日均持仓替换比例由约 76% 降至约 17%；按买卖合计 0.3% 的示意换仓费率估算，年成本由约 57% 降至约 12.5%。该结果为成本敏感性分析，实际投资效果需结合成交规则与真实费用验证。人工调整前后的清单及表现用于评估研究流程的增量价值。')+sh('5.5　上线监控与版本追溯')+p('上线前冻结模型与参数，开展前向观察，并预先确定 IC、换手率及预测分布的异常判定标准。连续异常时暂停组合建议，由研究员复核原因。每次更新保留数据快照、模型版本与人工调整记录，支持结果追溯。目前已完成建模、排序输出及回顾性验证；看板、业务接入与完整执行回测属于后续建设内容。')+sh('6　总结与展望')+p('项目形成了完整的量价排序流水线，解决了股票分组、连续窗口、截面归一化、内层早停与最终全量训练的一致性问题。36 因子与 EMA20 在已观察验证框架下得到 0.3520 竞赛复合分，测试期可核对预测换手率 0.2395，收益与 IC 尚待验证，验证分数含因子与参数筛选的选择效应。')+p('后续改进方向：前向观察（独立留出段），以及含涨跌停、停牌与 ST 限制的完整执行回测。')+sh('资料来源与复现依据')+p('数据来源为赛题五提供的训练集与测试集字段；评价依据为赛题公布的 Rank IC、Top 组年化超额收益和预测换手率加权规则。报告中的图表均来自本项目在训练数据上的时序验证结果，数据范围限定为赛题提供的行情与训练标签。')))
body='<div class="sheet"><section class="page cover"><div class="top">2026 FinTechathon · 数据分析赛道 · 赛题五</div><div class="titleblock"><h1>AlphaPulse</h1><div class="title-rule"></div><p class="subtitle">基于原始量价数据的<br>股票收益预测与截面排序</p><p class="lede">36 因子 · LightGBM 稳健回归 · 因果 EMA20</p></div><div class="covermeta">小组名称：LiSquaredLab<br>小组号：245<br>2026 年 10 月</div><div class="foot"><span>项目书 · 赛题五</span><span>245-赛题5</span></div></section>'
for i,(title,c) in enumerate(pages,1):
 body+=f'<section class="page"><div class="top">2026 FinTechathon · 赛题五数据分析报告</div><h2 class="storyline"><span class="hl">{title}</span></h2><div class="bodycontent">{c}</div><div class="foot"><span>36 因子 + EMA20 · 量价排序信号</span><span>{i} / {len(pages)}</span></div></section>'
body+='</div>'
out=t[:start]+body+'</body></html>'
out=out.replace('</head>',css+'</head>')
out=re.sub(r'<link[^>]+>','',out).replace('把回答写在明处 — Lieflat 报告模板 · 反色高亮','AlphaPulse | 基于原始量价数据的股票收益预测与截面排序')
out=out.replace('LIEFLAT 报告模板 05 · 影响力故事（中文版）','AlphaPulse | 基于原始量价数据的股票收益预测与截面排序')
out=out.replace('我们将它转化为','本项目将其转化为').replace('我们的目标是在','本方案目标是在').replace('我们主动保留','本方案主动保留')
out=out.replace('测试期竞赛换手率为 0.2395、可交易为 0.2869，两种口径换手率项差约 0.0142。测试期换手率高于全验证段的 0.1177 是预期内的：验证段 Top10% 中日均 72.15% 是无行情占位行，它们近乎不动，把验证段换手压得很低；测试期该比例仅 25.54%，换手更接近真实持仓变化，并非模型退化。','测试期竞赛口径换手率为 0.2395（可交易口径为 0.2869），高于全验证段的 0.1177。验证段 Top10% 中无行情行的日均占比为 72.15%，这些近乎不变的占位记录压低了换手率；测试期该比例降至 25.54%，解释换手率变化时需同时考虑样本结构。')
out=out.replace('当前测试预测采用冷启动，EMA 从测试集首日初始化；这是一个实现边界，可能影响测试初期的排名，后续应以固定留出段比较暖启动与冷启动。','当前测试预测采用冷启动，EMA 从测试集首日初始化；span=20 时初始状态权重按 (19/21)^n 指数衰减，约 20 次更新后剩 13.5%、30 次后剩 5%，因此早期预测受冷启动影响。暖启动与冷启动的测试期预测换手率分别为 0.2418 和 0.2395，按“暖启动减冷启动”计算，换手评分项差约为 -0.0007；IC、超额收益及综合分的差异需待测试标签公布后验证；本次采用冷启动提交。')
out=out.replace('若每次换仓买卖合计费率假设为 0.3%，','若每次换仓买卖合计费率假设为 0.3%（示意值，实际需按券商佣金、印花税、过户费和市场冲击成本重新测算），')
out=out.replace('0.3% 仅为示意费率，实盘需按券商佣金、印花税与冲击成本重新测算；','')
out=out.replace('次日回收 Rank IC、Top10% 超额、换手和缺失率四项指标。','次日回收日度 Rank IC、Top10% 超额、预测换手率和无行情行占比四项指标，同时监控预测分布漂移。')
out=out.replace('次日回收 IC、超额、换手、缺失率；漂移则暂停信号','次日回收 IC、超额、换手、无行情行占比；漂移则暂停信号')
out=out.replace('上线验证应监控 IC、超额、换手和预测分布漂移，增加实际成交价格、费用、停复牌与涨跌停执行规则，另做前向观察。','上线验证应增加实际成交价格、费用、停复牌与涨跌停执行规则，另做前向观察。')
(ROOT/'project_book.html').write_text(out,encoding='utf-8')
print('Built HTML from R05; 7 body pages + anonymous cover')

# -*- coding: utf-8 -*-
"""
================================================================================
2026 FinTechathon 赛题五 —— 基于原始量价数据的股票收益预测
完整可直接运行流水线
================================================================================
数据清洗 → 多维复合量价因子 → 逐日截面标准化 → Purged 时序交叉验证
→ LightGBM 截面排序模型 → 测试集预测 → submission.csv → 官方复合评分复现 → 300DPI 图表

【运行】
    python fintechathon_q5.py
    python fintechathon_q5.py --data_dir ./赛题五数据 --out_dir ./output

【依赖】
    pip install pandas numpy scipy lightgbm matplotlib seaborn

--------------------------------------------------------------------------------
【本数据集实测事实（脚本按此设计，不是猜测）】
  1. vol 单位 = 股（不是手）：000001.SZ 在 2025-01-02 的 amount/vol = 11.56，
     与该股真实股价一致。因此均价偏离度因子用 (amount/vol)/close，
     并相对其自身 20 日均值取偏离；若照搬 "amount/(vol*100)/close" 会得到
     近似恒等于 0 的常数，因子完全失效（已在本地数据上验证）。
  2. 训练集 2018-01-02 ~ 2024-12-31；测试集 2025-01-02 ~ 2026-06-08，共 344 个交易日。
  3. 训练集尾部 y_ret_1d 缺失率约 30%（全样本约 14.5%），缺失与停牌/退市/涨跌停相关，
     建模前必须剔除。
  4. 测试期覆盖 344 个交易日，官方换手率项 (1-turnover)×0.3 会持续影响排名，
     故本脚本把"压低换手率"作为一等目标。

--------------------------------------------------------------------------------
【与官方 evaluate.py 的严格对齐点】（逐条核对过）
  · Rank IC(40%)       ：逐日截面 Spearman(pred, y)，去掉 y 缺失，样本<30 的交易日跳过，取均值；
  · 年化超额(30%)      ：逐日剔涨停(flag_limit_up==0)与 y 缺失，样本<100 跳过，
                         按 pred 降序取前 10%（len//10）均值 减 当日全市场均值，再 ×252；
  · 预测换手率(30%)    ：逐日剔涨停，按 pred 降序取前 10% 成分股集合 S_t，
                         turnover_t = 1 - |S_t∩S_{t-1}| / |S_t∪S_{t-1}|，取均值；
                         样本<100 的交易日会重置 prev_set（已照做）；
  · 综合得分 = IC×0.4 + 年化超额×0.3 + (1-换手率)×0.3

--------------------------------------------------------------------------------
【换手率的两把武器：为什么能同时保住 IC】
  武器 A「稳健动量锚」：把 5/10/20 日动量 + MA 偏离度等低噪声方向性信号
    做截面百分位平均，与模型预测按权重 w 线性混合。这些因子时间自相关强，
    持仓切换慢，是压低换手的直接来源。
  武器 B「个股时序 EMA 平滑」：对每只股票的预测序列做因果 EMA(span)，
    抹平 Top10% 边界附近的排名抖动 —— 成分股大进大出正是抖动造成的。
  两者都是纯因果滤波（只用 t 及之前的信息），不产生任何未来数据泄露。
  脚本用验证集网格搜索 (w, span) 取综合得分最优组合，并打印完整对比表。

--------------------------------------------------------------------------------
【防未来数据泄露的三重保证】
  a. 滚动因子全部基于「截至 t 日收盘」可得信息（参考前一日处显式 shift(1)），
     并在 (ts_code, trade_date) 排序后的个股真实时间轴上计算；
  b. Purged Time-Series CV：验证段之前留 embargo 隔离带 —— t 日标签 y_ret_1d
     的收益实现窗口跨越到 t+1，不留间隔带就会与验证段起始日重叠；
  c. 因子标准化为逐日截面操作（当日均值/标准差），缺失按当日截面中位数填充，
     不跨日借用任何全局统计量。
================================================================================
"""

import argparse
import os
import sys
import time
import warnings
import json
import gc

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")  # 无界面后端：服务器 / CI 环境也能批量导图
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter

import lightgbm as lgb
from scipy.stats import spearmanr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ==============================================================================
# 0. 全局配置
# ==============================================================================

CFG = {
    "data_dir": os.path.join(SCRIPT_DIR, "赛题五数据"),
    "test_dir": None,                    # None → 与 data_dir 相同
    "out_dir": SCRIPT_DIR,
    "train_file": "训练集.csv",
    "test_file": "测试集_X.csv",

    "drop_limit_down_in_train": True,    # 训练时也剔除跌停（跌停次日收益同样失真）
    "n_folds": 5,                        # Purged 时序 CV 折数
    "embargo_days": 6,                   # purge 间隔带天数
    "top_ratio": 0.10,                   # Top10% 组合（官方口径）
    "smooth_span": 5,                    # 个股时序 EMA 平滑跨度（--tune 关闭时生效）
    "anchor_weight": 0.35,               # 稳健动量锚权重（--tune 关闭时生效）
    # 参数搜索窗口：只用在验证段末尾这段时间上的指标选参。
    # 理由：本数据集训练集到 2024-12-31 结束，测试期是 2025-01-02 起连续 344 个交易日。
    # 验证段的早期（2019~2021）与测试期隔着 3 年以上的市场结构变化，用全段平均
    # 选参会系统性偏向"高 IC 但高换手"的组合；而官方换手率项在 344 天上取平均，
    # 越靠近测试期的表现越有参考价值。0 = 用全部验证段。
    "tune_recent_days": 250,
    "halted_pred": -999.0,               # 可交易口径下无行情行（未上市/退市/停牌）的 pred（远低于 0~1 的正常值）
    "n_jobs": -1,                        # LightGBM 线程数，-1 = 全部 CPU 核心（--n_jobs 可覆盖）
    "seed": 42,

    # 因子构建时，测试集前面挂的训练集历史长度（交易日数），
    # 用于让测试期第一天的 20 日窗口因子不为空。
    "test_history_days": 90,
}

# 全部因子列（顺序即建模顺序）。命名前缀 fNN_ 便于特征重要性与论文描述对齐。
FACTOR_COLS = [
    # ---- 动量 / 反转 ----
    "f01_ret_1d",            # 1 日收益率
    "f02_ret_5d",            # 5 日滚动收益率
    "f03_ret_10d",           # 10 日滚动收益率
    "f04_ret_20d",           # 20 日滚动收益率
    "f05_rev_5d",            # 5 日反转（-5日收益）
    "f21_ma5_bias",          # close/MA5 - 1
    "f22_ma20_bias",         # close/MA20 - 1
    # ---- 日内形态 ----
    "f06_amp",               # 日内振幅 (high-low)/close
    "f07_hl_ratio",          # 区间占比 (high-low)/(high+low)
    "f08_close_pos",         # 收盘价在当日高低区间的位置
    "f09_gap",               # 跳空 open/prev_close - 1
    # ---- 波动率 ----
    "f10_std_5d",            # 5 日收益率标准差
    "f11_std_20d",           # 20 日收益率标准差
    "f12_vol_ratio_5_20",    # 短长波动率比
    "f25_ret_skew_20d",      # 20 日收益偏度
    # ---- 量价协同 ----
    "f13_vol_chg_5d",        # 成交量 5 日变化率
    "f14_vol_chg_20d",       # 成交量 20 日变化率
    "f15_amount_ratio_5d",   # 成交额相对 20 日均值强度
    "f16_price_dev",         # 均价偏离度 (amount/vol)/close 相对自身20日均值
    "f17_vp_corr_5d",        # 5 日量价相关性
    "f18_vp_corr_10d",       # 10 日量价相关性
    # ---- 流动性 / 规模 ----
    "f19_turnover",          # vol / 20日均量
    "f20_log_amount",        # log1p(成交额)
    "f23_vol_std_5d",        # 成交量 5 日波动率
    "f24_vol_std_20d",       # 成交量 20 日波动率
    # ---- Beta 代理 ----
    "f26_beta_proxy",        # 个股收益对"昨日截面中位收益"的 20 日滚动敏感度
]

# 用于构造"稳健动量锚"的因子（低噪声、时间自相关强 → 天然低换手）
ANCHOR_COLS = ["f02_ret_5d", "f03_ret_10d", "f04_ret_20d", "f21_ma5_bias", "f22_ma20_bias"]


# ==============================================================================
# 1. 基础工具
# ==============================================================================

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def setup_matplotlib() -> None:
    """中文字体 + 出版级绘图风格（300 DPI 导出）。"""
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
        "Source Han Sans SC", "Arial Unicode MS", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False       # 负号正常显示
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["axes.titleweight"] = "bold"


def read_csv_auto(path: str, **kwargs) -> pd.DataFrame:
    """自动尝试 utf-8 / gbk 编码。"""
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, **kwargs)


def cs_rank(x: np.ndarray) -> np.ndarray:
    """截面百分位排名（0~1），把不同量纲的信号统一到可加可比的空间。"""
    return pd.Series(x).rank(pct=True, method="average").to_numpy(dtype=np.float64)


def date_indices(dates: np.ndarray) -> dict:
    """
    预计算 {交易日: 行号数组}。
    逐日截面操作（IC/换手率/z-score）在 160 万行上若用 df[df.date==d] 过滤
    会退化到 O(n_days × n_rows)，用索引字典是 O(n_rows)。
    """
    order = np.argsort(dates, kind="stable")
    sorted_dates = dates[order]
    uniq, starts = np.unique(sorted_dates, return_index=True)
    bounds = list(starts) + [len(sorted_dates)]
    return {int(d): order[bounds[i]:bounds[i + 1]] for i, d in enumerate(uniq)}


# ==============================================================================
# 2. 因子工程
# ==============================================================================

def build_factors(raw: pd.DataFrame) -> pd.DataFrame:
    """
    多维复合量价因子工程。

    要点：
      · 先按 (ts_code, trade_date) 排序 —— A 股数据按股票分块存储，
        不排序直接 rolling 会把相邻股票的数据串期，是这类比赛最常见的隐性 bug；
      · 所有参考"前一日"的量显式 shift(1)，确保 t 日因子的信息集截止 t 日收盘；
      · 返回未标准化、未填缺的因子列，缺失交给 cs_zscore 按当日截面处理。
    """
    df = raw.sort_values(["ts_code", "trade_date"], kind="mergesort").reset_index(drop=True)
    g = df.groupby("ts_code", sort=False)
    eps = 1e-9

    c = df["close"].astype(np.float64)
    h = df["high"].astype(np.float64)
    lo = df["low"].astype(np.float64)
    o = df["open"].astype(np.float64)
    v = df["vol"].astype(np.float64)
    amt = df["amount"].astype(np.float64)

    def roll(s: pd.Series, n: int, minp: int, func: str) -> pd.Series:
        """在个股时间轴上的滚动统计（返回与 df 等长的 Series）。"""
        r = s.groupby(df["ts_code"], sort=False).rolling(n, min_periods=minp)
        out = getattr(r, func)()
        # groupby.rolling 返回 MultiIndex，drop 掉 ts_code 层后按原行序对齐
        return out.reset_index(level=0, drop=True).sort_index()

    # ---------- 1) 动量 / 反转 ----------
    # 必须沿每只股票自己的时间轴计算，不能让股票边界串入上一只股票的价格。
    # 显式关闭缺失值前值填充，避免停牌/缺失价格制造人工收益。
    df["f01_ret_1d"] = g["close"].pct_change(1, fill_method=None)
    df["f02_ret_5d"] = g["close"].pct_change(5, fill_method=None)
    df["f03_ret_10d"] = g["close"].pct_change(10, fill_method=None)
    df["f04_ret_20d"] = g["close"].pct_change(20, fill_method=None)
    df["f05_rev_5d"] = -df["f02_ret_5d"]

    # ---------- 2) 日内形态 ----------
    df["f06_amp"] = (h - lo) / (c + eps)
    df["f07_hl_ratio"] = (h - lo) / (h + lo + eps)
    df["f08_close_pos"] = (c - lo) / (h - lo + eps)
    df["f09_gap"] = o / (g["close"].shift(1) + eps) - 1.0

    # ---------- 3) 波动率 ----------
    r1 = df["f01_ret_1d"]
    df["f10_std_5d"] = roll(r1, 5, 3, "std")
    df["f11_std_20d"] = roll(r1, 20, 10, "std")
    df["f12_vol_ratio_5_20"] = df["f10_std_5d"] / (df["f11_std_20d"] + eps)
    df["f25_ret_skew_20d"] = roll(r1, 20, 10, "skew")

    # ---------- 4) 成交量 / 成交额 ----------
    # 变化率以"上一日起的滚动均量"为基准（shift(1) 排除当日自身，避免机械性相关）
    v_base5 = roll(v.groupby(df["ts_code"], sort=False).shift(1), 5, 3, "mean")
    v_base20 = roll(v.groupby(df["ts_code"], sort=False).shift(1), 20, 10, "mean")
    df["f13_vol_chg_5d"] = v / (v_base5 + eps) - 1.0
    df["f14_vol_chg_20d"] = v / (v_base20 + eps) - 1.0

    amt_ma20 = roll(amt, 20, 10, "mean")
    df["f15_amount_ratio_5d"] = amt / (amt_ma20 + eps)
    df["f20_log_amount"] = np.log1p(np.clip(amt, 0.0, None))

    # 均价偏离度：当日成交均价 amount/vol（vol 单位＝股，故无需再乘 100）
    # 原始比值 ≈ close（可差几个百分点），绝对水平无意义；真正携带信息的是
    # 它相对该股自身近期水平的偏离 —— 即"收盘价相对当日成交重心的位置"。
    vwap_ratio = amt / (v + eps) / (c + eps)
    vwap_ratio = vwap_ratio.replace([np.inf, -np.inf], np.nan)
    vwap_base = roll(vwap_ratio, 20, 10, "mean")
    df["f16_price_dev"] = vwap_ratio / (vwap_base + eps) - 1.0

    # 量价相关性：收益率 与 成交量 的滚动相关（价涨量增 → 正相关，趋势更可信）
    vlag = v.groupby(df["ts_code"], sort=False).shift(0)
    for n, col in ((5, "f17_vp_corr_5d"), (10, "f18_vp_corr_10d")):
        tmp = pd.DataFrame({"r": r1.to_numpy(), "v": vlag.to_numpy(),
                            "code": df["ts_code"].astype(str).to_numpy()})
        corr = (tmp.groupby("code", sort=False)
                   .apply(lambda d, n=n: d["r"].rolling(n, min_periods=max(3, n // 2))
                                             .corr(d["v"]), include_groups=False))
        corr = corr.reset_index(level=0, drop=True).sort_index()
        df[col] = corr.to_numpy()

    # ---------- 5) 流动性 / 均线结构 ----------
    df["f19_turnover"] = v / (roll(v, 20, 10, "mean") + eps)
    df["f23_vol_std_5d"] = roll(v, 5, 3, "std")
    df["f24_vol_std_20d"] = roll(v, 20, 10, "std")

    ma5 = roll(c, 5, 3, "mean")
    ma20 = roll(c, 20, 10, "mean")
    df["f21_ma5_bias"] = c / (ma5 + eps) - 1.0
    df["f22_ma20_bias"] = c / (ma20 + eps) - 1.0

    # ---------- 6) Beta 代理 ----------
    # 个股收益对"市场共同因子"（当日截面中位收益）的 20 日滚动敏感度。
    # 用截面中位数充当市场收益：无需额外指数数据，且对极端个股稳健。
    df["_cs_med_ret"] = df.groupby("trade_date")["f01_ret_1d"].transform("median")
    med_lag = df.groupby("ts_code", sort=False)["_cs_med_ret"].shift(1)
    cov = roll(pd.Series(r1.to_numpy() * med_lag.to_numpy(), index=df.index), 20, 10, "mean") \
        - roll(r1, 20, 10, "mean") * roll(med_lag, 20, 10, "mean")
    var = roll(pd.Series(med_lag.to_numpy() ** 2, index=df.index), 20, 10, "mean") \
        - roll(med_lag, 20, 10, "mean") ** 2
    df["f26_beta_proxy"] = cov / (var.abs() + eps)
    df.drop(columns=["_cs_med_ret"], inplace=True)

    # inf → NaN，交给标准化阶段按当日截面中位数填充
    for col in FACTOR_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)

    return df


def cs_zscore(df: pd.DataFrame, cols, date_idx: dict, date_col: str = "trade_date") -> pd.DataFrame:
    """
    逐日截面 Z-Score 标准化 —— 剥离大盘 Beta 与市场整体波动噪声。

        z_{i,t} = (x_{i,t} - mean_t(x)) / std_t(x)

    缺失值用"当日截面中位数"填充（而非全局 0）：缺失主要来自上市初期窗口不足，
    按当日中性值处理不会人为造出方向性信号。最后 clip 到 ±5 抑制极端值。

    逐日计算 → 天然无跨期泄露，也使得同日全市场被整体抬高（大盘 Beta）
    的效应被完全消掉，模型只能看到"个股相对当日市场"的强弱。
    """
    for col in cols:
        s = df[col]
        med = s.groupby(df[date_col], sort=False).transform("median")
        df[col] = s.fillna(med).fillna(0.0)

    mean_ = df.groupby(date_col, sort=False)[cols].transform("mean")
    std_ = df.groupby(date_col, sort=False)[cols].transform("std")
    std_ = std_.replace(0.0, np.nan)
    z = (df[cols] - mean_) / std_
    df[cols] = z.fillna(0.0).clip(-5.0, 5.0).astype(np.float32)
    return df


def anchor_signal(df: pd.DataFrame, date_idx: dict,
                  anchor_cols=ANCHOR_COLS) -> np.ndarray:
    """
    稳健动量锚：多个中长周期动量/均线偏离因子的等权截面百分位平均。

    为什么它存在：模型预测的训练目标是拟合 y_ret_1d，逐日噪声极大，
    输出天然抖动 → Top10% 成分股天天换 → 换手率惩罚吃满。
    而慢速动量因子的截面排名天天高度相似，是低换手的天然来源。
    取截面百分位平均，得到 0~1 的连续锚信号，与模型预测同量纲可直接加权。
    """
    cols = [c for c in anchor_cols if c in df.columns]
    anchor = np.zeros(len(df), dtype=np.float64)
    for c in cols:
        values = df[c].to_numpy(dtype=np.float64)
        ranked = np.zeros(len(df), dtype=np.float64)
        for idx in date_idx.values():
            ranked[idx] = cs_rank(values[idx])
        anchor += ranked
    anchor /= max(len(cols), 1)
    return anchor


# ==============================================================================
# 3. 数据加载与清洗
# ==============================================================================

def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    """统一字段类型并压缩内存（790 万行 × 数十列，float64 会吃掉数 GB）。"""
    df["trade_date"] = df["trade_date"].astype(np.int32)
    df["ts_code"] = df["ts_code"].astype(str)
    for col in ("open", "high", "low", "close", "vol", "amount", "y_ret_1d"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float32)
    for col in ("flag_limit_up", "flag_limit_down"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(np.int8)
    return df


def load_train(path: str, history_days: int = 0):
    """加载完整行情和原始尾部历史；清洗条件只用于模型拟合。"""
    t = time.time()
    log(f"读取训练集: {path}")
    df = read_csv_auto(path)
    log(f"  原始规模: {len(df):,} 行 | 交易日 {df['trade_date'].nunique()} 个 | "
        f"股票 {df['ts_code'].nunique():,} 只 | 用时 {time.time() - t:.1f}s")
    df = _coerce(df)

    # 测试期因子需要训练集尾部的连续原始行情。保留这段未清洗历史，
    # 避免删除标签缺失或涨跌停行后破坏滚动窗口的时间连续性。
    raw_history = None
    if history_days:
        raw_dates = np.sort(df["trade_date"].unique())
        hist_start = raw_dates[-history_days] if len(raw_dates) > history_days else raw_dates[0]
        history_cols = ["ts_code", "trade_date", "open", "high", "low", "close",
                        "vol", "amount", "flag_limit_up", "flag_limit_down"]
        raw_history = df.loc[df["trade_date"] >= hist_start, history_cols].copy()

    # 先保留完整行情计算滚动因子；清洗只作为模型拟合的掩码。
    return (df, raw_history) if history_days else df


def training_mask(df: pd.DataFrame) -> np.ndarray:
    eligible = (df['y_ret_1d'].notna() & (df['close'] > 0) &
                (df['high'] >= df['low']) & (df['vol'] >= 0) &
                (df['amount'] >= 0) & (df['flag_limit_up'] == 0))
    if CFG['drop_limit_down_in_train']:
        eligible &= df['flag_limit_down'] == 0
    return eligible.to_numpy()


def halted_mask(df: pd.DataFrame) -> np.ndarray:
    """
    无行情判定（下文统称"无行情行"）：vol 或 amount 为 0 或缺失，或 close 缺失。

    数据是"全部 ts_code × 全部交易日"的满面板，无行情行大多是占位行而非停牌：
    验证段 95.4% 是上市前、3.1% 是退市后、只有 1.5% 是区间内停牌；测试期分别为
    74.6% / 17.6% / 7.8%。这类行的 vol、amount 是 NaN 而不是 0（测试集 vol==0 的
    行数为 0），所以必须把缺失值也算进来，否则一行都匹配不到。
    """
    vol, amt = df["vol"], df["amount"]
    halted = vol.isna() | amt.isna() | (vol <= 0) | (amt <= 0) | df["close"].isna()
    return halted.to_numpy()


def demote_halted(pred: np.ndarray, halted: np.ndarray) -> np.ndarray:
    """
    可交易口径：把无行情行（未上市/退市/停牌）的 pred 压到 CFG['halted_pred']，使其不可能进入 Top10%。

    只用于可交易版本（submission_tradable.csv 与 *_tradable 指标），不用于竞赛提交：
    官方评分只剔除涨停股，不剔除无行情行；而 IC 与 Top 组超额只用有标签的行，无行情行
    几乎都没有标签，所以这一处理对竞赛得分只会通过换手率项产生净损失。
    这是交易约束，不是更准确的收益预测。
    必须在 EMA 平滑之后调用：压低值不进入平滑，复牌后的预测不受影响。
    用有限值而非 -inf：导出前的自检要求 pred 全部有限。
    """
    out = np.asarray(pred, dtype=np.float64).copy()
    out[halted] = CFG["halted_pred"]
    return out


def top_halted_share(df: pd.DataFrame, halted: np.ndarray, pred_col: str = "pred",
                     date_idx=None, top_ratio: float = 0.10, min_n: int = 100) -> float:
    """官方换手率口径下（剔涨停、按 pred 降序取前 10%），Top 组中无行情行的日均占比。"""
    if date_idx is None:
        date_idx = date_indices(df["trade_date"].to_numpy())
    pred = df[pred_col].to_numpy(dtype=np.float64)
    lim = df["flag_limit_up"].to_numpy()
    step = int(round(1.0 / top_ratio))
    shares = []
    for d in sorted(date_idx):
        idx = date_idx[d]
        idx = idx[lim[idx] == 0]
        if idx.size < min_n:
            continue
        order = idx[np.argsort(-pred[idx], kind="stable")]
        shares.append(halted[order[:max(order.size // step, 1)]].mean())
    return float(np.mean(shares)) if shares else 0.0


def load_test(path: str) -> pd.DataFrame:
    """加载测试集 X（不含标签）。"""
    log(f"读取测试集: {path}")
    df = _coerce(read_csv_auto(path))
    log(f"  规模: {len(df):,} 行 | 交易日 {df['trade_date'].nunique()} 个 "
        f"({df['trade_date'].min()} ~ {df['trade_date'].max()})")
    return df


# ==============================================================================
# 4. 官方评分公式复现（本地验证用，逐条对齐 evaluate.py）
# ==============================================================================

def daily_rank_ic(df: pd.DataFrame, pred_col="pred", y_col="y_ret_1d",
                  date_col="trade_date", date_idx=None, min_n=30):
    """逐日截面 Spearman Rank IC 序列 → (均值, std, ICIR, IC>0 占比, 序列)。"""
    if date_idx is None:
        date_idx = date_indices(df[date_col].to_numpy())
    pred = df[pred_col].to_numpy(dtype=np.float64)
    y = df[y_col].to_numpy(dtype=np.float64)

    ics = []
    for d in sorted(date_idx):
        idx = date_idx[d]
        yy = y[idx]
        ok = np.isfinite(yy)
        if ok.sum() < min_n:
            continue
        ic, _ = spearmanr(pred[idx][ok], yy[ok])
        if np.isfinite(ic):
            ics.append(ic)

    ics = np.asarray(ics, dtype=np.float64)
    if ics.size == 0:
        return 0.0, 0.0, 0.0, 0.0, ics
    std = float(ics.std(ddof=1)) if ics.size > 1 else 0.0
    return (float(ics.mean()), std,
            float(ics.mean() / std) if std > 0 else 0.0,
            float((ics > 0).mean()), ics)


def daily_top_excess(df: pd.DataFrame, pred_col="pred", y_col="y_ret_1d",
                     limit_col="flag_limit_up", top_ratio=0.10,
                     date_idx=None, min_n=100):
    """
    Top 组年化超额（官方口径）：
      逐日剔涨停 + 剔 y 缺失 → 样本<100 跳过 → 按 pred 降序取前 10%
      → (Top 均值 - 全市场均值) → 累加平均 → ×252
    同时返回逐日 Top 收益 / 全市场收益序列，供 fig2 画图。
    """
    if date_idx is None:
        date_idx = date_indices(df["trade_date"].to_numpy())
    pred = df[pred_col].to_numpy(dtype=np.float64)
    y = df[y_col].to_numpy(dtype=np.float64)
    lim = df[limit_col].to_numpy() if limit_col in df.columns else np.zeros(len(df), dtype=np.int8)
    step = int(round(1.0 / top_ratio))

    excess, top_rets, mkt_rets, dates = [], [], [], []
    for d in sorted(date_idx):
        idx = date_idx[d]
        ok = (lim[idx] == 0) & np.isfinite(y[idx])
        idx = idx[ok]
        if idx.size < min_n:
            continue
        order = idx[np.argsort(-pred[idx], kind="stable")]
        n_top = max(order.size // step, 1)
        top_ret = float(y[order[:n_top]].mean())
        mkt_ret = float(y[order].mean())
        excess.append(top_ret - mkt_ret)
        top_rets.append(top_ret)
        mkt_rets.append(mkt_ret)
        dates.append(int(d))

    excess = np.asarray(excess, dtype=np.float64)
    return (float(excess.mean() * 252) if excess.size else 0.0,
            float(np.mean(top_rets) * 252) if top_rets else 0.0,
            np.asarray(top_rets), np.asarray(mkt_rets), dates)


def daily_turnover(df: pd.DataFrame, pred_col="pred", limit_col="flag_limit_up",
                   top_ratio=0.10, date_idx=None, min_n=100):
    """
    预测换手率（官方口径）：
      turnover_t = 1 - |S_t ∩ S_{t-1}| / |S_t ∪ S_{t-1}|
    其中 S_t 为当日剔涨停后按 pred 降序的前 10% 成分股集合。
    注意官方在样本<100 的交易日会重置 prev_set，此处严格照做。
    """
    if date_idx is None:
        date_idx = date_indices(df["trade_date"].to_numpy())
    pred = df[pred_col].to_numpy(dtype=np.float64)
    codes = df["ts_code"].astype(str).to_numpy()
    lim = df[limit_col].to_numpy() if limit_col in df.columns else np.zeros(len(df), dtype=np.int8)
    step = int(round(1.0 / top_ratio))

    turnovers = []
    prev_set = None
    for d in sorted(date_idx):
        idx = date_idx[d]
        idx = idx[lim[idx] == 0]
        if idx.size < min_n:
            prev_set = None
            continue
        order = idx[np.argsort(-pred[idx], kind="stable")]
        n_top = max(order.size // step, 1)
        curr = set(codes[order[:n_top]])
        if prev_set:
            union = len(curr | prev_set)
            if union > 0:
                turnovers.append(1.0 - len(curr & prev_set) / union)
        prev_set = curr

    return (float(np.mean(turnovers)) if turnovers else 0.0,
            np.asarray(turnovers, dtype=np.float64))


def composite_score(ic_mean: float, annual_excess: float, mean_turnover: float) -> float:
    """官方综合得分公式。"""
    return ic_mean * 0.4 + annual_excess * 0.3 + (1.0 - mean_turnover) * 0.3


def evaluate_frame(df: pd.DataFrame, tag: str = "验证集", date_idx=None) -> dict:
    """对含 (ts_code, trade_date, pred, y_ret_1d, flag_limit_up) 的表做完整官方评分。"""
    if date_idx is None:
        date_idx = date_indices(df["trade_date"].to_numpy())
    ic_mean, ic_std, icir, ic_pos, ic_series = daily_rank_ic(df, date_idx=date_idx)
    annual_excess, top_annual, top_rets, mkt_rets, dates = daily_top_excess(df, date_idx=date_idx)
    mean_turnover, turnovers = daily_turnover(df, date_idx=date_idx)
    score = composite_score(ic_mean, annual_excess, mean_turnover)

    log(f"  ---------- {tag} ----------")
    log(f"  Rank IC  : mean={ic_mean:+.6f}  std={ic_std:.6f}  "
        f"ICIR={icir:+.4f}  IC>0={ic_pos:.2%}  (有效交易日 {len(ic_series)})")
    log(f"  年化超额 : {annual_excess:+.4f}   (Top 组年化绝对收益 {top_annual:+.4f})")
    log(f"  换手率   : {mean_turnover:.4f}   (1-turnover = {1 - mean_turnover:.4f})")
    log(f"  综合得分 : {score:+.6f} = {ic_mean:+.6f}×0.4 + "
        f"{annual_excess:+.4f}×0.3 + {1 - mean_turnover:.4f}×0.3")

    return {
        "ic_mean": ic_mean, "ic_std": ic_std, "icir": icir, "ic_positive_ratio": ic_pos,
        "ic_series": ic_series, "annual_excess": annual_excess, "top_annual": top_annual,
        "top_rets": top_rets, "mkt_rets": mkt_rets, "eval_dates": dates,
        "mean_turnover": mean_turnover, "turnovers": turnovers, "final_score": score,
    }


# ==============================================================================
# 5. 模型与交叉验证
# ==============================================================================

def build_model(seed: int) -> lgb.LGBMRegressor:
    """
    LightGBM 截面排序模型（回归口径）。

    抑制预测值剧烈波动的关键超参：
      · objective='huber'            : 对收益率厚尾稳健，避免个别极端样本主导分裂；
      · max_depth=6 / num_leaves=63  : 控制单树复杂度，不允许模型刻画噪声级交互；
      · min_child_samples=300        : 叶节点样本下限，强平滑；
      · lambda_l2=5.0 / lambda_l1=0.5: 叶子权重正则，直接压小输出幅度；
      · feature/bagging_fraction<1   : 双重随机化抑制过拟合；
      · learning_rate=0.03 + 早停     : 小步长多树，预测更接近因子的平滑组合。
    """
    return lgb.LGBMRegressor(
        objective="huber",
        alpha=0.9,                  # huber δ：覆盖约 90% 的日常收益波动
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=6,
        min_child_samples=300,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=5.0,
        n_jobs=CFG["n_jobs"],
        random_state=seed,
        verbose=-1,
    )


def purged_time_series_split(dates: np.ndarray, n_folds: int, embargo_days: int):
    """
    Purged Time-Series Cross-Validation（严格按 trade_date 切分）。

    做法：
      · 有序交易日序列切成 n_folds+1 段；
      · 第 k 折：训练段 = [起点, 第 k+1 段起点 - embargo)，验证段 = 第 k+1 段；
      · Purge 的理由：t 日标签 y_ret_1d 的收益实现窗口跨越到 t+1，训练段尾部
        若紧贴验证段起始，则存在标签-特征窗口重叠 → 必须留 embargo 隔离带；
      · 训练段永远在验证段之前，只有"过去预测未来"。
    """
    n = len(dates)
    bounds = np.linspace(0, n, n_folds + 2).astype(int)
    folds = []
    for k in range(n_folds):
        tr_end = max(bounds[k + 1] - embargo_days, 1)
        va_start, va_end = bounds[k + 1], bounds[k + 2]
        if tr_end <= 0 or va_end <= va_start:
            continue
        folds.append((dates[:tr_end], dates[va_start:va_end]))
    return folds


def apply_smoothing(df: pd.DataFrame, pred: np.ndarray, span: int) -> np.ndarray:
    """
    按 ts_code 的时间序列 EMA 平滑 —— 压低换手的核心手段。

    y_t = α·x_t + (1-α)·y_{t-1},  α = 2/(span+1)

    Top10% 成分股的大进大出，本质是"排名在边界附近抖动"。
    对每只股票的预测序列做因果 EMA 后，分数轨迹平滑，边界抖动被抹平，
    相邻日 Top 集合自然高度重叠 → 换手率骤降，而截面排序（IC）几乎不受影响。
    纯"过去→现在"滤波，不引入未来信息。
    """
    if not span or span <= 0:
        return pred.astype(np.float64)
    t = pd.DataFrame({
        "code": df["ts_code"].astype(str).to_numpy(),
        "date": df["trade_date"].to_numpy(),
        "p": pred.astype(np.float64),
    })
    order = np.lexsort((t["date"].to_numpy(), t["code"].to_numpy()))
    t = t.iloc[order]
    t["ps"] = t.groupby("code", sort=False)["p"].transform(
        lambda s: s.ewm(span=span, adjust=False).mean())
    out = np.empty(len(t), dtype=np.float64)
    out[order] = t["ps"].to_numpy()
    return out


def combine_and_smooth(df: pd.DataFrame, model_pred_rank: np.ndarray,
                       anchor_rank: np.ndarray, w: float, span: int) -> np.ndarray:
    """锚权重混合 + 个股时序平滑（两步顺序固定：先混合再平滑）。"""
    blended = (1.0 - w) * np.asarray(model_pred_rank, dtype=np.float64) + \
              w * np.asarray(anchor_rank, dtype=np.float64)
    return apply_smoothing(df, blended, span)


# ==============================================================================
# 6. 图表导出
# ==============================================================================

def fig1_ic_trend(ic_series: np.ndarray, out_dir: str) -> str:
    """fig1_ic_trend.png：逐日截面 Rank IC 柱状走势 + 累计 IC 曲线。"""
    fig, ax1 = plt.subplots(figsize=(14, 6.2))
    x = np.arange(len(ic_series))
    colors = ["#2E86AB" if v >= 0 else "#C0392B" for v in ic_series]
    ax1.bar(x, ic_series, color=colors, alpha=0.75, width=0.85, label="逐日 Rank IC")
    ax1.axhline(0, color="black", linewidth=1.0)
    ax1.axhline(float(ic_series.mean()), color="#F39C12", linestyle="--", linewidth=1.8,
                label=f"IC 均值 = {ic_series.mean():+.4f}")
    ax1.set_xlabel("验证期交易日序号")
    ax1.set_ylabel("Rank IC")
    ax1.set_title("逐日截面 Rank IC 走势")
    ax1.legend(loc="upper left", fontsize=11)

    ax2 = ax1.twinx()
    ax2.plot(x, np.cumsum(ic_series), color="#6C3483", linewidth=2.4,
             label="累计 IC", zorder=5)
    ax2.set_ylabel("累计 Rank IC", color="#6C3483")
    ax2.tick_params(axis="y", labelcolor="#6C3483")
    ax2.grid(False)
    ax2.legend(loc="upper right", fontsize=11)

    sd = ic_series.std(ddof=1) if ic_series.size > 1 else 0.0
    # 指标文本框放在右下角：累计 IC 曲线是从左下往右上走，左上/左下的固定位置
    # 会被曲线穿过（左下尤其明显），右下角是这张图里唯一长期空置的区域。
    ax1.text(0.988, 0.03,
             f"ICIR = {ic_series.mean() / sd if sd > 0 else 0:+.3f}    "
             f"IC>0 占比 = {(ic_series > 0).mean():.2%}    交易日数 = {len(ic_series)}",
             transform=ax1.transAxes, fontsize=11, va="bottom", ha="right",
             bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                       alpha=0.92, edgecolor="#999999"))

    path = os.path.join(out_dir, "fig1_ic_trend.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    log(f"  已导出: {path}")
    return path


def fig2_excess_return(top_rets: np.ndarray, mkt_rets: np.ndarray, out_dir: str) -> str:
    """fig2_excess_return.png：Top10% 组合 vs 全市场基准累计收益对比（均已剔涨停）。"""
    cum_top = (np.cumprod(1.0 + top_rets) - 1.0) * 100
    cum_mkt = (np.cumprod(1.0 + mkt_rets) - 1.0) * 100
    cum_exc = (np.cumprod(1.0 + (top_rets - mkt_rets)) - 1.0) * 100
    x = np.arange(len(top_rets))

    fig, ax = plt.subplots(figsize=(14, 6.2))
    ax.plot(x, cum_top, color="#C0392B", linewidth=2.6, label="Top10% 组合（剔除涨停）")
    ax.plot(x, cum_mkt, color="#2E86AB", linewidth=2.4, label="全市场等权基准（剔除涨停）")
    ax.fill_between(x, cum_mkt, cum_top, where=(cum_top >= cum_mkt),
                    color="#C0392B", alpha=0.12, label="累计超额区间")
    ax.fill_between(x, cum_mkt, cum_top, where=(cum_top < cum_mkt),
                    color="#2E86AB", alpha=0.12)
    ax.plot(x, cum_exc, color="#27AE60", linewidth=2.0, linestyle="--",
            label="超额收益（复利口径）")
    ax.axhline(0, color="black", linewidth=1.0)
    ax.set_xlabel("验证期交易日序号")
    ax.set_ylabel("累计收益率")
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=1, xmax=100))
    ax.set_title("Top10% 选股组合 vs 全市场基准：累计收益率对比")
    ax.legend(loc="best", fontsize=11)

    n = max(len(top_rets), 1)
    ann_top = np.prod(1.0 + top_rets) ** (252 / n) - 1
    ann_mkt = np.prod(1.0 + mkt_rets) ** (252 / n) - 1
    ax.text(0.012, 0.96,
            f"Top 组合年化 = {ann_top:.2%}    基准年化 = {ann_mkt:.2%}    "
            f"日均超额×252 = {np.mean(top_rets - mkt_rets) * 252:.2%}",
            transform=ax.transAxes, fontsize=11, va="top",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                      alpha=0.85, edgecolor="#999999"))

    path = os.path.join(out_dir, "fig2_excess_return.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    log(f"  已导出: {path}")
    return path


def fig3_feature_importance(importance: pd.Series, out_dir: str, top_n: int = 10) -> str:
    """fig3_feature_importance.png：特征重要性 Top10 柱状图（gain 口径）。"""
    top = importance.sort_values(ascending=False).head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    ax.barh(top.index, top.values, color=sns.color_palette("viridis", n_colors=len(top)),
            edgecolor="#333333", linewidth=0.6)
    for i, val in enumerate(top.values):
        ax.text(val + top.values.max() * 0.012, i, f"{val:,.0f}", va="center", fontsize=10.5)
    ax.set_xlabel("LightGBM 分裂增益（gain，跨折均值）")
    ax.set_title(f"特征重要性 Top {top_n}")
    ax.set_xlim(0, top.values.max() * 1.16)
    sns.despine(ax=ax, left=True)

    path = os.path.join(out_dir, "fig3_feature_importance.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    log(f"  已导出: {path}")
    return path


# ==============================================================================
# 7. 主流程
# ==============================================================================

def main():
    ap = argparse.ArgumentParser(description="2026 FinTechathon 赛题五 自动化流水线")
    ap.add_argument("--data_dir", default=CFG["data_dir"], help="训练集所在目录")
    ap.add_argument("--test_dir", default=CFG["test_dir"], help="测试集目录（默认同 data_dir）")
    ap.add_argument("--out_dir", default=CFG["out_dir"], help="输出目录")
    ap.add_argument("--train_file", default=CFG["train_file"])
    ap.add_argument("--test_file", default=CFG["test_file"])
    ap.add_argument("--folds", type=int, default=CFG["n_folds"])
    ap.add_argument("--embargo_days", type=int, default=CFG["embargo_days"])
    ap.add_argument("--smooth_span", type=int, default=CFG["smooth_span"]
                    if "smooth_span" in CFG else 5, help="个股时序 EMA 平滑跨度")
    ap.add_argument("--anchor_weight", type=float, default=CFG["anchor_weight"],
                    help="稳健动量锚权重")
    ap.add_argument("--tune", type=int, default=1,
                    help="1 = 网格搜索 (anchor_weight, smooth_span) 取最优；0 = 用命令行给定值")
    ap.add_argument("--tune_recent_days", type=int, default=CFG["tune_recent_days"],
                    help="仅用验证段末尾 N 个交易日选参；0 = 用全部验证段")
    ap.add_argument("--max_train_rows", type=int, default=0,
                    help="调试用：仅取末尾 N 行训练（0 = 全量）")
    ap.add_argument("--demote_halted", action="store_true",
                    help="额外导出可交易版 submission_tradable.csv（无行情行：未上市/退市/停牌，pred 置为 -999）；"
                         "竞赛版 submission.csv 不受影响")
    ap.add_argument("--n_jobs", type=int, default=CFG["n_jobs"],
                    help="LightGBM 线程数；-1 = 全部 CPU 核心")
    ap.add_argument("--skip_submission", action="store_true", help="只做本地验证，不导出提交文件")
    args = ap.parse_args()
    CFG["n_jobs"] = args.n_jobs              # build_model() 从 CFG 读取线程数

    test_dir = args.test_dir or args.data_dir
    os.makedirs(args.out_dir, exist_ok=True)
    setup_matplotlib()
    t_start = time.time()

    train_path = os.path.join(args.data_dir, args.train_file)
    test_path = os.path.join(test_dir, args.test_file)
    test_available = os.path.exists(test_path)
    if not os.path.exists(train_path):
        log(f"错误: 找不到训练集 {train_path}")
        sys.exit(1)
    if not test_available:
        log(f"警告: 找不到测试集 {test_path}，将只做本地验证（跳过提交导出）")

    # ------------------------------------------------------------------
    log("=" * 78)
    log("步骤 1/7  数据加载与交易限制处理（剔标签缺失 / 涨停 / 跌停）")
    log("=" * 78)
    train_result = load_train(train_path, history_days=CFG["test_history_days"] if test_available else 0)
    if test_available:
        train, raw_history = train_result
    else:
        train = train_result
        raw_history = None
    if args.max_train_rows and args.max_train_rows < len(train):
        # 原始数据按股票分块存放，tail() 截到的是排在最后几只股票的全部历史，
        # 而不是全市场最近的交易日。改为从最近的交易日往前取整日，直到行数
        # 达到 max_train_rows，保证每个交易日都是完整截面（截面标准化与评分需要）。
        rows_per_day = train.groupby("trade_date").size().sort_index(ascending=False)
        n_days = int(np.searchsorted(rows_per_day.cumsum().to_numpy(), args.max_train_rows)) + 1
        keep_dates = rows_per_day.index[:n_days]
        train = train[train["trade_date"].isin(keep_dates)].reset_index(drop=True)
        log(f"  [调试] 截取最近 {len(keep_dates)} 个交易日 "
            f"({keep_dates.min()} ~ {keep_dates.max()})：{len(train):,} 行 | "
            f"股票 {train['ts_code'].nunique():,} 只")

    # ------------------------------------------------------------------
    log("=" * 78)
    log("步骤 2/7  多维复合量价因子工程（动量 / 波动 / 量价协同）")
    log("=" * 78)
    t = time.time()
    train = build_factors(train)
    log(f"  因子构建完成：{len(FACTOR_COLS)} 个因子，用时 {time.time() - t:.1f}s")

    date_idx_tr = date_indices(train["trade_date"].to_numpy())
    train = cs_zscore(train, FACTOR_COLS, date_idx_tr)     # 逐日截面 Z-Score
    log("  逐日截面 Z-Score 标准化完成（当日均值0/标准差1，clip ±5）")

    # 稳健动量锚：由标准化后的因子再做截面百分位平均
    train["_anchor_rank"] = anchor_signal(train, date_idx_tr)
    anchor_rank_tr = train["_anchor_rank"].to_numpy()

    # ------------------------------------------------------------------
    all_dates = np.sort(train["trade_date"].unique())
    folds = purged_time_series_split(all_dates, args.folds, args.embargo_days)
    log("=" * 78)
    log(f"步骤 3/7  Purged 时序交叉验证：{len(folds)} 折，embargo = {args.embargo_days} 交易日")
    log("=" * 78)
    for i, (tr_d, va_d) in enumerate(folds, 1):
        log(f"  Fold {i}: 训练 {tr_d[0]}~{tr_d[-1]} ({len(tr_d)}日) | "
            f"隔离 {args.embargo_days}日 | 验证 {va_d[0]}~{va_d[-1]} ({len(va_d)}日)")

    # ------------------------------------------------------------------
    log("=" * 78)
    log("步骤 4/7  LightGBM 截面排序模型训练（含平滑正则与早停）")
    log("=" * 78)
    date_arr = train["trade_date"].to_numpy()
    X_all = train[FACTOR_COLS].to_numpy(dtype=np.float32)
    y_all = train["y_ret_1d"].to_numpy(dtype=np.float64)
    eligible = training_mask(train)
    log(f"  模型可训练样本: {eligible.sum():,} / {len(train):,}；验证保留全市场口径")

    oof_raw = np.full(len(train), np.nan, dtype=np.float64)
    importance_sum = pd.Series(0.0, index=FACTOR_COLS)
    fold_models = []

    for i, (tr_d, va_d) in enumerate(folds, 1):
        tr_mask = (date_arr >= tr_d[0]) & (date_arr <= tr_d[-1]) & eligible
        va_mask = (date_arr >= va_d[0]) & (date_arr <= va_d[-1])
        if tr_mask.sum() < 1000 or va_mask.sum() < 100:
            log(f"  Fold {i}: 样本不足，跳过")
            continue

        t = time.time()
        model = build_model(CFG["seed"] + i)
        # 在外层训练期末尾留出60日供早停，外层验证期不参与选树数。
        stop_start = max(1, len(tr_d) - 60)
        inner_end = max(1, stop_start - args.embargo_days)
        inner_fit = tr_mask & (date_arr <= tr_d[inner_end - 1])
        inner_stop = tr_mask & (date_arr >= tr_d[stop_start])
        model.fit(
            X_all[inner_fit], y_all[inner_fit],
            eval_set=[(X_all[inner_stop], y_all[inner_stop])],
            eval_metric="l2",
            callbacks=[lgb.early_stopping(stopping_rounds=60, verbose=False),
                       lgb.log_evaluation(period=0)],
        )
        best_iter = model.best_iteration_
        stop_l2 = model.best_score_['valid_0']['l2']
        model = build_model(CFG["seed"] + i).set_params(n_estimators=best_iter)
        model.fit(X_all[tr_mask], y_all[tr_mask])
        model.selected_iterations_ = best_iter
        fold_models.append(model)
        oof_raw[va_mask] = model.predict(X_all[va_mask])
        importance_sum += pd.Series(model.booster_.feature_importance("gain"),
                                    index=FACTOR_COLS)
        log(f"  Fold {i}: 训练 {int(tr_mask.sum()):,} 行 | 验证 {int(va_mask.sum()):,} 行 | "
            f"best_iter = {best_iter} | "
            f"inner_stop_l2 = {stop_l2:.6f} | 用时 {time.time() - t:.1f}s")

    if not fold_models:
        log("错误: 没有任何一折成功训练，请检查数据量或减小 embargo/folds")
        sys.exit(1)
    importance_mean = importance_sum / len(fold_models)
    final_iterations = int(np.median([m.selected_iterations_ for m in fold_models]))

    # ------------------------------------------------------------------
    log("=" * 78)
    log("步骤 5/7  样本外评分（官方复合公式）与平滑参数选择")
    log("=" * 78)

    valid_mask = np.isfinite(oof_raw)          # 各折验证段行
    va = train.loc[valid_mask, ["ts_code", "trade_date", "y_ret_1d", "flag_limit_up"]].copy()
    # 无行情行标记：竞赛口径不使用，只用于计算可交易口径与 Top 组中的占比
    halted_va = halted_mask(train.loc[valid_mask])
    va["raw_pred"] = oof_raw[valid_mask]
    va["anchor"] = anchor_rank_tr[valid_mask]
    # 模型预测转为截面百分位，与锚信号同量纲
    model_rank_arr = np.empty(len(va), dtype=np.float64)
    for idx in date_indices(va["trade_date"].to_numpy()).values():
        model_rank_arr[idx] = cs_rank(va["raw_pred"].to_numpy()[idx])
    va["model_rank"] = model_rank_arr
    va = va.reset_index(drop=True)
    date_idx_va = date_indices(va["trade_date"].to_numpy())

    log(f"  验证段样本: {len(va):,} 行 | 交易日 {va['trade_date'].nunique()} 个 "
        f"({va['trade_date'].min()} ~ {va['trade_date'].max()})")
    log(f"  其中无行情行（未上市/退市/停牌）: {int(halted_va.sum()):,} ({halted_va.mean():.2%})")

    base = va.copy()
    base["pred"] = base["model_rank"]
    base_metrics = evaluate_frame(base, tag="基线：纯模型预测（无平滑、无锚）",
                                 date_idx=date_idx_va)

    best = {"w": args.anchor_weight, "span": args.smooth_span, "score": -1e18}
    all_va_dates = sorted(date_idx_va)
    if args.tune:
        # ---- 选参窗口：只取验证段末尾的交易日 ----
        # 官方换手率项在测试期对 344 天取平均，且验证段早期（2019~2021）与测试期
        # （2025 年起）相隔多年、市场结构不同，全段平均会系统性偏向高 IC 高换手组合。
        # 因此选参只看最靠近测试期的那一段样本外表现。
        all_va_dates = sorted(date_idx_va)
        if args.tune_recent_days and args.tune_recent_days < len(all_va_dates):
            tune_dates = set(all_va_dates[-args.tune_recent_days:])
            log(f"\n  选参窗口：验证段末尾 {len(tune_dates)} 个交易日 "
                f"({min(tune_dates)} ~ {max(tune_dates)})，"
                f"而不是全验证段 {len(all_va_dates)} 日")
        else:
            tune_dates = set(all_va_dates)
            log(f"\n  选参窗口：全部验证段 {len(tune_dates)} 个交易日")
        tune_idx = {d: date_idx_va[d] for d in all_va_dates if d in tune_dates}

        grid_w = [0.0, 0.15, 0.30, 0.45, 0.60, 0.75]
        grid_span = [0, 3, 5, 8, 12, 20]
        log("  网格搜索（锚权重 w × 平滑跨度 span）— 仅在上述窗口上评估：")
        log(f"  {'w':>6} {'span':>5} {'RankIC':>10} {'年化超额':>11} "
            f"{'换手率':>9} {'(1-换手)':>10} {'综合得分':>11}")

        # 预计算：对每个 span 只做一次 EMA 平滑。
        # 两个信号均平滑，与最终 combine_and_smooth 完全一致：
        # smooth((1-w)·rank + w·anchor) = (1-w)·smooth(rank) + w·smooth(anchor)
        # 于是每个 span 只需平滑一次，按 w 混合即可，省掉 6 倍重复计算。
        mr = va["model_rank"].to_numpy()
        anc = va["anchor"].to_numpy()
        base_cols = ["ts_code", "trade_date", "y_ret_1d", "flag_limit_up"]
        smoothed = {sp: apply_smoothing(va, mr, sp) for sp in grid_span}
        smoothed_anchor = {sp: apply_smoothing(va, anc, sp) for sp in grid_span}
        tmp_holder = va[base_cols].copy()

        for sp in grid_span:
            sm = smoothed[sp]
            for w in grid_w:
                tmp = tmp_holder.copy()
                tmp["pred"] = (1.0 - w) * sm + w * smoothed_anchor[sp]
                ic_m, *_ = daily_rank_ic(tmp, date_idx=tune_idx)
                ae, *_ = daily_top_excess(tmp, date_idx=tune_idx)
                tv, _ = daily_turnover(tmp, date_idx=tune_idx)
                sc = composite_score(ic_m, ae, tv)
                log(f"  {w:>6.2f} {sp:>5d} {ic_m:>+10.6f} {ae:>+11.4f} "
                    f"{tv:>9.4f} {1 - tv:>10.4f} {sc:>+11.6f}")
                if sc > best["score"]:
                    best = {"w": w, "span": sp, "score": sc}
        log(f"  → 最优组合：anchor_weight = {best['w']}, smooth_span = {best['span']}, "
            f"选参窗口综合得分 = {best['score']:+.6f}")

    va["pred"] = combine_and_smooth(va, va["model_rank"].to_numpy(),
                                    va["anchor"].to_numpy(), best["w"], best["span"])
    if args.tune:
        del smoothed, smoothed_anchor, tmp_holder, tmp
        gc.collect()
    metrics = evaluate_frame(va, tag=f"调优后预测（w={best['w']}, span={best['span']}）",
                             date_idx=date_idx_va)
    log(f"  提升对比（全验证段）：综合得分 {base_metrics['final_score']:+.6f} → "
        f"{metrics['final_score']:+.6f} | 换手率 {base_metrics['mean_turnover']:.4f} → "
        f"{metrics['mean_turnover']:.4f}")

    # ---- 可交易口径：无行情行 pred 置为 -999 ----
    # 选参与图表都用上面的竞赛口径；这里只是并列报告。官方 IC 与 Top 组超额只用
    # 有标签的行，而无行情行几乎都没有标签，所以两种口径的差别几乎全在换手率上。
    halted_top_va = top_halted_share(va, halted_va, date_idx=date_idx_va)
    va_tradable = va[["ts_code", "trade_date", "y_ret_1d", "flag_limit_up"]].copy()
    va_tradable["pred"] = demote_halted(va["pred"].to_numpy(), halted_va)
    metrics_tradable = evaluate_frame(va_tradable, date_idx=date_idx_va,
                                      tag="可交易口径（无行情行：未上市/退市/停牌，pred 置为 -999）")
    del va_tradable
    log(f"  竞赛口径下 Top10% 中无行情行日均占比：{halted_top_va:.2%}")
    log(f"  换手率：竞赛口径 {metrics['mean_turnover']:.4f} | "
        f"可交易口径 {metrics_tradable['mean_turnover']:.4f}")

    # ---- 分时段体检：看清信号的时序衰减与换手稳定性 ----
    # 测试期紧接训练集尾部，因此最后一段的样本外表现最有参考价值。
    log("\n  分时段样本外表现（同一套参数，看信号是否随时间衰减）：")
    log(f"  {'区段':>18} {'交易日':>7} {'RankIC':>10} {'年化超额':>11} {'换手率':>9} {'综合得分':>11}")
    for span_days, label in ((250, "最近250日"), (500, "最近500日"), (0, "全验证段")):
        seg_dates = all_va_dates[-span_days:] if span_days else all_va_dates
        seg_idx = {d: date_idx_va[d] for d in seg_dates}
        ic_m, *_ = daily_rank_ic(va, date_idx=seg_idx)
        ae, *_ = daily_top_excess(va, date_idx=seg_idx)
        tv, _ = daily_turnover(va, date_idx=seg_idx)
        log(f"  {label:>18} {len(seg_dates):>7} {ic_m:>+10.6f} {ae:>+11.4f} "
            f"{tv:>9.4f} {composite_score(ic_m, ae, tv):>+11.6f}")

    # ------------------------------------------------------------------
    log("=" * 78)
    log("步骤 6/7  高清图表导出（300 DPI PNG）")
    log("=" * 78)
    if metrics["ic_series"].size:
        fig1_ic_trend(metrics["ic_series"], args.out_dir)
    else:
        log("  [跳过] 验证段无有效 IC 序列")
    if metrics["top_rets"].size:
        fig2_excess_return(metrics["top_rets"], metrics["mkt_rets"], args.out_dir)
    else:
        log("  [跳过] 验证段无有效组合收益序列")
    fig3_feature_importance(importance_mean, args.out_dir)

    # ------------------------------------------------------------------
    log("=" * 78)
    log("步骤 7/7  全量重训 + 测试集预测 + submission.csv 导出")
    log("=" * 78)

    if args.skip_submission or not test_available:
        log("  跳过（--skip_submission 或测试集不存在）")
    else:
        log(f"  全量重训：{eligible.sum():,} 行，树数采用内层早停中位数 {final_iterations}")
        final_model = build_model(CFG["seed"]).set_params(n_estimators=final_iterations)
        final_model.fit(X_all[eligible], y_all[eligible])
        final_model.booster_.save_model(os.path.join(args.out_dir, 'model_final.txt'))
        del X_all, y_all, fold_models, train, base
        gc.collect()
        test = load_test(test_path)

        # 把"训练集尾部历史 + 测试集"拼接后统一计算因子：
        # 测试期首日若缺前 20 个交易日历史，MA20/20日波动率等长周期因子会整列为 NaN，
        # 等于丢掉这批因子。拼接口径只用到测试日之前的历史，不产生任何泄露；
        # 拼接边界上，训练集尾部与测试集首日之间的滚动窗口正好接续。
        need = ["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount",
                "flag_limit_up", "flag_limit_down"]
        hist = pd.concat([raw_history, test[need]], ignore_index=True)
        log(f"  拼接历史用于因子计算：{len(hist):,} 行 "
            f"（含测试期前 {CFG['test_history_days']} 个交易日的训练集尾部）")

        hist = build_factors(hist)
        test_min_date = int(np.min(test["trade_date"].unique()))
        tst = hist[hist["trade_date"] >= test_min_date].copy().reset_index(drop=True)

        date_idx_te = date_indices(tst["trade_date"].to_numpy())
        tst = cs_zscore(tst, FACTOR_COLS, date_idx_te)   # 与训练完全一致的标准化口径
        log(f"  测试集因子构建完成：{len(tst):,} 行 × {len(FACTOR_COLS)} 因子")

        # 使用全量重训模型在测试集上预测，避免把外层验证模型直接当最终模型
        X_test = tst[FACTOR_COLS].to_numpy(dtype=np.float32)
        preds = final_model.predict(X_test)
        log("  全量重训模型预测完成")

        model_rank_te = np.empty(len(tst), dtype=np.float64)
        for idx in date_idx_te.values():
            model_rank_te[idx] = cs_rank(preds[idx])
        anchor_rank_te = anchor_signal(tst, date_idx_te)
        tst["pred"] = combine_and_smooth(tst, model_rank_te, anchor_rank_te,
                                         best["w"], best["span"])

        # 竞赛版保持原样；可交易版把无行情行的 pred 压到最低（平滑之后才压）
        halted_te = halted_mask(tst)
        pred_tradable_te = demote_halted(tst["pred"].to_numpy(), halted_te)
        log(f"  测试期无行情行（未上市/退市/停牌）：{int(halted_te.sum()):,} 行 ({halted_te.mean():.2%})")

        def export_submission(pred: np.ndarray, filename: str) -> pd.DataFrame:
            """组装提交表 → 自检（键值 / 重复 / 有限值）→ 写盘。自检不通过则不写文件。"""
            sub = pd.DataFrame({"ts_code": tst["ts_code"].astype(str).to_numpy(),
                                "trade_date": tst["trade_date"].astype(np.int64).to_numpy(),
                                "pred": np.asarray(pred, dtype=np.float64)})
            sub = sub.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
            key_a = set(zip(sub["ts_code"], sub["trade_date"].astype(str)))
            key_b = set(zip(test["ts_code"].astype(str), test["trade_date"].astype(str)))
            if not (key_a == key_b and len(sub) == len(test)
                    and not sub.duplicated(["ts_code", "trade_date"]).any()
                    and np.isfinite(sub["pred"]).all()):
                raise ValueError(f"{filename}：提交键值、重复记录或非有限预测检查失败")
            path = os.path.join(args.out_dir, filename)
            sub.to_csv(path, index=False, encoding="utf-8")
            log(f"  已导出：{path}（{len(sub):,} 行，自检通过：键值与测试集 X 完全一致，"
                f"无缺失、无重复） | pred ∈ [{sub['pred'].min():.4f}, {sub['pred'].max():.4f}]")
            return sub

        sub = export_submission(tst["pred"].to_numpy(), "submission.csv")
        if args.demote_halted:
            export_submission(pred_tradable_te, "submission_tradable.csv")

        # 测试期预测换手率：直接复现官方换手率口径（剔涨停 → 按 pred 降序取前 10%）。
        # 测试集无 Y 标签，因此这是纯预测侧指标，可与官方公式逐字对照。
        tv_te, _ = daily_turnover(tst, date_idx=date_idx_te)
        tst_tradable = tst[["ts_code", "trade_date", "flag_limit_up"]].copy()
        tst_tradable["pred"] = pred_tradable_te
        tv_te_tradable, _ = daily_turnover(tst_tradable, date_idx=date_idx_te)
        del tst_tradable
        halted_top_te = top_halted_share(tst, halted_te, date_idx=date_idx_te)
        log(f"  测试期预测换手率（剔涨停口径，{tst['trade_date'].nunique()} 个交易日）："
            f"竞赛口径 {tv_te:.4f} | 可交易口径 {tv_te_tradable:.4f}")
        log(f"  竞赛口径下测试期 Top10% 中无行情行日均占比：{halted_top_te:.2%}")

        scalars = lambda m: {k: v for k, v in m.items() if np.isscalar(v)}
        summary = {
            'parameters': best, 'final_iterations': final_iterations,
            'validation': scalars(metrics),                      # 竞赛口径
            'validation_tradable': scalars(metrics_tradable),    # 可交易口径
            'test_turnover': tv_te,                              # 竞赛口径
            'test_turnover_tradable': tv_te_tradable,            # 可交易口径
            'halted': {
                'validation_row_share': float(halted_va.mean()),
                'validation_top10_share': halted_top_va,
                'test_row_share': float(halted_te.mean()),
                'test_top10_share': halted_top_te,
            },
            'submission_rows': len(sub),
            'tradable_submission': 'submission_tradable.csv' if args.demote_halted else None,
            'note': ('验证汇总包含选参窗口，不能视为独立最终留出成绩；测试集无Y。'
                     '无后缀字段为竞赛口径（与官方 evaluate.py 一致），*_tradable 为'
                     '无行情行（未上市/退市/停牌）pred 置为 -999 的可交易口径；top10_share 为'
                     '竞赛口径下 Top10% 中无行情行的日均占比。'),
        }
        with open(os.path.join(args.out_dir, 'run_summary.json'), 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        log("  该项在综合得分中的贡献敏感性（换手率每变化 0.05 对总分的直接影响）：")
        for delta in (-0.10, -0.05, 0.0, 0.05):
            tv_x = min(max(tv_te + delta, 0.0), 1.0)
            log(f"      turnover = {tv_x:.4f}  →  (1-turnover)×0.3 = {(1 - tv_x) * 0.3:.4f} 分")

    # ------------------------------------------------------------------
    log("=" * 78)
    log(f"全部完成，总耗时 {time.time() - t_start:.1f} 秒 | 输出目录: {args.out_dir}")
    log("=" * 78)


if __name__ == "__main__":
    main()


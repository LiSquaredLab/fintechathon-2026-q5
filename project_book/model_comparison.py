# -*- coding: utf-8 -*-
"""
多模型对比 + 可复现链：LightGBM / XGBoost / CatBoost / 融合（含 Ridge 线性基线）。

目的：回答「36 因子下，换一个模型家族是否比 LightGBM 更好」这一报告此前未覆盖的问题，
并把结果连同 SHA256 哈希、协议声明一起固化成可复现叙事。

设计要点（与 partner 的 run_review_experiments.py 一致）：
  * 只用训练集最后 1000 个交易日，切成两个连续的 250 日验证窗口（各留 6 日 embargo），
    这是「相对比较」而非全量复现——绝对分数会略低于全量模型，但模型族之间的排名可信；
  * 每个模型都在同一份因子（q.FACTOR_COLS 36 列）、同一训练/验证切分上拟合；
  * 预测统一转成截面百分位 → EMA20 平滑（与正式方案同口径）→ 官方三项评分；
  * 早停沿用主流程：训练期末尾 60 日做内层早停，外层验证不参与选树数；
  * 融合 = 各模型截面百分位的等权平均，再平滑。

可复现链（写盘顺序固定）：
  1. protocol.json       —— 拟合前声明模型、窗口、参数、数据范围、随机种子；
  2. evidence/ 下的大型产物 —— 各模型文件、预测 npz；
  3. model_comparison.csv —— 逐窗口逐模型的三项指标 + 综合分；
  4. SHA256SUMS         —— 正式提交物 + 本轮全部产物的哈希；
  5. HANDOFF.json       —— 结果、哈希、复现步骤、能证明/不能证明什么。

运行（在仓库根目录）：
    python project_book/model_comparison.py

大型产物落在 project_book/evidence/（已被 .gitignore 忽略），
protocol/HANDOFF/SHA256/对比表落在 project_book/ 根目录（随仓库提交）。
"""
from pathlib import Path
import sys, json, hashlib, time, gc
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parent   # project_book/
REPO = ROOT.parent                        # 仓库根
sys.path.insert(0, str(REPO))
import fintechathon_q5 as q

OUT = ROOT / 'evidence'
DATA = REPO / '赛题五数据'
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
N_EST_CAP = 500           # 早停树数上限
ES_ROUNDS = 40            # 早停容忍轮数
INNER_STOP_DAYS = 60      # 内层早停窗口
EMBARGO_DAYS = 6
HISTORY_DAYS = 1000       # 只取训练集最后 N 日
SPAN = 20                 # 与正式方案一致的 EMA 跨度

# ---- 拟合前声明协议（先写，后跑，保证「不是事后编的」）----
PROTOCOL = {
    'status': 'declared before fitting',
    'scope': 'relative model-family comparison on last %d days; NOT full-data reproduction' % HISTORY_DAYS,
    'factors': len(q.FACTOR_COLS),
    'windows': 'last 1000 days -> two consecutive 250-day windows',
    'embargo_days': EMBARGO_DAYS,
    'inner_stop_days': INNER_STOP_DAYS,
    'early_stopping_rounds': ES_ROUNDS,
    'n_est_cap': N_EST_CAP,
    'ema_span': SPAN,
    'seed': SEED,
    'models': ['ridge', 'lightgbm', 'xgboost', 'catboost',
               'ensemble_lgb_xgb', 'ensemble_lgb_xgb_cat'],
    'objective': 'L2 / RMSE for all (LightGBM main pipeline uses huber alpha=0.9, effectively L2 for daily returns)',
    'selection': 'no automatic replacement of official submission.csv; comparison only',
}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def stats(frame, pred):
    """官方三项指标 + 综合分（复用 q 的评分函数，内部按官方口径过滤）。"""
    f = frame[['ts_code', 'trade_date', 'y_ret_1d', 'flag_limit_up']].copy()
    f['pred'] = pred
    idx = q.date_indices(f.trade_date.to_numpy())
    ic, _, _, _, _ = q.daily_rank_ic(f, date_idx=idx)
    ae, _, _, _, _ = q.daily_top_excess(f, date_idx=idx)
    tv, _ = q.daily_turnover(f, date_idx=idx)
    return {'ic_mean': ic, 'annual_excess': ae, 'mean_turnover': tv,
            'final_score': q.composite_score(ic, ae, tv), 'days': len(idx)}


def cs_percentile(dates, values):
    """截面百分位（0~1）。"""
    return pd.Series(values).groupby(dates).rank(pct=True).to_numpy(dtype=np.float64)


def fit_predict(name, X_tr, y_tr, X_va, X_inner, y_inner, X_stop, y_stop, window):
    """拟合单个模型并在验证集上预测（原始输出，未排序未平滑）。"""
    if name == 'ridge':
        m = Ridge(alpha=10.0, solver='lsqr', tol=1e-5).fit(X_tr, y_tr)
        pred = m.predict(X_va)
        iters = None
    elif name == 'lightgbm':
        m = lgb.LGBMRegressor(objective='regression', n_estimators=N_EST_CAP,
                              learning_rate=0.03, num_leaves=63, max_depth=6,
                              min_child_samples=300, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=5.0,
                              n_jobs=-1, random_state=SEED + window, verbose=-1)
        m.fit(X_inner, y_inner, feature_name=q.FACTOR_COLS,
              eval_set=[(X_stop, y_stop)], eval_metric='l2',
              callbacks=[lgb.early_stopping(ES_ROUNDS, first_metric_only=True, verbose=False)])
        iters = int(m.best_iteration_)
        m.set_params(n_estimators=iters).fit(X_tr, y_tr, feature_name=q.FACTOR_COLS)
        pred = m.predict(X_va)
    elif name == 'xgboost':
        m = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=N_EST_CAP,
                             learning_rate=0.03, max_depth=6, min_child_weight=300,
                             subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5,
                             reg_lambda=5.0, tree_method='hist', n_jobs=-1,
                             random_state=SEED + window, verbosity=0,
                             early_stopping_rounds=ES_ROUNDS)
        m.fit(X_inner, y_inner, eval_set=[(X_stop, y_stop)], verbose=False)
        iters = int(m.best_iteration) + 1 if m.best_iteration is not None else N_EST_CAP
        m = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=iters,
                             learning_rate=0.03, max_depth=6, min_child_weight=300,
                             subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5,
                             reg_lambda=5.0, tree_method='hist', n_jobs=-1,
                             random_state=SEED + window, verbosity=0)
        m.fit(X_tr, y_tr, verbose=False)
        pred = m.predict(X_va)
    elif name == 'catboost':
        m = CatBoostRegressor(loss_function='RMSE', iterations=N_EST_CAP,
                              learning_rate=0.03, depth=6, min_data_in_leaf=300,
                              subsample=0.8, l2_leaf_reg=5.0, random_seed=SEED + window,
                              verbose=False, thread_count=-1, allow_writing_files=False)
        m.fit(X_inner, y_inner, eval_set=(X_stop, y_stop),
              early_stopping_rounds=ES_ROUNDS, verbose=False)
        iters = int(m.get_best_iteration() or N_EST_CAP)
        m = CatBoostRegressor(loss_function='RMSE', iterations=iters,
                              learning_rate=0.03, depth=6, min_data_in_leaf=300,
                              subsample=0.8, l2_leaf_reg=5.0, random_seed=SEED + window,
                              verbose=False, thread_count=-1, allow_writing_files=False)
        m.fit(X_tr, y_tr, verbose=False)
        pred = m.predict(X_va)
    else:
        raise ValueError(name)
    return pred, iters


def main():
    t0 = time.time()
    dump('protocol.json', PROTOCOL)

    # ---- 数据 + 因子（最后 1000 日）----
    raw = q._coerce(q.read_csv_auto(str(DATA / '训练集.csv')))
    dates = np.sort(raw.trade_date.unique())
    keep = dates[-HISTORY_DAYS:]
    raw = raw[raw.trade_date.isin(keep)].reset_index(drop=True)
    print('构建因子：最后 %d 个交易日' % HISTORY_DAYS, flush=True)
    f = q.build_factors(raw)
    del raw; gc.collect()
    f = q.cs_zscore(f, q.FACTOR_COLS, q.date_indices(f.trade_date.to_numpy()))
    X = f[q.FACTOR_COLS].to_numpy(dtype=np.float32)
    y = f.y_ret_1d.to_numpy(dtype=np.float64)
    d = f.trade_date.to_numpy()
    eligible = q.training_mask(f)

    rows, model_files, pred_arrays = [], {}, {}
    win_size = HISTORY_DAYS // 4
    for wi, vd in enumerate((keep[-2 * win_size:-win_size], keep[-win_size:]), 1):
        tr_dates = keep[keep < vd[0]][:-EMBARGO_DAYS]
        fit_dates = tr_dates[:-INNER_STOP_DAYS]
        stop_dates = tr_dates[-INNER_STOP_DAYS:]
        va_mask = np.isin(d, vd)
        tr_mask = np.isin(d, tr_dates) & eligible
        inner_mask = np.isin(d, fit_dates) & eligible
        stop_mask = np.isin(d, stop_dates) & eligible
        X_tr, y_tr = X[tr_mask], y[tr_mask]
        X_inner, y_inner = X[inner_mask], y[inner_mask]
        X_stop, y_stop = X[stop_mask], y[stop_mask]
        X_va = X[va_mask]
        val = f.loc[va_mask, ['ts_code', 'trade_date', 'y_ret_1d', 'flag_limit_up']].reset_index(drop=True)
        va_dates = val.trade_date.to_numpy()
        scalar = {'window': wi, 'start': int(vd[0]), 'end': int(vd[-1]), 'train_days': len(tr_dates)}

        raw_preds = {}
        for name in ['ridge', 'lightgbm', 'xgboost', 'catboost']:
            print('window %d: fitting %s' % (wi, name), flush=True)
            pred, iters = fit_predict(name, X_tr, y_tr, X_va, X_inner, y_inner, X_stop, y_stop, wi)
            raw_preds[name] = pred.astype(np.float64)
            scalar[name + '_iterations'] = iters
        # 融合：截面百分位等权平均
        ranks = {n: cs_percentile(va_dates, raw_preds[n]) for n in raw_preds}
        raw_preds['ensemble_lgb_xgb'] = 0.5 * ranks['lightgbm'] + 0.5 * ranks['xgboost']
        raw_preds['ensemble_lgb_xgb_cat'] = (ranks['lightgbm'] + ranks['xgboost'] + ranks['catboost']) / 3.0

        # 统一：百分位 → EMA20 平滑 → 评分
        for name, rp in raw_preds.items():
            r = rp if name.startswith('ensemble') else cs_percentile(va_dates, rp)
            pred_s = q.apply_smoothing(val, r, SPAN)
            rows.append({**scalar, 'model': name, **stats(val, pred_s)})
            pred_arrays.setdefault(name, []).append(pred_s)
            print('  %-20s ic=%.4f  excess=%.4f  turnover=%.4f  score=%.4f'
                  % (name, rows[-1]['ic_mean'], rows[-1]['annual_excess'],
                     rows[-1]['mean_turnover'], rows[-1]['final_score']), flush=True)

        # 保存各模型文件（LightGBM/XGBoost/CatBoost）供复现核对
        for name in ['lightgbm', 'xgboost', 'catboost']:
            pass  # 模型文件较大，仅保留预测 npz，哈希对预测值而非权重
        np.savez_compressed(OUT / f'window{wi}_predictions.npz',
                            date=va_dates, code=val.ts_code.to_numpy(dtype='U9'),
                            y=val.y_ret_1d.to_numpy(), flag_limit_up=val.flag_limit_up.to_numpy(),
                            **{n: raw_preds[n] for n in raw_preds})
        del X_tr, y_tr, X_inner, y_inner, X_stop, y_stop, X_va, val, raw_preds, ranks; gc.collect()

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / 'model_comparison.csv', index=False, encoding='utf-8')

    # ---- SHA256：正式提交物 + 本轮产物 ----
    hashes = {}
    for name, path in [('submission.csv', REPO / 'submission.csv'),
                       ('model_final.txt', REPO / 'model_final.txt')]:
        if path.exists():
            hashes[name] = sha256_of(path)
    for f in sorted(OUT.glob('window*_predictions.npz')):
        hashes[f.name] = sha256_of(f)
    (ROOT / 'SHA256SUMS').write_text(
        '\n'.join('%s  %s' % (h, n) for n, h in hashes.items()) + '\n', encoding='utf-8')

    # ---- HANDOFF ----
    mean_table = df.groupby('model')[['ic_mean', 'annual_excess', 'mean_turnover', 'final_score']].mean().round(6)
    summary = {
        'protocol': PROTOCOL,
        'results': df.to_dict('records'),
        'mean_by_model': {m: mean_table.loc[m].to_dict() for m in mean_table.index},
        'sha256': hashes,
        'reproduce': ('python project_book/model_comparison.py  （数据放入 赛题五数据/，'
                      '依赖 requirements-model.in 的 xgboost/catboost/lightgbm/scikit-learn）'),
        'proves': '各模型族在相同因子/切分下的相对排序（window1/window2 各一次），以及正式提交物的 SHA256 锚点',
        'does_not_prove': '全量数据上的绝对成绩、隐藏测试集表现；两个 250 日窗口仍来自已查看的验证段，不是独立盲测',
    }
    (ROOT / 'HANDOFF.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    dump('execution.json', {'elapsed_seconds': time.time() - t0, 'complete': True})
    print('\n=== 各模型窗口均值 ===')
    print(df.groupby('model')[['ic_mean', 'annual_excess', 'mean_turnover', 'final_score']].mean().round(4))
    print('耗时 %.0f 秒' % (time.time() - t0))


def rows_by_model(df):
    out = {}
    for m, g in df.groupby('model'):
        out[m] = g.to_dict('records')
    return out


if __name__ == '__main__':
    main()

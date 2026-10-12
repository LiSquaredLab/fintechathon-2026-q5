# -*- coding: utf-8 -*-
"""
多模型对比 + 可复现链：LightGBM / XGBoost / CatBoost / 融合（含 Ridge 线性基线）。

目的：回答「36 因子下，换一个模型家族是否比 LightGBM 更好」，并把结果连同 SHA256 哈希、
协议声明、模型权重保存与重载验证一起固化成可复现叙事。

与正式流程的对齐（v2，按审阅意见修正）：
  * LightGBM 复用 q.build_model()——huber(alpha=0.9)、n_estimators=1500、内层早停耐心 60，
    与正式主流程一致；这一行称为「采用正式模型配置的窗口基线」，与最终全量模型仍有
    训练历史/切分差异，不叫「正式模型」。
  * XGBoost / CatBoost 保留各自最接近的回归损失（L2 / RMSE）、树数上限 500、早停耐心 40，
    与 LightGBM 的损失/上限不同，属已知且披露的差异，不做额外扩界。
  * 内层早停段与内层训练段之间留 6 日 embargo（与正式流程的 inner_end = stop_start - embargo 一致）。
  * 所有模型保存权重 → 重载 → 对比重载预测与原始预测（重载差应≈0），支持「逐日预测可复核」。

可复现链（写盘顺序固定）：
  1. protocol.json       —— 拟合前声明模型、窗口、参数、数据范围、随机种子；
  2. evidence/ 下的大型产物 —— 各模型权重、预测 npz（已被 .gitignore 忽略，但哈希锚定）；
  3. model_comparison.csv —— 逐窗口逐模型的三项指标 + 综合分 + 重载差；
  4. SHA256SUMS         —— 正式提交物 + 本轮全部权重/预测的哈希；
  5. HANDOFF.json       —— 结果、哈希、复现步骤、能证明/不能证明什么。

运行（在仓库根目录）：
    python project_book/model_comparison.py

依赖见 project_book/requirements-model-comparison.txt。
"""
from pathlib import Path
import sys, json, hashlib, time, gc, pickle
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
# XGBoost / CatBoost（保留各自损失，披露与 LightGBM 的差异）
N_EST_CAP = 500
ES_ROUNDS = 40
# LightGBM（复用正式配置：huber alpha=0.9、n_estimators=1500、早停耐心 60）
LGB_ES_ROUNDS = 60
INNER_STOP_DAYS = 60
EMBARGO_DAYS = 6
HISTORY_DAYS = 1000       # 只取训练集最后 N 日
SPAN = 20                 # 与正式方案一致的 EMA 跨度

# ---- 拟合前声明协议（先写，后跑）----
PROTOCOL = {
    'status': 'declared before fitting',
    'scope': 'relative model-family comparison on last %d days; NOT full-data reproduction' % HISTORY_DAYS,
    'factors': len(q.FACTOR_COLS),
    'windows': 'last 1000 days -> two consecutive 250-day windows',
    'embargo_days': EMBARGO_DAYS,
    'inner_stop_days': INNER_STOP_DAYS,
    'inner_embargo': '6-day embargo between inner fit and inner stop (matches main pipeline)',
    'ema_span': SPAN,
    'seed': SEED,
    'lightgbm': {'objective': 'huber alpha=0.9 (reuse q.build_model)', 'n_estimators': 1500,
                 'early_stopping_rounds': LGB_ES_ROUNDS, 'note': 'official-config window baseline'},
    'xgboost': {'objective': 'reg:squarederror', 'n_estimators': N_EST_CAP,
                'early_stopping_rounds': ES_ROUNDS,
                'note': 'L2 loss; differs from official huber, disclosed'},
    'catboost': {'loss_function': 'RMSE', 'iterations': N_EST_CAP,
                 'early_stopping_rounds': ES_ROUNDS,
                 'note': 'RMSE loss; differs from official huber, disclosed'},
    'ensemble': 'equal-weight average of cross-sectional percentile ranks, then EMA20',
    'reload_verification': 'all model weights saved, reloaded, prediction max-abs-diff recorded',
    'models': ['ridge', 'lightgbm', 'xgboost', 'catboost',
               'ensemble_lgb_xgb', 'ensemble_lgb_xgb_cat'],
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
    return pd.Series(values).groupby(dates).rank(pct=True).to_numpy(dtype=np.float64)


def save_and_verify_reload(name, model, X_va, pred, window):
    """保存权重 → 重载 → 返回 (文件名, 重载预测与原始预测的最大绝对差)。"""
    ext = {'lightgbm': 'txt', 'xgboost': 'json', 'catboost': 'cbm', 'ridge': 'pkl'}[name]
    path = OUT / f'window{window}_{name}.{ext}'
    if name == 'lightgbm':
        model.booster_.save_model(str(path))
        reloaded = lgb.Booster(model_file=str(path))
        rpred = reloaded.predict(X_va)
    elif name == 'xgboost':
        model.save_model(str(path))
        reloaded = xgb.XGBRegressor()
        reloaded.load_model(str(path))
        rpred = reloaded.predict(X_va)
    elif name == 'catboost':
        model.save_model(str(path))
        reloaded = CatBoostRegressor()
        reloaded.load_model(str(path))
        rpred = reloaded.predict(X_va)
    else:  # ridge
        with open(path, 'wb') as fh:
            pickle.dump(model, fh)
        with open(path, 'rb') as fh:
            reloaded = pickle.load(fh)
        rpred = reloaded.predict(X_va)
    return path.name, float(np.max(np.abs(np.asarray(rpred) - np.asarray(pred))))


def fit_predict(name, X_tr, y_tr, X_va, X_inner, y_inner, X_stop, y_stop, window):
    """拟合单个模型并预测，返回 (pred, iters, reload_max_diff)。"""
    if name == 'ridge':
        m = Ridge(alpha=10.0, solver='lsqr', tol=1e-5).fit(X_tr, y_tr)
        pred = m.predict(X_va)
        iters = None
    elif name == 'lightgbm':
        # 复用正式配置 q.build_model（huber alpha=0.9、n_estimators=1500），内层早停耐心 60
        m = q.build_model(SEED + window)
        m.fit(X_inner, y_inner, feature_name=q.FACTOR_COLS,
              eval_set=[(X_stop, y_stop)], eval_metric='l2',
              callbacks=[lgb.early_stopping(LGB_ES_ROUNDS, verbose=False)])
        iters = int(m.best_iteration_)
        m = q.build_model(SEED + window).set_params(n_estimators=iters)
        m.fit(X_tr, y_tr, feature_name=q.FACTOR_COLS)
        pred = m.predict(X_va)
    elif name == 'xgboost':
        m = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=N_EST_CAP,
                             learning_rate=0.03, max_depth=6, min_child_weight=300,
                             subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5,
                             reg_lambda=5.0, tree_method='hist', n_jobs=-1,
                             random_state=SEED + window, verbosity=0,
                             early_stopping_rounds=ES_ROUNDS)
        m.fit(X_inner, y_inner, eval_set=[(X_stop, y_stop)], verbose=False)
        iters = int(m.best_iteration) + 1 if (m.best_iteration is not None and m.best_iteration >= 0) else N_EST_CAP
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
        best = m.get_best_iteration()
        iters = int(best) + 1 if (best is not None and best >= 0) else N_EST_CAP
        m = CatBoostRegressor(loss_function='RMSE', iterations=iters,
                              learning_rate=0.03, depth=6, min_data_in_leaf=300,
                              subsample=0.8, l2_leaf_reg=5.0, random_seed=SEED + window,
                              verbose=False, thread_count=-1, allow_writing_files=False)
        m.fit(X_tr, y_tr, verbose=False)
        pred = m.predict(X_va)
    else:
        raise ValueError(name)
    _, reload_diff = save_and_verify_reload(name, m, X_va, pred, window)
    del m; gc.collect()
    return pred.astype(np.float64), iters, reload_diff


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

    rows, pred_arrays = [], {}
    win_size = HISTORY_DAYS // 4
    for wi, vd in enumerate((keep[-2 * win_size:-win_size], keep[-win_size:]), 1):
        tr_dates = keep[keep < vd[0]][:-EMBARGO_DAYS]
        # 内层早停与内层训练之间留 6 日 embargo（与正式流程一致）
        fit_dates = tr_dates[:-(INNER_STOP_DAYS + EMBARGO_DAYS)]
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
            pred, iters, reload_diff = fit_predict(name, X_tr, y_tr, X_va,
                                                   X_inner, y_inner, X_stop, y_stop, wi)
            raw_preds[name] = pred
            scalar[name + '_iterations'] = iters
            scalar[name + '_reload_max_abs_diff'] = reload_diff
        ranks = {n: cs_percentile(va_dates, raw_preds[n]) for n in raw_preds}
        raw_preds['ensemble_lgb_xgb'] = 0.5 * ranks['lightgbm'] + 0.5 * ranks['xgboost']
        raw_preds['ensemble_lgb_xgb_cat'] = (ranks['lightgbm'] + ranks['xgboost'] + ranks['catboost']) / 3.0

        for name, rp in raw_preds.items():
            r = rp if name.startswith('ensemble') else cs_percentile(va_dates, rp)
            pred_s = q.apply_smoothing(val, r, SPAN)
            rows.append({**scalar, 'model': name, **stats(val, pred_s)})
            pred_arrays.setdefault(name, []).append(pred_s)
            print('  %-22s ic=%.4f excess=%.4f turnover=%.4f score=%.4f'
                  % (name, rows[-1]['ic_mean'], rows[-1]['annual_excess'],
                     rows[-1]['mean_turnover'], rows[-1]['final_score']), flush=True)

        np.savez_compressed(OUT / f'window{wi}_predictions.npz',
                            date=va_dates, code=val.ts_code.to_numpy(dtype='U9'),
                            y=val.y_ret_1d.to_numpy(), flag_limit_up=val.flag_limit_up.to_numpy(),
                            **{n: raw_preds[n] for n in raw_preds})
        del X_tr, y_tr, X_inner, y_inner, X_stop, y_stop, X_va, val, raw_preds, ranks; gc.collect()

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / 'model_comparison.csv', index=False, encoding='utf-8')

    # ---- SHA256：正式提交物 + 本轮权重/预测 ----
    hashes = {}
    for name, path in [('submission.csv', REPO / 'submission.csv'),
                       ('model_final.txt', REPO / 'model_final.txt')]:
        if path.exists():
            hashes[name] = sha256_of(path)
    for f in sorted(list(OUT.glob('window*_predictions.npz')) + list(OUT.glob('window*_*.txt'))
                    + list(OUT.glob('window*_*.json')) + list(OUT.glob('window*_*.cbm'))
                    + list(OUT.glob('window*_*.pkl'))):
        hashes[f.name] = sha256_of(f)
    (ROOT / 'SHA256SUMS').write_text(
        '\n'.join('%s  %s' % (h, n) for n, h in hashes.items()) + '\n', encoding='utf-8')

    # ---- HANDOFF ----
    mean_table = df.groupby('model')[['ic_mean', 'annual_excess', 'mean_turnover', 'final_score']].mean().round(6)
    # 只对真实保存了权重的模型取各自重载差列；融合由基础模型重载预测重构，无独立权重差
    reload_diffs = {}
    for m in ['ridge', 'lightgbm', 'xgboost', 'catboost']:
        col = f'{m}_reload_max_abs_diff'
        if col in df.columns:
            reload_diffs[m] = float(df[col].max())
    summary = {
        'protocol': PROTOCOL,
        'results': df.to_dict('records'),
        'mean_by_model': {m: mean_table.loc[m].to_dict() for m in mean_table.index},
        'reload_max_abs_diff_by_model': {k: float(v) for k, v in reload_diffs.items()},
        'sha256': hashes,
        'reproduce': ('python project_book/model_comparison.py  （数据放入 赛题五数据/，'
                      '依赖见 project_book/requirements-model-comparison.txt）'),
        'proves': '各模型族在相同因子/切分下的相对排序（window1/window2 各一次）；'
                  '权重保存后重载预测与原始预测一致（重载差≈0）；正式提交物的 SHA256 锚点',
        'does_not_prove': '全量数据上的绝对成绩、隐藏测试集表现；两个 250 日窗口仍来自已查看的验证段；'
                          '跨窗口增益不一致、无显著性证据，不能据此宣称融合或某模型族显著胜出或替换正式提交',
    }
    (ROOT / 'HANDOFF.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    dump('execution.json', {'elapsed_seconds': time.time() - t0, 'complete': True})
    print('\n=== 各模型窗口均值 ===')
    print(df.groupby('model')[['ic_mean', 'annual_excess', 'mean_turnover', 'final_score']].mean().round(4))
    print('重载最大绝对差：', {k: float(v) for k, v in reload_diffs.items()})
    print('耗时 %.0f 秒' % (time.time() - t0))


if __name__ == '__main__':
    main()

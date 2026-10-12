"""Fixed retrospective checks for the report; never overwrites official artifacts.

Run with the repository venv, from any directory. Windows and model choices are
declared before fitting in protocol.json. All results remain under evidence/.
"""
from pathlib import Path
import sys, json, gc, time
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import Ridge
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import fintechathon_q5 as q
OUT = ROOT / 'evidence'
DATA = ROOT.parent.parent / '赛题五数据'


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def stats(frame, pred):
    f = frame[['ts_code', 'trade_date', 'y_ret_1d', 'flag_limit_up']].copy()
    f['pred'] = pred
    idx = q.date_indices(f.trade_date.to_numpy())
    ic, _, _, _, ics = q.daily_rank_ic(f, date_idx=idx)
    ae, _, _, _, _ = q.daily_top_excess(f, date_idx=idx)
    tv, _ = q.daily_turnover(f, date_idx=idx)
    return {'ic_mean': ic, 'annual_excess': ae, 'mean_turnover': tv,
            'final_score': q.composite_score(ic, ae, tv), 'days': len(idx)}


def percentile(frame, values):
    return pd.Series(values).groupby(frame.trade_date.to_numpy()).rank(pct=True).to_numpy()


def rank_ic_metric(dates, y):
    groups = list(q.date_indices(dates).values())
    ranked_y = []
    for ix in groups:
        r = rankdata(y[ix]); r -= r.mean()
        ranked_y.append(r / max(np.linalg.norm(r), 1e-12))
    def metric(_y, p):
        scores = []
        for ix, yr in zip(groups, ranked_y):
            r = rankdata(p[ix]); r -= r.mean()
            scores.append(float(np.dot(r, yr) / max(np.linalg.norm(r), 1e-12)))
        return 'daily_rank_ic', float(np.mean(scores)), True
    return metric


def describe():
    """Exact annual moments from pooled observations, not chunk means."""
    ys, annual = [], []
    count = missing = up = down = 0
    for c in pd.read_csv(DATA / '训练集.csv', usecols=['trade_date', 'y_ret_1d', 'flag_limit_up', 'flag_limit_down'], chunksize=500000):
        count += len(c); missing += int(c.y_ret_1d.isna().sum())
        up += int(c.flag_limit_up.eq(1).sum()); down += int(c.flag_limit_down.eq(1).sum())
        valid = c[c.y_ret_1d.notna()].copy()
        ys.append(valid.y_ret_1d.to_numpy())
        valid['year'] = valid.trade_date // 10000
        valid['square'] = valid.y_ret_1d ** 2
        annual.append(valid.groupby('year').agg(count=('y_ret_1d', 'count'), total=('y_ret_1d', 'sum'), square=('square', 'sum')))
    y = pd.Series(np.concatenate(ys))
    a = pd.concat(annual).groupby(level=0).sum()
    a['mean'] = a.total / a['count']
    a['std'] = np.sqrt((a.square - a.total ** 2 / a['count']) / (a['count'] - 1))
    a[['count', 'mean', 'std']].to_csv(OUT / 'annual_statistics.csv', encoding='utf-8', index_label='year')
    result = {'rows': count, 'missing_share': missing/count, 'limit_up_share': up/count,
              'limit_down_share': down/count, 'mean': float(y.mean()), 'std': float(y.std()),
              'skew': float(y.skew()), 'excess_kurtosis': float(y.kurt()),
              'quantiles': {str(k): float(v) for k, v in y.quantile([.01,.05,.25,.5,.75,.95,.99]).items()}}
    dump('distribution.json', result)


def train_checks():
    raw = q._coerce(q.read_csv_auto(str(DATA / '训练集.csv')))
    dates = np.sort(raw.trade_date.unique())
    keep = dates[-1000:]
    raw = raw[raw.trade_date.isin(keep)].reset_index(drop=True)
    print('Building factors on last 1000 days', flush=True)
    f = q.build_factors(raw)
    del raw; gc.collect()
    f = q.cs_zscore(f, q.FACTOR_COLS, q.date_indices(f.trade_date.to_numpy()))
    X = f[q.FACTOR_COLS].to_numpy(dtype=np.float32)
    y = f.y_ret_1d.to_numpy(dtype=np.float64)
    d = f.trade_date.to_numpy()
    eligible = q.training_mask(f)
    include_down = (f.y_ret_1d.notna() & (f.close > 0) & (f.high >= f.low)
                    & (f.vol >= 0) & (f.amount >= 0) & f.flag_limit_up.eq(0)).to_numpy()
    all_rows = []
    for wi, vd in enumerate((keep[-500:-250], keep[-250:]), 1):
        tr_dates = keep[keep < vd[0]][:-6]
        fit_dates = tr_dates[:-66]
        stop_dates = tr_dates[-60:]
        val_mask = np.isin(d, vd)
        # Warm-up uses only features from dates preceding evaluation; no y there.
        warm_dates = keep[keep < vd[0]][-90:]
        context_mask = np.isin(d, np.r_[warm_dates, vd])
        context = f.loc[context_mask].reset_index(drop=True)
        val = f.loc[val_mask].reset_index(drop=True)
        context_is_val = np.isin(context.trade_date.to_numpy(), vd)
        predictions = {}
        scalar = {'window': wi, 'start': int(vd[0]), 'end': int(vd[-1]), 'train_days': len(tr_dates)}
        base_tr = np.isin(d, tr_dates) & eligible
        ridge = Ridge(alpha=10.0, solver='lsqr', tol=1e-5).fit(X[base_tr], y[base_tr])
        predictions['ridge'] = ridge.predict(X[context_mask])
        predictions['ret1_factor'] = f.loc[context_mask, 'f01_ret_1d'].to_numpy(dtype=np.float64)
        for name in ['huber_l2', 'rank_target', 'huber_ic', 'huber_include_down']:
            print(f'window {wi}: fitting {name}', flush=True)
            allowed = include_down if name == 'huber_include_down' else eligible
            inner = np.isin(d, fit_dates) & allowed
            stop = np.isin(d, stop_dates) & allowed
            tr = np.isin(d, tr_dates) & allowed
            target = y
            model = q.build_model(42+wi)
            model.set_params(n_jobs=4, n_estimators=800)
            if name == 'rank_target':
                temp = pd.Series(y).where(allowed).groupby(d).rank(pct=True)
                target = temp.to_numpy(dtype=np.float64) - .5
                model.set_params(objective='regression', metric='l2')
            elif name == 'huber_ic':
                model.set_params(metric='None')
            metric = rank_ic_metric(d[stop], y[stop]) if name == 'huber_ic' else 'l2'
            model.fit(X[inner], target[inner], feature_name=q.FACTOR_COLS,
                      eval_set=[(X[stop], target[stop])], eval_metric=metric,
                      callbacks=[lgb.early_stopping(40, first_metric_only=True, verbose=False)])
            iterations = model.best_iteration_
            model.set_params(n_estimators=iterations)
            model.fit(X[tr], target[tr], feature_name=q.FACTOR_COLS)
            predictions[name] = model.predict(X[context_mask])
            scalar[name+'_iterations'] = int(iterations)
            model.booster_.save_model(str(OUT / f'window{wi}_{name}.txt'))
            del model
        arrays = {}
        for name, raw_pred in predictions.items():
            ranks = percentile(context, raw_pred)
            cold = q.apply_smoothing(val, ranks[context_is_val], 20)
            warm = q.apply_smoothing(context, ranks, 20)[context_is_val]
            arrays[name+'_cold'] = cold
            row = {**scalar, 'model': name, 'initialization': 'cold', **stats(val, cold)}
            all_rows.append(row)
            print(row, flush=True)
            if name == 'huber_l2':
                arrays[name+'_warm'] = warm
                all_rows.append({**scalar, 'model': name, 'initialization': 'warm', **stats(val, warm)})
                for span in [12, 30, 60]:
                    p = q.apply_smoothing(val, ranks[context_is_val], span)
                    all_rows.append({**scalar, 'model': name+f'_span{span}', 'initialization': 'cold', **stats(val, p)})
                # Observed impact is separated from analytic decay.
                for n in [20, 30, 60]:
                    ix = val.trade_date.isin(vd[:n]).to_numpy()
                    all_rows.append({**scalar, 'model': f'ema_boundary_first{n}', 'initialization': 'cold', **stats(val.loc[ix].reset_index(drop=True), cold[ix])})
                    all_rows.append({**scalar, 'model': f'ema_boundary_first{n}', 'initialization': 'warm', **stats(val.loc[ix].reset_index(drop=True), warm[ix])})
        np.savez_compressed(OUT / f'window{wi}_predictions.npz', **arrays,
                            date=val.trade_date.to_numpy(), code=val.ts_code.to_numpy(dtype='U9'),
                            y=y[val_mask], flag_limit_up=val.flag_limit_up.to_numpy())
        del context, val, predictions, arrays; gc.collect()
        pd.DataFrame(all_rows).to_csv(OUT / 'model_comparison.csv', index=False)
    del f, X, y; gc.collect()


def test_boundary():
    hist = q._coerce(q.read_csv_auto(str(DATA / '训练集.csv')))
    days = np.sort(hist.trade_date.unique())[-150:]
    hist = hist[hist.trade_date.isin(days)].drop(columns=['y_ret_1d']).reset_index(drop=True)
    test = q.load_test(str(DATA / '测试集_X.csv'))
    first_date = int(test.trade_date.min())
    f = q.build_factors(pd.concat([hist, test], ignore_index=True))
    f = q.cs_zscore(f, q.FACTOR_COLS, q.date_indices(f.trade_date.to_numpy()))
    model = lgb.Booster(model_file=str(ROOT.parent / 'model_final.txt'))
    ranks = percentile(f, model.predict(f[q.FACTOR_COLS].to_numpy(dtype=np.float32), num_threads=4))
    is_test = f.trade_date.ge(first_date).to_numpy()
    te = f.loc[is_test].reset_index(drop=True)
    warm = q.apply_smoothing(f, ranks, 20)[is_test]
    cold = q.apply_smoothing(te, ranks[is_test], 20)
    official = pd.read_csv(ROOT.parent / 'submission.csv')
    aligned = te[['ts_code','trade_date']].merge(official, how='left', validate='one_to_one')
    result = {'cold_reproduction_max_abs_diff': float(np.max(np.abs(aligned.pred.to_numpy()-cold))),
              'rows': len(te), 'note': 'Frozen official model; historical warm-up is in-sample but precedes test. No test labels or IC/returns.'}
    test_dates = np.sort(te.trade_date.unique())
    for name, pred in [('cold', cold), ('warm', warm)]:
        te['pred'] = pred
        result[name+'_turnover'] = q.daily_turnover(te)[0]
        te['pred'] = q.demote_halted(pred, q.halted_mask(te))
        result[name+'_tradable_turnover'] = q.daily_turnover(te)[0]
    result['initial_state_decay'] = {'after20': (19/21)**20, 'updates_below5percent': 30,
                                    'updates_below1percent': 47}
    changed = []
    te_dates = te.trade_date.to_numpy()
    for n in [1, 20, 30, 60, len(test_dates)]:
        ix = np.isin(te_dates, test_dates[:n]); dif = np.abs(warm[ix]-cold[ix])
        changed.append({'days': n, 'mean_abs_difference': float(dif.mean()), 'max_abs_difference': float(dif.max())})
    result['prediction_differences'] = changed
    np.savez_compressed(OUT / 'test_boundary_predictions.npz', cold=cold, warm=warm,
                        date=te.trade_date.to_numpy(), code=te.ts_code.to_numpy(dtype='U9'))
    dump('test_boundary.json', result)
    print(result, flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {'status': 'fixed before execution', 'scope': 'retrospective, NOT untouched holdout',
                'factor_history_days': 1000, 'windows': 'last 500 days split into two consecutive 250-day windows',
                'embargo_days': 6, 'inner_stop_days': 60, 'spans': [12,20,30,60],
                'ridge_alpha': 10, 'factor_baseline': 'positive contemporaneous 1-day return',
                'model_variants': ['huber_l2', 'rank_target', 'huber_ic', 'huber_include_down'],
                'selection': 'no automatic replacement of official artifacts; assess both windows',
                'future_validation': 'freeze factors/parameters and collect future labels as available; no re-labeling observed windows as independent'}
    dump('protocol.json', protocol)
    t = time.time()
    describe(); train_checks(); test_boundary()
    dump('execution.json', {'elapsed_seconds': time.time()-t, 'complete': True})


if __name__ == '__main__':
    main()

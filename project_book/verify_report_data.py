"""Recompute report evidence in an isolated directory; never write submission."""
from pathlib import Path
import sys, json
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import fintechathon_q5 as q

OUT = ROOT / 'evidence' / 'report_verify'
OUT.mkdir(parents=True, exist_ok=True)
original_evaluate = q.evaluate_frame


def evaluate(frame, *args, **kwargs):
    result = original_evaluate(frame, *args, **kwargs)
    if str(kwargs.get('tag', '')).startswith('调优后预测'):
        dates = sorted(frame.trade_date.unique())
        idx = q.date_indices(frame.trade_date.to_numpy())
        groups = {'recent': dates[-250:], 'early': dates[:-250]}
        rows = []
        for span in [0, 5, 12, 20, 30, 40, 50, 60]:
            temp = frame[['ts_code', 'trade_date', 'y_ret_1d', 'flag_limit_up']].copy()
            temp['pred'] = q.apply_smoothing(frame, frame.model_rank.to_numpy(), span)
            row = {'span': span, 'half_life': 0 if span == 0 else float(np.log(.5)/np.log(1-2/(span+1)))}
            for name, ds in groups.items():
                subset = {d: idx[d] for d in ds}
                ic, *_ = q.daily_rank_ic(temp, date_idx=subset)
                excess, *_ = q.daily_top_excess(temp, date_idx=subset)
                turnover, _ = q.daily_turnover(temp, date_idx=subset)
                row[name] = {'ic': ic, 'annual_excess': excess, 'turnover': turnover,
                             'score': q.composite_score(ic, excess, turnover)}
            rows.append(row)
            print('REPORT EMA', json.dumps(row), flush=True)
        (OUT / 'ema_sensitivity.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    return result


q.evaluate_frame = evaluate
sys.argv = ['verify_report_data', '--data_dir', str(ROOT.parent),
            '--out_dir', str(OUT), '--tune', '0', '--anchor_weight', '0',
            '--smooth_span', '20', '--skip_submission', '--n_jobs', '8']
q.main()

"""One run's results, from its JSONL files only.

TWO MAIN ROWS (epochs are 1-based):
  gcd   "Chuẩn GCD (theo code HypCD)" -- how HypCD / SelEx / GCD compute their published numbers
        (train/train_HypSelEx.py, upstream l. 226-234 and 385-431): every epoch upstream runs test_kmeans on the
        disjoint official test split and keeps the epoch whose Old accuracy there is strictly higher than all
        earlier ones (the "best" checkpoint); the reported All / Old / New are the training-set accuracies upstream
        paired with that epoch (`gcd` in epochs_test.jsonl): SS-k-means on all training images at the START of the
        epoch (weights after the previous epoch; for epoch 1 that is the pretrained model before any training),
        or test_kmeans on the unlabelled set after the epoch when (epoch + 1) % train_report_interval == 0.
        This reproduces upstream's "Best Train Accuracies" line (tests_freqcone/test_gcd_standard.py).
        Use it to compare with published numbers / SOTA.
  last  "Epoch cuối" -- upstream test() (SS-k-means, test transform) on the weights after the last epoch
        (`eval`); no epoch is chosen at all. Robustness check.
SECONDARY
  best_val  epoch with the highest validation accuracy (ties: later epoch), its `eval` numbers. Pilot runs only: the
            confirmation runs (B8-B10, F8-F10) hold out no validation split, so their tables do not have this row.
ORACLE (printed in a separate table, reference only, never for SOTA comparison or conclusions)
  oracle_all / oracle_old / oracle_new: the epoch with the highest `eval` All / Old / New (ties: first epoch). Each
  metric comes from a different epoch chosen on the test split, so the numbers are inflated.

usage: python -m freqcone.summarize_run runs/<run_id> [--json]
Test numbers of R1-R3 are refused until runs/selection.json is committed (freqcone/runs.py).
"""
import argparse
import json
import os
import sys

from freqcone.runs import LockError, load_run, utf8_stdout

ROWS = [
    ('gcd', 'Chuẩn GCD (theo code HypCD)', 'CHÍNH (1): so với số công bố / SOTA'),
    ('last', 'Epoch cuối', 'CHÍNH (2): độ bền, không chọn epoch'),
    ('best_val', 'Epoch có val cao nhất', 'phụ (chọn bằng val)'),
    ('oracle_all', 'Oracle: max All', 'oracle, chọn bằng test, chỉ tham khảo'),
    ('oracle_old', 'Oracle: max Old', 'oracle'),
    ('oracle_new', 'Oracle: max New', 'oracle'),
]
MAIN = ('gcd', 'last')
ORACLE = ('oracle_all', 'oracle_old', 'oracle_new')


def _first_strict_max(items, key):
    """Upstream's rule: best starts at 0 and moves only on a strictly higher value."""
    best, best_v = None, 0.0
    for it in items:
        v = key(it)
        if v is not None and v > best_v:
            best, best_v = it, v
    return best


def _argmax(items, key, prefer_later=False):
    best = None
    for it in items:
        v = key(it)
        if v is None:
            continue
        if best is None or v > key(best) or (prefer_later and v == key(best)):
            best = it
    return best


def table(val_lines, test_lines):
    tests = sorted(test_lines, key=lambda t: t['epoch'])
    vals = {v['epoch']: v for v in val_lines}
    with_eval = [t for t in tests if t.get('eval')]
    rows = []

    def add(key, line, metrics_key):
        label, role = next((lb, rl) for k, lb, rl in ROWS if k == key)
        m = (line or {}).get(metrics_key) or {}
        rows.append({'key': key, 'label': label, 'role': role, 'epoch': line['epoch'] if line else None,
                     'oracle': key in ORACLE, 'main': key in MAIN,
                     'all': m.get('all'), 'old': m.get('old'), 'new': m.get('new'),
                     **({'source': m.get('source'), 'weights_after_epoch': m.get('weights_after_epoch')}
                        if metrics_key == 'gcd' else {})})
    add('gcd', _first_strict_max(tests, lambda t: (t.get('disjoint_test') or {}).get('old')), 'gcd')
    add('last', tests[-1] if tests and tests[-1].get('eval') else None, 'eval')
    best_val = _argmax([t for t in with_eval if t['epoch'] in vals],
                       lambda t: vals[t['epoch']].get('val_acc'), prefer_later=True)
    if best_val is not None:
        add('best_val', best_val, 'eval')
    for key, m in (('oracle_all', 'all'), ('oracle_old', 'old'), ('oracle_new', 'new')):
        add(key, _argmax(with_eval, lambda t, m=m: t['eval'][m]), 'eval')
    return rows


def markdown(rows, title):
    pct = lambda x: '-' if x is None else f'{100 * x:.2f}'

    def block(keys, heading):
        out = [f'#### {heading}', '', '| Hàng | Epoch | All | Old (Known) | New (Novel) | Vai trò |',
               '| --- | --- | --- | --- | --- | --- |']
        for r in rows:
            if r['key'] in keys:
                out.append(f"| {r['label']} | {r['epoch'] if r['epoch'] is not None else '-'} | {pct(r['all'])} | "
                           f"{pct(r['old'])} | {pct(r['new'])} | {r['role']} |")
        return out
    lines = [f'### {title}', ''] + block(MAIN + ('best_val',), 'Kết quả') + ['']
    gcd = next(r for r in rows if r['key'] == 'gcd')
    if gcd['source']:
        lines.append(f"Hàng (1): epoch được chọn bằng Old trên tập test rời (quy tắc checkpoint của upstream); số lấy từ "
                     f"`{gcd['source']}` trên trọng số sau epoch {gcd['weights_after_epoch']}.")
        lines.append('')
    lines += block(ORACLE, 'Oracle (chọn bằng test, tham khảo; không dùng để so SOTA hay kết luận)')
    return '\n'.join(lines) + '\n'


def summarize(run_dir):
    r = load_run(run_dir, with_test=True)
    rows = table(r['val'], r['test'])
    meta = r['run']
    fc = meta['freqcone']
    title = (f"{meta['run_id']} ({'FreqCone ' + str(fc['variant']) + ', λ_ent ' + str(fc['w_ent']) + ', ' + str(fc['aperture']) if fc['enable'] else 'baseline'}"
             f", seed {meta['seed']}, {meta['dataset']})")
    return {'run_id': meta['run_id'], 'seed': meta['seed'], 'freqcone': fc, 'rows': rows,
            'epochs_recorded': len(r['test'])}, markdown(rows, title)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('--json', action='store_true', help='also write RUN_DIR/summary.json')
    a = ap.parse_args(argv)
    utf8_stdout()
    try:
        out, md = summarize(a.run_dir)
    except LockError as e:
        print(f'refused: {e}', file=sys.stderr)
        return 2
    print(md)
    if a.json:
        with open(os.path.join(a.run_dir, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=1, ensure_ascii=False)
    return 0


if __name__ == '__main__':
    sys.exit(main())

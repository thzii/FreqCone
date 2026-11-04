"""One run's table (plan, section 2.2), from its JSONL files only.

Rows (epochs are 1-based):
  last        the last epoch                                     MAIN RESULT
  best_val    epoch with the highest validation accuracy         secondary, valid (chosen on validation);
              (ties go to the later epoch)
  upstream    epoch with the highest Old accuracy on the         for comparison with published numbers (G0)
              disjoint test split, upstream's checkpoint rule
              (ties go to the first epoch, as upstream's '>')
  oracle_all  epoch with the highest All                         ORACLE: chosen on the test split, reference only
  oracle_old  epoch with the highest Old                         ORACLE
  oracle_new  epoch with the highest New                         ORACLE (ties: first epoch)
Every row reports All / Old / New (v2, transductive, unlabelled training images) of its epoch. The oracle rows take
each metric at a different epoch chosen on the test split: they are inflated and never support a conclusion.

usage: python -m freqcone.summarize_run runs/<run_id> [--json]
Test numbers of R1-R3 are refused until runs/selection.json is committed (freqcone/runs.py).
"""
import argparse
import json
import os
import sys

from freqcone.runs import LockError, load_run, utf8_stdout

ROWS = [
    ('last', 'Epoch cuối', 'KẾT QUẢ CHÍNH'),
    ('best_val', 'Epoch có val cao nhất', 'kết quả phụ (chọn bằng val)'),
    ('upstream', 'Tiêu chí upstream: Old cao nhất trên test rời', 'chỉ để đối chiếu số công bố (G0)'),
    ('oracle_all', 'Oracle: max All', 'oracle, chọn bằng test, không dùng để kết luận'),
    ('oracle_old', 'Oracle: max Old', 'oracle'),
    ('oracle_new', 'Oracle: max New', 'oracle'),
]
ORACLE = ('oracle_all', 'oracle_old', 'oracle_new')


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
    by_epoch = {t['epoch']: t for t in tests}
    vals = sorted(val_lines, key=lambda v: v['epoch'])
    picks = {
        'last': tests[-1]['epoch'],
        'best_val': (_argmax(vals, lambda v: v.get('val_acc'), prefer_later=True) or {}).get('epoch'),
        'upstream': (_argmax(tests, lambda t: (t.get('upstream_disjoint_test') or {}).get('old')) or {}).get('epoch'),
        'oracle_all': _argmax(tests, lambda t: t['all'])['epoch'],
        'oracle_old': _argmax(tests, lambda t: t['old'])['epoch'],
        'oracle_new': _argmax(tests, lambda t: t['new'])['epoch'],
    }
    rows = []
    for key, label, role in ROWS:
        e = picks[key]
        t = by_epoch.get(e) if e is not None else None
        rows.append({'key': key, 'label': label, 'role': role, 'epoch': e, 'oracle': key in ORACLE,
                     'all': t['all'] if t else None, 'old': t['old'] if t else None, 'new': t['new'] if t else None})
    return rows


def markdown(rows, title):
    pct = lambda x: '-' if x is None else f'{100 * x:.2f}'
    lines = [f'### {title}', '', '| Cách chọn epoch | Epoch | All | Old (Known) | New (Novel) | Vai trò |',
             '| --- | --- | --- | --- | --- | --- |']
    for r in rows:
        lines.append(f"| {r['label']} | {r['epoch'] if r['epoch'] is not None else '-'} | {pct(r['all'])} | "
                     f"{pct(r['old'])} | {pct(r['new'])} | {r['role']} |")
    lines.append('')
    lines.append('Các hàng oracle chọn epoch bằng chính tập test và lấy mỗi chỉ số ở một epoch khác nhau: số bị thổi '
                 'phồng, chỉ để tham khảo.')
    return '\n'.join(lines) + '\n'


def summarize(run_dir):
    r = load_run(run_dir, with_test=True)
    rows = table(r['val'], r['test'])
    meta = r['run']
    title = (f"{meta['run_id']} ({'FreqCone ' + meta['freqcone']['variant'] + ', λ_ent ' + str(meta['freqcone']['w_ent']) + ', ' + meta['freqcone']['aperture'] if meta['freqcone']['enable'] else 'baseline'}"
             f", seed {meta['seed']}, {meta['dataset']})")
    return {'run_id': meta['run_id'], 'seed': meta['seed'], 'freqcone': meta['freqcone'], 'rows': rows,
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
        with open(os.path.join(a.run_dir, 'summary.json'), 'w') as f:
            json.dump(out, f, indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())

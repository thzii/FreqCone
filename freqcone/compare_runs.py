"""Baseline against FreqCone, paired by seed, from the JSONL files only.

For every row of summarize_run: per-seed differences FreqCone - baseline of All / Old / New (points), their mean,
standard deviation and the number of seeds won. Gates and conclusions must pass on BOTH main rows
("Chuẩn GCD (theo code HypCD)" and "Epoch cuối"):
  --gate g1     one seed (pilot): dNew >= +1.0 and dAll >= -0.5, on each main row
  --gate final  seeds 8-10: mean dNew >= +1.0, wins on New >= 2/3 of the seeds, mean dAll >= -0.5, on each main row
SOTA table: row (1) of each arm, mean ± std over the seeds and every seed, next to HypCD's published Hyp-SelEx
numbers (one published run). The oracle rows are printed in a separate table titled
"Oracle (chọn bằng test, tham khảo)". --plot draws the last-epoch-style evaluation (`eval`) per epoch.

usage: python -m freqcone.compare_runs --base runs/base_s8 runs/base_s9 runs/base_s10 \
           --fc runs/fc_s8 runs/fc_s9 runs/fc_s10 --gate final --plot runs/compare_final.png --out runs/compare_final.json
"""
import argparse
import json
import math
import sys

import numpy as np

from freqcone.runs import LockError, load_run, utf8_stdout
from freqcone.summarize_run import MAIN, ORACLE, ROWS, table

G1_NEW, G1_ALL = 1.0, -0.5
FINAL_NEW, FINAL_ALL, FINAL_WIN_SHARE = 1.0, -0.5, 2 / 3
PUBLISHED = {'scars': {'all': 83.8, 'old': 93.3, 'new': 79.2},     # HypCD Table 1, Hyp-SelEx, DINOv2, one run
             'cub': {'all': 90.7, 'old': 85.3, 'new': 93.4}}
SERIES = {'base': ('Baseline (Hyp-SelEx)', '#2a78d6'), 'fc': ('FreqCone', '#eb6834')}   # categorical slots 1, 2
SURFACE, INK, INK_2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'


def pair_up(base_dirs, fc_dirs):
    base = {r['run']['seed']: r for r in (load_run(d, with_test=True) for d in base_dirs)}
    fc = {r['run']['seed']: r for r in (load_run(d, with_test=True) for d in fc_dirs)}
    if set(base) != set(fc):
        raise ValueError(f'seeds differ: baseline {sorted(base)}, FreqCone {sorted(fc)}')
    for s in base:
        if base[s]['run']['freqcone']['enable'] or not fc[s]['run']['freqcone']['enable']:
            raise ValueError(f'seed {s}: --base must be baselines and --fc FreqCone runs')
    return [(s, base[s], fc[s]) for s in sorted(base)]


def deltas(pairs):
    out = {}
    for key, label, role in ROWS:
        per_seed = []
        for s, b, f in pairs:
            rb = next(r for r in table(b['val'], b['test']) if r['key'] == key)
            rf = next(r for r in table(f['val'], f['test']) if r['key'] == key)
            if rb['all'] is None or rf['all'] is None:
                continue
            per_seed.append({'seed': s, 'epoch_base': rb['epoch'], 'epoch_fc': rf['epoch'],
                             **{m: 100 * (rf[m] - rb[m]) for m in ('all', 'old', 'new')},
                             **{f'base_{m}': 100 * rb[m] for m in ('all', 'old', 'new')},
                             **{f'fc_{m}': 100 * rf[m] for m in ('all', 'old', 'new')}})
        stats = {}
        for m in ('all', 'old', 'new'):
            d = np.array([p[m] for p in per_seed])
            stats[m] = {'mean': float(d.mean()) if d.size else None,
                        'std': float(d.std(ddof=1)) if d.size > 1 else None,
                        'wins': int((d > 0).sum()), 'n': int(d.size)}
            for arm in ('base', 'fc'):
                v = np.array([p[f'{arm}_{m}'] for p in per_seed])
                stats[f'{arm}_{m}'] = {'mean': float(v.mean()) if v.size else None,
                                       'std': float(v.std(ddof=1)) if v.size > 1 else None}
        out[key] = {'label': label, 'role': role, 'oracle': key in ORACLE, 'main': key in MAIN,
                    'per_seed': per_seed, 'stats': stats}
    return out


def _row_pass(st, kind):
    if st['new']['n'] == 0:
        return False
    if kind == 'g1':
        return st['new']['mean'] >= G1_NEW and st['all']['mean'] >= G1_ALL
    need = math.ceil(FINAL_WIN_SHARE * st['new']['n'] - 1e-9)
    return st['new']['mean'] >= FINAL_NEW and st['new']['wins'] >= need and st['all']['mean'] >= FINAL_ALL


def gate(d, kind):
    if kind == 'g1' and any(d[k]['stats']['new']['n'] != 1 for k in MAIN):
        raise ValueError('G1 compares one pilot pair')
    rule = (f'dNew >= +{G1_NEW} and dAll >= {G1_ALL}' if kind == 'g1' else
            f'mean dNew >= +{FINAL_NEW}, wins on New >= 2/3 of the seeds, mean dAll >= {FINAL_ALL}')
    per_row = {k: bool(_row_pass(d[k]['stats'], kind)) for k in MAIN}
    return {'gate': kind, 'rule': rule + ', on both main rows', 'per_row': per_row, 'pass': all(per_row.values()),
            'rows': {k: d[k]['stats'] for k in MAIN}}


def markdown(d, g=None, dataset='scars'):
    f = lambda x: '-' if x is None else f'{x:+.2f}'
    s = lambda x: '-' if x is None else f'{x:.2f}'

    def block(keys, title):
        lines = [f'### {title}', '', '| Hàng | ΔAll (TB ± SD) | ΔOld (TB ± SD) | ΔNew (TB ± SD) | Thắng New | n |',
                 '| --- | --- | --- | --- | --- | --- |']
        for k in keys:
            st = d[k]['stats']
            lines.append(f"| {d[k]['label']} | {f(st['all']['mean'])} ± {s(st['all']['std'])} | "
                         f"{f(st['old']['mean'])} ± {s(st['old']['std'])} | {f(st['new']['mean'])} ± {s(st['new']['std'])} | "
                         f"{st['new']['wins']}/{st['new']['n']} | {st['new']['n']} |")
        return lines
    lines = block(list(MAIN) + ['best_val'], 'FreqCone − baseline theo cặp seed (điểm %)') + ['']
    for k in MAIN:
        lines += [f"Theo từng seed, hàng \"{d[k]['label']}\":", '',
                  '| Seed | Epoch base / FreqCone | Base All/Old/New | FreqCone All/Old/New | ΔAll | ΔOld | ΔNew |',
                  '| --- | --- | --- | --- | --- | --- | --- |']
        for p in d[k]['per_seed']:
            lines.append(f"| {p['seed']} | {p['epoch_base']} / {p['epoch_fc']} | {p['base_all']:.2f} / {p['base_old']:.2f} / "
                         f"{p['base_new']:.2f} | {p['fc_all']:.2f} / {p['fc_old']:.2f} / {p['fc_new']:.2f} | "
                         f"{p['all']:+.2f} | {p['old']:+.2f} | {p['new']:+.2f} |")
        lines.append('')
    st = d['gcd']['stats']
    pub = PUBLISHED.get(dataset)
    lines += ['### So với số công bố (hàng "Chuẩn GCD (theo code HypCD)")', '',
              '| | All | Old | New | Ghi chú |', '| --- | --- | --- | --- | --- |']
    if pub:
        lines.append(f"| HypCD công bố (Hyp-SelEx, DINOv2) | {pub['all']:.1f} | {pub['old']:.1f} | {pub['new']:.1f} | "
                     'một lần chạy (Bảng 1 của HypCD) |')
    for arm, name in (('base', 'Baseline (bug đã sửa)'), ('fc', 'FreqCone')):
        seeds = ', '.join(f"{p['seed']}: {p[f'{arm}_all']:.2f}/{p[f'{arm}_old']:.2f}/{p[f'{arm}_new']:.2f}"
                          for p in d['gcd']['per_seed'])
        lines.append(f"| {name} | {s(st[f'{arm}_all']['mean'])} ± {s(st[f'{arm}_all']['std'])} | "
                     f"{s(st[f'{arm}_old']['mean'])} ± {s(st[f'{arm}_old']['std'])} | "
                     f"{s(st[f'{arm}_new']['mean'])} ± {s(st[f'{arm}_new']['std'])} | "
                     f"TB ± SD trên {st['new']['n']} seed; từng seed {seeds} |")
    lines += [''] + block(list(ORACLE), 'Oracle (chọn bằng test, tham khảo)') + ['']
    if g:
        rows = ', '.join(f"{d[k]['label']}: {'ĐẠT' if ok else 'KHÔNG ĐẠT'}" for k, ok in g['per_row'].items())
        lines.append(f"**Cổng {g['gate']}** ({g['rule']}): {rows} → **{'ĐẠT' if g['pass'] else 'KHÔNG ĐẠT'}**")
    return '\n'.join(lines) + '\n'


def plot(pairs, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    curves = {}
    for arm in ('base', 'fc'):
        runs = [b if arm == 'base' else f for _, b, f in pairs]
        epochs = sorted(set.intersection(*[{t['epoch'] for t in r['test'] if t.get('eval')} for r in runs]))
        if not epochs:
            raise ValueError('no per-epoch evaluation to plot (run with --eval_every_epoch True)')
        curves[arm] = {'epoch': np.array(epochs)}
        for m in ('all', 'old', 'new'):
            vals = np.array([[100 * next(t['eval'][m] for t in r['test'] if t['epoch'] == e) for e in epochs]
                             for r in runs])
            curves[arm][m] = (vals.mean(0), vals.min(0), vals.max(0))
    lo = min(c[m][1].min() for c in curves.values() for m in ('all', 'old', 'new'))
    hi = max(c[m][2].max() for c in curves.values() for m in ('all', 'old', 'new'))
    pad = 0.04 * (hi - lo + 1e-9)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True, facecolor=SURFACE)
    min_sep = 0.06 * (hi - lo + 2 * pad)                       # end labels closer than this are pushed apart
    for ax, (m, title) in zip(axes, (('all', 'All'), ('old', 'Old (Known)'), ('new', 'New (Novel)'))):
        ax.set_facecolor(SURFACE)
        ends = {}
        for arm in ('base', 'fc'):
            name, color = SERIES[arm]
            ep = curves[arm]['epoch']
            mean, mn, mx = curves[arm][m]
            if len(pairs) > 1:
                ax.fill_between(ep, mn, mx, color=color, alpha=0.15, linewidth=0)
            ax.plot(ep, mean, color=color, linewidth=2, label=name, solid_capstyle='round')
            ends[arm] = (ep[-1], mean[-1])
        # last-epoch values: a dot in the series colour (with a surface ring) carries identity, the number stays in
        # text ink; labels closer than min_sep are spread symmetrically so they never collide
        ys = {arm: v for arm, (_, v) in ends.items()}
        if abs(ys['base'] - ys['fc']) < min_sep:
            mid = (ys['base'] + ys['fc']) / 2
            upper = 'fc' if ys['fc'] >= ys['base'] else 'base'
            ys = {arm: mid + (min_sep / 2 if arm == upper else -min_sep / 2) for arm in ys}
        for arm, (x_end, y_end) in ends.items():
            ax.plot([x_end], [y_end], 'o', color=SERIES[arm][1], markersize=5, markeredgecolor=SURFACE,
                    markeredgewidth=1.5, zorder=5)
            ax.annotate(f'{y_end:.1f}', xy=(x_end, ys[arm]), xytext=(7, 0), textcoords='offset points', va='center',
                        fontsize=8, color=INK_2, annotation_clip=False)
        ax.set_title(title, color=INK, fontsize=11, loc='left')
        ax.set_xlabel('epoch', color=INK_2)
        ax.set_ylim(lo - pad, hi + pad)
        ax.grid(True, color=GRID, linewidth=0.8)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=8)
    axes[0].set_ylabel('accuracy (%)', color=INK_2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', ncol=2, frameon=False, fontsize=9, labelcolor=INK)
    n = len(pairs)
    fig.suptitle(f'Test accuracy after each epoch (SS-k-means, v2), mean over {n} seed{"s" if n > 1 else ""}'
                 + (' (band: min-max)' if n > 1 else ''), x=0.01, ha='left', color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', nargs='+', required=True)
    ap.add_argument('--fc', nargs='+', required=True)
    ap.add_argument('--gate', choices=['none', 'g1', 'final'], default='none')
    ap.add_argument('--plot', default=None)
    ap.add_argument('--out', default=None)
    a = ap.parse_args(argv)
    utf8_stdout()
    try:
        pairs = pair_up(a.base, a.fc)
    except LockError as e:
        print(f'refused: {e}', file=sys.stderr)
        return 2
    d = deltas(pairs)
    g = gate(d, a.gate) if a.gate != 'none' else None
    dataset = pairs[0][1]['run'].get('dataset', 'scars')
    print(markdown(d, g, dataset))
    if a.plot:
        print(f'plot written to {plot(pairs, a.plot)}')
    if a.out:
        with open(a.out, 'w', encoding='utf-8') as f:
            json.dump({'seeds': [s for s, _, _ in pairs], 'rows': d, 'gate': g}, f, indent=1, ensure_ascii=False)
    if g is not None and not g['pass']:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())

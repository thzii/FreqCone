"""Compute FreqCone's cut-offs f_a, f_b (SNR 5 / 10 dB, FourEx's definition) from the training images only.

Images read: the labelled training images without the held-out validation split, plus the unlabelled training
images, through HypCD's test transform at 224 px. Labels are never read (the loop keeps batch[0] only), and the
script checks that no validation image is among them. The definition, its citation and the choices FourEx
leaves open are in freqcone/cutoffs.py.

Writes cutoffs.json at the repository root (one entry per dataset, other entries kept) and the cut-off table
in freqcone/README.md. Commit cutoffs.json before the pilot. If a value differs from the handoff's 3 / 11
(Stanford Cars) or from FourEx Table 7, the computed value is the one used; the table records the difference.

usage (from anywhere): python scripts/freqcone/compute_cutoffs.py --datasets cub scars aircraft --device cuda
"""
import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from freqcone.cutoffs import CITATION, DEFINITION, FOUREX_TABLE7, LEVELS_DB, PowerAccumulator, cutoffs_from_power  # noqa: E402
from freqcone.data_checks import check_val_split, item_ids, list_sha256  # noqa: E402

HANDOFF = {'scars': (3, 11)}          # values written in the handoff, never verified
README_START, README_END = '<!-- cutoffs:start -->', '<!-- cutoffs:end -->'


def build_args(name):
    from data.get_datasets import get_class_splits
    ns = argparse.Namespace(dataset_name=name, use_ssb_splits=True, prop_train_labels=0.5, split_train_val=True,
                            smoke_frac=0, image_size=224, interpolation=3, crop_pct=0.875, transform='imagenet')
    return get_class_splits(ns)


def cutoff_image_set(train, val, known_classes):
    """Identity of the images the cut-offs are computed from; raises if a validation image is among them."""
    split = check_val_split(train.labelled_dataset, train.unlabelled_dataset, val, known_classes)
    ids = item_ids(train.labelled_dataset) + item_ids(train.unlabelled_dataset)
    used = {int(i.split(':', 1)[0]) for i in ids}
    if used & {int(i) for i in val.uq_idxs}:
        raise AssertionError('a validation image would enter the cut-off computation')
    return ids, split


def compute(name, device, batch_size=64, num_workers=4, max_images=0):
    from data.augmentations import get_transform
    from data.get_datasets import get_datasets
    args = build_args(name)
    _, test_tf = get_transform('imagenet', image_size=224, args=args)
    train, _, _, datasets = get_datasets(name, test_tf, test_tf, args)
    ids, split = cutoff_image_set(train, datasets['val'], args.train_classes)
    acc = PowerAccumulator()
    for batch in DataLoader(train, batch_size=batch_size, shuffle=False, num_workers=num_workers):
        acc.add(batch[0].to(device))                      # batch[0]: images; labels are not read
        if max_images and acc.n >= max_images:
            break
    levels, curve = cutoffs_from_power(acc.mean(), levels=LEVELS_DB + (15.0, 20.0, 25.0))
    f_a, f_b = levels[LEVELS_DB[0]], levels[LEVELS_DB[1]]
    if not f_a < f_b:
        raise ValueError(f'{name}: f_a = {f_a} is not below f_b = {f_b}; report before choosing other levels')
    return {'f_a': f_a, 'f_b': f_b, 'levels_db': list(LEVELS_DB), 'img_size': 224, 'n_images': acc.n,
            'complete': not max_images, 'image_list_sha256': list_sha256(ids), 'validation_excluded': split,
            'other_levels': {str(int(k)): v for k, v in levels.items() if k not in LEVELS_DB},
            'snr_curve_db': [None if v == float('inf') else round(float(v), 4) for v in curve[:61]],
            'snr_definition': DEFINITION, 'citation': CITATION,
            'fourex_table7': FOUREX_TABLE7.get(name), 'handoff': HANDOFF.get(name),
            'computed_utc': datetime.now(timezone.utc).isoformat()}


def readme_table(entries):
    lines = [README_START, '| Dataset | f_a (5 dB) | f_b (10 dB) | FourEx Table 7 (5 / 10 dB) | Handoff | Images |',
             '| --- | --- | --- | --- | --- | --- |']
    for name, e in sorted(entries.items()):
        t7 = e.get('fourex_table7') or {}
        ref = f"{t7.get('5', t7.get(5))} / {t7.get('10', t7.get(10))}" if t7 else '-'
        hand = f"{e['handoff'][0]} / {e['handoff'][1]}" if e.get('handoff') else '-'
        lines.append(f"| {name} | {e['f_a']} | {e['f_b']} | {ref} | {hand} | {e['n_images']} |")
    lines.append('Computed by `scripts/freqcone/compute_cutoffs.py`; the computed values are the ones used.')
    lines.append(README_END)
    return '\n'.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--datasets', nargs='+', default=['cub', 'scars', 'aircraft'])
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--max_images', type=int, default=0, help='testing only: stop early (entry marked incomplete)')
    ap.add_argument('--out', default=os.path.join(REPO_ROOT, 'cutoffs.json'))
    a = ap.parse_args(argv)
    os.chdir(REPO_ROOT)                                   # the SSB split files are read relative to the repo root
    entries = {}
    if os.path.exists(a.out):
        with open(a.out) as f:
            entries = json.load(f)
    for name in a.datasets:
        entries[name] = compute(name, a.device, a.batch_size, a.num_workers, a.max_images)
        e = entries[name]
        print(f"{name}: f_a {e['f_a']}, f_b {e['f_b']} from {e['n_images']} training images "
              f"(FourEx Table 7 5/10 dB: {e['fourex_table7'] and (e['fourex_table7'][5], e['fourex_table7'][10])}; "
              f"handoff: {e['handoff']})")
    with open(a.out, 'w') as f:
        json.dump(entries, f, indent=1)
    readme = os.path.join(REPO_ROOT, 'freqcone', 'README.md')
    with open(readme, encoding='utf-8') as f:
        text = f.read()
    pattern = re.compile(re.escape(README_START) + '.*?' + re.escape(README_END), re.S)
    with open(readme, 'w', encoding='utf-8') as f:
        f.write(pattern.sub(lambda _: readme_table(entries), text))
    print(f'written {a.out}; commit it (and freqcone/README.md) before the pilot')


if __name__ == '__main__':
    main()

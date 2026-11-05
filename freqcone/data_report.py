"""What the training runs will see: image counts per split, known / novel classes, the validation split, the labelled
images per known class, the comparison with upstream's split (no validation split held out), and the SHA-256 of every
image list (sorted 'uq_idx:file' entries). Run it on every machine and compare the hashes.

usage: python -m freqcone.data_report --dataset scars [--root DIR] [--json PATH] [--md docs/DATA_CHECK.md]
"""
import argparse
import json
import os
import sys

import numpy as np

from freqcone.data_checks import check_merged_train, check_val_split, item_ids, list_sha256, raw_targets
from freqcone.datasets import build_args, set_root


def report(dataset, root=None):
    from data.get_datasets import get_datasets
    if root:
        set_root(dataset, root)
    args = build_args(dataset)
    train, test, _, datasets = get_datasets(dataset, None, None, args)
    known, novel = set(int(k) for k in args.train_classes), set(int(k) for k in args.unlabeled_classes)
    if dataset == 'scars' and (len(known), len(novel)) != (98, 98):
        raise AssertionError(f'the SSB split of Stanford Cars has 98 known / 98 novel classes, got {len(known)} / {len(novel)}')
    lab, unl, val = train.labelled_dataset, train.unlabelled_dataset, datasets['val']
    split = check_val_split(lab, unl, val, args.train_classes)
    labelled = check_merged_train(train, args.train_classes)
    upstream, _, _, _ = get_datasets(dataset, None, None, build_args(dataset, split_train_val=False))
    n_upstream = len(upstream.labelled_dataset)
    unl_t = raw_targets(unl)
    per_class_val = np.bincount(raw_targets(val))
    out = {'dataset': dataset, 'classes': {'known': len(known), 'novel': len(novel)},
           'images': {'train_total': len(lab) + len(unl) + len(val), 'labelled': len(lab), 'validation': len(val),
                      'unlabelled': len(unl), 'unlabelled_known': int(np.isin(unl_t, list(known)).sum()),
                      'unlabelled_novel': int(np.isin(unl_t, list(novel)).sum()), 'test': len(test)},
           'validation_per_class': {'min': int(per_class_val[list(known)].min()),
                                    'max': int(per_class_val[list(known)].max())},
           'labelled_per_class': {'min': labelled['labelled_per_known_min'], 'max': labelled['labelled_per_known_max']},
           'upstream_labelled': {'n': n_upstream, 'ratio': len(lab) / n_upstream},
           'lists_sha256': {k: list_sha256(item_ids(ds)) for k, ds in
                            (('labelled', lab), ('unlabelled', unl), ('validation', val), ('test', test))},
           'val_check': split}
    return out


def markdown(out, root):
    im, cl = out['images'], out['classes']
    rows = [('Train, tổng', im['train_total']), ('Labelled (sau khi tách val)', im['labelled']),
            ('Unlabelled', f"{im['unlabelled']} (known {im['unlabelled_known']}, novel {im['unlabelled_novel']})"),
            ('Val', im['validation']), ('Test rời', im['test']), ('Lớp known / novel', f"{cl['known']} / {cl['novel']}"),
            ('Labelled mỗi lớp known, min–max', f"{out['labelled_per_class']['min']}–{out['labelled_per_class']['max']}"),
            ('Val mỗi lớp known, min–max', f"{out['validation_per_class']['min']}–{out['validation_per_class']['max']}"),
            ('Labelled của upstream (không tách val)', out['upstream_labelled']['n']),
            ('Tỉ lệ labelled so với upstream', f"{out['upstream_labelled']['ratio']:.4f}")]
    lines = [f"# Kiểm tra dữ liệu: {out['dataset']}", '', f'Sinh bởi `python -m freqcone.data_report`, dữ liệu tại '
             f'`{root or "đường dẫn trong config.py"}`.', '', '| Mục | Giá trị |', '| --- | --- |']
    lines += [f'| {k} | {v} |' for k, v in rows]
    lines += ['', '| Danh sách ảnh | SHA-256 |', '| --- | --- |']
    lines += [f'| {k} | `{v}` |' for k, v in out['lists_sha256'].items()]
    return '\n'.join(lines) + '\n'


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='scars', choices=['scars', 'cub', 'aircraft'])
    ap.add_argument('--root', default=None, help='dataset directory (default: the path in config.py)')
    ap.add_argument('--json', default=None, help='also write the report here')
    ap.add_argument('--md', default=None, help='also write a Markdown report here (docs/DATA_CHECK.md on the real data)')
    a = ap.parse_args(argv)
    out = report(a.dataset, a.root)
    im = out['images']
    print(f"{out['dataset']}: {out['classes']['known']} known / {out['classes']['novel']} novel classes")
    print(f"train {im['train_total']} = labelled {im['labelled']} + validation {im['validation']} + unlabelled "
          f"{im['unlabelled']} (known {im['unlabelled_known']}, novel {im['unlabelled_novel']}); test {im['test']}")
    print(f"validation images per known class: {out['validation_per_class']['min']}-{out['validation_per_class']['max']}")
    print(f"labelled images per known class: {out['labelled_per_class']['min']}-{out['labelled_per_class']['max']}")
    print(f"labelled vs upstream (no validation split): {im['labelled']} / {out['upstream_labelled']['n']} = "
          f"{out['upstream_labelled']['ratio']:.4f}")
    for k, v in out['lists_sha256'].items():
        print(f'sha256 {k:11s} {v}')
    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        with open(a.json, 'w') as f:
            json.dump(out, f, indent=1)
    if a.md:
        with open(a.md, 'w', encoding='utf-8') as f:
            f.write(markdown(out, a.root))
    return 0


if __name__ == '__main__':
    sys.exit(main())

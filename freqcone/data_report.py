"""What the training runs will see: image counts per split, known / novel classes, the validation split, and the
SHA-256 of every image list (sorted 'uq_idx:file' entries). Run it on every machine and compare the hashes.

usage: python -m freqcone.data_report --dataset scars [--root DIR] [--json PATH]
"""
import argparse
import json
import os
import sys

import numpy as np

from freqcone.data_checks import check_val_split, item_ids, list_sha256, raw_targets
from freqcone.datasets import build_args, set_root


def report(dataset, root=None):
    from data.get_datasets import get_datasets
    if root:
        set_root(dataset, root)
    args = build_args(dataset)
    train, test, _, datasets = get_datasets(dataset, None, None, args)
    known, novel = set(int(k) for k in args.train_classes), set(int(k) for k in args.unlabeled_classes)
    lab, unl, val = train.labelled_dataset, train.unlabelled_dataset, datasets['val']
    split = check_val_split(lab, unl, val, args.train_classes)
    unl_t = raw_targets(unl)
    per_class_val = np.bincount(raw_targets(val))
    out = {'dataset': dataset, 'classes': {'known': len(known), 'novel': len(novel)},
           'images': {'train_total': len(lab) + len(unl) + len(val), 'labelled': len(lab), 'validation': len(val),
                      'unlabelled': len(unl), 'unlabelled_known': int(np.isin(unl_t, list(known)).sum()),
                      'unlabelled_novel': int(np.isin(unl_t, list(novel)).sum()), 'test': len(test)},
           'validation_per_class': {'min': int(per_class_val[list(known)].min()),
                                    'max': int(per_class_val[list(known)].max())},
           'lists_sha256': {k: list_sha256(item_ids(ds)) for k, ds in
                            (('labelled', lab), ('unlabelled', unl), ('validation', val), ('test', test))},
           'val_check': split}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='scars', choices=['scars', 'cub', 'aircraft'])
    ap.add_argument('--root', default=None, help='dataset directory (default: the path in config.py)')
    ap.add_argument('--json', default=None, help='also write the report here')
    a = ap.parse_args(argv)
    out = report(a.dataset, a.root)
    im = out['images']
    print(f"{out['dataset']}: {out['classes']['known']} known / {out['classes']['novel']} novel classes")
    print(f"train {im['train_total']} = labelled {im['labelled']} + validation {im['validation']} + unlabelled "
          f"{im['unlabelled']} (known {im['unlabelled_known']}, novel {im['unlabelled_novel']}); test {im['test']}")
    print(f"validation images per known class: {out['validation_per_class']['min']}-{out['validation_per_class']['max']}")
    for k, v in out['lists_sha256'].items():
        print(f'sha256 {k:11s} {v}')
    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        with open(a.json, 'w') as f:
            json.dump(out, f, indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())

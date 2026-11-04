"""Synthetic Stanford Cars in the layout that data/stanford_cars.py reads, to run commands without the dataset.

Writes ROOT/cars_train/*.jpg, ROOT/cars_test/*.jpg, ROOT/devkit/cars_train_annos.mat and
ROOT/devkit/cars_test_annos_withlabels.mat (fields bbox_x1, bbox_y1, bbox_x2, bbox_y2, class (1-based), fname),
for all 196 classes. Images are smooth random colour fields: no real content, for tests and dry runs only.
With the default 20 training images per class the GCD validation split is non-empty (2 images per known class).

usage: python -m freqcone.synthetic_data --out /tmp/fake_scars [--per_class_train 20] [--per_class_test 4] [--size 48]
"""
import argparse
import os

import numpy as np

N_CLASSES = 196
FIELDS = ('bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'class', 'fname')


def _image(rng, size):
    from PIL import Image
    coarse = (rng.random((6, 6, 3)) * 255).astype(np.uint8)
    return Image.fromarray(coarse).resize((size, size), Image.BICUBIC)


def make_scars(root, per_class_train=20, per_class_test=4, size=48, seed=0):
    from scipy.io import savemat
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.join(root, 'devkit'), exist_ok=True)
    counts = {}
    for split, per, annos in (('train', per_class_train, 'cars_train_annos.mat'),
                              ('test', per_class_test, 'cars_test_annos_withlabels.mat')):
        img_dir = os.path.join(root, f'cars_{split}')
        os.makedirs(img_dir, exist_ok=True)
        order = [c for c in range(1, N_CLASSES + 1) for _ in range(per)]
        order = [order[i] for i in rng.permutation(len(order))]          # classes interleaved, as in the real files
        records = np.zeros((1, len(order)), dtype=[(f, 'O') for f in FIELDS])
        for i, cls in enumerate(order):
            fname = f'{i + 1:05d}.jpg'
            _image(rng, size).save(os.path.join(img_dir, fname), quality=90)
            records[0, i] = (np.array([[0]], np.uint16), np.array([[0]], np.uint16), np.array([[size - 1]], np.uint16),
                             np.array([[size - 1]], np.uint16), np.array([[cls]], np.uint16), np.array([fname]))
        savemat(os.path.join(root, 'devkit', annos), {'annotations': records})
        counts[split] = len(order)
    return counts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--per_class_train', type=int, default=20)
    ap.add_argument('--per_class_test', type=int, default=4)
    ap.add_argument('--size', type=int, default=48)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args(argv)
    counts = make_scars(a.out, a.per_class_train, a.per_class_test, a.size, a.seed)
    print(f"synthetic Stanford Cars written to {a.out}: {counts['train']} training and {counts['test']} test images")


if __name__ == '__main__':
    main()

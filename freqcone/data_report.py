"""What the training runs will see, in the two data modes of the plan (docs/PLAN.md, section 3):
  val      GCD's validation split held out: the pilot runs R0-R3 (seed 7), for the selection and G1;
  no_val   no split, 100 % of the labelled images as upstream: B8-B10 and F8-F10, G0 and the published numbers.

For each mode: image counts per split, known / novel classes, labelled images per known class, the labelled images
against upstream's, and the SHA-256 of every image list, computed on the image files ('cars_train/00001.jpg'), so no
numbering plays a part. Run it on every machine and compare the hashes.

Then upstream's pseudo-label lookup on EVERY image of the merged training set, before and after the renumbering
(data_checks.compact_uq_idxs, the function the runs use): an image must read its own true label and its own file, or
the report stops. The renumbering goes to a CSV under runs/ (--map); each run records the SHA-256 of the same map in
run.json (data_counts.uq_map_sha256).

Last, every place of the Hyp-SelEx training path that reads uq_idx, with the numbering it sees (scan_uq_uses refuses a
reader it does not know, and tests_freqcone/test_data_report.py checks the list on the real loaders).

Images are not decoded (the datasets' loader returns the path), so the report needs no GPU and takes seconds.

usage: python -m freqcone.data_report --dataset scars [--root DIR] [--json PATH] [--map PATH] [--md docs/DATA_CHECK.md]
"""
import argparse
import copy
import csv
import inspect
import json
import os
import re
import sys

import numpy as np

from freqcone.data_checks import (check_merged_train, check_uq_lookup, check_val_split, compact_uq_idxs, item_names,
                                  list_sha256, parts, raw_targets, source_ids, uq_map_sha256)
from freqcone.datasets import build_args, set_root
from freqcone.hypcd import REPO_ROOT, FCRun

MODES = (('val', True), ('no_val', False))
MODE_TITLES = {'val': 'Có tách val (pilot R0–R3)', 'no_val': 'Không tách val (B8–B10, F8–F10)'}
SCARS_CLASSES = (98, 98)
TRAIN_SCRIPT = 'train/train_HypSelEx.py'
UQ_READERS = {   # function of the training script -> what it reads; a reader that is not here stops the report
    'train': ('bảng nhãn giả của SelEx: `uq_index` từ `extract_labeled_protos`, rồi '
              '`preds_ind_list[i][np.argsort(uq_index)[uq_idxs]]` với `uq_idxs` của batch',
              '`merge_train_loader`, `train_loader`'),
    'extract_labeled_protos': ('`ids`, trả về thành `uq_index`', '`merge_train_loader`'),
    'test': ('`ids`, không dùng tiếp sau khi gom', '`merge_train_loader_test`'),
}
NO_UQ_READ = (FCRun.step, FCRun.record_supcon, FCRun.val_metric, FCRun._features, FCRun.geometry)


def scan_uq_uses(repo=REPO_ROOT):
    """Lines of the training script that touch uq_idx, per function."""
    lines, where = {}, None
    with open(os.path.join(repo, TRAIN_SCRIPT), encoding='utf-8') as f:
        for number, line in enumerate(f, 1):
            start = re.match(r'(?:def |class )(\w+)|if __name__', line)
            if start:
                where = start.group(1) or '__main__'
            if re.search(r'uq_idx|uq_index', line):
                if where not in UQ_READERS:
                    raise AssertionError(f'{TRAIN_SCRIPT}:{number} reads uq_idx in {where}: classify it in UQ_READERS')
                lines.setdefault(where, []).append(number)
    return lines


def uq_uses(repo=REPO_ROOT):
    """Every reader of uq_idx on the training path, and what does not read it. The numbering each loader really holds
    is checked by same_numbering; here is where each one comes from."""
    for fn in NO_UQ_READ:
        if 'uq_idx' in inspect.getsource(fn):
            raise AssertionError(f'{fn.__qualname__} reads uq_idx: add it to the table')
    with open(os.path.join(repo, 'freqcone', 'freqcone_torch.py'), encoding='utf-8') as f:
        if 'uq_idx' in f.read():
            raise AssertionError('freqcone_torch.py reads uq_idx: add it to the table')
    lines = scan_uq_uses(repo)
    rows = [{'place': f'`{fn}()`: {what}', 'lines': lines[fn], 'source': source, 'numbering': 'đánh lại'}
            for fn, (what, source) in UQ_READERS.items()]
    rows += [
        {'place': 'Nguồn: `CarsDataset.__getitem__` trả `self.uq_idxs[idx]`, `MergedDataset.__getitem__` chuyển tiếp',
         'lines': [], 'source': 'mọi loader ở trên', 'numbering': 'dãy của chính object: đã đánh lại'},
        {'place': 'Sampler `WeightedRandomSampler`: bốc **vị trí** trong `train_dataset`, không đọc `uq_idx`', 'lines': [],
         'source': '`train_loader`', 'numbering': 'không áp dụng; ảnh bốc được mang `uq_idx` đã đánh lại'},
        {'place': '`test_kmeans()`: bỏ qua `uq_idx` (`_`)', 'lines': [],
         'source': '`unlabelled_train_loader`, `test_loader`',
         'numbering': 'bản sao unlabelled: đánh lại; tập test: dãy riêng của nó'},
        {'place': 'Val: `FCRun.val_metric`, `FCRun._features` không đọc `uq_idx`; `check_val_split` so bằng số gốc',
         'lines': [], 'source': "`datasets['val']`",
         'numbering': 'số gốc: ảnh val đã rời tập train nên không có số trong dãy mới'},
        {'place': 'FreqCone: `freqcone_torch.py`, `FCRun.step`, `FCRun.geometry` không đọc `uq_idx`', 'lines': [],
         'source': '–', 'numbering': 'không áp dụng'},
        {'place': 'Dựng split: `subsample_dataset`, `unlabelled_indices`, `keep_every`', 'lines': [],
         'source': '`get_datasets`', 'numbering': 'số gốc: chạy xong trước khi đánh lại'},
        {'place': 'Kiểm tra của `FCRun`: `check_uq_lookup`; `check_val_split` và `uq_map_sha256`', 'lines': [],
         'source': 'tập train và các bản sao', 'numbering': 'dãy đánh lại; hai hàm sau đọc thêm số gốc (`source_uq_idxs`)'},
    ]
    return rows


def lookup_rates(merged):
    """Upstream's lookup for every image: the table is what SS-k-means fills in loader order, here each image's true
    label and file, read through the same expression as train(). WeightedRandomSampler draws positions of this same
    dataset, so every training batch is made of these items. An id out of bounds is a miss (train() raises on it)."""
    files, labels, uq, _ = zip(*(merged[i] for i in range(len(merged))))
    files, labels, uq = np.asarray(files), np.asarray(labels), np.asarray(uq).astype(np.int64)
    uq_index = uq.astype(float)                                  # upstream builds it with np.append on a float array
    inside = uq < uq.size
    position = np.argsort(uq_index)[uq[inside]]
    return {'label': float((labels[position] == labels[inside]).sum() / uq.size),
            'file': float((files[position] == files[inside]).sum() / uq.size), 'out_of_bounds': int((~inside).sum())}


def numbering(ds):
    return {name: int(i) for part in parts(ds) for name, i in zip(item_names(part), part.uq_idxs)}


def same_numbering(train, train_test, unlabelled_test):
    """One image has one number in the loaders' objects: merge_train_loader / train_loader (train),
    merge_train_loader_test (train_test), unlabelled_train_loader (unlabelled_test)."""
    expected = numbering(train)
    if numbering(train_test) != expected:
        return False
    return all(expected[name] == i for name, i in numbering(unlabelled_test).items())


def mode_report(dataset, split, n_upstream=None):
    from data.get_datasets import get_datasets
    args = build_args(dataset, split_train_val=split)
    train, test, unlabelled_test, datasets = get_datasets(dataset, None, None, args)
    known, novel = {int(k) for k in args.train_classes}, {int(k) for k in args.unlabeled_classes}
    if dataset == 'scars' and (len(known), len(novel)) != SCARS_CLASSES:
        raise AssertionError(f'the SSB split of Stanford Cars has 98 known / 98 novel classes, got {len(known)} / '
                             f'{len(novel)}')
    lab, unl, val = train.labelled_dataset, train.unlabelled_dataset, datasets.get('val')
    if split:
        check_val_split(lab, unl, val, args.train_classes)
    counts = check_merged_train(train, args.train_classes)
    train_test = copy.deepcopy(train)                            # as the training script builds merge_train_loader_test
    for part in parts(train) + parts(train_test) + (unlabelled_test, test):
        part.loader = str                                        # the path instead of the decoded image
    if val is not None:
        val.loader = str
    before = lookup_rates(train)
    gaps = compact_uq_idxs(lab, unl, (train_test, unlabelled_test))
    check_uq_lookup(train)
    check_uq_lookup(train_test)
    after = lookup_rates(train)
    if after != {'label': 1.0, 'file': 1.0, 'out_of_bounds': 0}:
        raise AssertionError(f'after the renumbering, the lookup gives {after}')
    if not same_numbering(train, train_test, unlabelled_test):
        raise AssertionError('the loaders disagree on the number of an image')
    every_file = [lab[i][0] for i in range(len(lab))] + [unl[i][0] for i in range(len(unl))] + \
        [test[i][0] for i in range(len(test))] + ([val[i][0] for i in range(len(val))] if val is not None else [])
    missing = sum(not os.path.isfile(p) for p in every_file)
    if missing:
        raise AssertionError(f'{missing} image files are missing')
    n_val = 0 if val is None else len(val)
    unl_t = raw_targets(unl)
    subsets = [('labelled', lab), ('unlabelled', unl), ('test', test)] + ([('validation', val)] if split else [])
    out = {'split_train_val': split,
           'images': {'train_total': len(lab) + len(unl) + n_val, 'labelled': len(lab), 'unlabelled': len(unl),
                      'unlabelled_known': int(np.isin(unl_t, list(known)).sum()),
                      'unlabelled_novel': int(np.isin(unl_t, list(novel)).sum()), 'validation': n_val,
                      'test': len(test)},
           'classes': {'known': len(known), 'novel': len(novel)},
           'labelled_per_class': {'min': counts['labelled_per_known_min'], 'max': counts['labelled_per_known_max']},
           'upstream_labelled': {'n': n_upstream or len(lab), 'ratio': len(lab) / (n_upstream or len(lab))},
           'lists_sha256': {k: list_sha256(item_names(ds)) for k, ds in subsets},
           'uq': {'gaps_closed': gaps, 'lookup_before': before, 'lookup_after': after, 'same_numbering': True,
                  'map_sha256': uq_map_sha256(train)}}
    if split:
        per_class = np.bincount(raw_targets(val))[sorted(known)]
        out['validation_per_class'] = {'min': int(per_class.min()), 'max': int(per_class.max())}
    return out, train, val


def uq_rows(train, val):
    """One row per training image, by original number: uq_idx_old, uq_idx_new (empty for a held-out image), subset,
    class, file."""
    rows = []
    for subset, ds in (('labelled', train.labelled_dataset), ('unlabelled', train.unlabelled_dataset),
                       ('validation', val)):
        if ds is None:
            continue
        new = [''] * len(ds) if subset == 'validation' else [int(i) for i in ds.uq_idxs]
        rows += zip((int(i) for i in source_ids(ds)), new, [subset] * len(ds), raw_targets(ds).tolist(), item_names(ds))
    return sorted(rows)


def write_map(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['uq_idx_old', 'uq_idx_new', 'subset', 'class', 'file'])
        w.writerows(rows)


def report(dataset, root=None):
    if root:
        set_root(dataset, root)
    no_val, _, _ = mode_report(dataset, False)
    n_upstream = no_val['images']['labelled']
    val, train, val_ds = mode_report(dataset, True, n_upstream)
    if no_val['images']['train_total'] != val['images']['train_total'] or \
            no_val['lists_sha256']['unlabelled'] != val['lists_sha256']['unlabelled'] or \
            no_val['lists_sha256']['test'] != val['lists_sha256']['test'] or \
            n_upstream != val['images']['labelled'] + val['images']['validation']:
        raise AssertionError('the validation split must only move labelled images out of training')
    out = {'dataset': dataset, 'classes': val['classes'], 'modes': {'val': val, 'no_val': no_val}, 'uq_uses': uq_uses()}
    return out, uq_rows(train, val_ds)


def markdown(out, root, map_path=None):
    modes = [out['modes'][k] for k, _ in MODES]
    pct = lambda x: f'{100 * x:.2f} %'

    def row(label, value):
        return f'| {label} | ' + ' | '.join(str(value(m)) for m in modes) + ' |'

    def span(m, key):
        return '–' if key not in m else f"{m[key]['min']}–{m[key]['max']}"
    lines = [f"# Kiểm tra dữ liệu: {out['dataset']}", '',
             f'Sinh bởi `python -m freqcone.data_report`, dữ liệu tại `{root or "đường dẫn trong config.py"}`. '
             'Val chỉ dùng để chọn cấu hình: pilot có tách val, các run xác nhận dùng 100 % ảnh labelled như upstream.',
             '', '| Mục | ' + ' | '.join(MODE_TITLES[k] for k, _ in MODES) + ' |', '| --- | --- | --- |',
             row('Train, tổng', lambda m: m['images']['train_total']),
             row('Labelled', lambda m: m['images']['labelled']),
             row('Unlabelled', lambda m: f"{m['images']['unlabelled']} (known {m['images']['unlabelled_known']}, "
                                         f"novel {m['images']['unlabelled_novel']})"),
             row('Val', lambda m: m['images']['validation']),
             row('Test rời', lambda m: m['images']['test']),
             row('Lớp known / novel', lambda m: f"{m['classes']['known']} / {m['classes']['novel']}"),
             row('Labelled mỗi lớp known, min–max', lambda m: span(m, 'labelled_per_class')),
             row('Val mỗi lớp known, min–max', lambda m: span(m, 'validation_per_class')),
             row('Labelled của upstream', lambda m: m['upstream_labelled']['n']),
             row('Tỉ lệ labelled so với upstream', lambda m: f"{m['upstream_labelled']['ratio']:.4f}"),
             '', '## `uq_idx` sau khi đánh lại', '',
             'Tra trên **mọi** ảnh của tập train, đúng biểu thức `np.argsort(uq_index)[uq_idxs]` của upstream.', '',
             '| Mục | ' + ' | '.join(MODE_TITLES[k] for k, _ in MODES) + ' |', '| --- | --- | --- |',
             row('Khe hở được đóng', lambda m: m['uq']['gaps_closed']),
             row('Trước khi đánh lại: đúng nhãn / đúng file / ngoài biên',
                 lambda m: f"{pct(m['uq']['lookup_before']['label'])} / {pct(m['uq']['lookup_before']['file'])} / "
                           f"{m['uq']['lookup_before']['out_of_bounds']}"),
             row('Sau khi đánh lại: đúng nhãn / đúng file / ngoài biên',
                 lambda m: f"{pct(m['uq']['lookup_after']['label'])} / {pct(m['uq']['lookup_after']['file'])} / "
                           f"{m['uq']['lookup_after']['out_of_bounds']}"),
             row('Mọi loader cùng một số cho mỗi ảnh', lambda m: 'có' if m['uq']['same_numbering'] else 'KHÔNG'),
             row('SHA-256 của ánh xạ (`uq_map_sha256` trong run.json)', lambda m: f"`{m['uq']['map_sha256']}`"), '']
    if map_path:
        lines += [f'Ánh xạ `uq_idx` cũ → mới (kèm tập con, lớp, file) ở `{map_path}`.', '']
    lines += ['## Mọi chỗ dùng `uq_idx`', '', '| Nơi | Dòng trong `train/train_HypSelEx.py` | Nguồn | Dãy số |',
              '| --- | --- | --- | --- |']
    lines += [f"| {u['place']} | {', '.join(map(str, u['lines'])) or '–'} | {u['source']} | {u['numbering']} |"
              for u in out['uq_uses']]
    lines += ['', '## SHA-256 của danh sách ảnh', '',
              'Tính trên đường dẫn file dưới thư mục dữ liệu (`cars_train/00001.jpg`), không dùng `uq_idx`.', '',
              '| Danh sách | Chế độ | SHA-256 |', '| --- | --- | --- |']
    for key, _ in MODES:
        lines += [f"| {name} | {key} | `{digest}` |" for name, digest in out['modes'][key]['lists_sha256'].items()]
    return '\n'.join(lines) + '\n'


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='scars', choices=['scars', 'cub', 'aircraft'])
    ap.add_argument('--root', default=None, help='dataset directory (default: the path in config.py)')
    ap.add_argument('--json', default=None, help='also write the report here')
    ap.add_argument('--map', default=None, help='write the uq_idx renumbering here (runs/uq_map_<dataset>.csv)')
    ap.add_argument('--md', default=None, help='also write a Markdown report here (docs/DATA_CHECK.md on the real data)')
    a = ap.parse_args(argv)
    from freqcone.runs import utf8_stdout
    utf8_stdout()
    out, rows = report(a.dataset, a.root)
    print(f"{out['dataset']}: {out['classes']['known']} known / {out['classes']['novel']} novel classes")
    for key, _ in MODES:
        m = out['modes'][key]
        im, uq = m['images'], m['uq']
        print(f"[{key}] train {im['train_total']} = labelled {im['labelled']} + validation {im['validation']} + "
              f"unlabelled {im['unlabelled']} (known {im['unlabelled_known']}, novel {im['unlabelled_novel']}); "
              f"test {im['test']}")
        print(f"[{key}] labelled per known class {m['labelled_per_class']['min']}-{m['labelled_per_class']['max']}; "
              f"labelled vs upstream {im['labelled']} / {m['upstream_labelled']['n']} = "
              f"{m['upstream_labelled']['ratio']:.4f}")
        print(f"[{key}] uq_idx: {uq['gaps_closed']} gaps closed; lookup before {uq['lookup_before']}, after "
              f"{uq['lookup_after']}; map sha256 {uq['map_sha256']}")
        for name, digest in m['lists_sha256'].items():
            print(f'[{key}] sha256 {name:11s} {digest}')
    print(f"uq_idx: {len(out['uq_uses'])} places listed in the Markdown report; every training loader holds one "
          f"number per image, the validation images keep their original ids")
    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        with open(a.json, 'w') as f:
            json.dump(out, f, indent=1)
    if a.map:
        write_map(a.map, rows)
    if a.md:
        with open(a.md, 'w', encoding='utf-8') as f:
            f.write(markdown(out, a.root, a.map))
    return 0


if __name__ == '__main__':
    sys.exit(main())

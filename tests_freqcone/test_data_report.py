"""The data report (docs/DATA_CHECK.md) on synthetic Stanford Cars: upstream's CarsDataset, the real SSB class split, both
data modes of the plan, the lookup on every image, the uq_idx map, and the list of every reader of uq_idx."""
import csv
import hashlib

import pytest
import torch

from freqcone import data_report as dr
from freqcone.data_checks import list_sha256
from freqcone.datasets import build_args
from freqcone.hypcd import FCRun
from helpers import fc_args


@pytest.fixture(scope='module')
def cars(tmp_path_factory):
    from freqcone.synthetic_data import make_scars
    root = tmp_path_factory.mktemp('cars')
    make_scars(str(root), per_class_train=20, per_class_test=2, size=16)
    return str(root)


@pytest.fixture
def report(cars, monkeypatch):
    import data.stanford_cars as sc
    monkeypatch.setattr(sc, 'car_root', cars)
    return dr.report('scars')


def test_both_modes(report):
    out, _ = report
    val, no_val = out['modes']['val'], out['modes']['no_val']
    assert out['classes'] == {'known': 98, 'novel': 98}
    assert val['images']['train_total'] == no_val['images']['train_total'] == 196 * 20
    assert val['images']['test'] == no_val['images']['test'] == 196 * 2
    assert no_val['images']['labelled'] == 98 * 10 and no_val['images']['validation'] == 0
    assert no_val['upstream_labelled'] == {'n': 98 * 10, 'ratio': 1.0}
    assert val['images']['labelled'] + val['images']['validation'] == 98 * 10
    assert val['upstream_labelled']['n'] == 98 * 10 and 0.8 <= val['upstream_labelled']['ratio'] < 0.9
    assert val['validation_per_class']['min'] >= 1 and 'validation_per_class' not in no_val
    assert no_val['labelled_per_class']['min'] >= val['labelled_per_class']['min'] >= 2
    assert val['lists_sha256']['unlabelled'] == no_val['lists_sha256']['unlabelled']
    assert val['lists_sha256']['test'] == no_val['lists_sha256']['test']
    assert val['lists_sha256']['labelled'] != no_val['lists_sha256']['labelled']
    assert 'validation' in val['lists_sha256'] and 'validation' not in no_val['lists_sha256']


def test_lookup_on_every_image(report):
    out, _ = report
    uq = out['modes']['val']['uq']
    assert uq['gaps_closed'] == out['modes']['val']['images']['validation'] > 0
    assert uq['lookup_after'] == {'label': 1.0, 'file': 1.0, 'out_of_bounds': 0} and uq['same_numbering']
    # the check can fail: on the ids the dataset code produces, almost every image reads another one
    assert uq['lookup_before']['file'] < 0.5 and uq['lookup_before']['out_of_bounds'] > 0
    uq = out['modes']['no_val']['uq']
    assert uq['gaps_closed'] == 0 and uq['lookup_before'] == uq['lookup_after'] == {'label': 1.0, 'file': 1.0,
                                                                                    'out_of_bounds': 0}


def test_a_wrong_renumbering_stops_the_report(cars, monkeypatch):
    import data.stanford_cars as sc
    monkeypatch.setattr(sc, 'car_root', cars)
    monkeypatch.setattr(dr, 'compact_uq_idxs', lambda *a, **k: 0)         # gaps left in place
    with pytest.raises(AssertionError, match='not 0'):
        dr.report('scars')


def test_missing_files_stop_the_report(tmp_path, monkeypatch):
    from freqcone.synthetic_data import make_scars
    make_scars(str(tmp_path), per_class_train=20, per_class_test=2, size=8)
    (tmp_path / 'cars_test' / '00001.jpg').unlink()
    import data.stanford_cars as sc
    monkeypatch.setattr(sc, 'car_root', str(tmp_path))
    with pytest.raises(AssertionError, match='1 image files are missing'):
        dr.report('scars')


def test_map_file_traces_every_image(report, tmp_path):
    out, rows = report
    path = tmp_path / 'runs' / 'uq_map_scars.csv'
    dr.write_map(str(path), rows)
    with open(path, newline='', encoding='utf-8') as f:
        table = list(csv.DictReader(f))
    val = out['modes']['val']
    assert len(table) == val['images']['train_total']
    assert [int(r['uq_idx_old']) for r in table] == list(range(len(table)))
    kept = [r for r in table if r['subset'] != 'validation']
    held = [r for r in table if r['subset'] == 'validation']
    assert len(held) == val['images']['validation'] and all(r['uq_idx_new'] == '' for r in held)
    assert sorted(int(r['uq_idx_new']) for r in kept) == list(range(len(kept)))
    assert all(int(r['uq_idx_new']) <= int(r['uq_idx_old']) for r in kept)
    known = {int(k) for k in build_args('scars').train_classes}
    assert all(int(r['class']) in known for r in table if r['subset'] != 'unlabelled')
    pairs = sorted(f"{r['uq_idx_old']},{r['uq_idx_new']}" for r in kept)
    assert hashlib.sha256('\n'.join(pairs).encode()).hexdigest() == val['uq']['map_sha256']
    # the validation list is identified by its files, without any number
    assert list_sha256([r['file'] for r in held]) == val['lists_sha256']['validation']
    assert held[0]['file'].startswith('cars_train/')


def test_runs_record_the_map_of_the_report(report):
    from data.get_datasets import get_datasets
    out, _ = report
    ns = build_args('scars')
    train, _, unlabelled_test, datasets = get_datasets('scars', None, None, ns)
    args = fc_args(train_classes=ns.train_classes, split_train_val=True, run_dir='')
    run = FCRun(args, torch.device('cpu'), train_dataset=train, val_dataset=datasets['val'], datasets=datasets,
                train_copies=(unlabelled_test,))
    assert run.data_counts['uq_map_sha256'] == out['modes']['val']['uq']['map_sha256']
    assert run.data_counts['uq_gaps_closed'] == out['modes']['val']['uq']['gaps_closed']


def test_markdown_report(report):
    out, _ = report
    text = dr.markdown(out, 'somewhere', 'runs/uq_map_scars.csv')
    for part in ('Có tách val (pilot R0–R3)', 'Không tách val (B8–B10, F8–F10)', '98 / 98', 'somewhere',
                 'Labelled mỗi lớp known, min–max', 'Tỉ lệ labelled so với upstream', '100.00 % / 100.00 % / 0',
                 'runs/uq_map_scars.csv', 'WeightedRandomSampler', 'merge_train_loader_test', 'không dùng `uq_idx`'):
        assert part in text, part


def write_script(tmp_path, body):
    (tmp_path / 'train').mkdir(exist_ok=True)
    (tmp_path / 'train' / 'train_HypSelEx.py').write_text(body)
    return str(tmp_path)


def test_every_reader_of_uq_idx_is_listed():
    lines = dr.scan_uq_uses()
    assert set(lines) == set(dr.UQ_READERS)
    assert len(lines['train']) == 5 and len(lines['extract_labeled_protos']) == 2 and len(lines['test']) == 2
    places = ' '.join(row['place'] for row in dr.uq_uses())
    for part in ('WeightedRandomSampler', 'test_kmeans', 'val_metric', 'FCRun.step', 'freqcone_torch.py'):
        assert part in places, part


@pytest.mark.parametrize('body, place', [
    ('def other(loader):\n    return uq_idx\n', 'other'),
    ('class Loss:\n    def forward(self, uq_idxs):\n        pass\n', 'Loss'),
    ('def test(l):\n    pass\n\nif __name__ == "__main__":\n    x = uq_index\n', '__main__'),
])
def test_an_unknown_reader_stops_the_report(tmp_path, body, place):
    with pytest.raises(AssertionError, match=place):
        dr.scan_uq_uses(write_script(tmp_path, body))


def test_a_new_reader_in_freqcone_stops_the_report(monkeypatch):
    def step(uq_idxs):
        return uq_idxs

    monkeypatch.setattr(dr, 'NO_UQ_READ', (step,))
    with pytest.raises(AssertionError, match='reads uq_idx'):
        dr.uq_uses()

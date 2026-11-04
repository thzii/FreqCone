"""End to end on synthetic Stanford Cars: upstream's CarsDataset loader, the real SSB class split, the data report
and the cut-off script's command line (the README's dry-run commands)."""
import importlib.util
import json
import os

from conftest import ROOT


def _script():
    spec = importlib.util.spec_from_file_location('compute_cutoffs', os.path.join(ROOT, 'scripts', 'freqcone',
                                                                                  'compute_cutoffs.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_synthetic_scars_report_and_cutoffs(tmp_path, monkeypatch):
    from freqcone.synthetic_data import make_scars
    root = tmp_path / 'cars'
    counts = make_scars(str(root), per_class_train=20, per_class_test=2, size=32)
    assert counts == {'train': 196 * 20, 'test': 196 * 2}
    import data.stanford_cars as sc
    monkeypatch.setattr(sc, 'car_root', str(root))
    from freqcone.data_report import report
    out = report('scars')
    assert out['classes'] == {'known': 98, 'novel': 98}
    im = out['images']
    assert im['train_total'] == 196 * 20 and im['test'] == 196 * 2
    assert im['labelled'] + im['validation'] + im['unlabelled'] == im['train_total']
    assert im['unlabelled_novel'] == 98 * 20 and out['validation_per_class']['min'] >= 1
    assert len(out['lists_sha256']['validation']) == 64
    mod = _script()
    mod.main(['--dataset', 'scars', '--root', str(root), '--out', str(tmp_path / 'cutoffs.json'), '--readme', '',
              '--num_workers', '0', '--device', 'cpu'])
    entry = json.loads((tmp_path / 'cutoffs.json').read_text())['scars']
    assert entry['complete'] and entry['n_images'] == im['labelled'] + im['unlabelled']
    assert entry['f_a'] < entry['f_b'] and entry['validation_excluded']['n_val'] == im['validation']
    assert entry['paper_table7'] == {'5': 3, '10': 11, '15': 27, '20': 47, '25': 55}

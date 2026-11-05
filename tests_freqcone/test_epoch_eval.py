"""Plan 2.1: per-epoch evaluation in isolated random state, split records, quiet output, resume."""
import json
import os
import types

import pytest
import torch

from freqcone.hypcd import FCRun, isolated_rng
from freqcone.pilot_tools import _has_test_numbers, check_run_dir
from helpers import fc_args, train_loop


def state(model, head):
    return {**{f'm.{k}': v for k, v in model.state_dict().items()}, **{f'h.{k}': v for k, v in head.state_dict().items()}}


@pytest.mark.parametrize('freqcone', [False, True])
def test_per_epoch_evaluation_does_not_change_training(tmp_path, freqcone):
    out = {}
    for on in (False, True):
        args = fc_args(freqcone=freqcone, run_dir=str(tmp_path / f'run_{on}'), eval_every_epoch=on, epochs=2)
        model, head, _ = train_loop(FCRun(args, torch.device('cpu')), epochs=2)
        out[on] = state(model, head)
    assert out[False].keys() == out[True].keys()
    assert all(torch.equal(out[False][k], out[True][k]) for k in out[False])


def test_isolated_rng_restores_every_generator():
    import random

    import numpy as np
    before = (random.random(), np.random.rand(), torch.rand(1))
    random.seed(1), np.random.seed(1), torch.manual_seed(1)
    ref = (random.random(), np.random.rand(), float(torch.rand(1)))
    random.seed(1), np.random.seed(1), torch.manual_seed(1)
    with isolated_rng(1234):
        random.random(), np.random.rand(100), torch.rand(100)
    assert (random.random(), np.random.rand(), float(torch.rand(1))) == ref
    assert before is not None


def test_records_keep_test_numbers_out_of_run_json(tmp_path):
    args = fc_args(freqcone=True, run_dir=str(tmp_path / 'R1'), eval_every_epoch=True, epochs=2, run_id='R1')
    fc = FCRun(args, torch.device('cpu'))
    model, head, merged = train_loop(fc, epochs=2)
    fc.final_eval(model, head, merged, lambda m, l, a: {'all_acc': 0.5, 'old_acc': 0.6, 'new_acc': 0.4})
    rec = json.loads((tmp_path / 'R1' / 'run.json').read_text())
    assert not _has_test_numbers(rec)
    tests = [json.loads(l) for l in (tmp_path / 'R1' / 'epochs_test.jsonl').read_text().splitlines()]
    vals = [json.loads(l) for l in (tmp_path / 'R1' / 'epochs_val.jsonl').read_text().splitlines()]
    assert [t['epoch'] for t in tests] == [1, 2] == [v['epoch'] for v in vals]
    assert set(tests[0]) == {'epoch', 'gcd', 'disjoint_test', 'eval'}
    assert tests[0]['gcd']['source'] == 'ss_kmeans_epoch_start' and tests[0]['gcd']['weights_after_epoch'] == 0
    assert all(set(t['eval']) == {'all', 'old', 'new'} for t in tests)
    assert not any(k in json.dumps(vals) for k in ('"all"', '"old"', '"new"'))
    assert vals[-1]['fc']['depth_b'] == pytest.approx(1.282, abs=1e-3) and vals[-1]['fc']['in_cone'] is not None
    assert check_run_dir(str(tmp_path / 'R1')) == []


def test_resume_continues_exactly(tmp_path):
    ref_args = fc_args(freqcone=True, run_dir=str(tmp_path / 'straight'), eval_every_epoch=True, epochs=3)
    ref_model, ref_head, _ = train_loop(FCRun(ref_args, torch.device('cpu')), epochs=3)
    run_dir = str(tmp_path / 'interrupted')
    args = fc_args(freqcone=True, run_dir=run_dir, eval_every_epoch=True, epochs=3)
    train_loop(FCRun(args, torch.device('cpu')), epochs=3, stop_after=2)          # "crash" after epoch 2
    with pytest.raises(FileExistsError):
        FCRun(fc_args(freqcone=True, run_dir=run_dir, eval_every_epoch=True, epochs=3), torch.device('cpu'))
    fc = FCRun(fc_args(freqcone=True, run_dir=run_dir, eval_every_epoch=True, epochs=3, resume=True), torch.device('cpu'))
    model, head, _ = train_loop(fc, epochs=3, seed=99)                                # other seed: state comes from ckpt
    assert fc.resumed_from == 2
    a, b = state(ref_model, ref_head), state(model, head)
    assert all(torch.equal(a[k], b[k]) for k in a)
    lines = (tmp_path / 'interrupted' / 'epochs_test.jsonl').read_text().splitlines()
    assert [json.loads(l)['epoch'] for l in lines] == [1, 2, 3]


def test_hide_test_stdout_filters_accuracy_lines(capsys):
    import project_utils.cluster_and_log_utils as clu
    module = types.ModuleType('fake_train')
    exec("def report():\n    print('Epoch: 3 Avg Loss: 1.000 | Constrastive: 0.900 ')\n"
         "    print('Train Accuracies: All 0.5000 | Old 0.6000 | New 0.4000')\n"
         "    print('Best ACC on new Classes on disjoint test set: 0.4...')\n"
         "    print('model saved to x_best.pt.')\n", module.__dict__)

    class Writer:
        def __init__(self):
            self.calls = []

        def add_scalars(self, *a, **k):
            self.calls.append('scalars')

        def add_scalar(self, *a, **k):
            self.calls.append('scalar')
    writer = Writer()
    args = fc_args(hide_test_stdout=True, writer=writer)
    try:
        FCRun(args, torch.device('cpu')).install_quiet_output(module)
        module.report()
        import numpy as np
        y = np.array([0, 1, 1, 2])
        clu.log_accs_from_preds(y, y, np.array([True, True, False, False]), ['v2'], 'SS-K-Means Train ACC', T=0,
                                writer=args.writer, print_output=True)
        args.writer.add_scalar('Loss', 1.0, 0)
    finally:
        clu.__dict__.pop('print', None)
    out = capsys.readouterr().out
    assert 'Avg Loss' in out and 'Accuracies' not in out and 'ACC' not in out and 'best' not in out.lower()
    assert writer.calls == ['scalar']


def test_profile_measures_evaluation_but_records_nothing(tmp_path):
    args = fc_args(freqcone=True, run_dir=str(tmp_path / 'probe'), eval_every_epoch=True, profile_steps=2, epochs=1)
    fc = FCRun(args, torch.device('cpu'))
    fc.start_epoch(0)
    assert [fc.profile_tick(i) for i in range(2)] == [False, True]
    fc.end_epoch(0, 1.0, 1.0, 0.0)
    from helpers import make_model, merged_loader, noisy_test
    model, head = make_model(args)
    data = torch.randn(6, 3, 32, 32)
    merged = merged_loader(data, torch.arange(6) % 3, 3)
    assert fc.epoch_done(0, 40, model, head, merged, noisy_test, (0.5, 0.5, 0.5)) is True
    rec = json.loads((tmp_path / 'probe' / 'run.json').read_text())
    assert rec['profile']['eval_s'] is not None and rec['profile']['steps_per_epoch'] == 40
    assert not os.path.exists(tmp_path / 'probe' / 'epochs_test.jsonl')
    assert check_run_dir(str(tmp_path / 'probe'), profile=True) == []

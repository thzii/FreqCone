"""Glue between FreqCone and HypCD's Hyp-SelEx training script (train/train_HypSelEx.py).

Every hook is called from a block marked ``# >>> freqcone`` ... ``# <<< freqcone`` in an upstream file.
With the default flags none of them changes the computation:
  * --freqcone False, --dist_impl upstream, --fix_label_order False, --split_train_val False,
  * --smoke_frac 0, --run_dir '', --eval_every_epoch False, --hide_test_stdout False, --profile_steps 0.
tests_freqcone/test_upstream_equivalence.py checks that deleting the marked blocks gives back the upstream
files byte for byte, and that these defaults leave every hook inactive.

Outputs of a run (--run_dir runs/<run_id>):
  run.json            setting, provenance, checksums, validation result of the last epoch, geometry, cost.
                      It holds no test number.
  epochs_val.jsonl    one line per epoch: validation accuracy, losses, FreqCone diagnostics, lr, time
  epochs_test.jsonl   one line per epoch: All / Old / New (v2, transductive) and upstream's disjoint-test
                      accuracies. Test numbers live only here; nothing prints them with --hide_test_stdout.
  ckpt_resume.pt      state after the last finished epoch (--resume continues from it); not committed

The per-epoch evaluation runs with every random generator saved and restored around it (isolated_rng), so a
run with it trains exactly like a run without it (tests_freqcone/test_epoch_eval.py).
"""
import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import re
import shlex
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from freqcone.data_checks import check_val_split, item_ids, list_sha256
from freqcone.freqcone_torch import (APERTURES, FreqConeLoss, aperture, cone_energy, depth, radial_lowpass,
                                     scaled_depth, to_fixed_depth)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHEMA = 'freqcone-run/2'
VAL_METRIC = 'ss_kmeans_known_hungarian'
VAL_DEFINITION = ("Hyp-SelEx's own prediction procedure applied to the held-out validation images: upstream "
                  "semi-supervised k-means (methods/clustering/faster_mix_k_means_pytorch.K_Means, settings of "
                  "test()) with the labelled training images as fixed anchors and the validation images as "
                  "unlabelled points, k = number of known classes, L2-normalised backbone features (test "
                  "transform); scored with upstream project_utils.cluster_utils.cluster_acc, the Hungarian "
                  "accuracy that split_cluster_acc_v2 reports as 'All'.")
TEST_LINE = re.compile(r'acc|best', re.IGNORECASE)        # upstream lines that carry test-derived numbers


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def add_args(parser):
    g = parser.add_argument_group('freqcone', 'FreqCone and protocol extensions; every default reproduces upstream')
    g.add_argument('--dist_impl', type=str, default='upstream', choices=['upstream', 'lowmem'],
                   help='hyperbolic distance matrix: upstream N x N x D Mobius tensor, or the equivalent O(N^2) form')
    g.add_argument('--fix_label_order', type=str2bool, default=False,
                   help="store SelEx's level >= 1 pseudo-labels in loader order (SelEx issue #11)")
    g.add_argument('--freqcone', type=str2bool, default=False, help='add the FreqCone loss')
    g.add_argument('--freqcone_variant', type=str, default='v2', choices=['v2', 'v3'],
                   help='v2: the cone never changes the sharp-view norm; v3: it may')
    g.add_argument('--w_ent', '--fc_w_ent', dest='fc_w_ent', type=float, default=0.03,
                   help='lambda_ent, weight of the entailment term')
    g.add_argument('--fc_w_dir', type=float, default=0.5, help='lambda_dir')
    g.add_argument('--fc_w_con', type=float, default=0.05, help='lambda_con')
    g.add_argument('--fc_K', type=float, default=0.1, help='cone aperture constant K')
    g.add_argument('--aperture', '--fc_aperture', dest='fc_aperture', type=str, default='A1', choices=sorted(APERTURES),
                   help='coarse-view radii (r_a, r_b): A1 (0.12, 0.2), A2 (0.104, 0.136), A3 (0.1, 0.114)')
    g.add_argument('--fc_r_a', type=float, default=None, help='custom r_a (needs --fc_r_b; overrides the preset)')
    g.add_argument('--fc_r_b', type=float, default=None, help='custom r_b (needs --fc_r_a)')
    g.add_argument('--fc_tau_a', type=float, default=0.5)
    g.add_argument('--fc_tau_b', type=float, default=0.3)
    g.add_argument('--fc_f_a', type=float, default=3.0, help='cut-off of x^a in cycles / image at the input size')
    g.add_argument('--fc_f_b', type=float, default=11.0, help='cut-off of x^b in cycles / image (> fc_f_a)')
    g.add_argument('--coarse_res', '--fc_coarse_size', dest='fc_coarse_size', type=int, default=0,
                   help='coarse-view resolution in px; 0 = input size (224); 112 needs fc_f_b < 56')
    g.add_argument('--fc_log_every', type=int, default=50, help='record the step diagnostics every N steps')
    g.add_argument('--fc_diag_images', type=int, default=2048, help='images for the final geometry (0 = all)')
    g.add_argument('--split_train_val', type=str2bool, default=False,
                   help="hold out GCD's known-class validation split (20%% of the labelled images of each class)")
    g.add_argument('--smoke_frac', type=int, default=0, help='smoke tests only: keep every k-th image of every split')
    g.add_argument('--run_dir', type=str, default='', help='write run.json and the per-epoch JSONL files here')
    g.add_argument('--eval_every_epoch', type=str2bool, default=False,
                   help='validation and test (v2) after every epoch, with isolated random state')
    g.add_argument('--hide_test_stdout', type=str2bool, default=False,
                   help='keep test accuracies out of stdout and TensorBoard (they still go to epochs_test.jsonl)')
    g.add_argument('--resume', type=str2bool, default=False, help='continue from RUN_DIR/ckpt_resume.pt')
    g.add_argument('--profile_steps', type=int, default=0, help='>0: train N steps of one epoch, record time and memory')
    g.add_argument('--run_id', type=str, default='', help='run id, e.g. R0 .. R3, base_s8, fc_s8, R0-repro')
    g.add_argument('--run_tag', type=str, default='', help='free-form run name')
    g.add_argument('--run_attempt', type=int, default=1, help='1 + number of earlier interrupted attempts')
    return parser


def coarse_radii(args):
    """(aperture name, r_a, r_b): the preset, or 'custom' when both radii are given."""
    if args.fc_r_a is None and args.fc_r_b is None:
        r_a, r_b = APERTURES[args.fc_aperture]
        return args.fc_aperture, r_a, r_b
    if args.fc_r_a is None or args.fc_r_b is None:
        raise ValueError('give both --fc_r_a and --fc_r_b, or neither')
    return 'custom', args.fc_r_a, args.fc_r_b


def loader_order(labels, mask_known):
    """SelEx issue #11 fix. fit_mix returns labels for cat(l_feats, u_feats). At hierarchy levels >= 1 the split
    is the predicted-known mask, so the labels arrive as [predicted known; predicted unknown] while every reader
    indexes them in loader order. Scatter them back to loader order."""
    mask = np.asarray(mask_known, dtype=bool)
    if labels.shape[0] != mask.size:
        raise ValueError(f'{labels.shape[0]} labels for a mask of {mask.size}')
    order = np.concatenate([np.nonzero(mask)[0], np.nonzero(~mask)[0]])
    out = torch.empty_like(labels)
    out[torch.as_tensor(order, device=labels.device, dtype=torch.long)] = labels
    return out


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def set_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


@contextlib.contextmanager
def isolated_rng(seed):
    """Run a block with every generator (python, numpy, torch CPU and CUDA) seeded with `seed`, then put back the
    exact states from before: the block cannot change what training draws next."""
    saved = rng_state()
    try:
        random.seed(seed)
        np.random.seed(seed % 2 ** 32)
        torch.manual_seed(seed)
        yield
    finally:
        set_rng_state(saved)


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def _jsonable(v):
    if isinstance(v, (bool, int, float, str)) or v is None:
        return v
    if isinstance(v, range):
        return list(v)
    if isinstance(v, (list, tuple)) and all(isinstance(x, (bool, int, float, str)) for x in v):
        return list(v)
    if isinstance(v, np.ndarray) and v.ndim == 1:
        return v.tolist()
    return None


def _git(*cmd):
    try:
        res = subprocess.run(['git', *cmd], cwd=REPO_ROOT, capture_output=True, text=True, timeout=30)
        return res.stdout.strip() if res.returncode == 0 else None
    except Exception:  # git missing or not a repository
        return None


def _command():
    argv = list(sys.argv)
    if argv and argv[0].replace('\\', '/').endswith('train/train_HypSelEx.py'):
        argv = ['-m', 'train.train_HypSelEx'] + argv[1:]
    return ' '.join(shlex.quote(a) for a in [sys.executable] + argv)


class _QuietWriter:
    """TensorBoard writer without add_scalars (upstream logs every accuracy with add_scalars, losses with add_scalar)."""

    def __init__(self, writer):
        self._writer = writer

    def add_scalars(self, *args, **kwargs):
        return None

    def __getattr__(self, name):
        return getattr(self._writer, name)


class _Inert:
    """Stand-in used until the training script builds an FCRun."""
    enabled = False


INERT = _Inert()


class FCRun:
    def __init__(self, args, device, train_dataset=None, val_dataset=None, datasets=None):
        self.args, self.device = args, device
        self.profiling = args.profile_steps > 0
        self.enabled = bool(args.freqcone or args.run_dir or self.profiling)
        self.aperture_name, self.r_a, self.r_b = coarse_radii(args)
        self.coarse_size = args.fc_coarse_size or None
        if not 0 < args.fc_f_a < args.fc_f_b:
            raise ValueError(f'need 0 < fc_f_a < fc_f_b, got {args.fc_f_a}, {args.fc_f_b}')
        if self.coarse_size is not None and not args.fc_f_b < self.coarse_size / 2:
            raise ValueError(f'coarse views at {self.coarse_size} px need fc_f_b < {self.coarse_size / 2} (no aliasing)')
        self.loss = None
        if args.freqcone:
            self.loss = FreqConeLoss(c=args.c, K=args.fc_K, r_a=self.r_a, r_b=self.r_b, tau_a=args.fc_tau_a,
                                     tau_b=args.fc_tau_b, w_dir=args.fc_w_dir, w_ent=args.fc_w_ent, w_con=args.fc_w_con,
                                     f_a=args.fc_f_a, f_b=args.fc_f_b,
                                     detach_child_norm=args.freqcone_variant == 'v2',
                                     clip_r=args.cr if args.cr else None)
        self.psi_a, self.psi_b = aperture(args.fc_K, self.r_a), aperture(args.fc_K, self.r_b)
        self.depth_a, self.depth_b = scaled_depth(self.r_a, args.c), scaled_depth(self.r_b, args.c)
        self.clip_depth = 2 * args.cr if args.cr else None   # every clipped point has depth d_c(0, exp0(C_r v)) = 2r
        self.val_dataset = val_dataset
        self.split = None
        if args.split_train_val:
            if train_dataset is None or val_dataset is None:
                raise ValueError('--split_train_val needs the training subsets and the validation split')
            self.split = check_val_split(train_dataset.labelled_dataset, train_dataset.unlabelled_dataset,
                                         val_dataset, args.train_classes)
            print(f'[FreqCone] validation split: {json.dumps(self.split)}')
        self.data_lists = self._data_lists(datasets) if (self.enabled and datasets) else None
        self.kmeans_cls = None           # tests inject a CPU stand-in; None = upstream K_Means (needs CUDA)
        self.command = _command()
        self.history, self.diagnostics, self.epoch_times, self.eval_times = [], [], [], []
        self._sum, self._n, self._win, self._win_n = {}, 0, {}, 0
        self._t_epoch, self._last_tick, self._step_times = None, None, []
        self._t_start = time.time()
        self.last_val = None
        self.resume_state, self.resumed_from = None, None
        self.run_dir = args.run_dir or None
        if self.run_dir:
            os.makedirs(self.run_dir, exist_ok=True)
            self.paths = {k: os.path.join(self.run_dir, f) for k, f in (('run', 'run.json'), ('val', 'epochs_val.jsonl'),
                                                                        ('test', 'epochs_test.jsonl'),
                                                                        ('ckpt', 'ckpt_resume.pt'))}
            if args.resume and os.path.exists(self.paths['ckpt']):
                self.resume_state = torch.load(self.paths['ckpt'], map_location='cpu', weights_only=False)
            elif any(os.path.exists(self.paths[k]) for k in ('val', 'test')) and not self.profiling:
                raise FileExistsError(f'{self.run_dir} already holds epoch records: use --resume True or a new run_dir')
        if self.loss is not None:
            print(f'[FreqCone] {args.freqcone_variant}: w_dir {args.fc_w_dir} w_ent {args.fc_w_ent} w_con '
                  f'{args.fc_w_con} K {args.fc_K} aperture {self.aperture_name} (r_a {self.r_a}, r_b {self.r_b}, '
                  f'psi {math.degrees(self.psi_a):.1f}/{math.degrees(self.psi_b):.1f} deg, depth '
                  f'{self.depth_a:.3f}/{self.depth_b:.3f}) f_a {args.fc_f_a} f_b {args.fc_f_b} coarse '
                  f'{self.coarse_size or "input size"}; clip depth {self.clip_depth}')

    @staticmethod
    def _data_lists(datasets):
        out = {}
        for key, ds in datasets.items():
            if ds is None:
                continue
            try:
                ids = item_ids(ds)
            except TypeError:
                continue
            out[key] = {'n': len(ids), 'sha256': list_sha256(ids)}
        return out

    # ------------------------------------------------------------------ output hygiene
    def install_quiet_output(self, train_module):
        """--hide_test_stdout: drop upstream's accuracy lines from stdout and its accuracy curves from TensorBoard."""
        if not self.args.hide_test_stdout:
            return
        import project_utils.cluster_and_log_utils as clu

        def quiet_print(*a, **k):
            if TEST_LINE.search(' '.join(str(x) for x in a)):
                return
            print(*a, **k)
        for module in (train_module, clu):
            setattr(module, 'print', quiet_print)
        if getattr(self.args, 'writer', None) is not None:
            self.args.writer = _QuietWriter(self.args.writer)

    # ------------------------------------------------------------------ training-time hooks
    def resume_skip(self, epoch, model, projection_head, optimizer, scheduler):
        """--resume: skip the finished epochs; at the last one, restore every state saved after it."""
        st = self.resume_state
        if st is None or epoch > st['epoch']:
            return False
        if epoch == st['epoch']:
            model.load_state_dict(st['model'])
            projection_head.load_state_dict(st['projection_head'])
            optimizer.load_state_dict(st['optimizer'])
            scheduler.load_state_dict(st['scheduler'])
            for key in ('history', 'diagnostics', 'epoch_times', 'eval_times'):
                setattr(self, key, st['fcrun'][key])
            for kind in ('val', 'test'):
                self._truncate(self.paths[kind], st['lines'][kind])
            set_rng_state(st['rng'])
            self.resumed_from = int(st['epoch']) + 1
            print(f'[FreqCone] resumed after epoch {self.resumed_from} from {self.paths["ckpt"]}')
            self.resume_state = None
        return True

    def start_epoch(self, epoch):
        self._t_epoch = time.time()
        if self.profiling and self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)

    def step(self, model, head_euclid, images, ball, batch_size, epoch, batch_idx):
        """FreqCone term for one batch (None when --freqcone False) and the depth / cone diagnostics.

        images: (2B, C, H, W), view 1 then view 2. ball: (2B, D) Poincare points of the base projection head
        (DINOHead -> ToPoincare), so u1, u2 use the base mapping. head_euclid: DINOHead alone (Euclidean).
        """
        args = self.args
        vals, loss_fc = {}, None
        u1 = ball[:batch_size]
        if self.loss is not None:
            u2, x1 = ball[batch_size:2 * batch_size], images[:batch_size]     # x1: the view that u1 came from
            with torch.no_grad():
                xa = radial_lowpass(x1, args.fc_f_a, out_size=self.coarse_size)
                xb = radial_lowpass(x1, args.fc_f_b, out_size=self.coarse_size)
            z = head_euclid(model(torch.cat([xa, xb], 0)))
            loss_fc, logs = self.loss(u1, u2, z[:batch_size], z[batch_size:])
            vals.update(logs)
        with torch.no_grad():
            d1 = depth(u1.detach().float(), args.c)
            vals['fc_depth_full'] = d1.mean()
            if self.clip_depth:
                vals['clip_frac'] = (d1 > self.clip_depth - 1e-3).float().mean()
        keys = sorted(vals)
        numbers = torch.stack([vals[k].detach().float().reshape(()) for k in keys]).tolist()   # one sync per step
        for k, x in zip(keys, numbers):
            self._sum[k] = self._sum.get(k, 0.0) + x
            self._win[k] = self._win.get(k, 0.0) + x
        self._n += 1
        self._win_n += 1
        if args.fc_log_every and (batch_idx + 1) % args.fc_log_every == 0:
            self._flush(epoch, batch_idx + 1)
        return loss_fc

    def _flush(self, epoch, step):
        m = {k: v / self._win_n for k, v in self._win.items()}
        rec = {'epoch': epoch + 1, 'step': step, 'depth_full': m.get('fc_depth_full'), 'in_cone': m.get('fc_in_cone'),
               'clip_frac': m.get('clip_frac'), 'angle_b_u': m.get('fc_angle_b_u'),
               'fc_dir': m.get('fc_dir'), 'fc_ent': m.get('fc_ent'), 'fc_con': m.get('fc_con')}
        self.diagnostics.append(rec)
        print('[FreqCone] ' + ' '.join(f'{k} {v:.4f}' if isinstance(v, float) else f'{k} {v}'
                                       for k, v in rec.items() if v is not None))
        self._win, self._win_n = {}, 0

    def check_finite(self, loss, epoch, batch_idx):
        """A NaN / inf loss stops the run; lambda and the learning rate are not touched."""
        if not bool(torch.isfinite(loss.detach()).all()):
            raise FloatingPointError(
                f'[FreqCone] non-finite loss at epoch {epoch + 1}, step {batch_idx + 1}: run stopped. Keep the log and '
                f'{self.paths["ckpt"] if self.run_dir else "the last checkpoint"} and report; do not lower lambda or '
                f'the learning rate.')

    def profile_tick(self, batch_idx):
        """True when --profile_steps steps are done (always False otherwise)."""
        if not self.profiling:
            return False
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        now = time.time()
        if self._last_tick is not None:
            self._step_times.append(now - self._last_tick)
        self._last_tick = now
        return batch_idx + 1 >= self.args.profile_steps

    def end_epoch(self, epoch, loss_avg, contrastive_avg, lambda_distance):
        """After the training steps of an epoch (before upstream's per-epoch evaluation)."""
        if self._win_n:
            self._flush(epoch, -1)
        rec = {'epoch': epoch + 1, 'loss': loss_avg, 'contrastive': contrastive_avg, 'lambda_distance': lambda_distance,
               'train_s': time.time() - self._t_epoch, 'steps': self._n}
        rec.update({k: v / max(self._n, 1) for k, v in self._sum.items()})
        self._sum, self._n = {}, 0
        self.history.append(rec)
        if self.loss is not None and rec.get('fc_depth_full', math.inf) < self.depth_b:
            print(f'[FreqCone] WARNING epoch {epoch + 1}: sharp views are shallower ({rec["fc_depth_full"]:.3f}) than '
                  f'the coarse view b (depth_b = {self.depth_b:.3f} at c = {self.args.c}, r_b = {self.r_b}); the cone '
                  f'cannot be satisfied (Lemma 3). This is a result to report, not an error.')

    def epoch_done(self, epoch, steps_per_epoch, model, projection_head, loader, test_fn, disjoint_test,
                   optimizer=None, scheduler=None):
        """End of an epoch (after upstream's own evaluation and checkpoints): per-epoch evaluation, records, resume
        state. Returns True when a --profile_steps run must stop here."""
        args = self.args
        if args.eval_every_epoch and self.run_dir:
            t0 = time.time()
            with isolated_rng(args.seed * 1000 + epoch + 1):
                m = test_fn(model, loader, args)
                val = self.val_metric(model, loader.dataset.labelled_dataset, args.seed * 1000 + epoch + 1) \
                    if self.split is not None else None
            self.eval_times.append(time.time() - t0)
            if not self.profiling:                         # a probe measures the evaluation time, records nothing
                self._append(self.paths['test'], {
                    'epoch': epoch + 1, 'all': float(m['all_acc']), 'old': float(m['old_acc']),
                    'new': float(m['new_acc']),
                    'upstream_disjoint_test': dict(zip(('all', 'old', 'new'), (float(x) for x in disjoint_test)))})
                self._append(self.paths['val'], self._val_line(epoch, val, optimizer))
        wall = time.time() - self._t_epoch
        self.epoch_times.append(wall)
        if self.history and self.history[-1]['epoch'] == epoch + 1:
            self.history[-1]['epoch_wall_s'] = wall
        if self.profiling:
            self._write_profile(epoch, steps_per_epoch, wall)
            return True
        if self.run_dir and optimizer is not None:
            self._save_resume(epoch, model, projection_head, optimizer, scheduler)
        return False

    def _val_line(self, epoch, val, optimizer):
        last = self.history[-1] if self.history else {}
        line = {'epoch': epoch + 1, 'val_acc': val['value'] if val else None,
                'losses': {k: last.get(k) for k in ('loss', 'contrastive', 'fc_dir', 'fc_ent', 'fc_con')},
                'fc': {'depth_full': last.get('fc_depth_full'), 'in_cone': last.get('fc_in_cone'),
                       'clip_frac': last.get('clip_frac'), 'angle_b_u': last.get('fc_angle_b_u'), 'depth_b': self.depth_b},
                'lambda_distance': last.get('lambda_distance'),
                'lr': optimizer.param_groups[0]['lr'] if optimizer is not None else None,
                'time_s': {'train': last.get('train_s'), 'eval': self.eval_times[-1] if self.eval_times else None}}
        if val is not None and epoch + 1 == self.args.epochs:
            line.update(val_per_sample_true=val['per_sample_true'], val_per_sample_pred=val['per_sample_pred'],
                        val_list_sha256=val['val_list_sha256'])
            self.last_val = val
        return line

    @staticmethod
    def _append(path, rec):
        with open(path, 'a') as f:
            f.write(json.dumps(rec) + '\n')

    @staticmethod
    def _count_lines(path):
        if not os.path.exists(path):
            return 0
        with open(path) as f:
            return sum(1 for _ in f)

    @staticmethod
    def _truncate(path, n_lines):
        if not os.path.exists(path):
            return
        with open(path) as f:
            lines = f.readlines()[:n_lines]
        with open(path, 'w') as f:
            f.writelines(lines)

    def _save_resume(self, epoch, model, projection_head, optimizer, scheduler):
        state = {'epoch': epoch, 'model': model.state_dict(), 'projection_head': projection_head.state_dict(),
                 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'rng': rng_state(),
                 'fcrun': {'history': self.history, 'diagnostics': self.diagnostics, 'epoch_times': self.epoch_times,
                           'eval_times': self.eval_times},
                 'lines': {k: self._count_lines(self.paths[k]) for k in ('val', 'test')}}
        tmp = self.paths['ckpt'] + '.tmp'
        torch.save(state, tmp)
        os.replace(tmp, self.paths['ckpt'])

    def _write_profile(self, epoch, steps_per_epoch, wall):
        st = self._step_times[2:] or self._step_times
        med = float(np.median(st)) if st else None
        done = len(self._step_times) + 1
        out = self._skeleton()
        out['profile'] = {
            'steps_trained': done, 'steps_per_epoch': int(steps_per_epoch), 'steps_timed': len(st),
            's_per_step_median': med, 's_per_step_mean': float(np.mean(st)) if st else None,
            'epoch_wall_s_measured': wall, 'eval_s': self.eval_times[-1] if self.eval_times else None,
            'est_epoch_s': wall + (steps_per_epoch - done) * med if med is not None else None,
            'note': 'est_epoch_s = measured wall time of this epoch (pseudo-label pass, the trained steps, upstream '
                    'per-epoch evaluation and checkpoints, and the per-epoch evaluation when enabled) + the untrained '
                    'steps at the median step time'}
        self._write(out)
        print(f'[FreqCone] profile: {json.dumps(out["profile"])} cost: {json.dumps(out["cost"])}')

    # ------------------------------------------------------------------ evaluation
    @torch.no_grad()
    def _features(self, model, dataset):
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False,
                                             num_workers=self.args.num_workers)
        model.eval()
        feats, targets = [], []
        for batch in loader:
            x = batch[0][0] if isinstance(batch[0], (list, tuple)) else batch[0]
            feats.append(F.normalize(model(x.to(self.device)), dim=-1).float().cpu())
            targets.append(torch.as_tensor(batch[1]).long())
        return torch.cat(feats), torch.cat(targets)

    def val_metric(self, model, labelled_dataset, random_state=0):
        """Selection metric (see VAL_DEFINITION); per-image predictions are returned for the bootstrap."""
        from project_utils.cluster_utils import cluster_acc
        km_cls = self.kmeans_cls
        if km_cls is None:
            from methods.clustering.faster_mix_k_means_pytorch import K_Means as km_cls
        args = self.args
        l_feats, l_targets = self._features(model, labelled_dataset)
        v_feats, v_targets = self._features(model, self.val_dataset)
        k = int(args.num_labeled_classes)
        n = len(l_targets) + len(v_targets)
        cluster_size = None if getattr(args, 'unbalanced', False) else math.ceil(n / k)
        km = km_cls(k=k, tolerance=1e-4, max_iterations=10, init='k-means++', n_init=1, random_state=int(random_state),
                    n_jobs=None, pairwise_batch_size=1024, mode=None, protos=None, cluster_size=cluster_size)
        km.fit_mix(v_feats.to(self.device), l_feats.to(self.device), l_targets.to(self.device))
        pred = km.labels_.cpu().numpy()[len(l_targets):].astype(int)
        true = v_targets.numpy().astype(int)
        return {'metric': VAL_METRIC, 'definition': VAL_DEFINITION, 'value': float(cluster_acc(true, pred)),
                'per_sample_true': true.tolist(), 'per_sample_pred': pred.tolist(),
                'val_list_sha256': self.split['val_list_sha256'], 'n_images': int(true.size),
                'n_anchor_images': int(len(l_targets)), 'n_clusters': k, 'cluster_size': cluster_size}

    @torch.no_grad()
    def geometry(self, model, projection_head, loader):
        """Depth and cone diagnostics on training images (test transform), for every aperture preset.

        Computed for the baseline too (same coarse views, fixed radii), on the same images: is the sharp view
        deeper than b, and how often is it inside b's cone?
        """
        args = self.args
        model.eval()
        stride = 1
        if args.fc_diag_images > 0:
            stride = max(1, math.ceil(len(loader) * args.batch_size / args.fc_diag_images))
        presets = dict(APERTURES)
        if self.aperture_name == 'custom':
            presets['custom'] = (self.r_a, self.r_b)
        rows = {'depth': [], 'known': [], 'angle': []}
        cone = {name: {'b_u': [], 'a_b': []} for name in presets}
        for i, batch in enumerate(loader):
            if i % stride:
                continue
            x, label = batch[0].to(self.device), torch.as_tensor(batch[1])
            u = projection_head(model(x)).float()
            za = projection_head[0](model(radial_lowpass(x, args.fc_f_a, out_size=self.coarse_size))).float()
            zb = projection_head[0](model(radial_lowpass(x, args.fc_f_b, out_size=self.coarse_size))).float()
            rows['depth'].append(depth(u, args.c).cpu())
            rows['known'].append((label < args.num_labeled_classes).cpu())
            rows['angle'].append(torch.rad2deg(torch.acos(F.cosine_similarity(zb, u, dim=-1).clamp(-1, 1))).cpu())
            for name, (r_a, r_b) in presets.items():
                ua, ub = to_fixed_depth(za, r_a, args.c), to_fixed_depth(zb, r_b, args.c)
                cone[name]['b_u'].append((cone_energy(ub, u, args.c, args.fc_K) < 1e-6).float().cpu())
                cone[name]['a_b'].append((cone_energy(ua, ub, args.c, args.fc_K) < 1e-6).float().cpu())
        d, known, ang = torch.cat(rows['depth']), torch.cat(rows['known']).bool(), torch.cat(rows['angle'])
        per = {}
        for name, (r_a, r_b) in presets.items():
            d_b = scaled_depth(r_b, args.c)
            per[name] = {'r_a': r_a, 'r_b': r_b, 'psi_a_deg': math.degrees(aperture(args.fc_K, r_a)),
                         'psi_b_deg': math.degrees(aperture(args.fc_K, r_b)), 'depth_a': scaled_depth(r_a, args.c),
                         'depth_b': d_b, 'frac_deeper_than_b': float((d > d_b).float().mean()),
                         'in_cone_b_u': float(torch.cat(cone[name]['b_u']).mean()),
                         'in_cone_a_b': float(torch.cat(cone[name]['a_b']).mean())}
        out = {'n_images': int(d.numel()), 'depth_full': float(d.mean()),
               'depth_full_known': float(d[known].mean()) if known.any() else None,
               'depth_full_novel': float(d[~known].mean()) if (~known).any() else None,
               'angle_b_u_deg': float(ang.mean()), 'run_aperture': self.aperture_name, 'per_aperture': per}
        if self.clip_depth:
            out['clip_frac'] = float((d > self.clip_depth - 1e-3).float().mean())
        return out

    def final_eval(self, model, projection_head, merge_train_loader_test, test_fn):
        """After the last epoch: run.json (no test number). Without --eval_every_epoch, the last-epoch test and
        validation are computed here once and the test goes to epochs_test.jsonl, never to run.json."""
        t0 = time.time()
        args = self.args
        if not args.eval_every_epoch:
            with isolated_rng(args.seed * 1000 + args.epochs):
                m = test_fn(model, merge_train_loader_test, args)
                if self.split is not None:
                    self.last_val = self.val_metric(model, merge_train_loader_test.dataset.labelled_dataset,
                                                    args.seed * 1000 + args.epochs)
            self._append(self.paths['test'], {'epoch': int(args.epochs), 'all': float(m['all_acc']),
                                              'old': float(m['old_acc']), 'new': float(m['new_acc']),
                                              'upstream_disjoint_test': None})
        out = self._skeleton()
        out['val'] = self.last_val
        out['geometry_final'] = self.geometry(model, projection_head, merge_train_loader_test)
        out['cost']['final_eval_s'] = time.time() - t0
        self._write(out)
        return out

    # ------------------------------------------------------------------ output
    def _skeleton(self):
        a = self.args
        cuda = self.device.type == 'cuda'
        modified = _git('diff', '--name-only', 'HEAD')
        ckpt = None
        try:
            from config import dinov2_pretrain_path
            if getattr(a, 'dino', None) == 'v2' and os.path.isfile(dinov2_pretrain_path):
                ckpt = {'path': dinov2_pretrain_path, 'sha256': sha256_file(dinov2_pretrain_path)}
        except Exception:  # config not filled in (tests)
            ckpt = None
        return {
            'schema': SCHEMA, 'run_id': a.run_id, 'run_tag': a.run_tag, 'attempt': a.run_attempt,
            'resumed_from_epoch': self.resumed_from,
            'git_commit': _git('rev-parse', 'HEAD'), 'git_tag': _git('describe', '--tags', '--exact-match'),
            'git_modified_files': modified.split('\n') if modified else [],
            'command': self.command, 'args': {k: _jsonable(v) for k, v in vars(a).items() if _jsonable(v) is not None},
            'base_method': 'Hyp-SelEx', 'dataset': a.dataset_name, 'seed': a.seed, 'c': a.c, 'r': a.cr,
            'dist_impl': a.dist_impl, 'fix_label_order': a.fix_label_order, 'split_train_val': a.split_train_val,
            'smoke_frac': a.smoke_frac, 'epochs': a.epochs, 'batch_size': a.batch_size,
            'eval_every_epoch': a.eval_every_epoch, 'hide_test_stdout': a.hide_test_stdout,
            'gpu_name': torch.cuda.get_device_name(self.device) if cuda else None,
            'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
            'dinov2_checkpoint': ckpt, 'data_lists': self.data_lists,
            'freqcone': {'enable': bool(a.freqcone), 'variant': a.freqcone_variant if a.freqcone else None,
                         'w_dir': a.fc_w_dir, 'w_ent': a.fc_w_ent, 'w_con': a.fc_w_con, 'K': a.fc_K,
                         'aperture': self.aperture_name, 'r_a': self.r_a, 'r_b': self.r_b,
                         'psi_a': math.degrees(self.psi_a), 'psi_b': math.degrees(self.psi_b),
                         'depth_a': self.depth_a, 'depth_b': self.depth_b, 'f_a': a.fc_f_a, 'f_b': a.fc_f_b,
                         'coarse_res': self.coarse_size or getattr(a, 'image_size', None),
                         'tau_a': a.fc_tau_a, 'tau_b': a.fc_tau_b},
            'split': self.split, 'val': None,
            'files': {'epochs_val': 'epochs_val.jsonl', 'epochs_test': 'epochs_test.jsonl'},
            'diagnostics': self.diagnostics, 'history': self.history,
            'cost': {'time_per_epoch_s': float(np.mean(self.epoch_times)) if self.epoch_times else None,
                     'eval_per_epoch_s': float(np.mean(self.eval_times)) if self.eval_times else None,
                     'epochs_completed': len(self.epoch_times), 'wall_s': time.time() - self._t_start,
                     'peak_mem_gb': torch.cuda.max_memory_allocated(self.device) / 2 ** 30 if cuda else None,
                     'peak_reserved_gb': torch.cuda.max_memory_reserved(self.device) / 2 ** 30 if cuda else None},
        }

    def _write(self, out):
        if not self.run_dir:
            return
        with open(self.paths['run'], 'w') as f:
            json.dump(out, f, indent=1)
        print(f'[FreqCone] run record written to {self.paths["run"]}')

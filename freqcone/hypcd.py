"""Glue between FreqCone and HypCD's Hyp-SelEx training script (train/train_HypSelEx.py).

Every hook is called from a block marked ``# >>> freqcone`` ... ``# <<< freqcone`` in an upstream file.
With the default flags none of them changes the computation:
  * --freqcone False         no coarse views, no extra loss term
  * --dist_impl upstream     hyptorch.pmath.dist_matrix, exactly as upstream
  * --fix_label_order False  SelEx's level >= 1 pseudo-labels as upstream stores them (SelEx issue #11)
  * --split_train_val False, --smoke_frac 0, --results_json '', --profile_steps 0
tests_freqcone/test_upstream_equivalence.py checks that deleting the marked blocks gives back the upstream
files byte for byte, and that these defaults leave every hook inactive.

Nothing here draws from the global random number generators, so a FreqCone run and its baseline see the
same sampler order and augmentations for the same seed. Result files follow SCHEMA (see README.md).
"""
import argparse
import json
import math
import os
import shlex
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from freqcone.data_checks import check_val_split
from freqcone.freqcone_torch import (APERTURES, FreqConeLoss, aperture, cone_energy, depth, radial_lowpass,
                                     scaled_depth, to_fixed_depth)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHEMA = 'freqcone-run/1'
VAL_METRIC = 'ss_kmeans_known_hungarian'
VAL_DEFINITION = ("Hyp-SelEx's own prediction procedure applied to the held-out validation images: upstream "
                  "semi-supervised k-means (methods/clustering/faster_mix_k_means_pytorch.K_Means, settings of "
                  "test()) with the labelled training images as fixed anchors and the validation images as "
                  "unlabelled points, k = number of known classes, L2-normalised backbone features (test "
                  "transform); scored with upstream project_utils.cluster_utils.cluster_acc, the Hungarian "
                  "accuracy that split_cluster_acc_v2 uses for 'All'.")


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
    g.add_argument('--freqcone', type=str2bool, default=False, help='add the FreqCone loss (v2, or v3 with --fc_v3)')
    g.add_argument('--fc_w_dir', type=float, default=0.5)
    g.add_argument('--fc_w_ent', type=float, default=0.03, help='lambda_ent: choose it on known-class validation')
    g.add_argument('--fc_w_con', type=float, default=0.05)
    g.add_argument('--fc_K', type=float, default=0.1, help='cone aperture constant K')
    g.add_argument('--fc_aperture', type=str, default='A1', choices=sorted(APERTURES),
                   help='coarse-view radii (r_a, r_b): A1 (0.12, 0.2), A2 (0.104, 0.136), A3 (0.1, 0.114)')
    g.add_argument('--fc_r_a', type=float, default=None, help='custom r_a (needs --fc_r_b; overrides the preset)')
    g.add_argument('--fc_r_b', type=float, default=None, help='custom r_b (needs --fc_r_a)')
    g.add_argument('--fc_tau_a', type=float, default=0.5)
    g.add_argument('--fc_tau_b', type=float, default=0.3)
    g.add_argument('--fc_f_a', type=float, default=3.0, help='cut-off of x^a in cycles / image at the input size')
    g.add_argument('--fc_f_b', type=float, default=11.0, help='cut-off of x^b in cycles / image (> fc_f_a)')
    g.add_argument('--fc_v3', type=str2bool, default=False, help='v3: L_ent may also change the sharp view norm')
    g.add_argument('--fc_coarse_size', type=int, default=0, help='0 = input size; e.g. 112 (needs fc_f_b < size / 2)')
    g.add_argument('--fc_log_every', type=int, default=50, help='record the diagnostics every N steps')
    g.add_argument('--fc_diag_images', type=int, default=2048, help='images for the final geometry (0 = all)')
    g.add_argument('--split_train_val', type=str2bool, default=False,
                   help="hold out GCD's known-class validation split (20%% of the labelled images of each class)")
    g.add_argument('--smoke_frac', type=int, default=0, help='smoke tests only: keep every k-th image of every split')
    g.add_argument('--results_json', type=str, default='', help='write the run record (schema freqcone-run/1)')
    g.add_argument('--profile_steps', type=int, default=0, help='>0: train N steps of one epoch, record time and memory')
    g.add_argument('--run_id', type=str, default='', help='pilot run id: R0-repro, R0 .. R8')
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


class _Inert:
    """Stand-in used until the training script builds an FCRun."""
    enabled = False


INERT = _Inert()


class FCRun:
    def __init__(self, args, device, train_dataset=None, val_dataset=None):
        self.args, self.device = args, device
        self.profiling = args.profile_steps > 0
        self.enabled = bool(args.freqcone or args.results_json or self.profiling)
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
                                     f_a=args.fc_f_a, f_b=args.fc_f_b, detach_child_norm=not args.fc_v3,
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
        self.kmeans_cls = None           # tests inject a CPU stand-in; None = upstream K_Means (needs CUDA)
        self.command = _command()
        self.history, self.diagnostics, self.epoch_times = [], [], []
        self._sum, self._n, self._win, self._win_n = {}, 0, {}, 0
        self._t_epoch, self._last_tick, self._step_times = None, None, []
        self._t_start = time.time()
        if self.loss is not None:
            print(f'[FreqCone] {"v3" if args.fc_v3 else "v2"}: w_dir {args.fc_w_dir} w_ent {args.fc_w_ent} w_con '
                  f'{args.fc_w_con} K {args.fc_K} aperture {self.aperture_name} (r_a {self.r_a}, r_b {self.r_b}, '
                  f'psi {math.degrees(self.psi_a):.1f}/{math.degrees(self.psi_b):.1f} deg, depth '
                  f'{self.depth_a:.3f}/{self.depth_b:.3f}) f_a {args.fc_f_a} f_b {args.fc_f_b} coarse '
                  f'{self.coarse_size or "input size"}; clip depth {self.clip_depth}')

    # ------------------------------------------------------------------ training-time hooks
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
        rec = {'epoch': epoch, 'step': step, 'depth_full': m.get('fc_depth_full'), 'in_cone': m.get('fc_in_cone'),
               'clip_frac': m.get('clip_frac'), 'angle_b_u': m.get('fc_angle_b_u'),
               'fc_dir': m.get('fc_dir'), 'fc_ent': m.get('fc_ent'), 'fc_con': m.get('fc_con')}
        self.diagnostics.append(rec)
        print('[FreqCone] ' + ' '.join(f'{k} {v:.4f}' if isinstance(v, float) else f'{k} {v}'
                                       for k, v in rec.items() if v is not None))
        self._win, self._win_n = {}, 0

    def check_finite(self, loss, epoch, batch_idx):
        """Troubleshooting rule: a NaN / inf loss stops the run; lambda and the learning rate are not touched."""
        if not bool(torch.isfinite(loss.detach()).all()):
            raise FloatingPointError(
                f'[FreqCone] non-finite loss at epoch {epoch}, step {batch_idx + 1}: run stopped. Keep the log and '
                f'the last checkpoint ({getattr(self.args, "model_path", "?")}, saved at the end of the previous '
                f'epoch) and report; do not lower lambda or the learning rate.')

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
        rec = {'epoch': epoch, 'loss': loss_avg, 'contrastive': contrastive_avg, 'lambda_distance': lambda_distance,
               'train_s': time.time() - self._t_epoch, 'steps': self._n}
        rec.update({k: v / max(self._n, 1) for k, v in self._sum.items()})
        self._sum, self._n = {}, 0
        self.history.append(rec)
        if self.loss is not None and rec.get('fc_depth_full', math.inf) < self.depth_b:
            print(f'[FreqCone] WARNING epoch {epoch}: sharp views are shallower ({rec["fc_depth_full"]:.3f}) than the '
                  f'coarse view b (depth_b = {self.depth_b:.3f} at c = {self.args.c}, r_b = {self.r_b}); the cone '
                  f'cannot be satisfied (Lemma 3). This is a result to report, not an error.')

    def epoch_done(self, epoch, steps_per_epoch):
        """End of the whole epoch (training, upstream evaluation, checkpoints). True: a profile run stops here."""
        wall = time.time() - self._t_epoch
        self.epoch_times.append(wall)
        if self.history and self.history[-1]['epoch'] == epoch:
            self.history[-1]['epoch_wall_s'] = wall
        if not self.profiling:
            return False
        st = self._step_times[2:] or self._step_times
        med = float(np.median(st)) if st else None
        done = len(self._step_times) + 1
        out = self._skeleton()
        out['profile'] = {
            'steps_trained': done, 'steps_per_epoch': int(steps_per_epoch), 'steps_timed': len(st),
            's_per_step_median': med, 's_per_step_mean': float(np.mean(st)) if st else None,
            'epoch_wall_s_measured': wall,
            'est_epoch_s': wall + (steps_per_epoch - done) * med if med is not None else None,
            'note': 'est_epoch_s = measured wall time of this epoch (pseudo-label pass, the trained steps, upstream '
                    'per-epoch evaluation, checkpoints) + the untrained steps at the median step time'}
        self._write(out)
        print(f'[FreqCone] profile: {json.dumps(out["profile"])} memory: {json.dumps(out["cost"])}')
        return True

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

    def val_metric(self, model, labelled_dataset):
        """Selection metric (see VAL_DEFINITION); per-image predictions are stored for the bootstrap."""
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
        km = km_cls(k=k, tolerance=1e-4, max_iterations=10, init='k-means++', n_init=1, random_state=None, n_jobs=None,
                    pairwise_batch_size=1024, mode=None, protos=None, cluster_size=cluster_size)
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

    def upstream_best(self, model, loader, test_fn, best_epoch, best_stats):
        """Upstream's checkpoint selected on Old accuracy of the disjoint test split, scored like test_last.
        G0 (reproduction) diagnostics only; never used to conclude."""
        args = self.args
        path = args.model_path[:-3] + '_best.pt'
        rec = {'epoch': int(best_epoch), 'all': None, 'old': None, 'new': None,
               'source': "upstream test() on the checkpoint '*_best.pt' that upstream keeps (best Old accuracy on "
                         'the disjoint test split); for G0 only',
               'printed_train_acc': [float(x) for x in best_stats] if best_stats else None}
        if not os.path.exists(path):
            rec['note'] = f'checkpoint not found: {path}'
            return rec
        final_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        try:
            model.load_state_dict(torch.load(path, map_location='cpu'))
            m = test_fn(model, loader, args)
        finally:
            model.load_state_dict(final_state)
        rec.update(all=float(m['all_acc']), old=float(m['old_acc']), new=float(m['new_acc']))
        return rec

    def final_eval(self, model, projection_head, merge_train_loader_test, test_fn, best_epoch, best_stats):
        """Last-epoch record: test_last (v2, transductive), val metric, geometry, upstream_best."""
        t0 = time.time()
        args = self.args
        out = self._skeleton()
        m = test_fn(model, merge_train_loader_test, args)
        out['test_last'] = {'all': float(m['all_acc']), 'old': float(m['old_acc']), 'new': float(m['new_acc']),
                            'epoch': int(args.epochs) - 1,
                            'protocol': 'v2 (one Hungarian matching) on the unlabelled training images, last epoch; '
                                        'upstream test(): SS-k-means over all training images'}
        if self.split is not None:
            out['val'] = self.val_metric(model, merge_train_loader_test.dataset.labelled_dataset)
        out['geometry_final'] = self.geometry(model, projection_head, merge_train_loader_test)
        out['upstream_best'] = self.upstream_best(model, merge_train_loader_test, test_fn, best_epoch, best_stats)
        out['cost']['final_eval_s'] = time.time() - t0
        self._write(out)
        return out

    # ------------------------------------------------------------------ output
    def _skeleton(self):
        a = self.args
        cuda = self.device.type == 'cuda'
        modified = _git('diff', '--name-only', 'HEAD')
        return {
            'schema': SCHEMA, 'run_id': a.run_id, 'run_tag': a.run_tag, 'attempt': a.run_attempt,
            'git_commit': _git('rev-parse', 'HEAD'), 'git_tag': _git('describe', '--tags', '--exact-match'),
            'git_modified_files': modified.split('\n') if modified else [],
            'command': self.command, 'args': {k: _jsonable(v) for k, v in vars(a).items() if _jsonable(v) is not None},
            'base_method': 'Hyp-SelEx', 'dataset': a.dataset_name, 'seed': a.seed, 'c': a.c, 'r': a.cr,
            'dist_impl': a.dist_impl, 'fix_label_order': a.fix_label_order, 'split_train_val': a.split_train_val,
            'smoke_frac': a.smoke_frac, 'epochs': a.epochs, 'batch_size': a.batch_size,
            'gpu_name': torch.cuda.get_device_name(self.device) if cuda else None,
            'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
            'freqcone': {'enable': bool(a.freqcone), 'variant': ('v3' if a.fc_v3 else 'v2') if a.freqcone else None,
                         'w_dir': a.fc_w_dir, 'w_ent': a.fc_w_ent, 'w_con': a.fc_w_con, 'K': a.fc_K,
                         'aperture': self.aperture_name, 'r_a': self.r_a, 'r_b': self.r_b,
                         'psi_a': math.degrees(self.psi_a), 'psi_b': math.degrees(self.psi_b),
                         'depth_a': self.depth_a, 'depth_b': self.depth_b, 'f_a': a.fc_f_a, 'f_b': a.fc_f_b,
                         'coarse_res': self.coarse_size or getattr(a, 'image_size', None),
                         'tau_a': a.fc_tau_a, 'tau_b': a.fc_tau_b},
            'split': self.split, 'val': None, 'test_last': None, 'upstream_best': None,
            'diagnostics': self.diagnostics, 'history': self.history,
            'cost': {'time_per_epoch_s': float(np.mean(self.epoch_times)) if self.epoch_times else None,
                     'epochs_completed': len(self.epoch_times), 'wall_s': time.time() - self._t_start,
                     'peak_mem_gb': torch.cuda.max_memory_allocated(self.device) / 2 ** 30 if cuda else None,
                     'peak_reserved_gb': torch.cuda.max_memory_reserved(self.device) / 2 ** 30 if cuda else None},
        }

    def _write(self, out):
        path = self.args.results_json
        if not path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(out, f, indent=1)
        print(f'[FreqCone] run record written to {path}')

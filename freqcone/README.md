# FreqCone in the official HypCD code

This branch (`freqcone`) is the official HypCD code at commit `ffc895b` (2025-11-07). It adds FreqCone, a memory-safe hyperbolic distance, a fix for SelEx's pseudo-label order bug, and a pre-registered pilot protocol.

Every change to an upstream file is a block between `# >>> freqcone` and `# <<< freqcone`. With the default flags no block changes the computation. `tests_freqcone/test_upstream_equivalence.py` checks two things:
- Deleting the blocks gives back the upstream files byte for byte.
- The defaults leave every hook inactive.

Run the tests with `python -m pytest tests_freqcone -q`. CPU is enough; the tests marked CUDA run on the GPU machine.

## Decisions fixed for the pilot (2026-11-04, before any SSB run)

| Item | Decision |
| --- | --- |
| Base method | Hyp-SelEx (`train/train_HypSelEx.py`). It predicts with semi-supervised k-means (`test()`), not with a classifier head. |
| Curvature c, clip r | The upstream DINOv2 values of each dataset (table below), for the baseline and FreqCone alike |
| SelEx label-order bug | Present upstream (see below). R0-R8 run with `--fix_label_order True`. R0-repro keeps the bug, for G0 only. |
| Cut-offs f_a, f_b | SNR 5 / 10 dB, FourEx's definition, computed per dataset on training images without labels (table below) |
| Cone aperture | K = 0.1 with the presets A1, A2, A3 instead of a grid over K (table below) |
| Depth threshold | δ_c(b) = (2/√c)·artanh(r_b), computed for each run and logged. It is never hard-coded. |
| Validation | GCD's split: 20 % of the labelled images of each known class, never trained on, identical across runs (`val_list_sha256`) |
| Selection | Validation only, with a paired, stratified bootstrap 1-SE rule (`freqcone/select_pilot.py`) |
| Seeds | Pilot 7. P2 uses 8, 9, 10. Branch B uses 11, 12, 13. |
| Coarse-view resolution, budget | Decided from the profile probes before R1 (section below) |

### c and r per dataset (upstream `scripts/train_HypSelEx.cmd`, DINOv2 rows)

| Dataset | c | r | Unsupervised smoothing | hyper_max_weight | Max scaled radius tanh(√c·r) |
| --- | --- | --- | --- | --- | --- |
| CUB | 0.1 | 1.5 | 1.0 | 1.0 | 0.442 |
| Stanford Cars | 0.1 | 2.0 | 0.5 | 0.5 | 0.559 |
| FGVC-Aircraft | 0.05 | 1.5 | 0.5 | 0.5 | 0.323 |

The draft paper currently states c = 0.05 and r = 2.3 (the HypCD paper's fine-grained setting). That will become "the configuration of the official scripts". A CUB baseline at 0.05 / 2.3 is only a later sensitivity check if budget remains; it is never used for selection.

### SelEx's pseudo-label order bug (SelEx issue #11, confirmed by the authors on 2026-10-20)

How the bug arises in the upstream code:
- `K_Means.fit_mix` returns labels for `cat(l_feats, u_feats)` (`methods/clustering/faster_mix_k_means_pytorch.py:184`).
- In `extract_labeled_protos`, hierarchy levels ≥ 1 split the features by the *predicted*-known mask (upstream l. 496–499).
- Those levels store their labels in `[predicted known; predicted unknown]` order (l. 517–521).
- The readers index them in loader order (l. 288, 356, 357).
- So coarse pseudo-labels land on other images. On CUB, whose `images.txt` is sorted by species, those images are mostly of the same or a neighbouring species, which leaks species information.

The issue reports SelEx with DINOv2 on CUB at All 89.4 in the shipped order, against 83.1 and 85.4 with shuffled orders.

`--fix_label_order True` scatters each level's labels back to loader order (`freqcone.hypcd.loader_order`). The published Hyp-SelEx numbers were very likely obtained with the bug, so G0 uses the run R0-repro, which keeps it.

### Coarse-view apertures (K = 0.1; r = √c·‖p‖)

| Preset | r_a | r_b | ψ_a | ψ_b | δ(a) / δ(b) at c = 0.1 | δ(a) / δ(b) at c = 0.05 |
| --- | --- | --- | --- | --- | --- | --- |
| A1 (default) | 0.120 | 0.200 | 55.2° | 28.7° | 0.763 / 1.282 | 1.079 / 1.813 |
| A2 | 0.104 | 0.136 | 72.0° | 46.2° | 0.660 / 0.866 | 0.934 / 1.224 |
| A3 (widest) | 0.100 | 0.114 | 81.9° | 60.0° | 0.635 / 0.724 | 0.897 / 1.024 |

`FreqConeLoss` raises an error unless ε_K < r_a < r_b < tanh(√c·r_clip), with ε_K = 2K/(1+√(1+4K²)) = 0.099 at K = 0.1. Below ε_K the half-aperture is clamped at 90°, so the cone becomes a half-space. That is why the old grid K ∈ {0.15, 0.2} with r_a = 0.12 was degenerate.

A2 and A3 change the aperture *and* the depth of the coarse views together. A shallower b also makes the cone easier to reach when the sharp views are shallow, so the two effects are confounded; this is accepted for a pilot.

ψ_b ≈ 74° is not used. It would need r_b ≈ 0.103 and r_a ∈ (0.099, 0.103), which leaves almost no depth gap between the two levels.

### FreqCone cut-offs

<!-- cutoffs:start -->
Not computed yet: run `python scripts/freqcone/compute_cutoffs.py --datasets cub scars aircraft` on the machine that holds the data. It fills this table and `cutoffs.json`. Reference values from FourEx Table 7 (5 / 10 dB) are CUB 2 / 8, Stanford Cars 3 / 11 and Aircraft 2 / 9. The computed values are the ones used.
<!-- cutoffs:end -->

The definition, citation and open choices are in `freqcone/cutoffs.py`. FourEx's code is not released, so its measurement details are fixed there before any run:
- radial metric, the same as FreqCone's filter;
- one mean spectrum over all training images, without labels;
- test transform at 224 px.

### Coarse-view resolution and budget

<!-- pilot-decision:start -->
Not decided yet: `bash scripts/freqcone/profile_gpu.sh` runs the probes and `python -m freqcone.pilot_tools decide` writes this section and `decision.json`.
<!-- pilot-decision:end -->

The rule, fixed before any probe:
- **Resolution.** Use 224 px if the 224 px probe runs at the upstream batch size (128) and its estimated epoch time is at most 1.8× the baseline's. Otherwise use 112 px for every FreqCone run of the pilot and of P2.
- **Budget.** GPU-hours = Σ over the planned runs (R0-repro, R0, R1–R8) of 200 × t_epoch.
- **Over 100 GPU-hours.** Drop aperture A2, i.e. runs R3 and R6 (λ_ent 0.03 and 0.1 with A2). The plan's text says "R4 and R5", but under its own run table A2 is R3 and R6.

## Validation metric and the separation of selection from reporting

**Validation metric** (`freqcone.hypcd.VAL_DEFINITION`). This is Hyp-SelEx's own prediction procedure applied to the validation images:
- upstream `K_Means` with the settings of `test()`;
- the labelled training images as fixed anchors and the validation images as unlabelled points;
- k = number of known classes;
- L2-normalised backbone features.

It is scored with upstream `cluster_acc`, which computes the same Hungarian accuracy that `split_cluster_acc_v2` reports as "All" (`tests_freqcone/test_hungarian.py`). Per-image predictions are stored so that the bootstrap can recompute everything.

**Separation of selection and reporting:**
- `python -m freqcone.select_pilot RESULTS` reads only the validation fields and writes `selection.json`. That file holds every score, its SE, the tie set, the choice, the R8 configuration and the SHA-256 of every input.
- `python -m freqcone.report_pilot RESULTS` prints test numbers only when `selection.json` is committed and unchanged and every input still has its recorded hash.
- While the pilot runs, do not open the logs or the JSON test fields of R1–R8.

## Run record (`--results_json`, schema `freqcone-run/1`)

| Key | Content |
| --- | --- |
| `git_commit`, `git_tag`, `git_modified_files`, `command`, `args` | Provenance |
| `dataset`, `seed`, `c`, `r`, `dist_impl`, `fix_label_order`, `split_train_val`, `gpu_name`, `torch_version`, `cuda_version` | Setting |
| `freqcone` | `enable, variant, w_dir, w_ent, w_con, K, aperture, r_a, r_b, psi_a, psi_b, depth_a, depth_b, f_a, f_b, coarse_res` |
| `split` | Validation-split identity: `val_list_sha256` and sizes, after the disjointness and known-class checks |
| `val` | `metric, value, per_sample_true, per_sample_pred, val_list_sha256` |
| `test_last` | All / Old / New at the last epoch: v2, transductive, on the unlabelled training images |
| `upstream_best` | Upstream's checkpoint chosen on test Old accuracy, scored like `test_last`. G0 diagnostics only. |
| `diagnostics` | Every `--fc_log_every` steps: `depth_full`, `in_cone`, `clip_frac`, `angle_b_u` (and the loss terms) |
| `geometry_final` | Sharp-view depth, overall and for known / novel images, plus `per_aperture` A1–A3 (δ(b), share deeper than b, in-cone shares, ψ) |
| `cost` | `time_per_epoch_s`, `peak_mem_gb`, wall time |

The baseline records sharp-view depth too: in its diagnostics and in `geometry_final`, on the same images as the FreqCone runs. `python -m freqcone.pilot_tools check FILE` validates a record.

## Runbook on the GPU machine

1. **Set up.** Check out tag `pilot-v1`. Install `requirements.txt`. In `config.py`, set `{DATA_PATH}`, `{PRETRAINED_PATH}` (`dinov2_vitb14_reg4_pretrain.pth`) and `{PROJECT_PATH}`; `config.py` is the only tracked file that may be modified. Run `python -m pytest tests_freqcone -q`; everything must pass, including the CUDA tests. Record the driver, CUDA and torch versions.
2. **Cut-offs.** Run `python scripts/freqcone/compute_cutoffs.py --datasets cub scars aircraft`. Commit `cutoffs.json` and `freqcone/README.md`.
3. **Smoke test.** Run `bash scripts/freqcone/smoke_gpu.sh` (two epochs on every 10th image). Then `rm -rf results_ssb/smoke_cub`.
4. **Profile.** Run `bash scripts/freqcone/profile_gpu.sh` (four probes, then `decide`). Commit `results_ssb/profile_cub/decision.json` and `freqcone/README.md`. Tag `pilot-v2` with the reason: "cut-offs and resolution / budget decision added after pilot-v1".
5. **Pilot.** Run `bash scripts/freqcone/pilot_cub.sh`. The script does the following in order:
   1. Preflight checks.
   2. R0-repro, then G0.
   3. R0, then R1–R7.
   4. Selection, committed.
   5. R8.
   6. Report and G1.

   Finished runs are skipped on restart.
6. **Send back.**
   - Every JSON (profile, R0-repro, R0–R8, `selection.json`, `g0.json`, `cutoffs.json`, `decision.json`, `pilot_report.json` / `.md`) and the logs.
   - The tag or commit used.
   - The conclusion against the fixed gates.
   - What did not run.

## Troubleshooting (no configuration change on our own)

| Situation | Action |
| --- | --- |
| Loss NaN / inf | The run stops with `FloatingPointError`. Keep the log and the last checkpoint and report. Do not lower λ or the learning rate. |
| Out of memory | Apply the resolution rule (112 px). If it is still out of memory, report. A smaller batch is allowed only for both arms and must be recorded, because it breaks comparability with the published numbers. |
| Interrupted run | The code cannot resume (upstream saves no optimiser state). Rerun from scratch with the same seed; `attempt` is recorded. |
| Sharp views shallower than δ(b), in-cone ≈ 0 | Not an error: this is the shallow regime and must be reported. R8 (v3) and A2/A3 test this case. |
| G0 fails | Stop before any FreqCone run. Check the SSB split, c / r, augmentation, epochs and batch, then report. |
| More than 30 % over the planned time | `pilot_tools budget` stops the pilot. Report before continuing. |

## Not done here

- Branch B (decoupled head `g_fc`, novel-to-parent attribution) waits for G1, by the user's decision.
- P2 runs only if G1 passes.
- No SSB number exists yet.

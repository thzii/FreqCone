# Sourced by the FreqCone run scripts. Needs $DATASET and $SEED ($EPOCHS defaults to 200). Defines:
#   UPSTREAM  upstream Hyp-SelEx exactly as scripts/train_HypSelEx.cmd runs it (DINOv2 row of the dataset),
#             including that row's curvature c and clip radius r, plus --seed and --dist_impl lowmem (same
#             distances as upstream, O(N^2) memory; tests_freqcone/test_dist_lowmem.py)
#   PILOT     flags of every pilot run except R0-repro: GCD's known-class validation split, and the fix of
#             SelEx's pseudo-label order bug (SelEx issue #11)
#   FC_CUTS   --fc_f_a / --fc_f_b from cutoffs.json (scripts/freqcone/compute_cutoffs.py); empty if missing
#
# Upstream c / r per dataset (DINOv2 rows of scripts/train_HypSelEx.cmd), used by every run, baseline included:
#   CUB c = 0.1, r = 1.5    Stanford Cars c = 0.1, r = 2.0    FGVC-Aircraft c = 0.05, r = 1.5
EPOCHS=${EPOCHS:-200}
case "$DATASET" in
  cub)      US=1.0; HMW=1.0; C=0.1;  CR=1.5 ;;
  scars)    US=0.5; HMW=0.5; C=0.1;  CR=2.0 ;;
  aircraft) US=0.5; HMW=0.5; C=0.05; CR=1.5 ;;
  *) echo "unknown DATASET=$DATASET" >&2; return 1 2>/dev/null || exit 1 ;;
esac
UPSTREAM=(--dataset_name "$DATASET" --batch_size 128 --grad_from_block 10 --epochs "$EPOCHS" --base_model vit_dino
          --num_workers 4 --use_ssb_splits True --sup_con_weight 0.35 --weight_decay 5e-5
          --contrast_unlabel_only False --transform imagenet --lr 0.1 --eval_funcs v2
          --c "$C" --cr "$CR" --dino v2 --mlp_out_dim 8192 --unsupervised_smoothing "$US" --hyper_max_weight "$HMW"
          --seed "$SEED" --dist_impl lowmem)
PILOT=(--split_train_val True --fix_label_order True)
FC_CUTS=()
if [ -f cutoffs.json ]; then
  read -r FA FB < <(python -c "import json; d = json.load(open('cutoffs.json')).get('$DATASET'); print(*(d and (d['f_a'], d['f_b'])) or ('', ''))")
  if [ -n "$FA" ]; then FC_CUTS=(--fc_f_a "$FA" --fc_f_b "$FB"); fi
fi

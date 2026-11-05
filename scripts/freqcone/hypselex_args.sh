# Sourced by the run scripts. Needs $DATASET (default scars); $EPOCHS defaults to 200. Defines:
#   UPSTREAM  upstream Hyp-SelEx exactly as scripts/train_HypSelEx.cmd runs it (DINOv2 row of the dataset), including
#             that row's curvature c and clip radius r, plus --dist_impl lowmem (same distances as upstream, O(N^2)
#             memory; tests_freqcone/test_dist_lowmem.py). Every run passes its own --seed.
#   CONFIRM   flags of the confirmation runs B8-B10, F8-F10: the fix of SelEx's pseudo-label order bug (SelEx issue
#             #11), test after every epoch, test numbers kept out of stdout. No validation split: 100 % of the labelled
#             images, as upstream, so that the numbers compare with the published ones
#   PILOT     flags of the pilot runs R0-R3 and the probes: CONFIRM plus GCD's known-class validation split, which
#             serves only to choose the configuration
#   FC_CUTS   --fc_f_a / --fc_f_b from cutoffs.json (scripts/freqcone/compute_cutoffs.py); empty if missing
#
# Upstream DINOv2 rows of scripts/train_HypSelEx.cmd (used by every run, baseline included):
#   Stanford Cars: c 0.1, r 2.0, unsupervised smoothing 0.5, hyper_max_weight 0.5
#   CUB:           c 0.1, r 1.5, smoothing 1.0, hyper_max_weight 1.0    Aircraft: c 0.05, r 1.5, 0.5, 0.5
#   all: batch 128, last 2 blocks trained (grad_from_block 10), 200 epochs, lr 0.1, weight decay 5e-5, D = 8192
DATASET=${DATASET:-scars}
EPOCHS=${EPOCHS:-200}
case "$DATASET" in
  scars)    US=0.5; HMW=0.5; C=0.1;  CR=2.0 ;;
  cub)      US=1.0; HMW=1.0; C=0.1;  CR=1.5 ;;
  aircraft) US=0.5; HMW=0.5; C=0.05; CR=1.5 ;;
  *) echo "unknown DATASET=$DATASET" >&2; return 1 2>/dev/null || exit 1 ;;
esac
UPSTREAM=(--dataset_name "$DATASET" --batch_size 128 --grad_from_block 10 --epochs "$EPOCHS" --base_model vit_dino
          --num_workers 4 --use_ssb_splits True --sup_con_weight 0.35 --weight_decay 5e-5
          --contrast_unlabel_only False --transform imagenet --lr 0.1 --eval_funcs v2
          --c "$C" --cr "$CR" --dino v2 --mlp_out_dim 8192 --unsupervised_smoothing "$US" --hyper_max_weight "$HMW"
          --dist_impl lowmem)
CONFIRM=(--fix_label_order True --eval_every_epoch True --hide_test_stdout True)
PILOT=("${CONFIRM[@]}" --split_train_val True)
FC_CUTS=()
if [ -f cutoffs.json ]; then
  read -r FA FB < <(python -c "import json; d = json.load(open('cutoffs.json')).get('$DATASET'); print(*(d and (d['f_a'], d['f_b'])) or ('', ''))")
  if [ -n "$FA" ]; then FC_CUTS=(--fc_f_a "$FA" --fc_f_b "$FB"); fi
fi

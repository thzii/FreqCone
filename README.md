# FreqCone trên Hyp-SelEx (Stanford Cars, DINOv2)

## 1. Tổng quan

Repo này là code chính thức của **HypCD** cộng với **FreqCone**.
- **HypCD:** Liu, He, Han, *Hyperbolic Category Discovery*, CVPR 2025, arXiv 2504.06120. Commit upstream `ffc895b`, giấy phép MIT (`LICENSE`). README gốc nằm ở `docs/UPSTREAM_README.md`.
- **FreqCone:** coi ảnh lọc thông thấp là "cha" của chính ảnh đó, rồi buộc điểm Poincaré của ảnh nét nằm trong *entailment cone* của các view mờ. Các view mờ đặt ở độ sâu cố định trong quả cầu.
  - FreqCone chỉ thêm loss lúc huấn luyện; suy luận giữ nguyên.
  - Bản thảo: `paper/en/freqcone_draft.pdf`.
  - Thí nghiệm kiểm chứng cơ chế: `mechanism_study/`.
  - Kế hoạch hiện hành: `docs/PLAN.md`.

**Phạm vi:** chỉ hướng Hyp-SelEx (đã sửa bug thứ tự nhãn) cộng FreqCone, trên **Stanford Cars**, với DINOv2 ViT-B/14.

**Quan hệ với code upstream.** Mọi thay đổi trên file upstream đều là khối chèn giữa `# >>> freqcone` và `# <<< freqcone`. Khi để mọi cờ ở mặc định, code trùng từng byte với `ffc895b`; điều này được kiểm bởi `tests_freqcone/test_upstream_equivalence.py`.

**Trạng thái:**
- Chưa có số SSB nào.
- Chưa chạy trên RTX 3090.
- Các con số trong README chỉ là tham số đã cố định trước, không phải kết quả.

## 2. Yêu cầu

**Phần cứng và hệ điều hành:**
- GPU mục tiêu: RTX 3090 24 GB. Chưa chạy thử; bước profile (mục 8) đo bộ nhớ và thời gian thật.
- Linux, driver hỗ trợ CUDA 12.1, Python 3.10.

**Môi trường huấn luyện.** Các phiên bản theo `requirements.txt` của upstream: torch 2.4.1, torchvision 0.19.1, xformers 0.0.28.post1, numpy 1.24.4, scipy 1.10.1, scikit-learn 1.1.1, kmeans-pytorch 0.3, tensorboard 2.14.0, matplotlib 3.5.2, pandas 2.0.3.

```bash
conda create -n freqcone python=3.10 -y
conda activate freqcone
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt -r requirements-dev.txt
```

**Thí nghiệm cơ chế** chạy trên CPU, trong một môi trường riêng vì JAX cần numpy ≥ 1.26:

```bash
pip install -r mechanism_study/requirements.txt
```

## 3. Lấy code

```bash
git clone git@github.com:<tài-khoản>/freqcone.git && cd freqcone
git checkout pilot-v1.2          # hoặc pilot-v2 sau bước profile
git rev-parse HEAD               # so với hash được gửi kèm tag
git remote add upstream https://github.com/Visual-AI/HypCD.git
git remote set-url --push upstream DISABLED      # upstream chỉ để fetch, không bao giờ push
```

**Hai remote:**
- `origin` là repo riêng tư. Sau mỗi mốc (cut-off, profile, selection, kết quả), commit trên 3090, push lên `origin`, rồi `git pull` ở máy kia.
- `upstream` là repo công khai của Visual-AI; chỉ dùng để fetch hoặc so sánh.

**Dự phòng** khi máy 3090 không vào được GitHub:
- Ở máy có repo: `git bundle create freqcone.bundle --all`.
- Ở máy 3090: `git clone freqcone.bundle freqcone`.

Cách này giữ nguyên hash commit.

**Cấu trúc thư mục:**

```
freqcone/                    # repo git (lịch sử HypCD + nhánh freqcone)
├── train/ data/ hyptorch/ models/ methods/ project_utils/ config.py   # HypCD upstream (+ khối chèn FreqCone)
├── freqcone/                # module FreqCone (freqcone_torch.py: bản duy nhất), hook, công cụ chọn và báo cáo
├── scripts/freqcone/        # compute_cutoffs.py, smoke_gpu.sh, profile_gpu.sh, pilot_cars.sh, hypselex_args.sh
├── tests_freqcone/          # pytest (CPU; 2 test cần CUDA)
├── mechanism_study/         # thí nghiệm tổng hợp: bằng chứng cho phần lý thuyết và cơ chế (kết quả thật)
├── paper/en/                # bản thảo tiếng Anh + bảng và macro sinh tự động
├── docs/                    # PLAN.md (kế hoạch hiện hành), UPSTREAM_README.md
└── runs/                    # kết quả: mỗi run một thư mục (mục 12)
```

## 4. Dữ liệu Stanford Cars

**Nguồn tải.** Trang gốc (`ai.stanford.edu/~jkrause/cars`) không còn phục vụ file, và `torchvision.datasets.StanfordCars(download=True)` cũng không tải được. Hãy dùng một bản sao có đủ các file dưới đây. Lưu ý về `cars_test_annos_withlabels.mat` (bản có nhãn của tập test):
- Nhiều bản sao không kèm file này, nên phải lấy riêng.
- Upstream đọc nó khi dựng dữ liệu.
- Tập test rời được dùng cho tiêu chí chọn epoch của upstream, tức là cổng G0.

**Cấu trúc** mà `data/stanford_cars.py` đọc:

```
<car_root>/
├── cars_train/00001.jpg …        8144 ảnh
├── cars_test/00001.jpg …         8041 ảnh
└── devkit/
    ├── cars_train_annos.mat
    └── cars_test_annos_withlabels.mat
```

**Split SSB:** dùng `data/ssb_splits/scars_osr_splits.pkl`, đã có sẵn trong repo (98 lớp known, 98 lớp novel).

**Kiểm tra số ảnh và SHA của danh sách file.** Chạy trên mọi máy và so hash:

```bash
python -m freqcone.data_report --dataset scars --json runs/data_report_scars.json
```

Lệnh này in số ảnh của từng phần: labelled, validation, unlabelled (chia known và novel), test. Nó cũng in SHA-256 của từng danh sách file. Commit `runs/data_report_scars.json`.

**Chạy thử khi chưa có dữ liệu thật** (ảnh ngẫu nhiên, chỉ để thử lệnh):

```bash
python -m freqcone.synthetic_data --out /tmp/fake_scars
python -m freqcone.data_report --dataset scars --root /tmp/fake_scars
python scripts/freqcone/compute_cutoffs.py --dataset scars --root /tmp/fake_scars --out /tmp/fake_cutoffs.json --readme ''
```

## 5. DINOv2

- **Biến thể:** ViT-B/14 có 4 register token, file `dinov2_vitb14_reg4_pretrain.pth`. Đây đúng là file mà `config.py` của upstream trỏ tới, và `models/vision_transformer2.py` (`vit_base(num_register_tokens=4)`) dựng.
- **Tải:**

  ```bash
  wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth -P <PRETRAINED_PATH>/
  sha256sum <PRETRAINED_PATH>/dinov2_vitb14_reg4_pretrain.pth
  ```

- **SHA-256 mong đợi:** chưa có giá trị tham chiếu. Ghi lại giá trị ở lần tải đầu và so trên mọi máy. Mỗi `run.json` tự ghi SHA của checkpoint vào trường `dinov2_checkpoint`.

## 6. `config.py`

Điền các đường dẫn. Đây là file duy nhất được phép sửa cục bộ (preflight cho phép), và không commit đường dẫn của máy.

```python
cub_root = '/data/cub'
car_root = '/data/cars'                     # thư mục ở mục 4
aircraft_root = '/data/fgvc-aircraft-2013b'
cifar_10_root = '/data/cifar10'             # các tập khác: chỉ cần đường dẫn hợp lệ nếu dùng
cifar_100_root = '/data/cifar100'
imagenet_root = '/data/imagenet'
herbarium_dataroot = '/data/herbarium_19'
pets_root = '/data/oxford_pets'
osr_split_dir = 'data/ssb_splits'           # giữ nguyên
dino_pretrain_path = '/models/dino_vitb16_pretrain.pth'                 # DINO v1, không dùng
dinov2_pretrain_path = '/models/dinov2_vitb14_reg4_pretrain.pth'       # mục 5
feature_extract_dir = 'osr_novel_categories/extracted_features_public_impl'
exp_root = '/work/freqcone_outputs'         # upstream lưu model.pt mỗi epoch và log TensorBoard ở đây
```

## 7. Kiểm tra trước khi train

```bash
python -m pytest tests_freqcone -q                        # phải đạt toàn bộ; 2 test cần CUDA chỉ chạy trên máy GPU
python -m freqcone.data_report --dataset scars --json runs/data_report_scars.json
python scripts/freqcone/compute_cutoffs.py --dataset scars # ghi cutoffs.json và bảng bên dưới
git add cutoffs.json README.md runs/data_report_scars.json
git commit -m "data: Stanford Cars cut-offs and data report"
bash scripts/freqcone/smoke_gpu.sh                         # 2 epoch trên 1/10 dữ liệu, 4 cấu hình
rm -rf runs/smoke_scars                                    # bắt buộc trước pilot (preflight kiểm tra)
```

**Cut-off của FreqCone** (SNR 5 / 10 dB, chu kỳ/ảnh ở 224 px):
- Script tính từ ảnh train, không đọc nhãn, và không dùng ảnh val.
- Định nghĩa và nguồn trích dẫn nằm trong `freqcone/cutoffs.py`: Rastegar et al., arXiv 2608.08963, §4 và Bảng 7.
- Metric là radial, đúng như bộ lọc `L_f` của FreqCone.
- Bài gốc không nêu metric, độ phân giải và cách chuẩn hóa. Các lựa chọn ở đây được cố định trước khi chạy và ghi trong file.

<!-- cutoffs:start -->
Chưa tính: chạy `compute_cutoffs.py` trên máy có dữ liệu. Bảng 7 của bài gốc (5 / 10 dB) cho Stanford Cars là 3 / 11. Giá trị tính được mới là giá trị dùng để train.
<!-- cutoffs:end -->

## 8. Profile và chọn mức ngân sách

```bash
bash scripts/freqcone/profile_gpu.sh          # 4 probe, có đánh giá mỗi epoch; rồi pilot_tools decide
cat runs/profile_scars/decision.json
git add runs/profile_scars README.md
git commit -m "data: Stanford Cars profile and budget decision"
git tag -a pilot-v2 -m "cut-offs and budget decision fixed before the runs"
git push origin freqcone --tags
```

**Luật quyết định** (cố định trong `freqcone/pilot_tools.py`):
- **Độ phân giải view thô:** dùng 224 px nếu probe 224 px chạy được ở batch 128 và thời gian mỗi epoch ≤ 1,8 lần baseline. Nếu không, dùng 112 px cho mọi run FreqCone.
- **Mức ngân sách:** chọn **mức A** nếu một run 200 epoch ≤ 10 giờ, ngược lại chọn **mức B** (mục 9).
- **Cách tính:** `GPU-giờ = Σ_run 200 × t_epoch`.

<!-- pilot-decision:start -->
Chưa quyết định: chạy `profile_gpu.sh` trên máy 3090.
<!-- pilot-decision:end -->

## 9. Lệnh train

**Cách khuyến nghị:**

```bash
bash scripts/freqcone/pilot_cars.sh
```

Script chạy đúng thứ tự, tự resume, dừng ở các cổng và tự commit `runs/selection.json`.
- **Mức A:**
  1. R0 (baseline, seed 7), rồi G0.
  2. R1, R2, R3 (FreqCone, seed 7).
  3. Chọn cấu hình, rồi G1.
  4. Xác nhận: `base_s8`–`base_s10` và `fc_s8`–`fc_s10`.
- **Mức B:** `base_s8` rồi G0, sau đó `base_s9`, `base_s10` và `fc_s8`–`fc_s10`. FreqCone dùng cấu hình cố định λ_ent = 0,03, A1, v2, không tinh chỉnh trên SSB.
- **R0-repro** (upstream nguyên trạng, seed 1) chỉ chạy khi G0 không đạt.

**Chạy tay** (copy-paste, từ gốc repo):

```bash
read FA FB < <(python -c "import json; d = json.load(open('cutoffs.json'))['scars']; print(d['f_a'], d['f_b'])")
read COARSE < <(python -c "import json; print(json.load(open('runs/profile_scars/decision.json'))['coarse_res_flag'])")
BASE="--dataset_name scars --batch_size 128 --grad_from_block 10 --epochs 200 --base_model vit_dino --num_workers 4 \
--use_ssb_splits True --sup_con_weight 0.35 --weight_decay 5e-5 --contrast_unlabel_only False --transform imagenet \
--lr 0.1 --eval_funcs v2 --c 0.1 --cr 2.0 --dino v2 --mlp_out_dim 8192 --unsupervised_smoothing 0.5 \
--hyper_max_weight 0.5 --dist_impl lowmem"
PILOT="--split_train_val True --fix_label_order True --eval_every_epoch True --hide_test_stdout True"
DIAG="--fc_f_a $FA --fc_f_b $FB --coarse_res $COARSE --fc_K 0.1"
FC="--freqcone True --freqcone_variant v2 $DIAG --fc_w_dir 0.5 --fc_w_con 0.05 --fc_tau_a 0.5 --fc_tau_b 0.3"

# Pilot (mức A), seed 7
mkdir -p runs/R0 && python -m train.train_HypSelEx $BASE $PILOT $DIAG --seed 7 --run_id R0 --run_dir runs/R0 > runs/R0/train.log 2>&1
python -m freqcone.pilot_tools g0 runs/R0
mkdir -p runs/R1 && python -m train.train_HypSelEx $BASE $PILOT $FC --w_ent 0.03 --aperture A1 --seed 7 --run_id R1 --run_dir runs/R1 > runs/R1/train.log 2>&1
mkdir -p runs/R2 && python -m train.train_HypSelEx $BASE $PILOT $FC --w_ent 0.1 --aperture A1 --seed 7 --run_id R2 --run_dir runs/R2 > runs/R2/train.log 2>&1
mkdir -p runs/R3 && python -m train.train_HypSelEx $BASE $PILOT $FC --w_ent 0.1 --aperture A3 --seed 7 --run_id R3 --run_dir runs/R3 > runs/R3/train.log 2>&1

# Xác nhận, seed 8-10. W, A = cấu hình được chọn (mức A, đọc từ selection.json) hoặc cố định (mức B)
read W A < <(python -c "import json; s = json.load(open('runs/selection.json'))['selected']; print(s['w_ent'], s['aperture'])")   # mức A
# W=0.03; A=A1                                                                                                    # mức B
for s in 8 9 10; do
  mkdir -p runs/base_s$s && python -m train.train_HypSelEx $BASE $PILOT $DIAG --seed $s --run_id base_s$s --run_dir runs/base_s$s > runs/base_s$s/train.log 2>&1
  mkdir -p runs/fc_s$s && python -m train.train_HypSelEx $BASE $PILOT $FC --w_ent $W --aperture $A --seed $s --run_id fc_s$s --run_dir runs/fc_s$s > runs/fc_s$s/train.log 2>&1
done

# Chỉ khi G0 không đạt: upstream nguyên trạng (còn bug, không tách val), seed mặc định của upstream
mkdir -p runs/R0-repro && python -m train.train_HypSelEx $BASE $DIAG --eval_every_epoch True --hide_test_stdout True --seed 1 --run_id R0-repro --run_dir runs/R0-repro > runs/R0-repro/train.log 2>&1
```

## 10. Cờ dòng lệnh của FreqCone

Mặc định của mọi cờ đều tái lập đúng upstream.

| Cờ | Ý nghĩa | Mặc định | Giá trị dùng |
| --- | --- | --- | --- |
| `--freqcone` | Bật loss FreqCone | `False` | `True` ở nhánh FreqCone |
| `--freqcone_variant v2\|v3` | v2: nón không được đổi chuẩn của ảnh nét; v3: được đổi | `v2` | `v2` |
| `--w_ent` (`--fc_w_ent`) | λ_ent | `0.03` | 0.03 hoặc 0.1 (R1–R3); sau đó là giá trị được chọn |
| `--aperture A1\|A2\|A3` (`--fc_aperture`) | Preset (r_a, r_b) của view thô | `A1` | A1 hoặc A3 |
| `--coarse_res` (`--fc_coarse_size`) | Độ phân giải view thô; 0 = 224 px; 112 cần f_b < 56 | `0` | Theo `decision.json` |
| `--fc_f_a`, `--fc_f_b` | Cut-off (chu kỳ/ảnh) của x^a, x^b | `3`, `11` | Từ `cutoffs.json` |
| `--fc_w_dir`, `--fc_w_con` | λ_dir, λ_con | `0.5`, `0.05` | 0.5, 0.05 |
| `--fc_K` | Hằng số độ mở K | `0.1` | 0.1 |
| `--fc_tau_a`, `--fc_tau_b` | Nhiệt độ của L_dir | `0.5`, `0.3` | 0.5, 0.3 |
| `--fc_r_a`, `--fc_r_b` | Bán kính tùy chỉnh, thay preset | – | Không dùng |
| `--dist_impl upstream\|lowmem` | Ma trận khoảng cách hyperbolic: upstream (tensor N×N×D) hoặc dạng N×N cho cùng giá trị | `upstream` | `lowmem` |
| `--fix_label_order` | Sửa bug thứ tự nhãn của SelEx (issue #11) | `False` | `True` (trừ R0-repro) |
| `--split_train_val` | Giữ 20 % ảnh có nhãn của mỗi lớp known làm val | `False` | `True` (trừ R0-repro) |
| `--eval_every_epoch` | Val và test sau mỗi epoch, với trạng thái ngẫu nhiên cô lập | `False` | `True` |
| `--hide_test_stdout` | Không in số test ra stdout và không ghi vào TensorBoard | `False` | `True` |
| `--run_dir` | Thư mục kết quả `runs/<run_id>` | `''` | `runs/<run_id>` |
| `--resume` | Tiếp tục từ `ckpt_resume.pt` | `False` | Tự động trong `pilot_cars.sh` |
| `--profile_steps` | Probe: train N bước, đo thời gian và bộ nhớ | `0` | 30 khi profile |
| `--smoke_frac` | Smoke: giữ mỗi ảnh thứ k | `0` | 10 khi smoke |
| `--run_id`, `--run_tag`, `--run_attempt` | Định danh run | – | Do script đặt |
| `--fc_log_every`, `--fc_diag_images` | Chu kỳ ghi chẩn đoán; số ảnh cho hình học cuối | `50`, `2048` | Mặc định |

## 11. Siêu tham số và giao thức

**Nền.** Lấy theo dòng DINOv2 cho Stanford Cars trong `scripts/train_HypSelEx.cmd` của upstream, áp cho cả hai nhánh:

| Tham số | Giá trị |
| --- | --- |
| Backbone | DINOv2 ViT-B/14-reg4; train 2 block cuối (`grad_from_block 10`) |
| Batch, epoch | 128, 200 |
| Optimizer | SGD, lr 0.1, momentum 0.9, weight decay 5e-5, cosine về lr × 1e-3 |
| Projection | DINOHead, D = 8192 |
| Hyperbolic | c = 0.1, clip r = 2.0; α tăng tuyến tính tới `hyper_max_weight` = 0.5 |
| SelEx | `unsupervised_smoothing` 0.5, `sup_con_weight` 0.35, temperature 1.0 |
| Augmentation | `imagenet`: resize 256, random crop 224, flip, ColorJitter; 2 view |
| Split | SSB 98 known / 98 novel, 50 % ảnh known có nhãn |

**FreqCone v2:**
- λ_dir = 0.5, λ_con = 0.05, τ_a = 0.5, τ_b = 0.3, K = 0.1.
- Cut-off lấy từ `cutoffs.json`.
- Độ mở dùng một trong ba preset:

| Preset | r_a | r_b | ψ_a | ψ_b | δ(a) / δ(b) ở c = 0.1 |
| --- | --- | --- | --- | --- | --- |
| A1 | 0.120 | 0.200 | 55.2° | 28.7° | 0.763 / 1.282 |
| A2 | 0.104 | 0.136 | 72.0° | 46.2° | 0.660 / 0.866 |
| A3 | 0.100 | 0.114 | 81.9° | 60.0° | 0.635 / 0.724 |

- `FreqConeLoss` báo lỗi nếu nón suy biến, tức nếu không thỏa ε_K < r_a < r_b < tanh(√c·r), với ε_K = 0.099 khi K = 0.1.
- A2 và A3 thay đổi đồng thời độ mở *và* độ sâu của view thô, nên hai hiệu ứng này bị trộn.

**Bug thứ tự nhãn** (SelEx issue #11, tác giả xác nhận ngày 20/10/2026):
- **Cơ chế bug:**
  - Ở các mức phân cấp ≥ 1, `extract_labeled_protos` tách đặc trưng theo mask "known dự đoán".
  - Hàm `fit_mix` trả nhãn theo thứ tự `[l_feats; u_feats]`.
  - Trong khi đó, nơi đọc nhãn coi chúng theo thứ tự của loader.
- **Hệ quả:** nhãn thô bị gắn nhầm ảnh và làm rò thông tin lớp qua thứ tự file.
- **Sửa:** `--fix_label_order True` trả nhãn về thứ tự của loader (`freqcone/hypcd.py: loader_order`).
- **Kiểm tra:** `tests_freqcone/test_label_order.py`; có thêm một test CUDA chạy `K_Means` thật.
- **Báo cáo:** chênh lệch do sửa bug (R0-repro so với R0) được báo riêng với chênh lệch do FreqCone.

**Chỉ số val.** Đây là chính quy trình dự đoán của nền, áp lên ảnh val:
- SS-k-means của upstream, với ảnh labelled train làm điểm neo, k bằng số lớp known.
- Chấm bằng `cluster_acc` của upstream, tức cùng phép ghép Hungarian mà "All" của giao thức test dùng.
- Dự đoán từng ảnh ở epoch cuối được lưu lại để bootstrap.

**Chọn cấu hình** (mức A, `freqcone/select_pilot.py`):
- Ứng viên là R1–R3, chấm bằng val ở epoch cuối.
- Luật 1-SE: bootstrap B = 2000, paired và phân tầng theo lớp, seed 0.
- Phá hòa: λ_ent nhỏ hơn trước, rồi nón rộng hơn trước.

**Cổng:** chỉ dùng hàng "Epoch cuối".

| Cổng | Điều kiện đạt |
| --- | --- |
| G0 | Hàng "tiêu chí upstream" của R0 (mức A) hoặc `base_s8` (mức B): \|All − 83,8\| ≤ 2,0 (Hyp-SelEx, Stanford Cars, DINOv2: 83,8 / 93,3 / 79,2, Bảng 1 của HypCD) |
| G1 | ΔNew ≥ +1,0 và ΔAll ≥ −0,5 so với R0 |
| Kết luận cuối (seed 8–10) | ΔNew trung bình ≥ +1,0; thắng ≥ 2/3 seed về New; ΔAll trung bình ≥ −0,5 |

## 12. Đầu ra của mỗi run

```
runs/<run_id>/
├── run.json            thiết lập, provenance, SHA (DINOv2, danh sách dữ liệu), val epoch cuối, hình học, chi phí. KHÔNG có số test
├── epochs_val.jsonl    mỗi epoch: epoch, val_acc, losses, fc{depth_full, in_cone, clip_frac, angle_b_u, depth_b}, lambda_distance, lr, time_s
│                       (dòng cuối thêm val_per_sample_true / _pred, val_list_sha256)
├── epochs_test.jsonl   mỗi epoch: epoch, all, old, new (v2, transductive), upstream_disjoint_test{all, old, new}
├── train.log           stdout (không có số test khi --hide_test_stdout)
├── ckpt_resume.pt      trạng thái sau epoch gần nhất (không commit)
├── g0.json             (run baseline dùng cho G0)
└── summary.json        (summarize_run --json)
runs/selection.json, runs/g1.json, runs/compare_final.{md,json,png}, runs/profile_scars/<probe>/run.json, runs/profile_scars/decision.json
```

- Epoch được đánh số 1–200.
- `run.json` theo schema `freqcone-run/2`. Kiểm tra bằng `python -m freqcone.pilot_tools check runs/<run_id>`; lệnh này cũng kiểm rằng `run.json` không chứa số test.

## 13. Theo dõi khi đang train

**Các dòng `[FreqCone]` trong `train.log`** (mỗi 50 bước):
- `depth_full`: độ sâu trung bình δ_c(u) của ảnh nét.
- `in_cone`: tỉ lệ ảnh nét nằm trong nón của view b.
- `clip_frac`: tỉ lệ đặc trưng bị clip; ảnh bị clip có độ sâu 2r = 4,0.
- `angle_b_u`: góc giữa b và u.
- `fc_dir`, `fc_ent`, `fc_con`: ba thành phần loss.

**Ngưỡng độ sâu:**
- δ(b) = (2/√c)·artanh(r_b). Với c = 0,1, ngưỡng là 1,282 ở A1 và 0,724 ở A3.
- Dòng `WARNING … shallower than the coarse view b` nghĩa là nón đang vô hiệu (chế độ nông). Đây là kết quả cần báo cáo, không phải lỗi.

**Nguyên tắc khi pilot đang chạy:**
- **Không mở `epochs_test.jsonl` của R1–R3 trước khi `runs/selection.json` được commit.**
- `summarize_run` và `compare_runs` tự từ chối đọc các file đó trước thời điểm ấy.
- `train.log` và TensorBoard không chứa số test, vì `--hide_test_stdout True` bỏ chúng đi.

## 14. Sau khi train

```bash
python -m freqcone.select_pilot runs                                # mức A: chỉ đọc val; ghi runs/selection.json
git add runs/selection.json && git commit -m "chore: lock the Stanford Cars pilot selection (validation only)"
python -m freqcone.summarize_run runs/R0 --json                     # bảng một run
python -m freqcone.compare_runs --base runs/R0 --fc runs/R2 --gate g1 --out runs/g1.json      # thay R2 bằng run được chọn
python -m freqcone.compare_runs --base runs/base_s8 runs/base_s9 runs/base_s10 \
    --fc runs/fc_s8 runs/fc_s9 runs/fc_s10 --gate final --plot runs/compare_final.png --out runs/compare_final.json
```

**Cách đọc bảng của `summarize_run`:**

| Hàng | Vai trò |
| --- | --- |
| Epoch cuối | **Kết quả chính**; mọi cổng và kết luận đều dùng hàng này |
| Epoch có val cao nhất | Kết quả phụ, hợp lệ vì chọn bằng val |
| Tiêu chí upstream (Old cao nhất trên test rời) | Chỉ để đối chiếu số công bố (G0) |
| Oracle max All / Old / New | Chọn epoch bằng chính test, mỗi chỉ số lấy ở một epoch khác nhau: số bị thổi phồng, chỉ để tham khảo |

`compare_runs` in hàng oracle ở một bảng riêng, có tiêu đề "Oracle (chọn bằng test, tham khảo)".

## 15. Tiếp tục khi bị ngắt, và chạy lại

**Tiếp tục:**
- Sau mỗi epoch, hệ thống lưu `runs/<id>/ckpt_resume.pt`, gồm:
  - mô hình, projection head, optimizer, scheduler;
  - trạng thái ngẫu nhiên (Python, NumPy, torch, CUDA);
  - lịch sử, và số dòng của hai file JSONL.
- Để tiếp tục, chạy lại đúng lệnh cũ và thêm `--resume True`. `pilot_cars.sh` tự làm việc này.
- Phần tiếp tục trùng khớp với chạy liền mạch (test `test_resume_continues_exactly` trên CPU). Trên GPU có thể lệch nhỏ do kernel không tất định.
- Logic "best checkpoint" của upstream bắt đầu lại sau khi resume; giao thức này không dùng nó.

**Chạy lại từ đầu:**
- Đổi tên thư mục cũ (`mv runs/R1 runs/R1.old`), không xóa, rồi chạy lại cùng seed.
- `pilot_cars.sh` tự đổi tên các run dở dang không có checkpoint, và ghi số lần thử vào trường `attempt`.

## 16. Xử lý sự cố

Không tự ý đổi cấu hình.

| Tình huống | Cách xử lý |
| --- | --- |
| Loss NaN hoặc inf | Run dừng với `FloatingPointError`. Giữ log và `ckpt_resume.pt`, rồi báo lại. Không tự hạ λ hay learning rate. |
| Hết bộ nhớ (OOM) | Áp luật độ phân giải (chuyển sang 112 px). Nếu vẫn OOM thì báo. Chỉ được giảm batch cho cả hai nhánh, và phải ghi rõ. |
| Chế độ nông (độ sâu < δ(b), `in_cone` ≈ 0) | Không phải lỗi; đây là kết quả cần báo cáo |
| G0 không đạt | Script chạy R0-repro. Nếu R0-repro đạt: chênh lệch đến từ việc sửa bug, ghi lại và đi tiếp. Nếu R0-repro cũng không đạt: dừng và báo. |
| Vượt ngân sách quá 30 % | `pilot_tools budget` dừng script; báo trước khi chạy tiếp |
| Preflight báo lỗi | Code khác tag, file chưa commit, hoặc còn kết quả smoke: sửa theo thông báo |

## 17. Tái lập

- **Seed:** pilot dùng 7; xác nhận dùng 8, 9, 10; R0-repro dùng 1 (mặc định của upstream). Giai đoạn sau (CUB, Aircraft) cũng dùng 8–10.
- **Provenance trong `run.json`:**
  - commit, tag và các file đã sửa;
  - lệnh chạy đầy đủ;
  - SHA của checkpoint DINOv2 và của các danh sách dữ liệu.
- **Đánh giá mỗi epoch không ảnh hưởng tới huấn luyện:** trạng thái ngẫu nhiên được lưu rồi khôi phục quanh bước đánh giá, và k-means dùng seed = `seed*1000 + epoch`. Có test `test_per_epoch_evaluation_does_not_change_training`.
- **Nguồn ngẫu nhiên còn lại:**
  - một số kernel CUDA không tất định, dù upstream đã bật `cudnn.deterministic = True`;
  - thứ tự số thực khi cộng dồn trên GPU.
- **Thời gian và GPU-giờ thực tế:**
  - ước tính lấy từ `decision.json` (mục 8);
  - giá trị đo được nằm trong `cost` của từng `run.json`;
  - chưa có số nào.

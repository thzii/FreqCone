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
- **Không đặt repo trong OneDrive** hay thư mục đồng bộ tương tự: mỗi run ghi lại `ckpt_resume.pt` (khoảng 470 MB) sau mỗi epoch. Trên máy chạy thật, clone ra thư mục ngoài OneDrive, ví dụ `D:\work\freqcone` hoặc `~/work/freqcone`.

**Bộ nhớ GPU của bước đánh giá (ước lượng, chưa đo).** `test_kmeans` của upstream gọi `kmeans_pytorch.pairwise_distance`. Hàm này tính `(A - B) ** 2.0` trên tensor N×K×D, nên hai tensor cùng tồn tại một lúc. Với tập test rời đầy đủ của Cars (N = 8041, K = 196, D = 768, float32):
- khoảng 4,8 GB mỗi tensor;
- đỉnh khoảng 9,7 GB, chưa tính phần mô hình đang nằm trên GPU.

Không sửa code upstream. GPU 6 GB không đủ cho bước này trên dữ liệu đầy đủ. Bước profile chạy `test_kmeans` này nên đo được đỉnh thật (`peak_mem_gb`).

**Môi trường huấn luyện.** Các phiên bản theo `requirements.txt` của upstream: torch 2.4.1, torchvision 0.19.1, xformers 0.0.28.post1, numpy 1.24.4, scipy 1.10.1, scikit-learn 1.1.1, tensorboard 2.14.0, matplotlib 3.5.2, pandas 2.0.3. `kmeans_pytorch` cài riêng, theo commit ghim bên dưới.

```bash
conda create -n freqcone python=3.10 -y
conda activate freqcone
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt -r requirements-dev.txt
pip install --force-reinstall --no-deps "git+https://github.com/subhadarship/kmeans_pytorch@f7f36bd1cb4e3a761d73d584866d0a9c6b4d2805"
python -m freqcone.pilot_tools env        # phải in "environment OK"
```

- **Bắt buộc cài `kmeans_pytorch` từ GitHub, ghim commit `f7f36bd`** (master, 31/03/2022; upstream cài từ repo này). `requirements.txt` ghim bản PyPI 0.3, bản này thiếu tham số `tqdm_flag` mà `test_kmeans` của upstream truyền vào. Thiếu bước này, run sẽ dừng với `TypeError` ngay cuối epoch 1.
- **Không dùng `--editable` hay chép tay.** `pilot_tools env` kiểm cả commit (pip ghi trong `direct_url.json`, chỉ khi cài bằng `git+...`) lẫn tham số `tqdm_flag`. `smoke_gpu.sh` và `pilot_cars.sh` tự gọi lệnh này.
- **Bản GitHub cần `numba`** (`requirements.txt` đã có).
- **Windows:** với NumPy 1.x, `np.arange` là int32 và `K_Means` của upstream báo lỗi kiểu ở bước gán cân bằng. Một khối chèn có đánh dấu ép chỉ số đó về int64; trên Linux khối này không đổi gì (`test_k_means_runs_with_an_int32_arange`).

**Thí nghiệm cơ chế** chạy trên CPU, trong một môi trường riêng vì JAX cần numpy ≥ 1.26:

```bash
pip install -r mechanism_study/requirements.txt
```

## 3. Lấy code

```bash
git clone https://github.com/thzii/FreqCone.git ~/work/freqcone && cd ~/work/freqcone    # ngoài OneDrive (mục 2); repo riêng tư, nhánh main
git describe --tags --exact-match    # phải in pilot-v1.4 (pilot-v2 sau bước profile), rồi so hash:
git rev-parse HEAD                   # so với hash được gửi kèm tag
git remote add upstream https://github.com/Visual-AI/HypCD.git
git remote set-url --push upstream DISABLED      # upstream chỉ để fetch, không bao giờ push
```

**Hai remote:**
- `origin` là repo riêng tư, chỉ có nhánh `main`. Sau mỗi mốc (cut-off, profile, selection, kết quả), commit trên 3090 ở nhánh `main`, chạy `git push origin main --tags`, rồi pull ở máy kia. Đừng `git checkout <tag>`: HEAD rời nhánh thì commit sẽ không lên `main`.
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
├── tests_freqcone/          # pytest (CPU; 8 test cần CUDA)
├── mechanism_study/         # thí nghiệm tổng hợp: bằng chứng cho phần lý thuyết và cơ chế (kết quả thật)
├── paper/en/                # bản thảo tiếng Anh + bảng và macro sinh tự động
├── docs/                    # PLAN.md (kế hoạch hiện hành), UPSTREAM_README.md, DATA_CHECK.md (sinh trên máy có dữ liệu)
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
python -m freqcone.data_report --dataset scars --json runs/data_report_scars.json --map runs/uq_map_scars.csv --md docs/DATA_CHECK.md
```

Lệnh này không đọc nội dung ảnh, không cần GPU và chỉ mất vài giây. Nó dừng với `AssertionError` nếu có kiểm tra nào không đạt.

**Báo cáo cho cả hai chế độ dữ liệu** (có tách val cho pilot, không tách val cho các run xác nhận):
- số ảnh: tổng train, labelled, unlabelled (chia known và novel), val, test rời;
- số lớp known / novel (dừng nếu khác 98 / 98);
- min–max số ảnh labelled mỗi lớp known;
- tỉ lệ labelled so với upstream (kỳ vọng ≈ 0,8 khi tách val, đúng 1 khi không tách);
- SHA-256 của từng danh sách ảnh, tính trên đường dẫn file dưới thư mục dữ liệu (`cars_train/00001.jpg`), không dùng `uq_idx`.

**Kiểm tra `uq_idx`:**
- Trên **mọi** ảnh của tập train, chạy đúng biểu thức tra nhãn của upstream, trước và sau khi đánh lại. Sau khi đánh lại, ảnh nào cũng phải ra đúng nhãn thật và đúng file của nó (100 %).
- Mọi loader phải cùng một số cho mỗi ảnh.
- Liệt kê mọi chỗ dùng `uq_idx` và dãy số mà chỗ đó thấy. Gặp nơi đọc `uq_idx` chưa được phân loại thì báo cáo dừng.
- `runs/uq_map_scars.csv` ghi ánh xạ cũ → mới của từng ảnh (kèm tập con, lớp, file). Mỗi run ghi SHA-256 của cùng ánh xạ vào `data_counts.uq_map_sha256`.

Commit `runs/data_report_scars.json`, `runs/uq_map_scars.csv` và `docs/DATA_CHECK.md`; preflight của pilot kiểm tra ba file này.

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
python -m freqcone.pilot_tools env                         # kmeans_pytorch đúng commit ghim
python -m pytest tests_freqcone -q                        # phải đạt toàn bộ; 8 test cần CUDA chỉ chạy trên máy GPU
python -m freqcone.data_report --dataset scars --json runs/data_report_scars.json --map runs/uq_map_scars.csv --md docs/DATA_CHECK.md
python scripts/freqcone/compute_cutoffs.py --dataset scars # ghi cutoffs.json và bảng bên dưới
git add cutoffs.json README.md runs/data_report_scars.json runs/uq_map_scars.csv docs/DATA_CHECK.md
git commit -m "data: Stanford Cars cut-offs and data report"
bash scripts/freqcone/smoke_gpu.sh                         # 2 epoch trên 1/10 dữ liệu, 6 cấu hình, cả hai chế độ dữ liệu
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
git push origin main --tags
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

**Val chỉ dùng để chọn cấu hình.**
- **Pilot (R0–R3, seed 7) có tách val**, để chọn cấu hình bằng luật 1-SE và xét G1. So sánh trong pilot vẫn công bằng vì cả hai nhánh cùng mất các ảnh đó.
- **Các run xác nhận (B8–B10, F8–F10) không tách val:** dùng 100 % ảnh labelled như upstream, cấu hình FreqCone lấy từ `selection.json`. Lý do: upstream huấn luyện trên toàn bộ ảnh labelled, nên chỉ các run này mới so được với số công bố.
- **G0 chạy đầu tiên, trên B8** (baseline, seed 8, không tách val, có `--fix_label_order`). B8 cũng là baseline seed 8 của giai đoạn xác nhận, nên không tốn thêm run.

**Thứ tự:**
- **Mức A (10 run):** B8 (G0) → R0, R1, R2, R3 (val) → chọn cấu hình → G1 → B9, B10, F8, F9, F10.
- **Mức B (6 run):** B8 (G0) → F8, B9, F9, B10, F10. Không có pilot và G1; FreqCone dùng cấu hình cố định λ_ent = 0,03, A1, v2, không tinh chỉnh trên SSB.
- **R0-repro** (upstream nguyên trạng, seed 1) chỉ chạy khi G0 không đạt, để xem chênh lệch có do bản sửa không.

**Chạy tay** (copy-paste, từ gốc repo):

```bash
read FA FB < <(python -c "import json; d = json.load(open('cutoffs.json'))['scars']; print(d['f_a'], d['f_b'])")
read COARSE < <(python -c "import json; print(json.load(open('runs/profile_scars/decision.json'))['coarse_res_flag'])")
BASE="--dataset_name scars --batch_size 128 --grad_from_block 10 --epochs 200 --base_model vit_dino --num_workers 4 \
--use_ssb_splits True --sup_con_weight 0.35 --weight_decay 5e-5 --contrast_unlabel_only False --transform imagenet \
--lr 0.1 --eval_funcs v2 --c 0.1 --cr 2.0 --dino v2 --mlp_out_dim 8192 --unsupervised_smoothing 0.5 \
--hyper_max_weight 0.5 --dist_impl lowmem"
CONFIRM="--fix_label_order True --eval_every_epoch True --hide_test_stdout True"    # không tách val
PILOT="$CONFIRM --split_train_val True"
DIAG="--fc_f_a $FA --fc_f_b $FB --coarse_res $COARSE --fc_K 0.1"
FC="--freqcone True --freqcone_variant v2 $DIAG --fc_w_dir 0.5 --fc_w_con 0.05 --fc_tau_a 0.5 --fc_tau_b 0.3"

# Chạy đầu tiên: B8, rồi G0
mkdir -p runs/B8 && python -m train.train_HypSelEx $BASE $CONFIRM $DIAG --seed 8 --run_id B8 --run_dir runs/B8 > runs/B8/train.log 2>&1
python -m freqcone.pilot_tools g0 runs/B8

# Pilot (mức A), seed 7, có tách val
mkdir -p runs/R0 && python -m train.train_HypSelEx $BASE $PILOT $DIAG --seed 7 --run_id R0 --run_dir runs/R0 > runs/R0/train.log 2>&1
mkdir -p runs/R1 && python -m train.train_HypSelEx $BASE $PILOT $FC --w_ent 0.03 --aperture A1 --seed 7 --run_id R1 --run_dir runs/R1 > runs/R1/train.log 2>&1
mkdir -p runs/R2 && python -m train.train_HypSelEx $BASE $PILOT $FC --w_ent 0.1 --aperture A1 --seed 7 --run_id R2 --run_dir runs/R2 > runs/R2/train.log 2>&1
mkdir -p runs/R3 && python -m train.train_HypSelEx $BASE $PILOT $FC --w_ent 0.1 --aperture A3 --seed 7 --run_id R3 --run_dir runs/R3 > runs/R3/train.log 2>&1

# Xác nhận, seed 8-10, không tách val. W, A = cấu hình được chọn (mức A, đọc từ selection.json) hoặc cố định (mức B)
read W A < <(python -c "import json; s = json.load(open('runs/selection.json'))['selected']; print(s['w_ent'], s['aperture'])")   # mức A
# W=0.03; A=A1                                                                                                    # mức B
for s in 9 10; do
  mkdir -p runs/B$s && python -m train.train_HypSelEx $BASE $CONFIRM $DIAG --seed $s --run_id B$s --run_dir runs/B$s > runs/B$s/train.log 2>&1
done
for s in 8 9 10; do
  mkdir -p runs/F$s && python -m train.train_HypSelEx $BASE $CONFIRM $FC --w_ent $W --aperture $A --seed $s --run_id F$s --run_dir runs/F$s > runs/F$s/train.log 2>&1
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
| `--split_train_val` | Giữ 20 % ảnh có nhãn của mỗi lớp known làm val | `False` | `True` ở pilot R0–R3; không bật ở B8–B10, F8–F10, R0-repro |
| `--eval_every_epoch` | Val và test sau mỗi epoch, với trạng thái ngẫu nhiên cô lập | `False` | `True` |
| `--hide_test_stdout` | Không in số test ra stdout và không ghi vào TensorBoard | `False` | `True` |
| `--run_dir` | Thư mục kết quả `runs/<run_id>` | `''` | `runs/<run_id>` |
| `--resume` | Tiếp tục từ `ckpt_resume.pt` | `False` | Tự động trong `pilot_cars.sh` |
| `--profile_steps` | Probe: train N bước, đo thời gian và bộ nhớ | `0` | 30 khi profile |
| `--smoke_frac` | Smoke: giữ mỗi ảnh thứ k của từng lớp (lớp nào cũng còn ảnh) | `0` | 10 khi smoke |
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
| Đánh giá | `eval_funcs v2` (Hungarian, transductive) |

- Các giá trị trên được áp qua mảng `UPSTREAM` trong `scripts/freqcone/hypselex_args.sh`.
- **Lưu ý về c:** paper HypCD ghi c = 0,05 cho Hyp-SelEx, nhưng dòng Stanford Cars trong `scripts/train_HypSelEx.cmd` của upstream dùng `--c 0.1`. Ở đây theo **script**, vì đó là cấu hình đã chạy ra số công bố. Mọi giá trị δ trong README tính ở c = 0,1.

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
- **Báo cáo:** chênh lệch do sửa bug (R0-repro so với B8, nếu R0-repro được chạy; hai run khác seed) được báo riêng với chênh lệch do FreqCone.

**`uq_idxs` khi giữ ảnh ra ngoài** (`--split_train_val`, `--smoke_frac`):
- **Cơ chế lỗi:** upstream tra nhãn giả bằng `preds[np.argsort(uq_index)[uq_idxs]]`, công thức này chỉ đúng khi `uq_idxs` của tập train là đúng 0..N−1. Bỏ ảnh val ra tạo khe hở trong dãy số.
- **Hệ quả nếu không sửa:** `IndexError` ở các bước đầu, hoặc nhãn giả gắn sai ảnh.
- **Sửa:** `FCRun` đánh số lại tập train theo thứ hạng (`freqcone/data_checks.py: compact_uq_idxs`), cả trên các bản sao mà loader đọc, nên mỗi ảnh chỉ có một số. Không có khe hở thì không đổi gì, nên các run không tách val và cấu hình upstream giữ nguyên. Số gốc nằm ở `source_uq_idxs`. Ảnh val giữ số gốc vì đã rời tập train. SHA của mọi danh sách ảnh tính theo file, không theo số.
- **Kiểm tra:** `tests_freqcone/test_merged_train.py` có test đối chứng trên dãy số chưa đánh lại; `data_report` kiểm trên mọi ảnh của dữ liệu thật (mục 4).

**Chặn lỗi sớm.** `FCRun` dừng run (`AssertionError`) trước khi train và trước mỗi lần SS-k-means nếu:
- loader không phải `MergedDataset`, hoặc không có ảnh labelled;
- một lớp known không có ảnh labelled, hoặc ảnh labelled thuộc lớp novel;
- `uq_idxs` của tập train không phải đúng 0..N−1, tức phép tra nhãn giả sẽ sai;
- tập val chồng lên labelled / unlabelled, hoặc chứa lớp novel.

Một epoch mà loss có giám sát không thấy ảnh labelled nào, hoặc có giá trị trung bình bằng 0, cũng dừng run (`RuntimeError`).

**Chỉ số val.** Đây là chính quy trình dự đoán của nền, áp lên ảnh val:
- SS-k-means của upstream, với ảnh labelled train làm điểm neo, k bằng số lớp known.
- Chấm bằng `cluster_acc` của upstream, tức cùng phép ghép Hungarian mà "All" của giao thức test dùng.
- Dự đoán từng ảnh ở epoch cuối được lưu lại để bootstrap.

**Chọn cấu hình** (mức A, `freqcone/select_pilot.py`):
- Ứng viên là R1–R3, chấm bằng val ở epoch cuối.
- Luật 1-SE: bootstrap B = 2000, paired và phân tầng theo lớp, seed 0.
- Phá hòa: λ_ent nhỏ hơn trước, rồi nón rộng hơn trước.

**Đánh giá theo chuẩn GCD (theo code HypCD).** Đây là cách các paper GCD (và HypCD) ra số, lặp lại đúng vòng train của upstream (`train/train_HypSelEx.py`, dòng 226–234 và 385–431):
- **Đầu mỗi epoch:** `extract_labeled_protos` chạy SS-k-means trên tập train (ảnh labelled làm neo) và chấm All / Old / New trên **phần unlabelled** theo v2. Trọng số lúc này là trọng số **sau epoch trước**.
- **Cuối mỗi epoch:** `test_kmeans` chạy trên **tập test chính thức, rời**. Upstream giữ checkpoint "best" khi Old ở đây **tăng ngặt** (bắt đầu từ 0; hòa thì giữ epoch sớm hơn).
- **Số báo cáo** là bộ All / Old / New mà upstream in ở dòng `Best Train Accuracies`, tức SS-k-means đầu epoch được chọn.
- **Ngoại lệ:** ở epoch có `(epoch + 1) % train_report_interval == 0` (với cấu hình này là epoch cuối), upstream thay bộ số đó bằng `test_kmeans` trên tập unlabelled, sau epoch.
- **Phép đo trước khi train ("epoch 0"):** SS-k-means đầu epoch 1 chạy trên trọng số DINOv2 chưa train. Vì vậy dòng epoch 1 có `weights_after_epoch = 0` và chính là phép đo này; không cần chạy thêm.
- **Tái dùng:** các hàm `extract_labeled_protos`, `test_kmeans`, `split_cluster_acc_v2` là của upstream, không viết lại. Hook `epoch_done` chỉ ghi lại nguyên văn các bộ số upstream đã tính.
- **Kiểm tra:** `tests_freqcone/test_gcd_standard.py` chạy chính hàm `train()` của upstream và đối chiếu hàng (1) với dòng `Best Train Accuracies` / `Best Train Epochs` upstream in ra. Test có hai trường hợp: hòa ở Old, và best rơi vào epoch cuối. Bản CUDA của test dùng hàm thật.

**Hai hàng chính** của mỗi run (`freqcone/summarize_run.py`):

| # | Hàng | Cách lấy | Dùng cho |
| --- | --- | --- | --- |
| (1) | **Chuẩn GCD (theo code HypCD)** | Như trên | So SOTA, G0 |
| (2) | **Epoch cuối** | `test()` của upstream (v2, transductive) sau epoch 200 | Kiểm tra độ vững |

Run xác nhận chỉ có hai hàng chính và các hàng oracle. Hàng "Epoch có val cao nhất" chỉ có ở pilot, vì run xác nhận không tách val.

- Hàng (1) chọn epoch bằng Old trên tập test rời. Đây là quy tắc của upstream: nó không chọn trên tập được báo cáo, nhưng vẫn là chọn bằng nhãn test. Vì vậy cần thêm hàng (2), không qua bất kỳ phép chọn nào.

**Cổng:** G1 và kết luận cuối phải đạt trên **CẢ HAI** hàng chính; G0 xét trên hàng (1).

| Cổng | Điều kiện đạt |
| --- | --- |
| G0 | Hàng (1) của B8: \|All − 83,8\| ≤ 2,0 (Hyp-SelEx, Stanford Cars, DINOv2: 83,8 / 93,3 / 79,2, Bảng 1 của HypCD). `pilot_tools g0` từ chối run có tách val |
| G1 | Trên mỗi hàng chính: ΔNew ≥ +1,0 và ΔAll ≥ −0,5 so với R0 |
| Kết luận cuối (B8–B10, F8–F10) | Trên mỗi hàng chính: ΔNew trung bình ≥ +1,0; thắng ≥ 2/3 seed về New; ΔAll trung bình ≥ −0,5 |

`compare_runs` từ chối so hai nhánh khác chế độ dữ liệu, từ chối G1 trên run không tách val và kết luận cuối trên run có tách val.

**So SOTA:**
- Dùng hàng (1): trung bình ± độ lệch chuẩn trên 3 seed (8–10), kèm số của từng seed.
- Đặt cạnh số công bố 83,8 / 93,3 / 79,2.
- **Số công bố của HypCD trên Cars là kết quả của một lần chạy**, không phải trung bình nhiều seed. Bảng của `compare_runs` ghi rõ điều này.

## 12. Đầu ra của mỗi run

```
runs/<run_id>/
├── run.json            thiết lập, provenance, SHA (DINOv2, danh sách dữ liệu), data_counts, val epoch cuối, hình học, chi phí. KHÔNG có số test
├── epochs_val.jsonl    mỗi epoch: epoch, val_acc, losses, sup_con{n_labelled, loss_distance, loss_angle},
│                       fc{depth_full, in_cone, clip_frac, angle_b_u, depth_b}, lambda_distance, lr, time_s
│                       (val_acc là null ở run không tách val; dòng cuối của pilot thêm val_per_sample_true / _pred, val_list_sha256)
├── epochs_test.jsonl   mỗi epoch: epoch,
│                         gcd{all, old, new, source, weights_after_epoch}   bộ số upstream gắn với epoch (hàng 1)
│                         disjoint_test{all, old, new}                       test_kmeans trên test rời (chọn epoch hàng 1)
│                         eval{all, old, new}                                test() v2 sau epoch (hàng 2, oracle)
├── train.log           stdout (không có số test khi --hide_test_stdout)
├── ckpt_resume.pt      trạng thái sau epoch gần nhất (không commit)
├── g0.json             (run baseline dùng cho G0)
└── summary.json        (summarize_run --json)
runs/selection.json, runs/g1.json, runs/compare_final.{md,json,png}, runs/profile_scars/<probe>/run.json, runs/profile_scars/decision.json
runs/data_report_scars.json, runs/uq_map_scars.csv (mục 4)
```

- Epoch được đánh số 1–200.
- `data_counts` gồm `n_labelled`, `n_unlabelled`, `n_val` (null nếu không tách val), `n_test_roi` (số ảnh, không phải kết quả), `labelled_per_known_min`, `labelled_per_known_max`, `uq_gaps_closed`, `uq_map_sha256`.
- `sup_con` là phần SupCon thuần có giám sát, trước khi cộng các cấp nhãn giả: số ảnh labelled đã đi qua (đếm cả lần lấy lặp của sampler) và giá trị trung bình theo ảnh, ở dạng khoảng cách Poincaré và dạng góc.
- `run.json` theo schema `freqcone-run/2`. Kiểm tra bằng `python -m freqcone.pilot_tools check runs/<run_id>`; lệnh này cũng kiểm rằng `run.json` không chứa số test.

## 13. Theo dõi khi đang train

**Các dòng `[FreqCone]` trong `train.log`** (mỗi 50 bước):
- `depth_full`: độ sâu trung bình δ_c(u) của ảnh nét.
- `in_cone`: tỉ lệ ảnh nét nằm trong nón của view b.
- `clip_frac`: tỉ lệ đặc trưng bị clip; ảnh bị clip có độ sâu 2r = 4,0.
- `angle_b_u`: góc giữa b và u.
- `fc_dir`, `fc_ent`, `fc_con`: ba thành phần loss.

**Dòng `[FreqCone] epoch N: supervised SupCon on K labelled images`** (mỗi epoch): K phải dương và giá trị trung bình phải khác 0.

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
python -m freqcone.compare_runs --base runs/B8 runs/B9 runs/B10 \
    --fc runs/F8 runs/F9 runs/F10 --gate final --plot runs/compare_final.png --out runs/compare_final.json
```

**Cách đọc bảng của `summarize_run`.** Bảng "Kết quả" có ba hàng ở pilot, hai hàng ở run xác nhận:

| Hàng | Vai trò |
| --- | --- |
| (1) Chuẩn GCD (theo code HypCD) | **CHÍNH**: so SOTA, G0, G1, kết luận cuối. Cột Epoch là epoch upstream chọn |
| (2) Epoch cuối | **CHÍNH**: kiểm tra độ vững; G1 và kết luận cuối phải đạt cả ở đây |
| Epoch có val cao nhất | Phụ, hợp lệ vì chọn bằng val. Chỉ có ở pilot |

- Dưới bảng là một dòng ghi nguồn số của hàng (1): `ss_kmeans_epoch_start` hoặc `test_kmeans_unlabelled`, và trọng số sau epoch nào.
- Bảng **"Oracle (chọn bằng test, tham khảo …)"** để riêng. Mỗi chỉ số max All / Old / New lấy ở một epoch khác nhau, nên số bị thổi phồng. Không dùng bảng này để so SOTA hay kết luận.

**Cách đọc bảng của `compare_runs`:**
- **Bảng Δ:** mỗi hàng chính có ΔAll / ΔOld / ΔNew (FreqCone − nền, điểm %), TB ± SD và số seed thắng về New. Cổng ghi đạt/không đạt cho từng hàng chính; chỉ đạt khi cả hai hàng đạt.
- **Bảng SOTA:** hàng (1) của mỗi nhánh, TB ± SD và từng seed, cạnh số công bố (một lần chạy).
- **Bảng oracle:** để riêng, chỉ tham khảo.

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
| `AssertionError` về labelled / val, hoặc `RuntimeError` "the supervised loss saw …" | Dữ liệu hoặc code tách val có vấn đề. Dừng, giữ log, báo lại. Không chạy tiếp. |
| `TypeError: kmeans() got an unexpected keyword argument 'tqdm_flag'`, hoặc `pilot_tools env` báo `not the pinned build` | Đang dùng `kmeans_pytorch` khác commit ghim, trên mọi hệ điều hành: cài lại đúng lệnh ở mục 2 |

**Chỉ gặp trên Windows** (máy phát triển; máy chạy thật dùng Linux):

| Hiện tượng | Nguyên nhân | Xử lý |
| --- | --- | --- |
| `OSError: [WinError 123]` trong `init_experiment` | Tên thư mục log của upstream chứa ký tự `\|` | Giữ nguyên upstream, nên không chạy trực tiếp `train_HypSelEx.py` trên Windows được |
| `RuntimeError: Index put requires the source and destination dtypes match` trong `K_Means` | `np.arange` là int32 với NumPy 1.x | Đã xử lý bằng khối chèn ép về int64 (mục 2) |
| Cảnh báo `KMeans is known to have a memory leak on Windows with MKL` | sklearn, MKL và OpenMP trên Windows | Chỉ là cảnh báo; đặt `OMP_NUM_THREADS=1` nếu muốn tắt |

**Diễn tập trên máy phát triển** (Windows, RTX 4050 6 GB dùng chung màn hình, RAM 7,2 GB). Dùng ảnh tổng hợp, trọng số ngẫu nhiên theo kiến trúc DINOv2 và cấu hình thu nhỏ riêng cho máy đó, không nằm trong repo. Số liệu của lần diễn tập không có ý nghĩa.
- **Đã xác nhận:**
  - `__main__` thật của `train/train_HypSelEx.py` chạy trọn cả 6 cấu hình của `smoke_gpu.sh`: hai chế độ dữ liệu, view thô 224 và 112 px, FreqCone v2 và v3.
  - `pilot_tools check` đạt trên cả 6 run; `select_pilot`, `summarize_run`, `compare_runs` chạy được trên các bản ghi đó.
  - Bảng của run không tách val không có hàng "val cao nhất". Các run cùng chế độ dữ liệu có cùng `uq_map_sha256`.
- **Chưa xác nhận:** DINOv2 thật, Stanford Cars thật, batch 128, thời gian và bộ nhớ thật. Các việc này phải chạy trên máy 3090.
- **OOM và segfault trong `kmeans_pytorch.pairwise_distance` trên máy đó là do cạn bộ nhớ hệ thống:** VRAM chỉ dùng tối đa 3,2 / 6,1 GiB, trong khi commit còn trống xuống tới 13 MiB và RAM trống xuống 157 MiB. Máy đó không đại diện cho 3090, không cần gỡ thêm.

## 17. Tái lập

- **Seed:** pilot (R0–R3) dùng 7; xác nhận (B8–B10, F8–F10) dùng 8, 9, 10; R0-repro dùng 1 (mặc định của upstream). Giai đoạn sau (CUB, Aircraft) cũng dùng 8–10.
- **Thư viện ngoài:** `kmeans_pytorch` ghim commit `f7f36bd`; `pilot_tools env` từ chối bản khác.
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

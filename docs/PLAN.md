# Kế hoạch v2: Hyp-SelEx + FreqCone trên Stanford Cars, DINOv2

> **Việc đầu tiên là gộp hai thư mục `261/HypCD` và `261/freqcone` thành một workspace duy nhất, `261/freqcone`** (mục 1A). Sau bước đó, mọi đường dẫn trong kế hoạch này đều tính từ `261/freqcone`.
> Kế hoạch này **thay thế** PLAN_P1_PILOT.md và PLAN_P1_ADDENDUM.md ở mọi chỗ mâu thuẫn. Những gì không nhắc lại ở đây vẫn giữ nguyên, ví dụ: luật 1-SE, preset A1–A3, ngưỡng δ(b) tính theo c và r_b, bug thứ tự nhãn, chuyển code bằng bundle.
> Nguyên tắc từ HANDOFF_PROMPT.md mục 0 vẫn bắt buộc:
> - không bịa số;
> - ghi luật trước khi chạy;
> - không chọn cấu hình bằng test;
> - số liệu sinh tự động từ JSON.

---

## 0. Thay đổi phạm vi theo yêu cầu mới của người dùng

| Trước | Bây giờ |
| --- | --- |
| Pilot trên CUB, P2 trên CUB, SCars và Aircraft, có thể hai nền | **Chỉ một hướng:** Hyp-SelEx đã sửa lỗi làm baseline, cộng FreqCone. **Chỉ Stanford Cars**, DINOv2 ViT-B/14 |
| Lưới pilot 9 run, gồm λ = 0, A2, v3 | Lưới tối thiểu 3 run FreqCone (mục 3). Ablation λ = 0, v3 và A2 để sang giai đoạn sau |
| Có FourEx làm đối chứng | **Bỏ FourEx khỏi các run và khỏi code.** Bài báo vẫn trích dẫn FourEx ở phần công trình liên quan; bỏ trích dẫn là sai chuẩn mực |
| Đánh giá test chỉ ở epoch cuối | **Đánh giá val và test ở mọi epoch**, rồi báo cáo các epoch đạt max All, max Old (Known), max New (Novel). Cách báo cáo cố định ở mục 2 |
| Workspace có nhiều thứ không liên quan | **Dọn sạch workspace** theo mục 1 (lưu trữ, không xóa vĩnh viễn) |

Mục tiêu trước mắt: chứng minh FreqCone có đóng góp trên Stanford Cars với bằng chứng đủ chặt để đưa vào bài báo.

Phải nói rõ với người dùng ở cuối giai đoạn này: một tập dữ liệu với 3 seed mới là bằng chứng *trên Cars*. Bài rank B nhiều khả năng sẽ bị yêu cầu thêm CUB và Aircraft cùng các ablation (mục 3.5).

---

## 1A. Gộp thành một workspace duy nhất: `261/freqcone`

**Vì sao đang có hai thư mục.**
- `261/freqcone` là gói nghiên cứu ban đầu: module PyTorch, thí nghiệm tổng hợp, kiểm tra toán, bản thảo. Gói này không có code GCD thật.
- `261/HypCD` là bản clone repo HypCD chính thức dùng để tích hợp FreqCone. `origin` của nó đang trỏ về repo công khai của Visual-AI.

Người dùng muốn **một** workspace để dễ quản lý và dễ push.

**Nguyên tắc gộp.** Giữ repo git của HypCD làm gốc, giữ nguyên lịch sử. Lý do:
- (a) test kiểm toán "tắt cờ thì trùng từng byte với upstream `ffc895b`" cần lịch sử này;
- (b) minh bạch với reviewer và đúng giấy phép HypCD, nên phải giữ file LICENSE và ghi công tác giả gốc;
- (c) các hash commit mà JSON đã ghi (`7d8c1c9`, …) vẫn còn nguyên giá trị.

Nội dung còn dùng của gói cũ được chuyển vào thành thư mục con.

**Các bước:**
1. Kiểm tra `git status` sạch trong `261/HypCD`. Tag `pilot-v1` đã có.
2. Đổi tên thư mục cũ `261/freqcone` thành `261/_archive_YYYYMMDD/freqcone_old/`. Không xóa.
3. Đổi tên `261/HypCD` thành `261/freqcone`. Đổi tên thư mục không làm thay đổi hash commit.
4. Chép từ `freqcone_old/` vào repo theo cấu trúc sau:

   ```
   261/freqcone/                 # repo git (lịch sử HypCD + nhánh freqcone)
   ├── <code HypCD upstream + phần chèn FreqCone + tests_freqcone/ + scripts/freqcone/>
   ├── mechanism_study/          # thí nghiệm tổng hợp: bằng chứng cho phần lý thuyết/cơ chế của bài
   │   ├── toy_experiment.py  verify_math.py  parity_check.py  aggregate.py
   │   ├── make_tables.py  make_tables_vi.py
   │   ├── run_sweep.sh  run_ablation_w0.sh  run_v3.sh
   │   └── results/  (pilot/, lambda_sweep/, results/, results_v3/, sweep.log: kết quả chạy thật, giữ nguyên)
   ├── paper/
   │   ├── en/  (freqcone_draft.tex + file sinh tự động)
   │   └── vi/  (freqcone_vi.tex + file sinh tự động)
   ├── docs/
   │   └── PLAN.md               # chỉ giữ kế hoạch hiện hành (file này)
   └── README.md                 # hướng dẫn train (mục 4)
   ```

5. **Chỉ một bản `freqcone_torch.py`.**
   - Nếu code tích hợp trong HypCD đã có module FreqCone riêng, đó là bản duy nhất.
   - So nó với `freqcone_old/freqcone_torch.py`. Mọi khác biệt phải là cố ý và được ghi lại, ví dụ mặc định `w_ent = 0.03`, preset A1–A3, assert miền nón.
   - Sửa `mechanism_study/parity_check.py` để import đúng bản đó, rồi chạy lại: chênh lệch loss so với JAX phải ≤ 1e-3.
6. Sửa đường dẫn trong `make_tables*.py` cho khớp cấu trúc mới. Chạy lại `make_tables.py` và `make_tables_vi.py`, rồi biên dịch lại `paper/en` (pdflatex) và `paper/vi` (xelatex). Mọi số trong hai bản thảo phải giữ nguyên.
7. Test kiểm toán upstream chỉ so các file thuộc upstream. Các thư mục mới (`mechanism_study/`, `paper/`, `docs/`) là phần thêm và không làm test này hỏng.
8. Thêm `.gitignore`: dữ liệu, checkpoint (`*.pt`, `*.pth`), `__pycache__`, file LaTeX trung gian, `runs/*/ckpt*`. **Không** ignore các JSON hay JSONL kết quả, vì chúng nhỏ và là bằng chứng.
9. Commit "Gộp workspace", gắn tag `pilot-v1.1`.

## 1B. Push lên GitHub (repo riêng tư của người dùng)

- **Không push lên repo Visual-AI.**
  - Đổi tên remote: `git remote rename origin upstream`.
  - Từ nay `upstream` chỉ dùng để fetch hoặc so sánh, không bao giờ push.
- Tạo một repo **riêng tư mới** trên tài khoản của người dùng, ví dụ `<tài-khoản>/freqcone`. Không dùng nút "Fork", vì GitHub không cho đặt fork của repo công khai về chế độ riêng tư.

  ```bash
  gh repo create <tài-khoản>/freqcone --private      # hoặc tạo trên web
  git remote add origin git@github.com:<tài-khoản>/freqcone.git
  git push -u origin freqcone
  git push origin --tags
  ```

- Nếu cần đăng nhập GitHub mà máy chưa có quyền, dừng lại và nhờ người dùng chạy `gh auth login`.
- Trên máy 3090:

  ```bash
  git clone git@github.com:<tài-khoản>/freqcone.git && cd freqcone
  git checkout <tag mới nhất: pilot-v1.2, hoặc pilot-v2 sau khi profile>
  git rev-parse HEAD     # kiểm hash
  ```

- `git bundle` chỉ còn là phương án dự phòng, khi máy 3090 không truy cập được GitHub.
- Sau mỗi mốc trên 3090 (`cutoffs.json`, quyết định profile, `selection.json`, kết quả), commit trên 3090, push lên `origin`, rồi `git pull` trên máy này.

## 1. Dọn workspace (an toàn, có kiểm kê)

Làm sau 1A và 1B, trong `261/freqcone`. Các mục chuyển ra ngoài đều để trong `261/_archive_YYYYMMDD/`, không nằm trong repo.

Yêu cầu của người dùng: bỏ mọi thứ thừa hoặc không liên quan (planora_sf và các nội dung không còn ý nghĩa); bỏ các file JSON không phải kết quả chạy thật.

Cách làm bắt buộc, vì xóa nhầm là không lấy lại được:

1. **Kiểm kê.**
   - Liệt kê toàn bộ cây thư mục `261/` (bỏ qua `.git`), ghi vào `CLEANUP_MANIFEST.md`.
   - Với mỗi mục, ghi một trong ba nhãn: **GIỮ**, **LƯU TRỮ** hoặc **XÓA** (chỉ cho file sinh lại được hoặc file rác), kèm một dòng lý do.
2. **Gửi manifest cho người dùng duyệt trước khi di chuyển bất cứ thứ gì.** Riêng planora_sf là code đồ án của người dùng, nên phải được người dùng xác nhận đích danh.
3. **LƯU TRỮ.** Chuyển các mục này ra **ngoài** repo, vào `261/_archive_YYYYMMDD/`. Nếu môi trường cho phép, nén thành `.tar.gz` và ghi SHA-256. Không xóa vĩnh viễn. Nếu một mục đã nằm trong lịch sử git thì xóa khỏi cây làm việc bằng một commit riêng, ghi rõ trong commit message.
4. **XÓA**, chỉ áp dụng cho:
   - kết quả smoke test;
   - `__pycache__` và `.pytest_cache`;
   - file tạm;
   - JSON **không phải** kết quả chạy thật (JSON giả, mẫu, sinh tay).

   Test cần dữ liệu mẫu thì tự sinh trong `tmp_path` khi chạy, không commit file mẫu.

**Phải GIỮ:**
- code HypCD upstream cùng phần chèn FreqCone và các test;
- `cutoffs.json` (khi đã tính);
- mọi kết quả chạy thật;
- `mechanism_study/` cùng các JSON kết quả thật của thí nghiệm tổng hợp. Đây là bằng chứng cho phần lý thuyết và cơ chế của bài báo. Riêng run baseline `fourex_partition` trong đó là kết quả chạy thật, đã được báo cáo trong bản thảo, nên giữ nguyên.
- `paper/en`, `paper/vi`;
- `docs/PLAN.md`;
- `README.md`.

**LƯU TRỮ** (ngoài repo):
- mọi thứ liên quan planora_sf;
- code dùng FourEx làm đối chứng trên SSB, nếu có;
- các kế hoạch cũ (HANDOFF_PROMPT.md, PLAN_P1_PILOT.md, PLAN_P1_ADDENDUM.md);
- script phục vụ Claude Docs (`doc_chart.py`, `doc_tables.py`, `doc_v3.py`);
- file zip cũ;
- thư mục `freqcone_old/` sau khi đã chép xong ở 1A.

**Kiểm tra sau khi dọn:**
- `pytest` vẫn đạt toàn bộ;
- khi tắt cờ FreqCone, code vẫn trùng từng byte với upstream `ffc895b` (test kiểm toán hiện có);
- `grep -ri "planora\|fourex"` trong repo không còn kết quả nào, ngoại trừ `mechanism_study/` (run `fourex_partition` thật) và các trích dẫn trong `paper/`;
- commit "Dọn workspace", gắn tag `pilot-v1.2`, rồi push lên `origin` (repo riêng tư). Trên máy 3090, checkout `pilot-v1.2` (hoặc `pilot-v2` sau khi profile).

---

## 2. Đánh giá ở mọi epoch và cách báo cáo (cố định trước khi chạy)

### 2.1 Cài đặt
- Sau **mỗi epoch** (1…200), ở `model.eval()` và `torch.no_grad()`, tính:
  - **val:** chỉ số SS-k-means đã chốt;
  - **test:** All/Old/New theo giao thức v2, trên phần unlabelled của train (GCD transductive).
- **Cô lập trạng thái ngẫu nhiên của bước đánh giá.** Lưu rồi khôi phục `torch.get_rng_state()`, `torch.cuda.get_rng_state_all()`, `np.random.get_state()` và `random.getstate()` quanh bước đánh giá. k-means dùng generator riêng, seed = `seed*1000 + epoch`. Nhờ vậy đánh giá mỗi epoch **không làm thay đổi quá trình huấn luyện**.
- Thêm test: trên CPU với mô hình tí hon, train 2 epoch có và không có đánh giá mỗi epoch, trọng số thu được phải trùng nhau.
- **Tách file để không vô tình nhìn thấy số test:**
  - `runs/<run_id>/epochs_val.jsonl`, mỗi dòng: `{epoch, val_acc, val_per_sample_pred (chỉ ở epoch cuối), losses, fc:{depth_full, in_cone, clip_frac, angle_b_u, depth_b}, lr, time_s}`;
  - `runs/<run_id>/epochs_test.jsonl`, mỗi dòng: `{epoch, all, old, new}`;
  - stdout **không in** số test.
- File `runs/<run_id>/run.json` giữ schema đã chốt (git commit, args, môi trường, SHA của checkpoint DINOv2 và danh sách dữ liệu, chi phí), cộng thêm đường dẫn tới hai file JSONL trên.

### 2.2 `summarize_run.py`: bảng của **một** run

| Cách chọn epoch | Epoch | All | Old (Known) | New (Novel) | Vai trò |
| --- | --- | --- | --- | --- | --- |
| Epoch cuối (200) | 200 | | | | **KẾT QUẢ CHÍNH** |
| Epoch có val cao nhất | e_val | | | | kết quả phụ, hợp lệ (chọn bằng val) |
| Tiêu chí upstream: Old cao nhất trên test | e_up | | | | chỉ để đối chiếu số công bố (G0) |
| Oracle: max All | e1 | | | | **oracle, chọn bằng test, không dùng để kết luận** |
| Oracle: max Old | e2 | | | | oracle |
| Oracle: max New | e3 | | | | oracle |

Ba hàng oracle là cách lọc "epoch có All/Known/Novel cao nhất" theo yêu cầu của người dùng. Chúng được báo cáo đầy đủ nhưng **luôn kèm nhãn oracle**, vì:
- mỗi chỉ số lấy ở một epoch khác nhau;
- epoch được chọn bằng chính tập test.

Hai điều này làm số bị thổi phồng, và reviewer sẽ không chấp nhận chúng làm kết quả chính.

### 2.3 `compare_runs.py`: baseline so với FreqCone theo cặp seed
- Tính ΔAll, ΔOld, ΔNew cho **từng hàng** của bảng 2.2.
- Báo trung bình, độ lệch chuẩn và số seed thắng.
- Mọi cổng và mọi kết luận (mục 3) **chỉ dùng hàng "Epoch cuối"**. Hàng "val cao nhất" là kết quả phụ. Các hàng oracle in ở bảng riêng, có tiêu đề "Oracle (chọn bằng test, tham khảo)".
- Thêm biểu đồ All/Old/New theo epoch cho baseline và FreqCone (trung bình các seed) để thấy độ ổn định. Script chỉ đọc JSONL.

---

## 3. Kế hoạch chạy trên Stanford Cars (RTX 3090)

### 3.0 Cố định
- Dữ liệu: Stanford Cars theo split SSB của GCD (98 known / 98 novel).
- Nền: Hyp-SelEx của HypCD với `--fix_label_order` (đã sửa lỗi). Backbone DINOv2 ViT-B/14 đúng biến thể mà script upstream dùng (ghi SHA checkpoint). Lịch huấn luyện 200 epoch.
- c, r, batch size, learning rate và augmentation: lấy theo **script upstream cho SCars**, áp cho cả hai nhánh. Ghi bảng vào README.
- FreqCone v2:
  - λ_dir = 0,5, λ_con = 0,05, τ_a = 0,5, τ_b = 0,3, K = 0,1;
  - cut-off lấy từ `cutoffs.json` mục `scars`, do `compute_cutoffs.py` tính;
  - độ phân giải view thô theo luật profile (224 px, hoặc 112 px nếu 224 px không vừa hay quá chậm).

### 3.1 Profile
- Chạy `DATASET=scars bash scripts/freqcone/profile_gpu.sh`, **có** đánh giá mỗi epoch, để đo thời gian thật.
- Ước tính `GPU-giờ = Σ_run 200 × t_epoch`.
- Chọn mức ngân sách theo 3.2, ghi vào README, commit, gắn tag `pilot-v2`. Mọi run sau chạy từ `pilot-v2`.

### 3.2 Hai mức ngân sách (chọn một sau khi profile; ghi lý do)

**Mức A (khuyến nghị nếu một run ≤ 10 giờ): 10 run, cộng 1 run có điều kiện.**

| Giai đoạn | Run | Seed |
| --- | --- | --- |
| Pilot | R0: baseline | 7 |
| Pilot | R1: FreqCone λ_ent = 0,03, A1 (giá trị chọn ở thí nghiệm tổng hợp) | 7 |
| Pilot | R2: FreqCone λ_ent = 0,1, A1 | 7 |
| Pilot | R3: FreqCone λ_ent = 0,1, A3 (nón rộng, nới giới hạn của Mệnh đề 5) | 7 |
| Xác nhận | baseline × {8, 9, 10}; FreqCone (cấu hình được chọn) × {8, 9, 10} | 8–10 |
| Có điều kiện | R0-repro: upstream nguyên trạng (còn bug), seed mặc định của upstream; chỉ chạy nếu G0 không đạt | upstream |

**Mức B (nếu ngân sách chỉ đủ khoảng 6 run): không có pilot.**
- FreqCone dùng cấu hình cố định từ thí nghiệm tổng hợp: λ_ent = 0,03, A1, v2. Ghi rõ "không tinh chỉnh trên SSB".
- Chạy baseline và FreqCone × seed {8, 9, 10}.
- Không có cổng G1; G0 dùng baseline seed 8.

### 3.3 Cổng (luật cố định)
- **G0 (tái lập baseline).**
  - Hàng "Tiêu chí upstream" của R0 phải có All lệch số công bố của Hyp-SelEx trên Stanford Cars (**83,8 / 93,3 / 79,2**, Bảng 1 của HypCD) không quá ±2,0 điểm.
  - Không đạt: chạy R0-repro. Nếu R0-repro đạt, ghi rằng chênh lệch do sửa bug và đi tiếp. Nếu R0-repro cũng không đạt, **dừng và báo người dùng**.
- **Chọn cấu hình (chỉ ở mức A).**
  - `select_pilot.py` chọn trong R1–R3 bằng val **ở epoch cuối**, theo luật 1-SE paired và stratified, B = 2000.
  - Phá hòa: λ nhỏ hơn trước, rồi nón rộng hơn trước.
  - Commit `selection.json` trước khi chạy `summarize_run.py` hay `compare_runs.py` trên R1–R3.
- **G1 (mức A).** Ở hàng "Epoch cuối", cấu hình được chọn so với R0 phải có ΔNew ≥ +1,0 **và** ΔAll ≥ −0,5. Không đạt thì dừng nhánh "tăng ACC" và báo người dùng; đây là lúc cân nhắc nhánh B (đầu chiếu tách rời) trong PLAN_P1_PILOT mục 5.
- **Kết luận cuối (sau seed 8–10), cố định trước:** FreqCone được coi là có đóng góp trên Stanford Cars nếu, ở hàng "Epoch cuối":
  - ΔNew trung bình ≥ +1,0;
  - thắng ≥ 2/3 seed về New;
  - ΔAll trung bình ≥ −0,5.

  Kết quả âm vẫn phải báo cáo đầy đủ.

### 3.4 Chẩn đoán phải có trong báo cáo
- Độ sâu ảnh nét của baseline và FreqCone so với δ(b) của preset đang dùng: chế độ sâu hay nông.
- Tỉ lệ ảnh nét nằm trong nón, và góc (b, u) theo epoch.
- Thời gian và bộ nhớ thực tế.

### 3.5 Giai đoạn sau (chỉ khi kết luận cuối là dương; chưa chạy bây giờ)
- CUB và Aircraft với **cùng cấu hình**, không chọn lại, seed 8–10.
- Ablation trên Cars: λ = 0 (bỏ L_ent), v3, A2.
- Gán Novel → hãng xe trên Cars.

---

## 4. README: hướng dẫn train chi tiết (Claude Code viết; phải chạy được đúng như ghi)

README ở gốc repo, tiếng Việt, gồm các mục:

1. **Tổng quan ngắn.**
   - Repo này là HypCD upstream cộng FreqCone (mô tả một đoạn, kèm link tới bản thảo).
   - Chỉ hỗ trợ hướng Hyp-SelEx + FreqCone trên Stanford Cars.
2. **Yêu cầu.** GPU (đã kiểm trên RTX 3090 24 GB), phiên bản Python, PyTorch, CUDA, và lệnh tạo môi trường (conda hoặc pip, có `requirements.txt` ghi rõ phiên bản).
3. **Lấy code.**
   - `git clone` từ repo riêng tư của người dùng, checkout tag, kiểm hash. Dự phòng: `git bundle`.
   - Giải thích hai remote: `origin` là repo riêng tư; `upstream` là Visual-AI, chỉ fetch, **không push**.
   - Giới thiệu cấu trúc thư mục ở mục 1A.
4. **Dữ liệu Stanford Cars.**
   - Nguồn tải hợp lệ, cấu trúc thư mục mong đợi, file split SSB (`scars_osr_splits.pkl` hoặc tương đương).
   - Lệnh kiểm tra số ảnh (train/test, known/novel) và SHA danh sách file.
5. **DINOv2.** Biến thể chính xác, cách tải, đường dẫn trong `config.py`, SHA-256 mong đợi.
6. **`config.py`.** Mọi trường cần điền, có ví dụ.
7. **Kiểm tra trước khi train.** `pytest`, `compute_cutoffs.py --dataset scars`, smoke test, rồi xóa kết quả smoke.
8. **Profile và chọn mức ngân sách.** Lệnh chạy và cách đọc kết quả.
9. **Lệnh train**, copy-paste được, cho:
   - baseline;
   - FreqCone (từng cấu hình R1–R3);
   - các run xác nhận seed 8–10.
10. **Bảng mọi cờ dòng lệnh của FreqCone.** Mỗi cờ: ý nghĩa, giá trị mặc định, giá trị dùng trong bài. Gồm `--freqcone`, `--freqcone_variant`, `--w_ent`, `--aperture A1|A2|A3`, `--coarse_res`, `--dist_impl lowmem`, `--fix_label_order`, cờ tắt in test ra stdout.
11. **Bảng siêu tham số** của nền (theo script upstream SCars) và của FreqCone.
12. **Đầu ra của mỗi run.** Cây thư mục `runs/<run_id>/`, schema của `run.json`, `epochs_val.jsonl` và `epochs_test.jsonl`.
13. **Theo dõi khi đang train.**
    - Ý nghĩa của `fc_depth_full`, `fc_in_cone`, `clip_frac`.
    - Cảnh báo "độ sâu < δ(b)" nghĩa là nón đang vô hiệu.
    - Nhắc: không mở `epochs_test.jsonl` của R1–R3 trước khi `selection.json` được commit.
14. **Sau khi train.** Thứ tự chạy `select_pilot.py`, `summarize_run.py`, `compare_runs.py`, và cách đọc bảng 2.2 (hàng chính, hàng phụ, hàng oracle).
15. **Tiếp tục khi bị ngắt** (resume), và cách chạy lại một run.
16. **Xử lý sự cố.**
    - NaN: dừng và báo, không tự đổi λ.
    - OOM: chuyển sang 112 px; chỉ giảm batch cho cả hai nhánh và phải ghi rõ.
    - Chế độ nông: không phải lỗi.
    - G0 không đạt.
17. **Tái lập.** Seed, commit, nguồn ngẫu nhiên còn lại trên GPU; thời gian và GPU-giờ thực tế đo được.

Sau khi viết xong: chạy lại **đúng từng lệnh** trong README trên máy hiện tại cho các bước không cần GPU (pytest, `compute_cutoffs` với dữ liệu giả lập trong `tmp_path`, smoke trên CPU nếu có thể), rồi sửa mọi lệnh sai.

---

## 5. Giữ nguyên từ các bản trước
- Luật 1-SE: paired, stratified, B = 2000, phá hòa theo độ đơn giản; chỉ đọc val qua danh sách trắng.
- Preset A1–A3 với K = 0,1, có assert chặn nón suy biến. Ngưỡng δ(b) tính từ c và r_b. Baseline cũng log độ sâu.
- Bug thứ tự nhãn: README ghi rõ bug, có test. Báo cáo tách riêng chênh lệch do sửa bug và chênh lệch do FreqCone.
- Chuyển code qua repo GitHub riêng tư (mục 1B); `git bundle` chỉ là dự phòng. Ghi SHA của checkpoint DINOv2 và của dữ liệu.
- Commit style: theo xác nhận của người dùng.

## 6. Checklist
- [ ] 1A. Gộp về `261/freqcone`: một bản `freqcone_torch.py`, parity ≤ 1e-3, hai bản thảo biên dịch lại không đổi số, `.gitignore`, tag `pilot-v1.1`
- [ ] 1B. `origin` thành `upstream`; repo GitHub riêng tư làm `origin`; push nhánh `freqcone` và các tag; clone thử, kiểm hash
- [ ] 1. `CLEANUP_MANIFEST.md`, người dùng duyệt, lưu trữ hoặc xóa, kiểm tra sau dọn (commit mới, tag `pilot-v1.2`)
- [ ] 2.1 Đánh giá mỗi epoch, cô lập RNG có test, tách file val/test, stdout không in test
- [ ] 2.2–2.3 `summarize_run.py`, `compare_runs.py`, biểu đồ theo epoch; hàng oracle có nhãn
- [ ] 3.0 Bảng c/r, batch, lr theo script upstream SCars; `compute_cutoffs.py --dataset scars`
- [ ] 3.1 Profile SCars có đánh giá mỗi epoch, chọn mức A hoặc B, tag `pilot-v2`
- [ ] 3.3 G0, chọn cấu hình (mức A), G1, xác nhận seed 8–10, kết luận theo luật cố định
- [ ] 4. README đầy đủ 17 mục, đã chạy thử các lệnh không cần GPU
- [ ] Báo người dùng ở các mốc: manifest dọn dẹp, profile, G0, G1, kết luận cuối (kèm JSON)

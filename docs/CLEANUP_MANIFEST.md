# Manifest dọn workspace (kế hoạch v2, mục 1)

Kiểm kê ngày 2026-11-04. Đã thực hiện ngày 2026-11-05 theo phần được duyệt (xem mục E).

**Quyết định của người dùng (2026-11-05):**
- `model/` (planora_sf): GIỮ NGUYÊN.
- `planogram.zip` và hai thư mục backup của planogram: GIỮ NGUYÊN.
- Phần còn lại của manifest: CHƯA DUYỆT. Người dùng sẽ sửa manifest trước khi thực hiện.

**Nguyên tắc:**
- **LƯU TRỮ** nghĩa là chuyển nguyên trạng vào `261/_archive_20261104/` (ngoài repo), nén `.tar.gz` và ghi SHA-256. Không xóa vĩnh viễn.
- **XÓA** chỉ áp dụng cho file sinh lại được hoặc file rác.

## A. Trong repo `261/freqcone`

| Mục | Nhãn | Lý do |
| --- | --- | --- |
| Code HypCD upstream cùng các khối chèn FreqCone (`train/`, `data/`, `hyptorch/`, `models/`, `methods/`, `project_utils/`, `config.py`, `scripts/` upstream, `LICENSE`, `requirements.txt`) | GIỮ | Nền của phương pháp; giấy phép MIT |
| `freqcone/`, `scripts/freqcone/`, `tests_freqcone/`, `requirements-dev.txt` | GIỮ | Code và test của FreqCone |
| `mechanism_study/` (gồm 52 JSON kết quả thật và `sweep.log`) | GIỮ | Bằng chứng cho phần cơ chế của bài. Run `fourex_partition` là kết quả thật, đã báo cáo trong bản thảo |
| `paper/en/` (`.tex`, `.pdf`, bảng và macro sinh tự động) | GIỮ | Bản thảo. Không có `paper/vi`, vì chưa từng có bản LaTeX tiếng Việt trên máy |
| `docs/PLAN.md`, `docs/UPSTREAM_README.md`, `README.md`, file manifest này | GIỮ | Kế hoạch hiện hành, ghi công tác giả gốc, hướng dẫn |
| `__pycache__/` (đã bị git bỏ qua) | XÓA | Sinh lại được |
| `paper/en/*.aux`, `*.log`, `*.out` (đã bị git bỏ qua) | XÓA | File trung gian LaTeX |
| JSON giả, mẫu hoặc sinh tay | (không có) | Test tự sinh dữ liệu trong `tmp_path` |
| Kết quả smoke | (không có) | Chưa chạy smoke |

## B. Trong `261/` (ngoài repo)

| Mục | Kích thước | Nhãn đề xuất | Lý do |
| --- | --- | --- | --- |
| `_archive_20261104/freqcone_old/` (75 file, có `freqcone_old.sha256`) | 0,8 MB | LƯU TRỮ (đã nằm ngoài repo) → nén `.tar.gz` | Gói nghiên cứu cũ, đã chép phần còn dùng vào repo. Gồm `doc_chart.py`, `doc_tables.py`, `doc_v3.py` (script cho Claude Docs) và `freqcone_torch.py` bản cũ |
| `model/`: tài liệu PLANORA và repo `research/planora_sf` (remote `thzii/model`, HEAD `de9492e`, cây sạch) | 4,2 MB | GIỮ (người dùng quyết định 2026-11-05) | Code đồ án planora_sf; không còn dùng cho hướng FreqCone |
| `paper/`: DebGCB.pdf, FourEx.pdf, HyperGCD.pdf, Online Continuous GCD.pdf, SelEx.pdf | 64 MB | GIỮ | Tài liệu tham khảo cho phần công trình liên quan |
| `paper/FourEx — Paper Brief.pptx` | 0,2 MB | LƯU TRỮ (đề xuất) | Slide tóm tắt từ giai đoạn planora |
| `planogram.zip` | 86 MB | GIỮ (người dùng quyết định 2026-11-05) | File zip cũ (04/10) của mã nguồn RetailVision |
| `planogram/` | 245 MB | GIỮ — ngoài phạm vi | Mã nguồn ứng dụng RetailVision (đồ án) |
| `planogram-backend-git-preserved-20261006-183838/`, `planogram-safety-backup-20261006-180107/` | 0,4 MB; 495 MB | GIỮ (người dùng quyết định 2026-11-05) | Bản sao lưu ngày 06/10 của ứng dụng |
| `Thesis_LeDucDong/`, `baba/`, `ctrr/`, `gttkd/`, `hkien/`, `sida/`, `style/`, `tthcm/`, `vpbank/`, `DL/` | — | GIỮ — ngoài phạm vi, không đụng | Môn học và tài liệu khác |
| Các file PDF, `.doc`, `.xlsx` ở gốc `261/` (đề thi cũ, CV, quy định BTL, Score.xlsx…) | — | GIỮ — ngoài phạm vi, không đụng | Tài liệu cá nhân |

## C. Các mục trong kế hoạch nhưng không tồn tại trên máy

- `HANDOFF_PROMPT.md`, `PLAN_P1_PILOT.md`, `PLAN_P1_ADDENDUM.md`: chỉ có trong tin nhắn chat, không có file. Kế hoạch hiện hành đã lưu ở `docs/PLAN.md`.
- Code dùng FourEx làm đối chứng trên SSB: không có trong repo. Code FourEx của planora_sf nằm trong `model/`, nên được xử lý cùng mục đó.

## D. Kiểm tra sau khi dọn

- `pytest` đạt toàn bộ.
- Test kiểm toán upstream đạt.
- `git grep -i "planora\|fourex"` không còn kết quả, ngoại trừ ba nơi:
  - `mechanism_study/`;
  - `paper/`;
  - `docs/PLAN.md` và manifest này. Đây là ngoại lệ bắt buộc, vì hai file này mô tả chính việc dọn dẹp.
- Commit, gắn tag `pilot-v1.2`, rồi push.

## E. Đã thực hiện (2026-11-05)

**Phần được duyệt:**
- Xóa mọi `__pycache__` và file trung gian LaTeX trong repo; thêm các mẫu này vào `.gitignore`.
- Giữ `.tex`, `tab_*.tex`, `macros*.tex`, `*.dat`, `numbers.json` và PDF.
- Nén `freqcone_old` và kiểm tra trước khi xóa bản chưa nén.

**Trong repo:**
- Đã xóa 12 thư mục `__pycache__`, cùng `paper/en/freqcone_draft.aux`, `.log`, `.out`. Không file nào trong số đó được git theo dõi.
- `.gitignore` đã có sẵn `__pycache__/` và `paper/**/*.{aux,log,out,toc,fls,fdb_latexmk,synctex.gz}`.
- `mechanism_study/results/sweep.log` là log thí nghiệm (mục A, GIỮ), không phải file LaTeX, nên được giữ.
- Biên dịch lại `paper/en/freqcone_draft.tex` (`pdflatex` × 2, ra thư mục tạm): cả hai lượt exit 0, PDF 10 trang, không có reference hay citation nào undefined. Không có `paper/vi` để biên dịch.

**Ngoài repo (`261/_archive_20261104/`):**
- `freqcone_old/` → `freqcone_old.tar.gz` (425 524 byte).
- SHA-256 `9191cec6d9c3ca8025eed9515155b93b673df26b5c0724ffe8358a37af1e3a81`, ghi trong `freqcone_old.tar.gz.sha256`.
- Kiểm tra:
  - tar chứa đúng 75 file.
  - Giải nén ra thư mục tạm: SHA-256 của cả 75 file khớp bản gốc. Manifest cũ `freqcone_old.sha256` cũng khớp.
  - Chỉ sau đó mới xóa `freqcone_old/` chưa nén và thư mục tạm.

**Kiểm tra sau dọn (mục D):**
- `pytest tests_freqcone`: 68 passed, 3 skipped (cả 3 là test CUDA).
- `test_upstream_equivalence.py`: 7 passed.
- `git grep -i planora`: không có kết quả.
- `git grep -i fourex`: chỉ còn trong `mechanism_study/` (run baseline `fourex_partition` thật) và `paper/en/` (trích dẫn FourEx, bảng số).

Các mục B khác (`paper/FourEx — Paper Brief.pptx`, …) chưa được duyệt nên giữ nguyên.

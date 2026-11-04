# Thí nghiệm kiểm chứng cơ chế (dữ liệu tổng hợp SpecHier-48)

Đây là bằng chứng cho phần lý thuyết và cơ chế của bản thảo (`paper/en`, mục "Mechanism study"). Mọi kết quả trong `results/` là kết quả chạy thật trên CPU, bằng JAX. **Không sửa tay** các file này.

| File | Nội dung |
| --- | --- |
| `toy_experiment.py` | Dữ liệu SpecHier-48, mô hình và các phương pháp so sánh |
| `verify_math.py` | Kiểm tra bằng số cho 8 nhóm mệnh đề của phần lý thuyết |
| `parity_check.py` | So loss và gradient của `freqcone/freqcone_torch.py` (bản duy nhất trong repo) với JAX |
| `aggregate.py`, `make_tables.py` | Tổng hợp kết quả; `make_tables.py` sinh bảng, macro và dữ liệu biểu đồ vào `paper/en/` |
| `run_sweep.sh`, `run_ablation_w0.sh`, `run_v3.sh` | Các lần chạy đã thực hiện; luật chọn và câu hỏi nằm ở đầu file |
| `results/` | `pilot/` và `lambda_sweep/` (seed 0), `results/` (seed 1–3), `results_v3/` (seed 4–6), `sweep.log` |

Run baseline `fourex_partition` trong `results/` là kết quả chạy thật đã được báo cáo trong bản thảo, nên được giữ nguyên.

## Chạy lại

Cần `torch`, `numpy`, `jax[cpu]`, `optax`, `scipy`, `scikit-learn`.

```bash
python mechanism_study/verify_math.py
python mechanism_study/parity_check.py
cd mechanism_study && python aggregate.py results/results --seeds 1,2,3 && cd ..
python mechanism_study/make_tables.py
cd paper/en && pdflatex freqcone_draft.tex && pdflatex freqcone_draft.tex
```

## Khác biệt giữa `freqcone/freqcone_torch.py` và bản dùng trong thí nghiệm này (đều cố ý)

- `pairwise_dist` dùng ma trận Gram (bộ nhớ O(N²)); dạng Möbius cũ vẫn giữ là `pairwise_dist_mobius`.
- `radial_lowpass` nhận thêm `out_size` (resample chính xác bằng cách cắt phổ).
- Thêm preset độ mở `APERTURES` (A1–A3), `cone_min_radius`, `aperture` và `scaled_depth`.
- `FreqConeLoss` báo lỗi khi nón suy biến (ε_K < r_a < r_b) hoặc khi `r_b ≥ tanh(√c·clip_r)`, và log thêm góc (b, u).
- Mặc định `w_ent = 0.03` (λ* chọn trong thí nghiệm này).

`parity_check.py` xác nhận giá trị loss và gradient không đổi: chênh lệch loss ≤ 1e-6 ở float32, gradient ≤ 2e-12 ở float64.

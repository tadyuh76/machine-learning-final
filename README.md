# Machine Learning Final - DENCLUE

Project này dùng DENCLUE để phân cụm dữ liệu 2D và tìm hotspot sạt lở trong NASA Landslide Events.

DENCLUE là thuật toán chính của bài. Các mô hình như DBSCAN, KMeans, GMM và MeanShift được chạy thêm để đối chiếu cách định nghĩa cụm và đưa ra cái nhìn khách quan hơn.

## Dữ liệu

- File chính: `data/raw/landslide_catalog.csv`
- Dataset benchmark: `data/raw/benchmark_cases/`
- Feature đưa vào clustering: `longitude`, `latitude`
- Cột dùng để diễn giải sau clustering: `landslide_type`, `landslide_size`, `trigger`, `country_name`

Các cột mô tả không được đưa vào thuật toán phân cụm. Nhóm chỉ dùng chúng sau khi đã có nhãn cụm để đọc profile từng hotspot.

## Chạy lại trên macOS

```bash
cd machine-learning-final
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Chạy pipeline chính:

```bash
python3 src/denclue_pipeline.py
```

Chạy thêm benchmark và seed stability:

```bash
python3 src/denclue_benchmark_cases.py
python3 src/denclue_seed_stability.py
```

## Chạy lại trên Windows PowerShell

```powershell
cd machine-learning-final
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
```

Nếu PowerShell không cho activate virtual environment, chạy tạm lệnh này trong cửa sổ PowerShell hiện tại rồi activate lại:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Chạy pipeline chính:

```powershell
py src\denclue_pipeline.py
```

Chạy thêm benchmark và seed stability:

```powershell
py src\denclue_benchmark_cases.py
py src\denclue_seed_stability.py
```

## Output cần kiểm tra

Sau khi chạy xong, xem kết quả trong `outputs/`. Một số file chính:

- `outputs/tables/ml_dataset_summary.csv`
- `outputs/tables/ml_preprocessing_validation.csv`
- `outputs/tables/ml_denclue_tuning.csv`
- `outputs/tables/ml_clustering_metrics.csv`
- `outputs/tables/ml_denclue_cluster_profile.csv`
- `outputs/tables/ml_benchmark_case_metrics.csv`
- `outputs/tables/ml_seed_stability_summary.csv`
- `outputs/figures/ml_denclue_clusters.png`
- `outputs/figures/ml_denclue_density.png`
- `outputs/figures/ml_benchmark_aggregation_comparison.png`
- `outputs/figures/ml_failure_spiral_comparison.png`
- `outputs/figures/ml_failure_pathbased_comparison.png`

## Ghi chú ngắn

- `src/denclue_pipeline.py` là pipeline chính cho NASA Landslide.
- `src/denclue_benchmark_cases.py` dùng Aggregation, Spiral và Pathbased để kiểm tra điểm mạnh/yếu của DENCLUE.
- `src/denclue_seed_stability.py` chạy lại vài seed để xem kết quả có dao động mạnh không.
- Kết quả trong báo cáo nên đọc cùng hình scatter, không chỉ đọc một metric riêng lẻ.


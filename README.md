# Machine Learning Final - DENCLUE

Repository này giữ các phần cần để chạy lại thí nghiệm DENCLUE trên dữ liệu NASA Landslide Events và benchmark clustering 2D.

## Dataset

- Nguồn: Kaggle - NASA Landslide Events
- File chính: `data/raw/landslide_catalog.csv`
- Feature dùng cho clustering: `longitude`, `latitude`
- Cột dùng để diễn giải sau clustering: `landslide_type`, `landslide_size`, `trigger`, `country_name`

Các cột mô tả như landslide type/size/trigger không được dùng làm input clustering. Chúng chỉ dùng để giải thích cụm sau khi thuật toán chạy xong.

## Pipeline

```bash
pip install -r requirements.txt
```

```bash
python src/denclue_pipeline.py
```

Pipeline sẽ:

1. đọc landslide catalog,
2. lọc các dòng có tọa độ hợp lệ,
3. chuẩn hóa `longitude` và `latitude`,
4. tune DENCLUE theo `sigma` và `xi_quantile`,
5. tune DBSCAN theo `eps` và `min_samples`,
6. so sánh DENCLUE với KMeans, DBSCAN, GMM, MeanShift,
7. xuất bảng và hình vào `outputs/`.

## Output chính

- `outputs/tables/ml_dataset_summary.csv`
- `outputs/tables/ml_preprocessing_validation.csv`
- `outputs/tables/ml_denclue_tuning.csv`
- `outputs/tables/ml_dbscan_tuning.csv`
- `outputs/tables/ml_clustering_metrics.csv`
- `outputs/tables/ml_denclue_cluster_profile.csv`
- `outputs/figures/ml_landslide_locations.png`
- `outputs/figures/ml_denclue_clusters.png`
- `outputs/figures/ml_denclue_density.png`
- `outputs/figures/ml_dbscan_clusters.png`
- `outputs/figures/ml_kmeans_clusters.png`
- `outputs/figures/ml_gmm_clusters.png`
- `outputs/figures/ml_meanshift_clusters.png`

## Benchmark cases

Chạy thêm các case benchmark:

```bash
python src/denclue_benchmark_cases.py
```

Script này dùng 3 dataset nhỏ trong `data/raw/benchmark_cases/`:

- `aggregation.txt`: case DENCLUE hoạt động rất tốt.
- `spiral.txt`: case DENCLUE hoạt động kém hơn DBSCAN.
- `pathbased.txt`: case cấu trúc dạng path/connectivity, DBSCAN hợp hơn.

Output chính:

- `outputs/tables/ml_benchmark_case_summary.csv`
- `outputs/tables/ml_benchmark_case_metrics.csv`
- `outputs/figures/ml_benchmark_aggregation_comparison.png`
- `outputs/figures/ml_failure_spiral_comparison.png`
- `outputs/figures/ml_failure_pathbased_comparison.png`

## Kiểm tra độ ổn định theo seed

```bash
python src/denclue_seed_stability.py
```

Phần này chạy lại các mô hình với vài seed khác nhau để kiểm tra kết quả có phụ thuộc quá mạnh vào random state hay không.

Output chính:

- `outputs/tables/ml_seed_stability_metrics.csv`
- `outputs/tables/ml_seed_stability_summary.csv`

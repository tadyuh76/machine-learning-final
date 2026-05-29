#!/usr/bin/env python3
"""
Các bộ dữ liệu mô phỏng nhỏ để kiểm tra DENCLUE.

Phạm vi được giữ gọn:
- Aggregation cho thấy trường hợp DENCLUE chạy tốt;
- Spiral và pathbased cho thấy trường hợp DBSCAN phù hợp hơn.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from denclue_pipeline import RANDOM_STATE, SimpleDENCLUE, cluster_count, noise_rate
from sklearn.cluster import DBSCAN, KMeans, MeanShift, estimate_bandwidth
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    file_name: str
    role: str
    case_note: str


CASES = [
    BenchmarkCase(
        case_id="aggregation",
        file_name="aggregation.txt",
        role="denclue_strong_case",
        case_note="DENCLUE xử lý cụm mật độ không đều tốt hơn các baseline dạng centroid/model.",
    ),
    BenchmarkCase(
        case_id="spiral",
        file_name="spiral.txt",
        role="dbscan_strong_case",
        case_note="DENCLUE dễ chia nhỏ cụm cong liên tục; DBSCAN phù hợp hơn trong case này.",
    ),
    BenchmarkCase(
        case_id="pathbased",
        file_name="pathbased.txt",
        role="dbscan_strong_case",
        case_note="Cấu trúc dạng đường đi hợp với density connectivity hơn density attractor.",
    ),
]

SIGMA_GRID = [0.08, 0.10, 0.12, 0.16, 0.20, 0.24, 0.30, 0.40]
XI_GRID = [0.00, 0.005, 0.01, 0.02, 0.05, 0.10]
EPS_GRID = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.55]
MIN_SAMPLES_GRID = [3, 5, 8, 10, 15]


def read_case(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=["x", "y", "true_label"])


def scale_xy(df: pd.DataFrame) -> np.ndarray:
    return StandardScaler().fit_transform(df[["x", "y"]].to_numpy(dtype=float))


def evaluate_labels(X: np.ndarray, y_true: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels)
    metrics = {
        "clusters": cluster_count(labels),
        "noise_rate": noise_rate(labels),
        "ari": float(adjusted_rand_score(y_true, labels)),
        "nmi": float(normalized_mutual_info_score(y_true, labels)),
        "silhouette": np.nan,
    }
    unique_count = len(np.unique(labels))
    if 1 < unique_count < len(labels):
        try:
            metrics["silhouette"] = float(silhouette_score(X, labels))
        except ValueError:
            pass
    return metrics


def tune_denclue(X: np.ndarray, y_true: np.ndarray) -> tuple[pd.DataFrame, SimpleDENCLUE, np.ndarray]:
    rows = []
    best_score = -np.inf
    best_model: SimpleDENCLUE | None = None
    best_labels: np.ndarray | None = None

    for sigma in SIGMA_GRID:
        for xi_quantile in XI_GRID:
            start = time.perf_counter()
            model = SimpleDENCLUE(
                sigma=sigma,
                xi_quantile=xi_quantile,
                max_iter=35,
                max_anchors=2000,
                random_state=RANDOM_STATE,
            )
            labels = model.fit_predict(X)
            runtime_seconds = time.perf_counter() - start
            metrics = evaluate_labels(X, y_true, labels)
            score = metrics["ari"]
            rows.append(
                {
                    "sigma": sigma,
                    "xi_quantile": xi_quantile,
                    "runtime_seconds": runtime_seconds,
                    **metrics,
                    "selection_score": score,
                }
            )
            if score > best_score:
                best_score = score
                best_model = model
                best_model.runtime_seconds_ = runtime_seconds
                best_labels = labels

    if best_model is None or best_labels is None:
        raise RuntimeError("Tuning DENCLUE không tạo được kết quả.")

    tuning = pd.DataFrame(rows)
    tuning["selected"] = (tuning["sigma"] == best_model.sigma) & (
        tuning["xi_quantile"] == best_model.xi_quantile
    )
    return tuning.sort_values(["selected", "ari"], ascending=[False, False]), best_model, best_labels


def tune_dbscan(X: np.ndarray, y_true: np.ndarray) -> tuple[pd.DataFrame, np.ndarray, dict[str, float]]:
    rows = []
    best_score = -np.inf
    best_labels: np.ndarray | None = None
    best_row: dict[str, float] | None = None

    for eps in EPS_GRID:
        for min_samples in MIN_SAMPLES_GRID:
            start = time.perf_counter()
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
            runtime_seconds = time.perf_counter() - start
            metrics = evaluate_labels(X, y_true, labels)
            score = metrics["ari"]
            row = {
                "eps": eps,
                "min_samples": min_samples,
                "runtime_seconds": runtime_seconds,
                **metrics,
                "selection_score": score,
            }
            rows.append(row)
            if score > best_score:
                best_score = score
                best_labels = labels
                best_row = row

    if best_labels is None or best_row is None:
        raise RuntimeError("Tuning DBSCAN không tạo được kết quả.")

    tuning = pd.DataFrame(rows)
    tuning["selected"] = (tuning["eps"] == best_row["eps"]) & (
        tuning["min_samples"] == best_row["min_samples"]
    )
    return tuning.sort_values(["selected", "ari"], ascending=[False, False]), best_labels, best_row


def run_baselines(X: np.ndarray, y_true: np.ndarray) -> dict[str, tuple[np.ndarray, dict[str, float]]]:
    true_k = len(np.unique(y_true))
    output: dict[str, tuple[np.ndarray, dict[str, float]]] = {}

    start = time.perf_counter()
    labels = KMeans(n_clusters=true_k, n_init=20, random_state=RANDOM_STATE).fit_predict(X)
    output["KMeans"] = (labels, {"n_clusters": true_k, "runtime_seconds": time.perf_counter() - start})

    start = time.perf_counter()
    labels = GaussianMixture(n_components=true_k, random_state=RANDOM_STATE).fit_predict(X)
    output["GMM"] = (labels, {"n_components": true_k, "runtime_seconds": time.perf_counter() - start})

    start = time.perf_counter()
    bandwidth = estimate_bandwidth(X, quantile=0.20, n_samples=min(500, len(X)), random_state=RANDOM_STATE)
    if bandwidth and not np.isnan(bandwidth) and bandwidth > 0:
        labels = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit_predict(X)
        output["MeanShift"] = (
            labels,
            {"bandwidth": float(bandwidth), "runtime_seconds": time.perf_counter() - start},
        )

    return output


def plot_case_comparison(
    df: pd.DataFrame,
    labels_by_model: dict[str, np.ndarray],
    case: BenchmarkCase,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    fig.suptitle(f"Benchmark {case.case_id.title()}", fontsize=15, weight="bold")

    for ax, (model_name, labels) in zip(axes.ravel(), labels_by_model.items()):
        labels = np.asarray(labels)
        values = sorted(np.unique(labels))
        cmap = plt.get_cmap("tab20", max(len(values), 1))

        color_idx = 0
        for value in values:
            mask = labels == value
            if value == -1:
                ax.scatter(df.loc[mask, "x"], df.loc[mask, "y"], s=18, c="#9CA3AF", alpha=0.65, linewidths=0)
            else:
                ax.scatter(
                    df.loc[mask, "x"],
                    df.loc[mask, "y"],
                    s=18,
                    color=cmap(color_idx % 20),
                    alpha=0.78,
                    linewidths=0,
                )
                color_idx += 1

        if model_name == "True Label":
            title = f"Nhãn thật (k={len(np.unique(labels))})"
        else:
            y_true = df["true_label_code"].to_numpy()
            metrics = evaluate_labels(scale_xy(df), y_true, labels)
            title = (
                f"{model_name}: ARI={metrics['ari']:.3f}, "
                f"k={metrics['clusters']}, noise={metrics['noise_rate']:.2f}"
            )
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#DDDDDD")

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_case(case: BenchmarkCase, data_dir: Path, figures_dir: Path, tables_dir: Path) -> tuple[list[dict], pd.DataFrame]:
    raw_df = read_case(data_dir / case.file_name)
    raw_df["true_label_code"] = raw_df["true_label"].astype("category").cat.codes
    X = scale_xy(raw_df)
    y_true = raw_df["true_label_code"].to_numpy()

    denclue_tuning, denclue_model, denclue_labels = tune_denclue(X, y_true)
    dbscan_tuning, dbscan_labels, dbscan_best = tune_dbscan(X, y_true)
    baseline_outputs = run_baselines(X, y_true)

    denclue_tuning.insert(0, "case_id", case.case_id)
    dbscan_tuning.insert(0, "case_id", case.case_id)
    denclue_tuning.to_csv(tables_dir / f"ml_{case.case_id}_denclue_tuning.csv", index=False)
    dbscan_tuning.to_csv(tables_dir / f"ml_{case.case_id}_dbscan_tuning.csv", index=False)

    labels_by_model = {
        "True Label": y_true,
        "DENCLUE": denclue_labels,
        "DBSCAN": dbscan_labels,
    }
    labels_by_model.update({model_name: labels for model_name, (labels, _) in baseline_outputs.items()})

    figure_prefix = "ml_benchmark" if case.role == "denclue_strong_case" else "ml_failure"
    plot_case_comparison(
        raw_df,
        labels_by_model,
        case,
        figures_dir / f"{figure_prefix}_{case.case_id}_comparison.png",
    )

    label_frame = raw_df[["x", "y", "true_label"]].copy()
    label_frame.insert(0, "case_id", case.case_id)
    for model_name, labels in labels_by_model.items():
        if model_name != "True Label":
            label_frame[f"{model_name.lower()}_label"] = labels

    rows = []
    denclue_metrics = evaluate_labels(X, y_true, denclue_labels)
    rows.append(
        {
            "case_id": case.case_id,
            "role": case.role,
            "model": "DENCLUE",
            "parameters": f"sigma={denclue_model.sigma}, xi={denclue_model.xi_quantile}",
            "runtime_seconds": getattr(denclue_model, "runtime_seconds_", np.nan),
            **denclue_metrics,
            "case_note": case.case_note,
        }
    )
    rows.append(
        {
            "case_id": case.case_id,
            "role": case.role,
            "model": "DBSCAN",
            "parameters": f"eps={dbscan_best['eps']}, min_samples={dbscan_best['min_samples']}",
            "runtime_seconds": dbscan_best["runtime_seconds"],
            **evaluate_labels(X, y_true, dbscan_labels),
            "case_note": case.case_note,
        }
    )
    for model_name, (labels, params) in baseline_outputs.items():
        rows.append(
            {
                "case_id": case.case_id,
                "role": case.role,
                "model": model_name,
                "parameters": ", ".join(f"{key}={value}" for key, value in params.items() if key != "runtime_seconds"),
                "runtime_seconds": params["runtime_seconds"],
                **evaluate_labels(X, y_true, labels),
                "case_note": case.case_note,
            }
        )

    return rows, label_frame


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(project_dir / "data" / "raw" / "benchmark_cases"))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    figures_dir = project_dir / "outputs" / "figures"
    tables_dir = project_dir / "outputs" / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    metric_rows = []
    label_frames = []
    for case in CASES:
        rows, labels = run_case(case, data_dir, figures_dir, tables_dir)
        metric_rows.extend(rows)
        label_frames.append(labels)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(tables_dir / "ml_benchmark_case_metrics.csv", index=False)
    pd.concat(label_frames, ignore_index=True).to_csv(tables_dir / "ml_benchmark_case_labels.csv", index=False)

    summary = (
        metrics.sort_values(["case_id", "ari"], ascending=[True, False])
        .groupby("case_id", as_index=False)
        .first()[["case_id", "role", "model", "ari", "nmi", "clusters", "noise_rate", "case_note"]]
        .rename(columns={"model": "best_model", "ari": "best_ari", "nmi": "best_nmi"})
    )
    summary.to_csv(tables_dir / "ml_benchmark_case_summary.csv", index=False)

    print("Xong.")
    print(metrics.sort_values(["case_id", "ari"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()

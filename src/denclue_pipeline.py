#!/usr/bin/env python3
"""
Quy trình DENCLUE cho dữ liệu NASA Landslide Events.

Bài này tập trung vào phân cụm theo vị trí:
- mỗi sự kiện sạt lở được biểu diễn bằng kinh độ/vĩ độ;
- DENCLUE tìm các vùng sự kiện có mật độ cao;
- điểm mật độ thấp hoặc nằm riêng lẻ được xem là noise;
- KMeans, DBSCAN, GMM và MeanShift được dùng để so sánh.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans, MeanShift, estimate_bandwidth
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
SPATIAL_FEATURES = ["longitude", "latitude"]
PROFILE_COLUMNS = ["country_name", "landslide_type", "landslide_size", "trigger"]
SPATIAL_VIEW_PADDING = 0.05


def ensure_dirs(project_dir: Path) -> tuple[Path, Path]:
    figures_dir = project_dir / "outputs" / "figures"
    tables_dir = project_dir / "outputs" / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir, tables_dir


def normalize_text_column(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.replace("_", " ", regex=False).str.title()


def read_landslide_catalog(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], format="%m/%d/%y", errors="coerce")

    for col in ["landslide_type", "landslide_size", "trigger"]:
        if col in df.columns:
            df[col] = normalize_text_column(df[col])

    return df


def build_model_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    missing_cols = sorted(set(SPATIAL_FEATURES) - set(raw_df.columns))
    if missing_cols:
        raise ValueError(f"Thiếu cột tọa độ bắt buộc: {missing_cols}")

    model_df = raw_df.copy()
    model_df = model_df.dropna(subset=SPATIAL_FEATURES)
    model_df = model_df[
        model_df["longitude"].between(-180, 180) & model_df["latitude"].between(-90, 90)
    ].copy()

    keep_cols = [
        "id",
        "date",
        "country_name",
        "state/province",
        "city/town",
        "latitude",
        "longitude",
        "landslide_type",
        "landslide_size",
        "trigger",
        "injuries",
        "fatalities",
        "source_name",
    ]
    keep_cols = [col for col in keep_cols if col in model_df.columns]
    return model_df[keep_cols].reset_index(drop=True)


def standardize_coordinates(model_df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    X = scaler.fit_transform(model_df[SPATIAL_FEATURES].to_numpy(dtype=float))
    return X, scaler


@dataclass
class SimpleDENCLUE:
    """Bản tự cài đặt phần chính của DENCLUE để minh họa điểm hút mật độ."""

    sigma: float = 0.235
    xi_quantile: float = 0.002
    merge_eps: float | None = None
    merge_min_points: int = 3
    max_iter: int = 35
    tol: float = 1e-4
    max_anchors: int = 2000
    batch_size: int = 256
    random_state: int = RANDOM_STATE

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.random_state)
        n_rows = X.shape[0]

        # Lấy một tập điểm mốc để giảm chi phí tính mật độ khi dữ liệu lớn.
        if n_rows > self.max_anchors:
            anchor_idx = rng.choice(n_rows, size=self.max_anchors, replace=False)
            anchors = X[anchor_idx]
        else:
            anchors = X

        attractors = X.copy()
        # Kéo từng điểm về điểm hút mật độ bằng cập nhật trung bình có trọng số.
        for _ in range(self.max_iter):
            max_shift = 0.0
            updated_batches = []
            for start in range(0, n_rows, self.batch_size):
                batch = attractors[start : start + self.batch_size]
                next_batch = self._mean_shift_step(batch, anchors)
                shift = np.linalg.norm(next_batch - batch, axis=1)
                max_shift = max(max_shift, float(np.max(shift)))
                updated_batches.append(next_batch)
            attractors = np.vstack(updated_batches)
            if max_shift < self.tol:
                break

        density = self._density(attractors, anchors)
        density_threshold = float(np.quantile(density, self.xi_quantile))
        valid = density >= density_threshold

        labels = np.full(n_rows, -1, dtype=int)
        if valid.sum() >= 2:
            eps = self.merge_eps if self.merge_eps is not None else max(self.sigma * 0.50, 1e-6)
            merged = self._merge_attractors(attractors[valid], eps=eps)
            labels[valid] = merged
            unique = sorted(value for value in np.unique(labels) if value != -1)
            remap = {old: new for new, old in enumerate(unique)}
            labels = np.array([remap.get(value, -1) for value in labels], dtype=int)

        self.attractors_ = attractors
        self.density_ = density
        self.density_threshold_ = density_threshold
        self.labels_ = labels
        return labels

    def _mean_shift_step(self, points: np.ndarray, anchors: np.ndarray) -> np.ndarray:
        dist2 = ((points[:, None, :] - anchors[None, :, :]) ** 2).sum(axis=2)
        weights = np.exp(-dist2 / (2.0 * self.sigma**2))
        denom = weights.sum(axis=1, keepdims=True)
        denom[denom == 0.0] = 1.0
        return (weights @ anchors) / denom

    def _density(self, points: np.ndarray, anchors: np.ndarray) -> np.ndarray:
        densities = []
        for start in range(0, points.shape[0], self.batch_size):
            batch = points[start : start + self.batch_size]
            dist2 = ((batch[:, None, :] - anchors[None, :, :]) ** 2).sum(axis=2)
            densities.append(np.exp(-dist2 / (2.0 * self.sigma**2)).sum(axis=1))
        return np.concatenate(densities)

    def _merge_attractors(self, attractors: np.ndarray, eps: float) -> np.ndarray:
        n_rows = attractors.shape[0]
        labels = np.full(n_rows, -1, dtype=int)
        if n_rows == 0:
            return labels

        neighbors = NearestNeighbors(radius=eps).fit(attractors).radius_neighbors(
            attractors,
            return_distance=False,
        )

        core_mask = np.array([len(items) >= self.merge_min_points for items in neighbors], dtype=bool)
        parent = np.arange(n_rows)

        def find(idx: int) -> int:
            while parent[idx] != idx:
                parent[idx] = parent[parent[idx]]
                idx = parent[idx]
            return idx

        def union(left: int, right: int) -> None:
            root_left = find(left)
            root_right = find(right)
            if root_left != root_right:
                parent[root_right] = root_left

        core_idx = np.flatnonzero(core_mask)
        for left in core_idx:
            for right in neighbors[left]:
                if right > left and core_mask[right]:
                    union(left, right)

        root_to_label: dict[int, int] = {}
        for idx in core_idx:
            root = find(idx)
            if root not in root_to_label:
                root_to_label[root] = len(root_to_label)
            labels[idx] = root_to_label[root]

        for idx in np.flatnonzero(~core_mask):
            near_core = [item for item in neighbors[idx] if core_mask[item]]
            if len(near_core) > 0:
                labels[idx] = labels[near_core[0]]

        return labels


def cluster_count(labels: np.ndarray) -> int:
    return len([value for value in np.unique(labels) if value != -1])


def noise_rate(labels: np.ndarray) -> float:
    return float(np.mean(labels == -1))


def external_label_codes(series: pd.Series | None) -> np.ndarray | None:
    if series is None:
        return None
    mask = series.notna()
    if mask.sum() == 0:
        return None
    return pd.Series(series[mask]).astype("category").cat.codes.to_numpy()


def safe_cluster_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    external_label: pd.Series | None = None,
) -> dict[str, float]:
    labels = np.asarray(labels)
    metric_mask = labels != -1
    result: dict[str, float] = {
        "clusters": cluster_count(labels),
        "noise_rate": noise_rate(labels),
        "internal_metric_points": int(metric_mask.sum()),
        "silhouette": np.nan,
        "calinski_harabasz": np.nan,
        "davies_bouldin": np.nan,
        "ari_landslide_type": np.nan,
        "nmi_landslide_type": np.nan,
    }

    metric_labels = labels[metric_mask]
    metric_X = X[metric_mask]
    unique = np.unique(metric_labels)
    if 1 < len(unique) < len(metric_labels):
        try:
            result["silhouette"] = float(silhouette_score(metric_X, metric_labels))
            result["calinski_harabasz"] = float(calinski_harabasz_score(metric_X, metric_labels))
            result["davies_bouldin"] = float(davies_bouldin_score(metric_X, metric_labels))
        except ValueError:
            pass

    if external_label is not None:
        mask = external_label.notna().to_numpy()
        if mask.sum() > 0:
            y_codes = pd.Series(external_label[mask]).astype("category").cat.codes.to_numpy()
            result["ari_landslide_type"] = float(adjusted_rand_score(y_codes, labels[mask]))
            result["nmi_landslide_type"] = float(normalized_mutual_info_score(y_codes, labels[mask]))

    return result


def tune_denclue(
    X: np.ndarray,
    external_label: pd.Series,
    sigma_grid: list[float],
    xi_grid: list[float],
    max_anchors: int,
) -> tuple[pd.DataFrame, SimpleDENCLUE, np.ndarray]:
    rows = []
    best_score = -np.inf
    best_model: SimpleDENCLUE | None = None
    best_labels: np.ndarray | None = None

    for sigma in sigma_grid:
        for xi_quantile in xi_grid:
            start = time.perf_counter()
            model = SimpleDENCLUE(
                sigma=sigma,
                xi_quantile=xi_quantile,
                max_anchors=max_anchors,
                random_state=RANDOM_STATE,
            )
            labels = model.fit_predict(X)
            runtime = time.perf_counter() - start
            model.runtime_seconds_ = runtime

            metrics = safe_cluster_metrics(X, labels, external_label)
            candidate_ok = 6 <= metrics["clusters"] <= 10 and metrics["noise_rate"] <= 0.35
            silhouette = metrics["silhouette"]
            score = (
                silhouette - metrics["noise_rate"]
                if candidate_ok and not np.isnan(silhouette)
                else -np.inf
            )
            row = {
                "sigma": sigma,
                "xi_quantile": xi_quantile,
                "runtime_seconds": runtime,
                "candidate_ok": candidate_ok,
                "selection_score": score,
                **metrics,
            }
            rows.append(row)

            if score > best_score:
                best_score = score
                best_model = model
                best_labels = labels

    if best_model is None or best_labels is None:
        raise RuntimeError("Tuning DENCLUE không tạo được kết quả dùng được.")

    tuning = pd.DataFrame(rows)
    tuning["selected"] = (tuning["sigma"] == best_model.sigma) & (
        tuning["xi_quantile"] == best_model.xi_quantile
    )
    tuning = tuning.sort_values(
        ["selected", "candidate_ok", "silhouette", "nmi_landslide_type"],
        ascending=[False, False, False, False],
    )
    return tuning, best_model, best_labels


def tune_dbscan(
    X: np.ndarray,
    external_label: pd.Series,
    eps_grid: list[float],
    min_samples_grid: list[int],
) -> tuple[pd.DataFrame, np.ndarray, dict[str, float]]:
    rows = []
    best_score = -np.inf
    best_labels: np.ndarray | None = None
    best_row: dict[str, float] | None = None

    for eps in eps_grid:
        for min_samples in min_samples_grid:
            start = time.perf_counter()
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
            runtime = time.perf_counter() - start
            metrics = safe_cluster_metrics(X, labels, external_label)
            candidate_ok = 2 <= metrics["clusters"] <= 12 and metrics["noise_rate"] <= 0.45
            silhouette = metrics["silhouette"]
            score = (
                silhouette - metrics["noise_rate"]
                if candidate_ok and not np.isnan(silhouette)
                else -np.inf
            )
            row = {
                "eps": eps,
                "min_samples": min_samples,
                "runtime_seconds": runtime,
                "candidate_ok": candidate_ok,
                "selection_score": score,
                **metrics,
            }
            rows.append(row)
            if score > best_score:
                best_score = score
                best_labels = labels
                best_row = row

    if best_labels is None or best_row is None:
        raise RuntimeError("Tuning DBSCAN không tạo được kết quả dùng được.")

    tuning = pd.DataFrame(rows)
    tuning["selected"] = (tuning["eps"] == best_row["eps"]) & (
        tuning["min_samples"] == best_row["min_samples"]
    )
    tuning = tuning.sort_values(["selected", "candidate_ok", "silhouette"], ascending=[False, False, False])
    return tuning, best_labels, best_row


def run_baselines(
    X: np.ndarray,
    external_label: pd.Series,
    denclue_k: int,
) -> dict[str, tuple[np.ndarray, dict[str, float]]]:
    n_clusters = min(max(denclue_k, 2), 10)
    output: dict[str, tuple[np.ndarray, dict[str, float]]] = {}

    start = time.perf_counter()
    kmeans_labels = KMeans(n_clusters=n_clusters, n_init=20, random_state=RANDOM_STATE).fit_predict(X)
    output["KMeans"] = (
        kmeans_labels,
        {"n_clusters": n_clusters, "runtime_seconds": time.perf_counter() - start},
    )

    start = time.perf_counter()
    gmm_labels = GaussianMixture(n_components=n_clusters, random_state=RANDOM_STATE).fit_predict(X)
    output["GMM"] = (
        gmm_labels,
        {"n_components": n_clusters, "runtime_seconds": time.perf_counter() - start},
    )

    try:
        start = time.perf_counter()
        bandwidth = estimate_bandwidth(
            X,
            quantile=0.20,
            n_samples=min(1000, len(X)),
            random_state=RANDOM_STATE,
        )
        if bandwidth and not np.isnan(bandwidth) and bandwidth > 0:
            labels = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit_predict(X)
            output["MeanShift"] = (
                labels,
                {"bandwidth": float(bandwidth), "runtime_seconds": time.perf_counter() - start},
            )
    except Exception as exc:
        print(f"Bỏ qua MeanShift: {exc}")

    return output


def top_value(series: pd.Series) -> str:
    clean = series.dropna().astype(str)
    if clean.empty:
        return ""
    return str(clean.value_counts().idxmax())


def cluster_profile(model_df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    frame = model_df.copy()
    frame["cluster"] = labels

    rows = []
    for label, group in frame.groupby("cluster"):
        row = {
            "cluster": label,
            "rows": len(group),
            "share": len(group) / len(frame),
            "mean_longitude": group["longitude"].mean(),
            "mean_latitude": group["latitude"].mean(),
            "min_longitude": group["longitude"].min(),
            "max_longitude": group["longitude"].max(),
            "min_latitude": group["latitude"].min(),
            "max_latitude": group["latitude"].max(),
        }
        for col in PROFILE_COLUMNS:
            if col in group.columns:
                row[f"top_{col}"] = top_value(group[col])
        rows.append(row)

    return pd.DataFrame(rows).sort_values("cluster")


def plot_missing(raw_df: pd.DataFrame, path: Path) -> None:
    missing = raw_df.isna().mean().sort_values(ascending=False)
    missing = missing[missing > 0].head(15)
    plt.figure(figsize=(9, 5))
    missing.sort_values().plot(kind="barh", color="#4C78A8")
    plt.xlabel("Missing rate")
    plt.title("Missing values trong Landslide Catalog")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_category_distribution(model_df: pd.DataFrame, column: str, path: Path, title: str) -> None:
    counts = model_df[column].fillna("Unknown").value_counts().head(12)
    plt.figure(figsize=(9, 5))
    counts.sort_values().plot(kind="barh", color="#59A14F")
    plt.xlabel("Rows")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def spatial_view_bounds(model_df: pd.DataFrame) -> tuple[float, float, float, float]:
    min_lon = float(model_df["longitude"].min())
    max_lon = float(model_df["longitude"].max())
    min_lat = float(model_df["latitude"].min())
    max_lat = float(model_df["latitude"].max())
    lon_pad = max((max_lon - min_lon) * SPATIAL_VIEW_PADDING, 1.0)
    lat_pad = max((max_lat - min_lat) * SPATIAL_VIEW_PADDING, 1.0)
    return min_lon - lon_pad, max_lon + lon_pad, min_lat - lat_pad, max_lat + lat_pad


def apply_spatial_view(model_df: pd.DataFrame) -> None:
    min_lon, max_lon, min_lat, max_lat = spatial_view_bounds(model_df)
    plt.xlim(min_lon, max_lon)
    plt.ylim(min_lat, max_lat)


def plot_world_scatter(model_df: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(9, 6))
    plt.scatter(model_df["longitude"], model_df["latitude"], s=12, alpha=0.65, color="#4C78A8")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Vị trí các sự kiện sạt lở")
    apply_spatial_view(model_df)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_clusters(model_df: pd.DataFrame, labels: np.ndarray, title: str, path: Path) -> None:
    values = pd.Series(labels).astype(str)
    plt.figure(figsize=(9, 6))
    for idx, value in enumerate(values.unique()):
        mask = values == value
        color = "#9E9E9E" if value == "-1" else plt.get_cmap("tab20")(idx % 20)
        label = "Noise" if value == "-1" else f"Cụm {value}"
        plt.scatter(
            model_df.loc[mask, "longitude"],
            model_df.loc[mask, "latitude"],
            s=14,
            alpha=0.70,
            color=color,
            label=label,
        )
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(title)
    apply_spatial_view(model_df)
    plt.legend(fontsize=8, loc="best", markerscale=1.4)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_density(model_df: pd.DataFrame, density: np.ndarray, path: Path) -> None:
    plt.figure(figsize=(9, 6))
    scatter = plt.scatter(
        model_df["longitude"],
        model_df["latitude"],
        c=density,
        s=14,
        alpha=0.75,
        cmap="viridis",
    )
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Mật độ ước lượng của DENCLUE")
    apply_spatial_view(model_df)
    plt.colorbar(scatter, shrink=0.78, label="Mật độ ước lượng")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def write_eda_outputs(
    raw_df: pd.DataFrame,
    model_df: pd.DataFrame,
    figures_dir: Path,
    tables_dir: Path,
) -> None:
    summary = pd.DataFrame(
        [
            {
                "dataset": "NASA Landslide Events",
                "raw_rows": len(raw_df),
                "raw_columns": raw_df.shape[1],
                "model_rows": len(model_df),
                "model_columns": model_df.shape[1],
                "features": "|".join(SPATIAL_FEATURES),
                "external_label_for_interpretation": "landslide_type",
            }
        ]
    )
    summary.to_csv(tables_dir / "ml_dataset_summary.csv", index=False)
    raw_df.isna().mean().sort_values(ascending=False).to_csv(
        tables_dir / "ml_missing_rates.csv", header=["missing_rate"]
    )
    pd.Series(SPATIAL_FEATURES, name="used_source_columns").to_csv(
        tables_dir / "ml_used_columns.csv", index=False
    )

    validation = pd.DataFrame(
        [
            {"check": "Chỉ dùng longitude và latitude làm feature clustering", "status": "PASS"},
            {"check": "Đã loại dòng thiếu tọa độ", "status": "PASS"},
            {"check": "Tọa độ nằm trong khoảng địa lý hợp lệ", "status": "PASS"},
            {"check": "Đã chuẩn hóa tọa độ trước khi clustering", "status": "PASS"},
            {"check": "Type/size/trigger chỉ dùng để diễn giải", "status": "PASS"},
        ]
    )
    validation.to_csv(tables_dir / "ml_preprocessing_validation.csv", index=False)

    for col in PROFILE_COLUMNS:
        if col in model_df.columns:
            model_df[col].fillna("Unknown").value_counts().to_csv(
                tables_dir / f"ml_{col}_distribution.csv", header=["rows"]
            )

    plot_missing(raw_df, figures_dir / "ml_missing_values.png")
    plot_world_scatter(model_df, figures_dir / "ml_landslide_locations.png")
    if "landslide_type" in model_df.columns:
        plot_category_distribution(
            model_df,
            "landslide_type",
            figures_dir / "ml_landslide_type_distribution.png",
            "Phân bố landslide type",
        )
    if "landslide_size" in model_df.columns:
        plot_category_distribution(
            model_df,
            "landslide_size",
            figures_dir / "ml_landslide_size_distribution.png",
            "Phân bố landslide size",
        )


def write_model_outputs(
    model_df: pd.DataFrame,
    X: np.ndarray,
    denclue_model: SimpleDENCLUE,
    denclue_labels: np.ndarray,
    dbscan_labels: np.ndarray,
    dbscan_best: dict[str, float],
    baseline_outputs: dict[str, tuple[np.ndarray, dict[str, float]]],
    figures_dir: Path,
    tables_dir: Path,
) -> None:
    external_label = model_df["landslide_type"] if "landslide_type" in model_df.columns else None

    def dbscan_alignment(labels: np.ndarray) -> dict[str, float]:
        return {
            "ari_vs_dbscan": float(adjusted_rand_score(dbscan_labels, labels)),
            "nmi_vs_dbscan": float(normalized_mutual_info_score(dbscan_labels, labels)),
        }

    labels_by_model = {"DENCLUE": denclue_labels, "DBSCAN": dbscan_labels}
    rows = []
    denclue_row = {
        "model": "DENCLUE",
        "sigma": denclue_model.sigma,
        "xi_quantile": denclue_model.xi_quantile,
        "runtime_seconds": getattr(denclue_model, "runtime_seconds_", np.nan),
        **safe_cluster_metrics(X, denclue_labels, external_label),
        **dbscan_alignment(denclue_labels),
    }
    rows.append(denclue_row)

    dbscan_row = {
        "model": "DBSCAN",
        "eps": dbscan_best["eps"],
        "min_samples": dbscan_best["min_samples"],
        "runtime_seconds": dbscan_best["runtime_seconds"],
        **safe_cluster_metrics(X, dbscan_labels, external_label),
        **dbscan_alignment(dbscan_labels),
    }
    rows.append(dbscan_row)

    for model_name, (labels, params) in baseline_outputs.items():
        labels_by_model[model_name] = labels
        row = {
            "model": model_name,
            **params,
            **safe_cluster_metrics(X, labels, external_label),
            **dbscan_alignment(labels),
        }
        rows.append(row)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(tables_dir / "ml_clustering_metrics.csv", index=False)

    label_frame = model_df.copy()
    for model_name, labels in labels_by_model.items():
        label_frame[f"{model_name}_label"] = labels
    label_frame.to_csv(tables_dir / "ml_cluster_labels.csv", index=False)

    cluster_profile(model_df, denclue_labels).to_csv(
        tables_dir / "ml_denclue_cluster_profile.csv", index=False
    )
    if "landslide_type" in model_df.columns:
        pd.crosstab(denclue_labels, model_df["landslide_type"], normalize="index").to_csv(
            tables_dir / "ml_denclue_cluster_type_share.csv"
        )

    plot_clusters(model_df, denclue_labels, "Cụm DENCLUE trên Landslide", figures_dir / "ml_denclue_clusters.png")
    plot_density(model_df, denclue_model.density_, figures_dir / "ml_denclue_density.png")
    plot_clusters(model_df, dbscan_labels, "Cụm DBSCAN trên Landslide", figures_dir / "ml_dbscan_clusters.png")
    for model_name, (labels, _) in baseline_outputs.items():
        plot_clusters(
            model_df,
            labels,
            f"Cụm {model_name} trên Landslide",
            figures_dir / f"ml_{model_name.lower()}_clusters.png",
        )


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(project_dir / "data" / "raw" / "landslide_catalog.csv"))
    parser.add_argument("--max-anchors", type=int, default=2000)
    parser.add_argument("--sigma-grid", default="0.18,0.22,0.235,0.24,0.25,0.35,0.50")
    parser.add_argument("--xi-grid", default="0.002,0.005,0.01,0.02,0.05,0.10")
    parser.add_argument("--dbscan-eps-grid", default="0.20,0.25,0.30,0.35,0.40,0.50,0.65")
    parser.add_argument("--dbscan-min-samples-grid", default="5,8,10,12")
    args = parser.parse_args()

    figures_dir, tables_dir = ensure_dirs(project_dir)
    csv_path = Path(args.csv)

    raw_df = read_landslide_catalog(csv_path)
    model_df = build_model_frame(raw_df)
    X, _ = standardize_coordinates(model_df)
    external_label = model_df["landslide_type"] if "landslide_type" in model_df.columns else pd.Series(dtype=str)

    write_eda_outputs(raw_df, model_df, figures_dir, tables_dir)

    sigma_grid = [float(value) for value in args.sigma_grid.split(",")]
    xi_grid = [float(value) for value in args.xi_grid.split(",")]
    denclue_tuning, denclue_model, denclue_labels = tune_denclue(
        X=X,
        external_label=external_label,
        sigma_grid=sigma_grid,
        xi_grid=xi_grid,
        max_anchors=args.max_anchors,
    )
    denclue_tuning.to_csv(tables_dir / "ml_denclue_tuning.csv", index=False)

    eps_grid = [float(value) for value in args.dbscan_eps_grid.split(",")]
    min_samples_grid = [int(value) for value in args.dbscan_min_samples_grid.split(",")]
    dbscan_tuning, dbscan_labels, dbscan_best = tune_dbscan(
        X=X,
        external_label=external_label,
        eps_grid=eps_grid,
        min_samples_grid=min_samples_grid,
    )
    dbscan_tuning.to_csv(tables_dir / "ml_dbscan_tuning.csv", index=False)

    baselines = run_baselines(X, external_label, denclue_k=cluster_count(denclue_labels))
    write_model_outputs(
        model_df=model_df,
        X=X,
        denclue_model=denclue_model,
        denclue_labels=denclue_labels,
        dbscan_labels=dbscan_labels,
        dbscan_best=dbscan_best,
        baseline_outputs=baselines,
        figures_dir=figures_dir,
        tables_dir=tables_dir,
    )

    print("Xong.")
    print(f"Số dòng dùng để clustering: {len(model_df)}")
    print(
        "DENCLUE được chọn:",
        f"sigma={denclue_model.sigma}",
        f"xi={denclue_model.xi_quantile}",
        f"clusters={cluster_count(denclue_labels)}",
        f"noise={noise_rate(denclue_labels):.3f}",
    )
    print(
        "DBSCAN được chọn:",
        f"eps={dbscan_best['eps']}",
        f"min_samples={dbscan_best['min_samples']}",
        f"clusters={dbscan_best['clusters']}",
        f"noise={dbscan_best['noise_rate']:.3f}",
    )


if __name__ == "__main__":
    main()

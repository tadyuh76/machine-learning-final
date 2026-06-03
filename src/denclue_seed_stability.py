#!/usr/bin/env python3
"""
Kiểm tra nhanh độ ổn định theo seed cho các mô hình clustering.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from denclue_pipeline import (
    SimpleDENCLUE,
    build_model_frame,
    ensure_dirs,
    read_landslide_catalog,
    safe_cluster_metrics,
    standardize_coordinates,
)
from sklearn.cluster import DBSCAN, KMeans, MeanShift, estimate_bandwidth
from sklearn.mixture import GaussianMixture


def parse_seeds(raw: str) -> list[int]:
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def stability_summary(rows: list[dict[str, float | int | str]]) -> pd.DataFrame:
    metrics = pd.DataFrame(rows)
    summary = metrics.groupby("model").agg(
        runs=("seed", "count"),
        clusters_mean=("clusters", "mean"),
        clusters_std=("clusters", "std"),
        noise_mean=("noise_rate", "mean"),
        noise_std=("noise_rate", "std"),
        silhouette_mean=("silhouette", "mean"),
        silhouette_std=("silhouette", "std"),
        ari_mean=("ari_landslide_type", "mean"),
        ari_std=("ari_landslide_type", "std"),
        nmi_mean=("nmi_landslide_type", "mean"),
        nmi_std=("nmi_landslide_type", "std"),
    )
    return summary.reset_index()


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(project_dir / "data" / "raw" / "landslide_catalog.csv"))
    parser.add_argument("--seeds", default="11,23,42,58,91")
    parser.add_argument("--denclue-sigma", type=float, default=0.235)
    parser.add_argument("--denclue-xi", type=float, default=0.002)
    parser.add_argument("--max-anchors", type=int, default=2000)
    parser.add_argument("--dbscan-eps", type=float, default=0.40)
    parser.add_argument("--dbscan-min-samples", type=int, default=12)
    parser.add_argument("--n-clusters", type=int, default=7)
    args = parser.parse_args()

    _, tables_dir = ensure_dirs(project_dir)
    seeds = parse_seeds(args.seeds)

    raw_df = read_landslide_catalog(Path(args.csv))
    model_df = build_model_frame(raw_df)
    X, _ = standardize_coordinates(model_df)
    external_label = model_df["landslide_type"] if "landslide_type" in model_df.columns else pd.Series(dtype=str)

    rows: list[dict[str, float | int | str]] = []
    for seed in seeds:
        models = []
        models.append(
            (
                "DENCLUE",
                lambda seed=seed: SimpleDENCLUE(
                    sigma=args.denclue_sigma,
                    xi_quantile=args.denclue_xi,
                    max_anchors=args.max_anchors,
                    random_state=seed,
                ).fit_predict(X),
            )
        )
        models.append(
            (
                "DBSCAN",
                lambda: DBSCAN(eps=args.dbscan_eps, min_samples=args.dbscan_min_samples).fit_predict(X),
            )
        )
        models.append(
            (
                "KMeans",
                lambda seed=seed: KMeans(n_clusters=args.n_clusters, n_init=20, random_state=seed).fit_predict(X),
            )
        )
        models.append(
            (
                "GMM",
                lambda seed=seed: GaussianMixture(n_components=args.n_clusters, random_state=seed).fit_predict(X),
            )
        )
        models.append(
            (
                "MeanShift",
                lambda seed=seed: MeanShift(
                    bandwidth=estimate_bandwidth(
                        X,
                        quantile=0.20,
                        n_samples=min(1000, len(X)),
                        random_state=seed,
                    ),
                    bin_seeding=True,
                ).fit_predict(X),
            )
        )

        for name, runner in models:
            start = time.perf_counter()
            labels = runner()
            row = {
                "seed": seed,
                "model": name,
                "runtime_seconds": time.perf_counter() - start,
                **safe_cluster_metrics(X, labels, external_label),
            }
            rows.append(row)

    metrics = pd.DataFrame(rows)
    summary = stability_summary(rows)
    metrics.to_csv(tables_dir / "ml_seed_stability_metrics.csv", index=False)
    summary.to_csv(tables_dir / "ml_seed_stability_summary.csv", index=False)

    print("Xong kiểm tra seed.")
    print(summary[["model", "runs", "clusters_mean", "clusters_std", "silhouette_mean", "silhouette_std"]])


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REPLICATE_COLUMNS = ["Gab", "Leo", "Nic"]


def short_plant_label(plant_id: str) -> str:
    plant_id = str(plant_id)
    return plant_id.split("_")[-1] if "_" in plant_id else plant_id


def add_automatic_image_index(auto: pd.DataFrame) -> pd.DataFrame:
    auto = auto.copy()

    auto["plant_label"] = auto["plant_id"].apply(short_plant_label)

    if "relative_path" in auto.columns:
        sort_columns = ["plant_label", "relative_path"]
    else:
        sort_columns = ["plant_label", "image"]

    auto = auto.sort_values(sort_columns).reset_index(drop=True)
    auto["image_index"] = auto.groupby("plant_label").cumcount() + 1

    return auto


def prepare_manual_counts(manual: pd.DataFrame, ignore_zero: bool = True) -> pd.DataFrame:
    manual = manual.copy()

    manual["plant_label"] = manual["plant_id"].apply(short_plant_label)

    for column in REPLICATE_COLUMNS:
        manual[column] = pd.to_numeric(manual[column], errors="coerce")

    if ignore_zero:
        manual[REPLICATE_COLUMNS] = manual[REPLICATE_COLUMNS].replace(0, np.nan)

    manual["manual_n"] = manual[REPLICATE_COLUMNS].count(axis=1)
    manual["manual_mean"] = manual[REPLICATE_COLUMNS].mean(axis=1)
    manual["manual_sd"] = manual[REPLICATE_COLUMNS].std(axis=1)

    return manual


def calculate_metrics(df: pd.DataFrame) -> dict:
    y_true = df["manual_mean"].to_numpy(dtype=float)
    y_pred = df["stomatacount"].to_numpy(dtype=float)

    valid = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true != 0)

    y_true = y_true[valid]
    y_pred = y_pred[valid]

    if len(y_true) == 0:
        return {
            "n_images": 0,
            "manual_mean_count": np.nan,
            "stomatacount_mean": np.nan,
            "bias": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
            "mape_percent": np.nan,
            "correlation": np.nan,
        }

    difference = y_pred - y_true

    correlation = (
        np.corrcoef(y_true, y_pred)[0, 1]
        if len(y_true) > 1
        else np.nan
    )

    return {
        "n_images": int(len(y_true)),
        "manual_mean_count": round(float(np.mean(y_true)), 2),
        "stomatacount_mean": round(float(np.mean(y_pred)), 2),
        "bias": round(float(np.mean(difference)), 2),
        "mae": round(float(np.mean(np.abs(difference))), 2),
        "rmse": round(float(np.sqrt(np.mean(difference**2))), 2),
        "mape_percent": round(float(np.mean(np.abs(difference) / y_true) * 100), 2),
        "correlation": round(float(correlation), 4) if np.isfinite(correlation) else np.nan,
    }


def build_metrics_by_plant(comparison: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for plant, group in comparison.groupby("plant"):
        metrics = calculate_metrics(group)
        metrics["plant"] = plant
        rows.append(metrics)

    columns = [
        "plant",
        "n_images",
        "manual_mean_count",
        "stomatacount_mean",
        "bias",
        "mae",
        "rmse",
        "mape_percent",
        "correlation",
    ]

    return pd.DataFrame(rows)[columns]


def save_wide_matrices(comparison: pd.DataFrame, output_dir: Path) -> None:
    automatic_matrix = comparison.pivot(
        index="image_index",
        columns="plant",
        values="stomatacount",
    )

    manual_matrix = comparison.pivot(
        index="image_index",
        columns="plant",
        values="manual_mean",
    )

    difference_matrix = comparison.pivot(
        index="image_index",
        columns="plant",
        values="difference",
    )

    automatic_matrix.index.name = "Image"
    manual_matrix.index.name = "Image"
    difference_matrix.index.name = "Image"

    automatic_matrix.to_csv(output_dir / "automatic_count_matrix.csv")
    manual_matrix.to_csv(output_dir / "manual_mean_matrix.csv")
    difference_matrix.to_csv(output_dir / "difference_matrix.csv")


def build_comparison(
    auto: pd.DataFrame,
    manual: pd.DataFrame,
    output_dir: Path,
    require_complete_triplicate: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if require_complete_triplicate:
        manual = manual[manual["manual_n"] == 3].copy()

    merged = auto.merge(
        manual,
        on=["plant_label", "image_index"],
        suffixes=("_auto", "_manual"),
        how="inner",
    )

    comparison = pd.DataFrame(
        {
            "plant": merged["plant_label"],
            "image_index": merged["image_index"],
            "image_file": merged["image"],
            "relative_path": merged.get("relative_path", merged["image"]),
            "stomatacount": pd.to_numeric(merged["total_stomata"], errors="coerce"),
            "Gab": merged["Gab"],
            "Leo": merged["Leo"],
            "Nic": merged["Nic"],
            "manual_n": merged["manual_n"],
            "manual_mean": merged["manual_mean"],
            "manual_sd": merged["manual_sd"],
        }
    )

    comparison["difference"] = comparison["stomatacount"] - comparison["manual_mean"]
    comparison["absolute_error"] = comparison["difference"].abs()
    comparison["percent_error"] = (
        comparison["absolute_error"] / comparison["manual_mean"] * 100
    )

    comparison["manual_mean"] = comparison["manual_mean"].round(2)
    comparison["manual_sd"] = comparison["manual_sd"].round(2)
    comparison["difference"] = comparison["difference"].round(2)
    comparison["absolute_error"] = comparison["absolute_error"].round(2)
    comparison["percent_error"] = comparison["percent_error"].round(2)

    comparison.to_csv(output_dir / "comparison_by_image.csv", index=False)

    metrics_by_plant = build_metrics_by_plant(comparison)
    metrics_by_plant.to_csv(output_dir / "comparison_by_plant.csv", index=False)

    global_metrics = pd.DataFrame([calculate_metrics(comparison)])
    global_metrics.to_csv(output_dir / "global_metrics.csv", index=False)

    save_wide_matrices(comparison, output_dir)

    print()
    print(f"Saved comparison to: {output_dir}")
    print()
    print("Global metrics:")
    print(global_metrics)
    print()
    print("Metrics by plant:")
    print(metrics_by_plant)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate StomataCount against manual stomata counts."
    )

    parser.add_argument(
        "--automatic",
        required=True,
        help="Path to StomataCount results.csv.",
    )

    parser.add_argument(
        "--manual",
        required=True,
        help="Path to manual_counts.csv.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Base output directory.",
    )

    args = parser.parse_args()

    automatic_path = Path(args.automatic)
    manual_path = Path(args.manual)
    output_base = Path(args.output)

    auto = pd.read_csv(automatic_path)
    manual = pd.read_csv(manual_path)

    auto = add_automatic_image_index(auto)
    manual = prepare_manual_counts(manual, ignore_zero=True)

    automatic_order = auto[
        ["plant_label", "image_index", "image", "relative_path"]
        if "relative_path" in auto.columns
        else ["plant_label", "image_index", "image"]
    ]

    automatic_order.to_csv(output_base / "automatic_image_order.csv", index=False)

    build_comparison(
        auto=auto,
        manual=manual,
        output_dir=output_base / "manual_comparison_all",
        require_complete_triplicate=False,
    )

    build_comparison(
        auto=auto,
        manual=manual,
        output_dir=output_base / "manual_comparison_complete_triplicates",
        require_complete_triplicate=True,
    )

    print()
    print("Evaluation completed.")
    print(f"Automatic image order saved to: {output_base / 'automatic_image_order.csv'}")


if __name__ == "__main__":
    main()
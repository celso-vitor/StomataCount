import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def confidence_interval_mean(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    n = len(values)

    mean = np.mean(values)
    se = stats.sem(values)

    if n < 2:
        return np.nan, np.nan

    ci_low, ci_high = stats.t.interval(
        confidence,
        df=n - 1,
        loc=mean,
        scale=se,
    )

    return ci_low, ci_high


def tost_equivalence_test(
    differences: np.ndarray,
    equivalence_margin: float,
    alpha: float = 0.05,
) -> dict:
    """
    Two One-Sided Tests for equivalence.

    differences = software - manual
    Equivalence is concluded if the mean difference is statistically inside:
    [-equivalence_margin, +equivalence_margin]
    """

    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]

    n = len(differences)
    mean_diff = np.mean(differences)
    se = stats.sem(differences)
    df = n - 1

    t_lower = (mean_diff + equivalence_margin) / se
    p_lower = 1 - stats.t.cdf(t_lower, df)

    t_upper = (mean_diff - equivalence_margin) / se
    p_upper = stats.t.cdf(t_upper, df)

    tost_p = max(p_lower, p_upper)

    return {
        "equivalence_margin": equivalence_margin,
        "tost_t_lower": t_lower,
        "tost_p_lower": p_lower,
        "tost_t_upper": t_upper,
        "tost_p_upper": p_upper,
        "tost_p": tost_p,
        "equivalent": tost_p < alpha,
    }


def calculate_statistics(df: pd.DataFrame, equivalence_margin: float) -> dict:
    manual = df["manual_mean"].to_numpy(dtype=float)
    software = df["stomatacount"].to_numpy(dtype=float)
    differences = software - manual

    n = len(df)
    bias = np.mean(differences)
    sd_diff = np.std(differences, ddof=1)
    ci_low, ci_high = confidence_interval_mean(differences)

    paired_t = stats.ttest_rel(software, manual)
    wilcoxon = stats.wilcoxon(differences)
    shapiro = stats.shapiro(differences)

    pearson = stats.pearsonr(manual, software)
    spearman = stats.spearmanr(manual, software)

    mae = np.mean(np.abs(differences))
    rmse = np.sqrt(np.mean(differences**2))
    mape = np.mean(np.abs(differences) / manual) * 100

    loa_low = bias - 1.96 * sd_diff
    loa_high = bias + 1.96 * sd_diff

    tost = tost_equivalence_test(
        differences=differences,
        equivalence_margin=equivalence_margin,
    )

    return {
        "n": n,
        "manual_mean": round(float(np.mean(manual)), 4),
        "software_mean": round(float(np.mean(software)), 4),
        "bias_software_minus_manual": round(float(bias), 4),
        "bias_ci95_low": round(float(ci_low), 4),
        "bias_ci95_high": round(float(ci_high), 4),
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "mape_percent": round(float(mape), 4),
        "paired_t_statistic": round(float(paired_t.statistic), 4),
        "paired_t_pvalue": round(float(paired_t.pvalue), 6),
        "wilcoxon_statistic": round(float(wilcoxon.statistic), 4),
        "wilcoxon_pvalue": round(float(wilcoxon.pvalue), 6),
        "shapiro_statistic": round(float(shapiro.statistic), 4),
        "shapiro_pvalue": round(float(shapiro.pvalue), 6),
        "pearson_r": round(float(pearson.statistic), 4),
        "pearson_pvalue": round(float(pearson.pvalue), 6),
        "spearman_r": round(float(spearman.statistic), 4),
        "spearman_pvalue": round(float(spearman.pvalue), 6),
        "bland_altman_loa_low": round(float(loa_low), 4),
        "bland_altman_loa_high": round(float(loa_high), 4),
        **{
            key: round(float(value), 6) if isinstance(value, (float, np.floating)) else value
            for key, value in tost.items()
        },
    }


def save_statistics(df: pd.DataFrame, output_dir: Path, equivalence_margin: float) -> None:
    global_stats = pd.DataFrame(
        [calculate_statistics(df, equivalence_margin=equivalence_margin)]
    )

    global_stats.to_csv(output_dir / "statistical_summary_global.csv", index=False)

    rows = []

    for plant, group in df.groupby("plant"):
        stats_row = calculate_statistics(group, equivalence_margin=equivalence_margin)
        stats_row["plant"] = plant
        rows.append(stats_row)

    by_plant = pd.DataFrame(rows)

    columns = ["plant"] + [column for column in by_plant.columns if column != "plant"]
    by_plant = by_plant[columns]

    by_plant.to_csv(output_dir / "statistical_summary_by_plant.csv", index=False)

    print()
    print("Global statistics:")
    print(global_stats)
    print()
    print("Statistics by plant:")
    print(by_plant)


def plot_scatter(df: pd.DataFrame, output_dir: Path) -> None:
    manual = df["manual_mean"]
    software = df["stomatacount"]

    plt.figure(figsize=(7, 6))
    plt.scatter(manual, software)

    min_value = min(manual.min(), software.min())
    max_value = max(manual.max(), software.max())

    plt.plot([min_value, max_value], [min_value, max_value], linestyle="--")

    plt.xlabel("Manual mean count")
    plt.ylabel("StomataCount")
    plt.title("Manual counting vs StomataCount")
    plt.tight_layout()
    plt.savefig(output_dir / "scatter_manual_vs_stomatacount.png", dpi=300)
    plt.close()


def plot_bland_altman(df: pd.DataFrame, output_dir: Path) -> None:
    manual = df["manual_mean"].to_numpy(dtype=float)
    software = df["stomatacount"].to_numpy(dtype=float)

    mean_counts = (manual + software) / 2
    differences = software - manual

    bias = np.mean(differences)
    sd_diff = np.std(differences, ddof=1)

    loa_low = bias - 1.96 * sd_diff
    loa_high = bias + 1.96 * sd_diff

    plt.figure(figsize=(7, 6))
    plt.scatter(mean_counts, differences)

    plt.axhline(bias, linestyle="-")
    plt.axhline(loa_low, linestyle="--")
    plt.axhline(loa_high, linestyle="--")
    plt.axhline(0, linestyle=":")

    plt.xlabel("Mean of manual and StomataCount")
    plt.ylabel("StomataCount - manual mean")
    plt.title("Bland-Altman plot")
    plt.tight_layout()
    plt.savefig(output_dir / "bland_altman.png", dpi=300)
    plt.close()


def plot_difference_by_plant(df: pd.DataFrame, output_dir: Path) -> None:
    plot_df = df.copy()
    plot_df["difference"] = plot_df["stomatacount"] - plot_df["manual_mean"]

    plt.figure(figsize=(7, 6))
    plot_df.boxplot(column="difference", by="plant")
    plt.axhline(0, linestyle="--")
    plt.xlabel("Plant")
    plt.ylabel("StomataCount - manual mean")
    plt.title("Difference by plant")
    plt.suptitle("")
    plt.tight_layout()
    plt.savefig(output_dir / "difference_by_plant.png", dpi=300)
    plt.close()


def plot_counts_by_method(df: pd.DataFrame, output_dir: Path) -> None:
    plot_df = df[["manual_mean", "stomatacount"]].rename(
        columns={
            "manual_mean": "Manual mean",
            "stomatacount": "StomataCount",
        }
    )

    plt.figure(figsize=(7, 6))
    plot_df.boxplot()
    plt.ylabel("Stomata count")
    plt.title("Manual mean and StomataCount distributions")
    plt.tight_layout()
    plt.savefig(output_dir / "method_count_distribution.png", dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Statistical analysis comparing StomataCount against complete manual triplicates."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to comparison_by_image.csv from complete triplicates.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for statistics and plots.",
    )

    parser.add_argument(
        "--equivalence-margin",
        type=float,
        default=5.0,
        help="Equivalence margin in stomata per image for TOST.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    required_columns = {"plant", "image_index", "stomatacount", "manual_mean"}

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df["stomatacount"] = pd.to_numeric(df["stomatacount"], errors="coerce")
    df["manual_mean"] = pd.to_numeric(df["manual_mean"], errors="coerce")

    df = df.dropna(subset=["stomatacount", "manual_mean"]).copy()

    save_statistics(
        df=df,
        output_dir=output_dir,
        equivalence_margin=args.equivalence_margin,
    )

    plot_scatter(df, output_dir)
    plot_bland_altman(df, output_dir)
    plot_difference_by_plant(df, output_dir)
    plot_counts_by_method(df, output_dir)

    print()
    print("Statistical analysis completed.")
    print(f"Results saved to: {output_dir}")
    print()
    print("Generated files:")
    print(output_dir / "statistical_summary_global.csv")
    print(output_dir / "statistical_summary_by_plant.csv")
    print(output_dir / "scatter_manual_vs_stomatacount.png")
    print(output_dir / "bland_altman.png")
    print(output_dir / "difference_by_plant.png")
    print(output_dir / "method_count_distribution.png")


if __name__ == "__main__":
    main()
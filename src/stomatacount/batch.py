import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from stomatacount.roboflow_detector import RoboflowStomataDetector
from stomatacount.visualization import draw_predictions


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".bmp",
}


def collect_images(input_path: str | Path) -> list[Path]:
    input_path = Path(input_path)

    if input_path.is_file():
        if input_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            return [input_path]
        raise ValueError(f"Unsupported image extension: {input_path.suffix}")

    if input_path.is_dir():
        images = [
            path
            for path in input_path.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        return sorted(images)

    raise FileNotFoundError(f"Input path not found: {input_path}")


def get_relative_path_and_plant_id(image_path: Path, root_dir: Path) -> tuple[Path, str]:
    relative_path = image_path.relative_to(root_dir)

    if len(relative_path.parts) > 1:
        plant_id = relative_path.parts[0]
    else:
        plant_id = image_path.parent.name

    return relative_path, plant_id


def create_plant_summary(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    plant_summary = (
        df.groupby("plant_id", as_index=False)
        .agg(
            n_images=("image", "count"),
            total_stomata=("total_stomata", "sum"),
            mean_stomata_per_image=("total_stomata", "mean"),
            min_stomata_per_image=("total_stomata", "min"),
            max_stomata_per_image=("total_stomata", "max"),
            mean_confidence=("mean_confidence", "mean"),
        )
    )

    plant_summary["mean_stomata_per_image"] = plant_summary[
        "mean_stomata_per_image"
    ].round(2)

    plant_summary["mean_confidence"] = plant_summary["mean_confidence"].round(4)

    plant_summary.to_csv(output_dir / "plant_summary.csv", index=False)

    return plant_summary


def process_images(
    input_path: str | Path,
    output_dir: str | Path,
    confidence_threshold: float = 0.30,
    save_json: bool = True,
) -> pd.DataFrame:
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    annotated_dir = output_dir / "annotated"
    json_dir = output_dir / "json"

    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)

    if save_json:
        json_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(input_path)

    if not images:
        raise ValueError(f"No supported images found in: {input_path}")

    root_dir = input_path if input_path.is_dir() else input_path.parent

    detector = RoboflowStomataDetector()
    rows = []

    for image_path in tqdm(images, desc="Analyzing images"):
        relative_path, plant_id = get_relative_path_and_plant_id(
            image_path=image_path,
            root_dir=root_dir,
        )

        analysis = detector.analyze_image(
            image_path,
            confidence_threshold=confidence_threshold,
        )

        predictions = analysis["predictions"]

        annotated_path = (
            annotated_dir
            / relative_path.parent
            / f"{image_path.stem}_annotated{image_path.suffix}"
        )

        draw_predictions(
            image_path=image_path,
            predictions=predictions,
            output_path=annotated_path,
            show_labels=True,
        )

        if save_json:
            json_path = json_dir / relative_path.with_suffix(".json")
            json_path.parent.mkdir(parents=True, exist_ok=True)

            with json_path.open("w", encoding="utf-8") as handle:
                json.dump(analysis["raw_result"], handle, indent=2)

        rows.append(
            {
                "plant_id": plant_id,
                "image": image_path.name,
                "relative_path": str(relative_path),
                "image_path": str(image_path),
                "annotated_image": str(annotated_path),
                "model_id": analysis["model_id"],
                "confidence_threshold": analysis["confidence_threshold"],
                "total_stomata": analysis["total_stomata"],
                "mean_confidence": analysis["mean_confidence"],
                "min_confidence": analysis["min_confidence"],
                "max_confidence": analysis["max_confidence"],
            }
        )

    df = pd.DataFrame(rows)

    results_csv = output_dir / "results.csv"
    df.to_csv(results_csv, index=False)

    create_plant_summary(df, output_dir)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run Roboflow stomata detection on one image, "
            "one folder, or plant-organized image folders."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input image file or folder containing images.",
    )

    parser.add_argument(
        "--output",
        default="data/output",
        help="Output directory.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.30,
        help="Confidence threshold for filtering predictions.",
    )

    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Do not save raw Roboflow JSON outputs.",
    )

    args = parser.parse_args()

    df = process_images(
        input_path=args.input,
        output_dir=args.output,
        confidence_threshold=args.threshold,
        save_json=not args.no_json,
    )

    print()
    print("Analysis completed.")
    print(f"Images analyzed: {len(df)}")
    print(f"Plants analyzed: {df['plant_id'].nunique()}")
    print(f"Total stomata detected: {int(df['total_stomata'].sum())}")
    print(f"Results saved to: {Path(args.output) / 'results.csv'}")
    print(f"Plant summary saved to: {Path(args.output) / 'plant_summary.csv'}")


if __name__ == "__main__":
    main()

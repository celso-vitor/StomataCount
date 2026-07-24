from pathlib import Path

import cv2


def prediction_to_xyxy(prediction: dict) -> tuple[int, int, int, int]:
    """
    Convert Roboflow center-based bounding box format to xyxy format.

    Roboflow usually returns:
    - x: center x
    - y: center y
    - width: bounding box width
    - height: bounding box height
    """

    x = float(prediction.get("x", 0))
    y = float(prediction.get("y", 0))
    width = float(prediction.get("width", 0))
    height = float(prediction.get("height", 0))

    x1 = int(round(x - width / 2))
    y1 = int(round(y - height / 2))
    x2 = int(round(x + width / 2))
    y2 = int(round(y + height / 2))

    return x1, y1, x2, y2


def draw_predictions(
    image_path: str | Path,
    predictions: list[dict],
    output_path: str | Path,
    show_labels: bool = True,
) -> Path:
    image_path = Path(image_path)
    output_path = Path(output_path)

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    for prediction in predictions:
        x1, y1, x2, y2 = prediction_to_xyxy(prediction)

        confidence = float(prediction.get("confidence", 0.0))
        class_name = prediction.get("class", "stoma")

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2,
        )

        if show_labels:
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(
                image,
                label,
                (x1, max(y1 - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 0, 0),
                1,
                cv2.LINE_AA,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)

    return output_path

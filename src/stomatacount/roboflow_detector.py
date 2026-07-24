import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient


class RoboflowStomataDetector:
    """
    Roboflow-based stomata detector.

    This class handles:
    - loading Roboflow credentials from .env
    - sending images to the Roboflow hosted API
    - filtering predictions by confidence
    - summarizing stomata counts
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_id: str | None = None,
        api_url: str | None = None,
    ) -> None:
        load_dotenv(".env", override=True, encoding="utf-8-sig")

        self.api_key = api_key or os.getenv("ROBOFLOW_API_KEY")
        self.model_id = model_id or os.getenv("ROBOFLOW_MODEL_ID")
        self.api_url = api_url or os.getenv(
            "ROBOFLOW_API_URL",
            "https://serverless.roboflow.com",
        )

        if not self.api_key:
            raise ValueError("ROBOFLOW_API_KEY is missing. Add it to your .env file.")

        if not self.model_id:
            raise ValueError("ROBOFLOW_MODEL_ID is missing. Add it to your .env file.")

        self.client = InferenceHTTPClient(
            api_url=self.api_url,
            api_key=self.api_key,
        )

    def predict(self, image_path: str | Path) -> dict[str, Any]:
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        return self.client.infer(str(image_path), model_id=self.model_id)

    @staticmethod
    def extract_predictions(result: dict[str, Any]) -> list[dict[str, Any]]:
        predictions = result.get("predictions", [])

        if not isinstance(predictions, list):
            return []

        return predictions

    @staticmethod
    def filter_predictions(
        predictions: list[dict[str, Any]],
        confidence_threshold: float = 0.30,
    ) -> list[dict[str, Any]]:
        return [
            prediction
            for prediction in predictions
            if float(prediction.get("confidence", 0.0)) >= confidence_threshold
        ]

    @staticmethod
    def summarize_predictions(
        predictions: list[dict[str, Any]],
    ) -> dict[str, float | int]:
        confidences = [
            float(prediction.get("confidence", 0.0))
            for prediction in predictions
        ]

        if not confidences:
            return {
                "total_stomata": 0,
                "mean_confidence": 0.0,
                "min_confidence": 0.0,
                "max_confidence": 0.0,
            }

        return {
            "total_stomata": len(predictions),
            "mean_confidence": round(sum(confidences) / len(confidences), 4),
            "min_confidence": round(min(confidences), 4),
            "max_confidence": round(max(confidences), 4),
        }

    def analyze_image(
        self,
        image_path: str | Path,
        confidence_threshold: float = 0.30,
    ) -> dict[str, Any]:
        result = self.predict(image_path)
        predictions = self.extract_predictions(result)
        filtered_predictions = self.filter_predictions(
            predictions,
            confidence_threshold=confidence_threshold,
        )

        summary = self.summarize_predictions(filtered_predictions)

        return {
            "image": str(image_path),
            "model_id": self.model_id,
            "confidence_threshold": confidence_threshold,
            "predictions": filtered_predictions,
            "raw_result": result,
            **summary,
        }


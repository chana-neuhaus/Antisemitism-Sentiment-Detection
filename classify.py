"""Prediction utilities for the DSC 360 bias classifier app."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer


def normalize_text(text: str) -> str:
    """Apply light text cleanup before embedding.

    This is intentionally simple. If your notebook used a custom normalizer,
    replace this function with that same preprocessing logic so the app matches
    your training pipeline.
    """
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_model(model_path: str | Path) -> Any:
    """Load the saved sklearn/joblib model."""
    return joblib.load(model_path)


def load_embedding_model(model_name: str) -> SentenceTransformer:
    """Load the BERT/SentenceTransformer embedding model."""
    return SentenceTransformer(model_name)


def classify_text(
    text: str,
    model: Any,
    embedding_model: SentenceTransformer,
    labels: dict[int, str],
) -> dict[str, Any]:
    """Classify one text sample and return prediction details."""
    clean_text = normalize_text(text)

    if not clean_text:
        raise ValueError("Please enter text before classifying.")

    embedding = embedding_model.encode([clean_text])
    prediction = int(model.predict(embedding)[0])
    predicted_label = labels.get(prediction, str(prediction))

    result: dict[str, Any] = {
        "clean_text": clean_text,
        "prediction": prediction,
        "label": predicted_label,
        "probabilities": None,
        "confidence": None,
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(embedding)[0]
        probabilities = np.asarray(probabilities, dtype=float)
        result["probabilities"] = probabilities
        result["confidence"] = float(probabilities[prediction])

    return result

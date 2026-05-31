"""Prediction utilities for the DSC 360 bias classifier app."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from text_normalizer_correct import TextNormalizer


normalizer = TextNormalizer()


def normalize_text(text: str) -> str:
    """Apply the same text normalization used during model training."""
    clean_text = normalizer.normalize_series(
        pd.Series([text])
    ).iloc[0]

    return str(clean_text).strip()


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

from pathlib import Path

import joblib
import numpy as np
from scipy.special import softmax

from src.config import LABEL_NAMES, MODEL_PATH
from src.data import build_input_frame


def load_model(path: Path | str = MODEL_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run `python -m src.train` first."
        )
    return joblib.load(path)


def predict_job_level(
    model,
    *,
    title: str,
    location: str,
    description: str,
    function: str,
    industry: str,
) -> dict:
    frame = build_input_frame(title, location, description, function, industry)
    predicted_label = model.predict(frame)[0]
    classifier = model.named_steps["classifier"]
    classes = classifier.classes_

    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(frame)[0]
        score_name = "probability"
    else:
        decision_scores = np.asarray(model.decision_function(frame))
        if decision_scores.ndim == 1:
            decision_scores = np.column_stack([-decision_scores, decision_scores])
        scores = softmax(decision_scores[0])
        score_name = "relative_score"

    ranking = sorted(
        (
            {
                "label": label,
                "display_name": LABEL_NAMES.get(label, label),
                "score": float(score),
            }
            for label, score in zip(classes, scores, strict=True)
        ),
        key=lambda item: item["score"],
        reverse=True,
    )
    return {
        "label": predicted_label,
        "display_name": LABEL_NAMES.get(predicted_label, predicted_label),
        "score_type": score_name,
        "ranking": ranking,
    }

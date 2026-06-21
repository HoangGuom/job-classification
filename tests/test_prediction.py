import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import MODEL_PATH
from src.predict import load_model, predict_job_level


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Train the model first")
def test_saved_model_can_make_prediction():
    model = load_model()
    result = predict_job_level(
        model,
        title="Senior Software Engineering Manager",
        location="Austin, TX",
        description="Lead engineers, mentor staff and manage project delivery.",
        function="information_technology_telecommunications",
        industry="Information Technology",
    )
    assert result["label"]
    assert len(result["ranking"]) >= 3
    assert result["ranking"][0]["label"] == result["label"]

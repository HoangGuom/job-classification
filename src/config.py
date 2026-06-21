from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "final_project.ods"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "job_level_classifier.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"
CONFUSION_MATRIX_PATH = ARTIFACT_DIR / "confusion_matrix.png"

RANDOM_STATE = 100
TARGET = "career_level"
RAW_COLUMNS = ["title", "location", "description", "function", "industry", TARGET]
FEATURE_COLUMNS = ["combined_text", "location", "function"]

LABEL_NAMES = {
    "specialist": "Specialist",
    "senior_specialist_or_project_manager": "Senior Specialist / Project Manager",
    "manager_team_leader": "Manager / Team Leader",
    "bereichsleiter": "Department Head",
    "director_business_unit_leader": "Director / Business Unit Leader",
    "managing_director_small_medium_company": "Managing Director (SME)",
}

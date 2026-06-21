from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC

from src.config import RANDOM_STATE


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "text",
                TfidfVectorizer(
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=3,
                    max_df=0.98,
                    max_features=50_000,
                    sublinear_tf=True,
                ),
                "combined_text",
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", min_frequency=2),
                ["location", "function"],
            ),
        ]
    )


def build_candidate_models() -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    LogisticRegression(
                        C=4.0,
                        class_weight="balanced",
                        max_iter=1_500,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Linear SVM": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    LinearSVC(
                        C=1.0,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }

import json
from datetime import datetime, timezone

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split

from src.config import (
    ARTIFACT_DIR,
    CONFUSION_MATRIX_PATH,
    DATA_PATH,
    METADATA_PATH,
    METRICS_PATH,
    MODEL_PATH,
    RANDOM_STATE,
)
from src.data import load_and_clean_data, split_features_target
from src.modeling import build_candidate_models


def calculate_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_and_clean_data(DATA_PATH)
    X, y = split_features_target(data)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    results = {
        "Most-frequent baseline": calculate_metrics(
            y_test, baseline.predict(X_test)
        )
    }

    fitted_models = {}
    predictions = {}
    for name, model in build_candidate_models().items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        fitted_models[name] = model
        predictions[name] = y_pred
        results[name] = calculate_metrics(y_test, y_pred)

    best_name = max(fitted_models, key=lambda name: results[name]["macro_f1"])
    best_model = fitted_models[best_name]
    best_predictions = predictions[best_name]
    joblib.dump(best_model, MODEL_PATH)

    reports = {
        name: classification_report(
            y_test,
            predictions[name],
            output_dict=True,
            zero_division=0,
        )
        for name in fitted_models
    }
    payload = {
        "selected_model": best_name,
        "selection_metric": "macro_f1",
        "dataset_rows_after_cleaning": len(data),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "results": results,
        "classification_reports": reports,
    }
    METRICS_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_model": best_name,
        "target_classes": sorted(y.unique().tolist()),
        "functions": sorted(data["function"].unique().tolist()),
        "industries": sorted(data["industry"].unique().tolist()),
        "class_distribution": y.value_counts().to_dict(),
    }
    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(11, 9))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        best_predictions,
        normalize="true",
        xticks_rotation=45,
        cmap="Blues",
        values_format=".2f",
        ax=ax,
    )
    ax.set_title(f"Normalized confusion matrix — {best_name}")
    fig.tight_layout()
    fig.savefig(CONFUSION_MATRIX_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)

    comparison = pd.DataFrame(results).T.sort_values("macro_f1", ascending=False)
    print("\nModel comparison:")
    print(comparison.round(4).to_string())
    print(f"\nSelected model: {best_name}")
    print(f"Saved model to: artifacts/{MODEL_PATH.name}")


if __name__ == "__main__":
    main()

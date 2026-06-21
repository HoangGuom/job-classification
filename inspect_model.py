import json

import joblib

from src.config import METRICS_PATH, MODEL_PATH


def main() -> None:
    model = joblib.load(MODEL_PATH)
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

    print("JOB LEVEL CLASSIFIER")
    print("=" * 60)
    print(f"Saved object:       {type(model).__name__}")
    print(f"Selected model:     {metrics['selected_model']}")
    print(f"Classifier:         {type(classifier).__name__}")
    print(f"Training rows:      {metrics['train_rows']:,}")
    print(f"Test rows:          {metrics['test_rows']:,}")
    print(f"Number of classes:  {len(classifier.classes_)}")
    print()

    print("PIPELINE")
    print("-" * 60)
    for name, step in model.named_steps.items():
        print(f"{name}: {type(step).__name__}")
    print()

    print("INPUT TRANSFORMERS")
    print("-" * 60)
    for name, transformer, columns in preprocessor.transformers_:
        print(f"{name}: {type(transformer).__name__} <- {columns}")
    print()

    print("TARGET CLASSES")
    print("-" * 60)
    for index, label in enumerate(classifier.classes_, start=1):
        print(f"{index}. {label}")
    print()

    print("MODEL METRICS")
    print("-" * 60)
    for model_name, values in metrics["results"].items():
        print(f"\n{model_name}")
        for metric_name, value in values.items():
            print(f"  {metric_name:20} {value:.4f}")


if __name__ == "__main__":
    main()

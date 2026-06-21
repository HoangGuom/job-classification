import argparse

from src.predict import load_model, predict_job_level


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict a career level from a job advertisement."
    )
    parser.add_argument("--title", required=True)
    parser.add_argument("--location", required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument("--function", required=True)
    parser.add_argument("--industry", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = predict_job_level(
        load_model(),
        title=args.title,
        location=args.location,
        description=args.description,
        function=args.function,
        industry=args.industry,
    )
    print(result["display_name"])
    for position, item in enumerate(result["ranking"][:3], start=1):
        print(f"{position}. {item['display_name']}: {item['score']:.2%}")


if __name__ == "__main__":
    main()

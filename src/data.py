import html
import re
from pathlib import Path

import pandas as pd

from src.config import FEATURE_COLUMNS, RAW_COLUMNS, TARGET


def clean_text(value: object) -> str:
    """Normalize common encoding artifacts and repeated whitespace."""
    value = html.unescape(str(value)).replace("Â", " ")
    return re.sub(r"\s+", " ", value).strip()


def normalize_location(value: object) -> str:
    """Reduce US locations such as 'Houston, TX' to their state code."""
    value = clean_text(value)
    match = re.search(r"(?:,|\s)\s*([A-Z]{2})$", value)
    return match.group(1) if match else value


def make_combined_text(title: object, description: object, industry: object) -> str:
    """Create the text field consumed by TF-IDF, emphasizing the job title."""
    title = clean_text(title)
    return f"{title} {title} {clean_text(description)} {clean_text(industry)}"


def load_and_clean_data(path: Path | str) -> pd.DataFrame:
    data = pd.read_excel(path, dtype=str, engine="odf")
    missing_columns = sorted(set(RAW_COLUMNS) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    data = data[RAW_COLUMNS].dropna().drop_duplicates().copy()
    for column in RAW_COLUMNS:
        data[column] = data[column].map(clean_text)

    data = data[(data[RAW_COLUMNS] != "").all(axis=1)].copy()
    data["location"] = data["location"].map(normalize_location)
    data["combined_text"] = [
        make_combined_text(title, description, industry)
        for title, description, industry in zip(
            data["title"], data["description"], data["industry"], strict=True
        )
    ]
    return data


def build_input_frame(
    title: str,
    location: str,
    description: str,
    function: str,
    industry: str,
) -> pd.DataFrame:
    row = {
        "combined_text": make_combined_text(title, description, industry),
        "location": normalize_location(location),
        "function": clean_text(function),
    }
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def split_features_target(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return data[FEATURE_COLUMNS], data[TARGET]

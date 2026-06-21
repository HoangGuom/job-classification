import html
import re
from pathlib import Path
from typing import cast

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
    data = cast(pd.DataFrame, pd.read_excel(path, dtype=str, engine="odf"))
    missing_columns = sorted(set(RAW_COLUMNS) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    data = cast(
        pd.DataFrame,
        data.loc[:, RAW_COLUMNS].dropna().drop_duplicates().copy(),
    )
    for column in RAW_COLUMNS:
        values = cast(pd.Series, data.loc[:, column])
        data.loc[:, column] = values.map(clean_text)

    non_empty_rows = cast(
        pd.Series,
        (data.loc[:, RAW_COLUMNS] != "").all(axis=1),
    )
    data = cast(pd.DataFrame, data.loc[non_empty_rows].copy())
    locations = cast(pd.Series, data.loc[:, "location"])
    data.loc[:, "location"] = locations.map(normalize_location)
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
    features = cast(pd.DataFrame, data.loc[:, FEATURE_COLUMNS].copy())
    target = cast(pd.Series, data.loc[:, TARGET].copy())
    return features, target

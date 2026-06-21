import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import build_input_frame, clean_text, normalize_location


def test_clean_text_removes_encoding_artifact_and_extra_spaces():
    assert clean_text("HelloÂ   world") == "Hello world"


def test_normalize_us_location():
    assert normalize_location("Houston, TX") == "TX"


def test_non_us_location_is_preserved():
    assert normalize_location("North Carolina") == "North Carolina"


def test_build_input_frame_has_expected_columns():
    frame = build_input_frame(
        title="Engineering Manager",
        location="Austin, TX",
        description="Lead an engineering team.",
        function="information_technology_telecommunications",
        industry="Information Technology",
    )
    assert frame.columns.tolist() == ["combined_text", "location", "function"]
    assert frame.loc[0, "location"] == "TX"
    combined_text = str(frame.loc[0, "combined_text"])
    assert "Engineering Manager Engineering Manager" in combined_text

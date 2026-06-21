import html
import json
import re
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

if get_script_run_ctx(suppress_warning=True) is None:
    print("Hãy chạy ứng dụng bằng lệnh: python -m streamlit run app.py")
    raise SystemExit(0)

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "artifacts" / "job_level_classifier.joblib"
METADATA_PATH = PROJECT_ROOT / "artifacts" / "metadata.json"

LABEL_NAMES = {
    "specialist": "Specialist",
    "senior_specialist_or_project_manager": "Senior Specialist / Project Manager",
    "manager_team_leader": "Manager / Team Leader",
    "bereichsleiter": "Department Head",
    "director_business_unit_leader": "Director / Business Unit Leader",
    "managing_director_small_medium_company": "Managing Director (SME)",
}


def clean_text(value: object) -> str:
    value = html.unescape(str(value)).replace("Â", " ")
    return re.sub(r"\s+", " ", value).strip()


def normalize_location(value: object) -> str:
    value = clean_text(value)
    match = re.search(r"(?:,|\s)\s*([A-Z]{2})$", value)
    return match.group(1) if match else value


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata() -> dict:
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


st.set_page_config(page_title="Job Level Classification", page_icon="💼")
st.title("💼 Job Level Classification")
st.caption("Dự đoán cấp bậc nghề nghiệp từ nội dung tin tuyển dụng.")

if not MODEL_PATH.exists() or not METADATA_PATH.exists():
    st.error(
        "Chưa có model. Hãy chạy toàn bộ notebook "
        "`notebooks/job_level_classification.ipynb` trước."
    )
    st.stop()

model = load_model()
metadata = load_metadata()

with st.form("prediction_form"):
    title = st.selectbox(
        "Job title",
        metadata["job_titles"],
        index=None,
        placeholder="Chọn hoặc nhập job title...",
        accept_new_options=True,
    )
    location = st.selectbox(
        "Location",
        metadata["locations"],
        index=None,
        placeholder="Chọn hoặc nhập location...",
        accept_new_options=True,
    )
    function = st.selectbox("Function", metadata["functions"])
    industry = st.selectbox("Industry", metadata["industries"])
    description = st.text_area(
        "Job description",
        placeholder="Nhập mô tả công việc...",
        height=180,
    )
    submitted = st.form_submit_button("Dự đoán")

if submitted:
    if not title or not location or not description.strip():
        st.warning("Vui lòng nhập đầy đủ job title, location và description.")
    else:
        title_clean = clean_text(title)
        input_data = pd.DataFrame(
            [
                {
                    "combined_text": (
                        f"{title_clean} {title_clean} "
                        f"{clean_text(description)} {clean_text(industry)}"
                    ),
                    "location": normalize_location(location),
                    "function": clean_text(function),
                }
            ]
        )
        prediction = model.predict(input_data)[0]
        st.success(
            f"Cấp bậc dự đoán: **{LABEL_NAMES.get(prediction, prediction)}**"
        )

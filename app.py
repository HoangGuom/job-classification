import json

import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

from src.config import LABEL_NAMES, METADATA_PATH, MODEL_PATH
from src.predict import load_model, predict_job_level

if get_script_run_ctx(suppress_warning=True) is None:
    print("Hãy chạy ứng dụng bằng lệnh: python -m streamlit run app.py")
    raise SystemExit(0)

st.set_page_config(
    page_title="Job Level Classifier",
    page_icon="💼",
    layout="wide",
)


@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)


@st.cache_data
def get_metadata() -> dict:
    if not METADATA_PATH.exists():
        return {"functions": [], "industries": []}
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


st.title("💼 Job Level Classifier")
st.caption(
    "Dự đoán cấp bậc nghề nghiệp từ nội dung tin tuyển dụng bằng TF-IDF và Linear SVM."
)

try:
    model = get_model()
except FileNotFoundError:
    st.error("Chưa tìm thấy mô hình. Hãy chạy `python -m src.train` trước.")
    st.stop()

metadata = get_metadata()
function_options = metadata.get("functions", [])
industry_options = metadata.get("industries", [])

with st.sidebar:
    st.header("Các cấp bậc có thể dự đoán")
    for display_name in LABEL_NAMES.values():
        st.write(f"• {display_name}")
    st.info(
        "Dữ liệu mất cân bằng mạnh. Kết quả ở các lớp rất hiếm nên được xem là gợi ý, không phải quyết định nhân sự."
    )

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        title = st.text_input(
            "Job title *",
            value="Senior Software Engineering Manager",
        )
        location = st.text_input("Location *", value="Austin, TX")
        function = st.selectbox(
            "Function *",
            options=function_options or ["information_technology_telecommunications"],
            index=(
                function_options.index("information_technology_telecommunications")
                if "information_technology_telecommunications" in function_options
                else 0
            ),
        )
    with col2:
        industry = st.selectbox(
            "Industry *",
            options=industry_options or ["Information Technology"],
            index=(
                industry_options.index("Information Technology")
                if "Information Technology" in industry_options
                else 0
            ),
        )
        description = st.text_area(
            "Job description *",
            value=(
                "Lead a team of software engineers, plan product delivery, "
                "mentor staff, manage stakeholders and own project outcomes."
            ),
            height=180,
        )

    submitted = st.form_submit_button(
        "Dự đoán cấp bậc",
        type="primary",
    )

if submitted:
    if not all(value.strip() for value in [title, location, description, function, industry]):
        st.warning("Vui lòng nhập đầy đủ các trường bắt buộc.")
    else:
        result = predict_job_level(
            model,
            title=title,
            location=location,
            description=description,
            function=function,
            industry=industry,
        )
        st.success(f"Kết quả dự đoán: **{result['display_name']}**")

        ranking = pd.DataFrame(result["ranking"][:3])
        ranking["score"] = (ranking["score"] * 100).round(2)
        ranking = ranking.rename(
            columns={"display_name": "Career level", "score": "Relative score (%)"}
        )
        st.subheader("Top 3 kết quả")
        st.dataframe(
            ranking[["Career level", "Relative score (%)"]],
            hide_index=True,
        )
        if result["score_type"] == "relative_score":
            st.caption(
                "Relative score dùng để xếp hạng đầu ra; đây không phải xác suất đã được hiệu chỉnh."
            )

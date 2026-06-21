import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streamlit.testing.v1 import AppTest


def test_streamlit_app_submits_and_displays_prediction():
    app = AppTest.from_file("app.py", default_timeout=30).run()
    assert not app.exception
    assert app.title[0].value == "💼 Job Level Classifier"

    app.selectbox[0].set_value(app.selectbox[0].options[0])
    app.selectbox[1].set_value(app.selectbox[1].options[0])
    app.button[0].click().run()
    assert not app.exception
    assert "Kết quả dự đoán" in app.success[0].value
    assert len(app.dataframe) == 1

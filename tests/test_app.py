"""Run the Streamlit app headlessly and make sure it works."""

from streamlit.testing.v1 import AppTest

from analyzer import PROJECT_DIR

APP = str(PROJECT_DIR / "app.py")


def run_app():
    return AppTest.from_file(APP, default_timeout=60).run()


def test_app_runs_without_errors():
    at = run_app()
    assert not at.exception
    assert not at.error
    assert at.title[0].value == "Student Exam Score Predictor"


def test_moving_sliders_changes_prediction():
    at = run_app()
    homework = at.sidebar.slider[0]   # first sidebar slider is homework

    homework.set_value(20).run()
    low = [m.value for m in at.markdown]
    homework.set_value(95).run()
    high = [m.value for m in at.markdown]
    assert not at.exception
    assert low != high

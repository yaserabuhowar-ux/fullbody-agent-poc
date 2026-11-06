from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app/streamlit_app.py")


def test_app_runs_and_simulates():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    at.button(key="sim").click().run()
    assert not at.exception
    assert any("Verdict" in s.value for s in at.subheader)


def test_reverse_tab():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.button(key="rev").click().run()
    assert not at.exception
    assert len(at.dataframe) >= 1

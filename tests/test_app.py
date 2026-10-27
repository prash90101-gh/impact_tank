"""Smoke tests for the Streamlit dashboard (app.py)."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_dashboard_renders_without_errors_on_default_scenario() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path))
    app.run(timeout=30)

    assert not app.exception
    assert len(app.metric) >= 4
    assert len(app.dataframe) >= 1


def test_switching_to_custom_mode_recomputes_the_dispatch() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path))
    app.run(timeout=30)

    app.toggle[0].set_value(False)
    app.run(timeout=30)

    assert not app.exception


def test_each_standard_scenario_renders_without_errors() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path))
    app.run(timeout=30)

    for option in app.selectbox[0].options:
        app.selectbox[0].set_value(option)
        app.run(timeout=30)
        assert not app.exception

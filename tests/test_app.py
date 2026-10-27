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


def test_custom_tariff_editor_is_present_and_editable_in_custom_mode() -> None:
    """The reviewer-requested fix: users must be able to enter their own hourly
    tariff, not just pick from the three built-in presets.

    `AppTest` (streamlit==1.66.1) exposes `data_editor` widgets read-only via
    `get_by_key`, without a scriptable `set_value` for editable grids, so this
    test verifies the editor renders with real, editable per-hour values and
    that changing the pre-fill selection changes its contents. End-to-end
    correctness of arbitrary custom tariffs flowing through the optimizer is
    covered directly in tests/test_scenarios.py via
    `generate_synthetic_commercial_profiles(custom_price_inr_per_kwh=...)`.
    """
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path))
    app.run(timeout=30)

    app.toggle[0].set_value(False)
    app.run(timeout=30)
    assert not app.exception

    editor = app.get_by_key("tariff_editor")
    assert editor is not None
    tariff_df = editor.value
    assert len(tariff_df) == 24
    assert list(tariff_df["hour"]) == list(range(24))
    # Default pre-fill is the normal Time-of-Use preset: not flat, not all one value.
    assert tariff_df["price_inr_per_kwh"].nunique() > 1

    # Switching the pre-fill source (not the only 3 presets — just a starting
    # point) must update the editable table to a different set of values.
    prefill_select = next(sb for sb in app.selectbox if sb.label == "Pre-fill the table from")
    prefill_select.set_value("flat")
    app.run(timeout=30)

    assert not app.exception
    flat_tariff_df = app.get_by_key("tariff_editor").value
    assert (flat_tariff_df["price_inr_per_kwh"] == 8.0).all()

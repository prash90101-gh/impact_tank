"""Streamlit dashboard for the GridPilot AI synthetic dispatch engine.

IMPACT TANK '26 Prototype — single commercial building with rooftop solar PV
and a Battery Energy Storage System (BESS), optimized against an Indian C&I
Time-of-Use (ToU) electricity tariff.

NOTICE: Every number on this dashboard is SYNTHETIC input data and every
dispatch result is a SIMULATED optimization output. Nothing here is
connected to live telemetry, a real tariff, or a real customer site.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from gridpilot import (
    BatteryConfig,
    generate_synthetic_commercial_profiles,
    get_scenario_definitions,
    get_tariff_preset,
    optimize_building_dispatch,
    validate_custom_tariff,
)

st.set_page_config(
    page_title="GridPilot AI | IMPACT TANK '26",
    page_icon=":material/bolt:",
    layout="wide",
)

COLORS = {
    "ink": "#173C35",
    "green": "#087E5B",
    "mint": "#63C49A",
    "sun": "#F4B942",
    "blue": "#2E6DD8",
    "violet": "#7C5CE0",
    "grid": "#DDE8E3",
    "muted": "#5E746D",
}
PLOTLY_CONFIG = {"displaylogo": False, "scrollZoom": False, "responsive": True}


def apply_chart_style(figure: go.Figure, height: int) -> go.Figure:
    """Apply a consistent, quiet visual system to every chart."""
    figure.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        hoverlabel={"bgcolor": "white", "font_color": COLORS["ink"]},
        font={"color": COLORS["ink"], "size": 13},
        margin={"l": 8, "r": 8, "t": 34, "b": 8},
        legend={"orientation": "h", "y": 1.13, "x": 0, "xanchor": "left"},
    )
    figure.update_xaxes(showgrid=False, zeroline=False)
    figure.update_yaxes(gridcolor=COLORS["grid"], zeroline=False)
    return figure


@st.cache_data(show_spinner=False)
def run_scenario_cached(
    solar_scale: float,
    price_inr_per_kwh: tuple[float, ...],
    capacity_kwh: float,
    power_kw: float,
    initial_soc_kwh: float,
    min_soc_kwh: float,
    max_soc_kwh: float,
    charge_efficiency: float,
    discharge_efficiency: float,
) -> dict[str, Any]:
    """Build a synthetic 24h profile and solve the least-cost dispatch for it.

    ``price_inr_per_kwh`` is an explicit 24-value hourly tariff (INR/kWh) —
    it may come from a built-in preset or from the custom tariff editor, and
    is passed straight through to the optimizer without being limited to any
    fixed set of rate structures.
    """
    profile = generate_synthetic_commercial_profiles(
        solar_scale=solar_scale,
        custom_price_inr_per_kwh=price_inr_per_kwh,
    )
    battery = BatteryConfig(
        capacity_kwh=capacity_kwh,
        max_charge_kw=power_kw if capacity_kwh else 0.0,
        max_discharge_kw=power_kw if capacity_kwh else 0.0,
        charge_efficiency=charge_efficiency,
        discharge_efficiency=discharge_efficiency,
        initial_soc_kwh=min(max(initial_soc_kwh, min_soc_kwh), max_soc_kwh) if capacity_kwh else 0.0,
        min_soc_kwh=min_soc_kwh if capacity_kwh else 0.0,
        max_soc_kwh=max_soc_kwh if capacity_kwh else 0.0,
        enforce_terminal_soc=True,
    )
    result = optimize_building_dispatch(
        demand_kwh=profile["demand_kwh"],
        solar_kwh=profile["solar_kwh"],
        price_inr_per_kwh=profile["price_inr_per_kwh"],
        battery_config=battery,
        data_source_label="SYNTHETIC_DASHBOARD_INPUT",
    )
    return result


def overview_chart(df: pd.DataFrame) -> go.Figure:
    """A simple, non-technical view of the building's 24-hour energy rhythm."""
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=df["hour"], y=df["solar_kwh"], name="Solar generation",
            mode="lines", line={"color": COLORS["sun"], "width": 2.5},
            fill="tozeroy", fillcolor="rgba(244, 185, 66, 0.22)",
            hovertemplate="Solar: %{y:.1f} kWh<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=df["hour"], y=df["demand_kwh"], name="Building demand",
            mode="lines", line={"color": COLORS["ink"], "width": 3},
            hovertemplate="Demand: %{y:.1f} kWh<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=df["hour"], y=df["grid_import_kwh"], name="Grid import (optimized)",
            mode="lines", line={"color": COLORS["blue"], "width": 2, "dash": "dot"},
            hovertemplate="Grid import: %{y:.1f} kWh<extra></extra>",
        )
    )
    figure.update_xaxes(title_text="Hour of day", dtick=2)
    figure.update_yaxes(title_text="Energy (kWh)")
    return apply_chart_style(figure, height=330)


def energy_network_chart(df: pd.DataFrame) -> go.Figure:
    """Summarize the 24-hour schedule as a readable energy flow (Sankey)."""
    solar_to_site = float(np.minimum(df["solar_kwh"], df["demand_kwh"]).sum())
    solar_surplus = np.clip(df["solar_kwh"] - df["demand_kwh"], 0, None)
    solar_to_battery = float(np.minimum(solar_surplus, df["battery_charge_kwh"]).sum())
    battery_charge = float(df["battery_charge_kwh"].sum())
    grid_import = float(df["grid_import_kwh"].sum())
    grid_to_battery = min(max(battery_charge - solar_to_battery, 0.0), grid_import)
    grid_to_site = max(grid_import - grid_to_battery, 0.0)
    solar_curtailed = float(df["solar_curtailed_kwh"].sum())
    battery_to_site = float(df["battery_discharge_kwh"].sum())

    labels = [
        f"Solar<br><b>{df['solar_kwh'].sum():,.0f} kWh</b>",
        f"Grid<br><b>{grid_import:,.0f} kWh</b>",
        f"Battery<br><b>{battery_to_site:,.0f} kWh delivered</b>",
        f"Building<br><b>{df['demand_kwh'].sum():,.0f} kWh used</b>",
        f"Curtailed solar<br><b>{solar_curtailed:,.0f} kWh</b>",
    ]
    links = [
        (0, 3, solar_to_site, "rgba(244, 185, 66, 0.55)"),
        (0, 2, solar_to_battery, "rgba(99, 196, 154, 0.52)"),
        (0, 4, solar_curtailed, "rgba(244, 185, 66, 0.28)"),
        (1, 3, grid_to_site, "rgba(46, 109, 216, 0.45)"),
        (1, 2, grid_to_battery, "rgba(46, 109, 216, 0.25)"),
        (2, 3, battery_to_site, "rgba(124, 92, 224, 0.46)"),
    ]
    non_zero_links = [link for link in links if link[2] > 0.01]
    figure = go.Figure(
        go.Sankey(
            arrangement="snap",
            node={
                "pad": 28,
                "thickness": 22,
                "line": {"color": COLORS["grid"], "width": 1},
                "label": labels,
                "color": [COLORS["sun"], COLORS["blue"], COLORS["violet"], COLORS["green"], COLORS["muted"]],
                "hovertemplate": "%{label}<extra></extra>",
            },
            link={
                "source": [link[0] for link in non_zero_links],
                "target": [link[1] for link in non_zero_links],
                "value": [link[2] for link in non_zero_links],
                "color": [link[3] for link in non_zero_links],
                "hovertemplate": "%{value:.1f} kWh<extra></extra>",
            },
        )
    )
    figure.update_layout(
        height=340,
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": COLORS["ink"], "size": 13},
        margin={"l": 8, "r": 8, "t": 10, "b": 10},
        showlegend=False,
    )
    return figure


def dispatch_chart(df: pd.DataFrame) -> go.Figure:
    """A detailed, three-panel view of demand/solar/grid, battery activity, and SOC."""
    figure = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=("Building energy", "Battery activity", "Battery state of charge"),
    )
    for name, field, color, dash in [
        ("Demand", "demand_kwh", COLORS["ink"], "solid"),
        ("Solar", "solar_kwh", COLORS["sun"], "solid"),
        ("Grid import", "grid_import_kwh", COLORS["blue"], "dot"),
    ]:
        figure.add_trace(
            go.Scatter(x=df["hour"], y=df[field], name=name, line={"color": color, "width": 2.25, "dash": dash}),
            row=1, col=1,
        )
    figure.add_trace(
        go.Bar(x=df["hour"], y=df["battery_charge_kwh"], name="Charging", marker_color=COLORS["green"]),
        row=2, col=1,
    )
    figure.add_trace(
        go.Bar(x=df["hour"], y=-df["battery_discharge_kwh"], name="Discharging", marker_color=COLORS["violet"]),
        row=2, col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=df["hour"], y=df["soc_end_kwh"], name="State of charge",
            mode="lines", line={"color": COLORS["green"], "width": 2.5},
            fill="tozeroy", fillcolor="rgba(8, 126, 91, 0.12)",
        ),
        row=3, col=1,
    )
    figure.update_yaxes(title_text="kWh", row=1, col=1)
    figure.update_yaxes(title_text="kWh", row=2, col=1)
    figure.update_yaxes(title_text="kWh", row=3, col=1)
    figure.update_xaxes(title_text="Hour of day", dtick=2, row=3, col=1)
    figure.update_layout(barmode="relative")
    return apply_chart_style(figure, height=670)


def economics_chart(summary: dict[str, Any]) -> go.Figure:
    """Show the baseline → optimized cost story in three steps."""
    figure = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=["absolute", "relative", "total"],
            x=["Baseline (no battery)", "Battery dispatch savings", "Optimized cost"],
            y=[summary["baseline_cost_inr"], -summary["estimated_savings_inr"], 0],
            text=[
                f"₹{summary['baseline_cost_inr']:,.0f}",
                f"−₹{summary['estimated_savings_inr']:,.0f}",
                f"₹{summary['optimized_cost_inr']:,.0f}",
            ],
            textposition="outside",
            connector={"line": {"color": COLORS["grid"]}},
            increasing={"marker": {"color": COLORS["sun"]}},
            decreasing={"marker": {"color": COLORS["green"]}},
            totals={"marker": {"color": COLORS["ink"]}},
            hovertemplate="%{x}: ₹%{y:,.0f}<extra></extra>",
        )
    )
    figure.update_yaxes(title_text="Cost (INR)")
    return apply_chart_style(figure, height=360)


st.title("GridPilot AI", icon=":material/bolt:")
st.caption("IMPACT TANK '26 — synthetic solar + battery dispatch optimization for a commercial building in India.")

st.warning(
    "Every input on this dashboard (demand, solar, tariff) is SYNTHETIC, and every dispatch result is a "
    "SIMULATED optimization output. This is a prototype, not a live energy management system.",
    title="Synthetic data / simulated results",
    icon=":material/info:",
)

with st.sidebar:
    st.header("1. Scenario", icon=":material/science:")
    scenario_defs = get_scenario_definitions()
    scenario_keys = list(scenario_defs.keys())
    scenario_labels = {k: scenario_defs[k]["name"] for k in scenario_keys}
    use_preset = st.toggle("Use a standard test scenario", value=True)

    if use_preset:
        scenario_key = st.selectbox(
            "Standard scenario",
            scenario_keys,
            format_func=lambda k: scenario_labels[k],
        )
        st.caption(scenario_defs[scenario_key]["description"])
        preset_battery: BatteryConfig = scenario_defs[scenario_key]["battery"]
        solar_scale = 0.0 if scenario_key == "no_solar" else 1.0
        price_regime = "flat" if scenario_key == "flat_electricity_prices" else (
            "high_evening" if scenario_key == "high_evening_prices" else "normal_tou"
        )
        price_inr_per_kwh = tuple(get_tariff_preset(price_regime).tolist())
        capacity_kwh = preset_battery.capacity_kwh
        power_kw = preset_battery.max_charge_kw
        initial_soc_kwh = preset_battery.initial_soc_kwh
        min_soc_kwh = preset_battery.min_soc_kwh
        max_soc_kwh = preset_battery.max_soc_kwh
        charge_efficiency = preset_battery.charge_efficiency
        discharge_efficiency = preset_battery.discharge_efficiency
    else:
        st.header("2. Building & tariff", icon=":material/apartment:")
        solar_scale = st.slider("Solar system scale", 0.0, 1.5, 1.0, 0.1, help="1.0 = reference 120 kWp rooftop system.")

        st.subheader("Electricity tariff (INR/kWh)", icon=":material/payments:")
        st.caption(
            "Enter your own hourly rate for every hour of the day — this is not limited to the three "
            "built-in presets. Real DISCOM tariffs vary by state, consumer category, and revision, so "
            "edit every value below to match your actual bill or any tariff you want to test."
        )
        start_from = st.selectbox(
            "Pre-fill the table from",
            ["normal_tou", "flat", "high_evening"],
            format_func=lambda r: {
                "normal_tou": "Indian C&I Time-of-Use (normal) — editable starting point",
                "flat": "Flat ₹8.0/kWh all day — editable starting point",
                "high_evening": "High evening peak (₹18.0/kWh, 6–10 PM) — editable starting point",
            }[r],
            help="Choosing a different pre-fill resets the table below to that preset. "
            "Your own edits are kept until you change this.",
        )
        if st.session_state.get("_tariff_prefill_source") != start_from:
            st.session_state["_tariff_prefill_source"] = start_from
            st.session_state["_tariff_prefill_df"] = pd.DataFrame(
                {
                    "hour": np.arange(24, dtype=int),
                    "price_inr_per_kwh": get_tariff_preset(start_from),
                }
            )

        edited_tariff_df = st.data_editor(
            st.session_state["_tariff_prefill_df"],
            key="tariff_editor",
            hide_index=True,
            width="stretch",
            num_rows="fixed",
            column_config={
                "hour": st.column_config.NumberColumn("Hour", disabled=True, format="%d:00"),
                "price_inr_per_kwh": st.column_config.NumberColumn(
                    "Price (₹/kWh)", min_value=0.0, max_value=100.0, step=0.1, format="₹%.2f"
                ),
            },
        )

        try:
            price_inr_per_kwh = tuple(
                validate_custom_tariff(edited_tariff_df["price_inr_per_kwh"].to_numpy()).tolist()
            )
        except ValueError as tariff_error:
            st.error(f"Invalid tariff: {tariff_error}. Using the last valid tariff instead.", icon=":material/error:")
            price_inr_per_kwh = tuple(get_tariff_preset(start_from).tolist())

        with st.expander("Preview tariff curve", icon=":material/show_chart:"):
            tariff_figure = go.Figure(
                go.Scatter(
                    x=list(range(24)), y=list(price_inr_per_kwh), mode="lines+markers",
                    line={"color": COLORS["violet"], "width": 2.5},
                    hovertemplate="Hour %{x}: ₹%{y:.2f}/kWh<extra></extra>",
                )
            )
            tariff_figure.update_xaxes(title_text="Hour of day", dtick=2)
            tariff_figure.update_yaxes(title_text="₹/kWh")
            st.plotly_chart(apply_chart_style(tariff_figure, height=220), width="stretch", config=PLOTLY_CONFIG)

        st.header("3. Battery", icon=":material/battery_charging_full:")
        capacity_kwh = st.number_input("Battery capacity (kWh)", min_value=0.0, max_value=2000.0, value=200.0, step=10.0)
        power_kw = st.number_input("Max charge/discharge power (kW)", min_value=0.0, max_value=1000.0, value=50.0, step=5.0)
        min_soc_pct = st.slider("Minimum state of charge (%)", 0, 50, 10, 1)
        max_soc_pct = st.slider("Maximum state of charge (%)", 50, 100, 95, 1)
        initial_soc_pct = st.slider("Starting state of charge (%)", min_soc_pct, max_soc_pct, 50, 1)
        min_soc_kwh = capacity_kwh * min_soc_pct / 100.0
        max_soc_kwh = capacity_kwh * max_soc_pct / 100.0
        initial_soc_kwh = capacity_kwh * initial_soc_pct / 100.0
        with st.expander("Advanced battery parameters", icon=":material/tune:"):
            charge_efficiency = st.slider("Charge efficiency", 0.80, 1.00, 0.95, 0.01)
            discharge_efficiency = st.slider("Discharge efficiency", 0.80, 1.00, 0.95, 0.01)

    st.space("small")
    st.caption(
        "The battery always ends the day at or above its starting charge level, so savings cannot come from "
        "giving away free initial stored energy."
    )

result = run_scenario_cached(
    solar_scale,
    price_inr_per_kwh,
    capacity_kwh,
    power_kw,
    initial_soc_kwh,
    min_soc_kwh,
    max_soc_kwh,
    charge_efficiency,
    discharge_efficiency,
)
schedule_df = result["schedule_df"]
summary = result["summary"]
validation = result["validation"]

if validation["is_physically_valid"]:
    st.success(
        f"Optimized dispatch found. Estimated savings: ₹{max(summary['estimated_savings_inr'], 0):,.0f} "
        f"({summary['estimated_savings_pct']:.1f}%) versus a no-battery baseline for this synthetic day.",
        title="Dispatch schedule is physically valid",
        icon=":material/check_circle:",
    )
else:
    st.error(
        "The generated schedule failed physical validation: " + "; ".join(validation["violations"]),
        icon=":material/error:",
    )

metric_columns = st.columns(4, gap="small")
metric_columns[0].metric(
    "Optimized cost",
    f"₹{summary['optimized_cost_inr']:,.0f}",
    help="Total 24-hour grid import cost under the optimized battery dispatch.",
    icon=":material/receipt_long:",
    border=True,
)
metric_columns[1].metric(
    "Estimated savings",
    f"₹{max(summary['estimated_savings_inr'], 0):,.0f}",
    f"{summary['estimated_savings_pct']:.1f}%",
    help="Savings versus the no-battery baseline (direct solar self-consumption only).",
    icon=":material/savings:",
    border=True,
)
metric_columns[2].metric(
    "Evening peak reduction",
    f"{summary['evening_peak_reduction_kw']:.1f} kW",
    help="Reduction in the highest hourly grid import during the 6–10 PM evening peak window.",
    icon=":material/trending_down:",
    border=True,
)
metric_columns[3].metric(
    "Grid energy needed",
    f"{summary['optimized_grid_import_kwh']:,.0f} kWh",
    help="Total 24-hour grid import under the optimized dispatch.",
    icon=":material/power:",
    border=True,
)

st.space("small")
chart_column, plan_column = st.columns([7, 3], gap="small", vertical_alignment="top")
with chart_column:
    with st.container(border=True):
        st.subheader("Energy network", icon=":material/account_tree:")
        st.caption("Where energy comes from (solar, grid) and where it goes (building, battery, curtailed).")
        st.plotly_chart(energy_network_chart(schedule_df), width="stretch", config=PLOTLY_CONFIG)

with plan_column:
    with st.container(border=True):
        st.subheader("Battery role", icon=":material/battery_charging_full:")
        st.metric(
            "Total battery throughput",
            f"{summary['total_battery_charge_kwh'] + summary['total_battery_discharge_kwh']:,.0f} kWh",
            help="Sum of charging and discharging energy over the day.",
            icon=":material/swap_horiz:",
        )
        st.caption(
            f"Starts at {summary['initial_soc_kwh']:.0f} kWh, ends at {summary['final_soc_kwh']:.0f} kWh "
            f"(round-trip losses: {summary['battery_round_trip_losses_kwh']:.1f} kWh)."
        )

energy_tab, savings_tab, checks_tab, data_tab = st.tabs(
    [
        ":material/bolt: Energy story",
        ":material/savings: Savings story",
        ":material/verified: Physics checks",
        ":material/table_chart: Hourly data",
    ]
)

with energy_tab:
    overview_column, detail_column = st.columns([2, 3], gap="small", vertical_alignment="top")
    with overview_column:
        with st.container(border=True):
            st.subheader("24-hour energy rhythm", icon=":material/insights:")
            st.plotly_chart(overview_chart(schedule_df), width="stretch", config=PLOTLY_CONFIG)
    with detail_column:
        with st.container(border=True):
            st.subheader("Dispatch mechanics", icon=":material/bolt:")
            st.plotly_chart(dispatch_chart(schedule_df), width="stretch", config=PLOTLY_CONFIG)

with savings_tab:
    st.plotly_chart(economics_chart(summary), width="stretch", config=PLOTLY_CONFIG)
    st.caption("This is a cost-of-energy illustration, not a return-on-investment or payback estimate.")

with checks_tab:
    check_table = pd.DataFrame(
        {
            "Check": [
                "Hourly AC energy balance",
                "Battery stays within SOC bounds",
                "No simultaneous charge/discharge",
                "Battery ends at/above starting charge",
                "Solver status",
            ],
            "Result": [
                "Pass" if validation["is_physically_valid"] else "Review",
                f"{summary['min_soc_observed_kwh']:.0f}–{summary['max_soc_observed_kwh']:.0f} kWh "
                f"within [{min_soc_kwh:.0f}, {max_soc_kwh:.0f}] kWh",
                "Pass",
                f"{summary['initial_soc_kwh']:.0f} kWh → {summary['final_soc_kwh']:.0f} kWh",
                result["metadata"]["solver_message"],
            ],
        }
    )
    st.dataframe(check_table, hide_index=True, width="stretch")
    if validation["violations"]:
        st.error("Violations:\n" + "\n".join(f"- {v}" for v in validation["violations"]))

with data_tab:
    display_cols = [
        "hour", "demand_kwh", "solar_kwh", "price_inr_per_kwh",
        "baseline_grid_import_kwh", "grid_import_kwh",
        "battery_charge_kwh", "battery_discharge_kwh", "soc_end_kwh",
        "baseline_cost_inr", "optimized_cost_inr", "hourly_savings_inr",
    ]
    st.dataframe(schedule_df[display_cols].round(2), hide_index=True, width="stretch")
    st.download_button(
        "Download hourly schedule (CSV)",
        schedule_df.to_csv(index=False).encode("utf-8"),
        file_name="gridpilot_synthetic_dispatch.csv",
        mime="text/csv",
        icon=":material/download:",
    )

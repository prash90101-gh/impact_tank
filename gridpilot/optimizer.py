"""
optimizer.py
------------
Transparent Linear Programming (LP) optimization engine for a single commercial
building with rooftop solar PV and a Battery Energy Storage System (BESS)
(GridPilot AI - IMPACT TANK '26 Prototype).

Mathematical Formulation (SciPy HiGHS Linear Program):
------------------------------------------------------
For each hour t in {0, ..., T-1} (with step duration dt = 1 hour):
  Decision variables:
    - g[t] >= 0 : Grid electricity import (kWh)
    - c[t] >= 0 : Battery charging energy (kWh, AC-side into inverter)
    - d[t] >= 0 : Battery discharging energy (kWh, AC-side delivered to building)
    - u[t] >= 0 : Solar PV energy utilized on-site (kWh, for building load or battery charging)
    - P_peak >= 0 : Maximum hourly grid import across the horizon (kW)

  Objective:
    Minimize sum_{t=0}^{T-1} (price[t] * g[t] + eps_cycle * (c[t] + d[t])) + eps_peak * P_peak
    where:
      - price[t] * g[t] is the exact hourly electricity import bill (INR),
      - eps_cycle (1e-6 INR/kWh) prevents physically impossible simultaneous
        charging and discharging and eliminates pointless zero-value battery cycling,
      - eps_peak (1e-3 INR/kW) acts as a secondary tie-breaker across equally
        priced hours so battery charging never creates artificial grid import spikes
        and battery discharging smooths the highest peak demand hours first.

  Subject to:
    1. Hourly AC Energy Balance (for all t):
         g[t] - c[t] + d[t] + u[t] == demand[t]
    2. Peak Grid Import Envelope (for all t):
         g[t] - P_peak <= 0
    3. Solar Availability Bound (for all t):
         0 <= u[t] <= solar[t]
    4. Battery Power Limits (for all t):
         0 <= c[t] <= max_charge_kw * dt
         0 <= d[t] <= max_discharge_kw * dt
    5. State of Charge (SOC) Trajectory & Bounds (for all t):
         soc_end[t] = initial_soc + sum_{k=0}^{t} (eta_c * c[k] - d[k] / eta_d)
         min_soc_kwh <= soc_end[t] <= max_soc_kwh
    6. Terminal SOC Neutrality Constraint (t = T-1):
         soc_end[T-1] >= target_final_soc_kwh (defaults to initial_soc_kwh)
       Ensures the optimizer cannot claim fake bill savings by draining free
       initial battery energy without replenishing it by the end of the day.
"""

from typing import Any, Dict, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from scipy.optimize import linprog

from .energy_model import (
    BatteryConfig,
    compute_baseline_dispatch,
    get_scenario_definitions,
    validate_physical_schedule,
)


def optimize_building_dispatch(
    demand_kwh: Union[Sequence[float], np.ndarray, pd.Series],
    solar_kwh: Union[Sequence[float], np.ndarray, pd.Series],
    price_inr_per_kwh: Union[Sequence[float], np.ndarray, pd.Series],
    battery_capacity_kwh: float = 200.0,
    max_charge_kw: float = 50.0,
    max_discharge_kw: float = 50.0,
    charge_efficiency: float = 0.95,
    discharge_efficiency: float = 0.95,
    initial_soc_kwh: float = 100.0,
    min_soc_kwh: float = 20.0,
    max_soc_kwh: float = 190.0,
    enforce_terminal_soc: bool = True,
    terminal_soc_kwh: Optional[float] = None,
    battery_config: Optional[BatteryConfig] = None,
    data_source_label: str = "SYNTHETIC_INPUT_DATA",
) -> Dict[str, Any]:
    """
    Solve the single-building solar + battery dispatch optimization problem and
    return Streamlit-ready hourly schedules, KPIs, and physics validation status.
    """
    demand = np.asarray(demand_kwh, dtype=float).flatten()
    solar = np.asarray(solar_kwh, dtype=float).flatten()
    price = np.asarray(price_inr_per_kwh, dtype=float).flatten()

    n = len(demand)
    if n == 0:
        raise ValueError("Input hourly series cannot be empty.")
    if not (len(solar) == n and len(price) == n):
        raise ValueError(
            f"Length mismatch: demand ({len(demand)}), solar ({len(solar)}), and price ({len(price)}) must have identical length."
        )
    if (demand < 0).any():
        raise ValueError("Hourly building demand_kwh must be non-negative.")
    if (solar < 0).any():
        raise ValueError("Hourly solar_kwh must be non-negative.")
    if (price < 0).any():
        raise ValueError("Hourly price_inr_per_kwh must be non-negative.")

    if battery_config is None:
        battery = BatteryConfig(
            capacity_kwh=float(battery_capacity_kwh),
            max_charge_kw=float(max_charge_kw),
            max_discharge_kw=float(max_discharge_kw),
            charge_efficiency=float(charge_efficiency),
            discharge_efficiency=float(discharge_efficiency),
            initial_soc_kwh=float(initial_soc_kwh),
            min_soc_kwh=float(min_soc_kwh),
            max_soc_kwh=float(max_soc_kwh),
            enforce_terminal_soc=bool(enforce_terminal_soc),
            terminal_soc_kwh=terminal_soc_kwh,
        )
    else:
        battery = battery_config

    battery.validate()

    # Compute unoptimized baseline first
    baseline_df = compute_baseline_dispatch(
        demand_kwh=demand,
        solar_kwh=solar,
        price_inr_per_kwh=price,
        initial_soc_kwh=battery.initial_soc_kwh,
    )

    # Fast path if battery has zero usable capacity or zero charge/discharge power
    if (
        battery.capacity_kwh <= 1e-9
        or battery.usable_capacity_kwh <= 1e-9
        or (battery.max_charge_kw <= 1e-9 and battery.max_discharge_kw <= 1e-9)
    ):
        schedule_df = baseline_df.copy()
        schedule_df["result_type"] = "SIMULATED_OPTIMIZED_NO_BATTERY"
        is_valid, violations = validate_physical_schedule(schedule_df, battery)
        summary = _build_summary_metrics(schedule_df, battery)
        return {
            "schedule_df": schedule_df,
            "summary": summary,
            "battery_config": battery.to_dict(),
            "validation": {
                "is_physically_valid": is_valid,
                "violations": violations,
            },
            "metadata": {
                "data_source_label": data_source_label,
                "simulation_label": "SIMULATED_OPTIMIZATION_RESULT",
                "solver": "scipy.optimize.linprog (HiGHS - bypass for zero usable battery)",
                "solver_success": True,
                "solver_message": "Zero usable battery capacity; baseline schedule is uniquely feasible.",
            },
        }

    # Variable layout for T = n hours:
    #   [g_0..g_{n-1}, c_0..c_{n-1}, d_0..d_{n-1}, u_0..u_{n-1}, P_peak] -> total 4*n + 1 variables
    eps_cycle = 1e-6
    eps_peak = 1e-3
    c_obj = np.concatenate([
        price,                          # g[t]: grid import tariff (INR/kWh)
        np.full(n, eps_cycle),          # c[t]: tiny regularization on battery charge
        np.full(n, eps_cycle),          # d[t]: tiny regularization on battery discharge
        np.zeros(n),                    # u[t]: zero marginal cost for using on-site solar
        np.array([eps_peak]),           # P_peak: tie-breaking peak grid import smoothing
    ])

    # 1. Equality constraints: g[t] - c[t] + d[t] + u[t] + 0 * P_peak == demand[t]
    eye_n = np.eye(n)
    zeros_col = np.zeros((n, 1))
    A_eq = np.hstack([eye_n, -eye_n, eye_n, eye_n, zeros_col])
    b_eq = demand.copy()

    # 2. Inequality constraints:
    # (a) SOC upper and lower bounds at each hour t
    tril = np.tril(np.ones((n, n)))
    eta_c = battery.charge_efficiency
    inv_eta_d = 1.0 / battery.discharge_efficiency

    # Upper bound: eta_c * sum c[k] - (1/eta_d) * sum d[k] <= max_soc - soc_0
    A_soc_max = np.hstack([
        np.zeros((n, n)),
        eta_c * tril,
        -inv_eta_d * tril,
        np.zeros((n, n)),
        zeros_col,
    ])
    b_soc_max = np.full(n, battery.max_soc_kwh - battery.initial_soc_kwh, dtype=float)

    # Lower bound: -eta_c * sum c[k] + (1/eta_d) * sum d[k] <= soc_0 - min_soc
    min_soc_vec = np.full(n, battery.min_soc_kwh, dtype=float)
    if battery.enforce_terminal_soc:
        min_soc_vec[-1] = max(min_soc_vec[-1], battery.target_final_soc_kwh)

    A_soc_min = np.hstack([
        np.zeros((n, n)),
        -eta_c * tril,
        inv_eta_d * tril,
        np.zeros((n, n)),
        zeros_col,
    ])
    b_soc_min = battery.initial_soc_kwh - min_soc_vec

    # (b) Peak grid import envelope: g[t] - P_peak <= 0
    A_peak = np.hstack([
        eye_n,
        np.zeros((n, n)),
        np.zeros((n, n)),
        np.zeros((n, n)),
        -np.ones((n, 1)),
    ])
    b_peak = np.zeros(n, dtype=float)

    A_ub = np.vstack([A_soc_max, A_soc_min, A_peak])
    b_ub = np.concatenate([b_soc_max, b_soc_min, b_peak])

    # 3. Variable bounds
    bounds = (
        [(0.0, None)] * n
        + [(0.0, float(battery.max_charge_kw))] * n
        + [(0.0, float(battery.max_discharge_kw))] * n
        + [(0.0, float(s)) for s in solar]
        + [(0.0, None)]
    )

    res = linprog(
        c=c_obj,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    if not res.success:
        raise RuntimeError(f"Optimization failed to find a feasible schedule: {res.message}")

    x = res.x
    grid_import = np.clip(x[0:n], 0.0, None)
    batt_charge = np.clip(x[n:2 * n], 0.0, None)
    batt_discharge = np.clip(x[2 * n:3 * n], 0.0, None)
    solar_used = np.clip(x[3 * n:4 * n], 0.0, solar)

    # Guard against any floating-point solver noise (< 1e-9) in simultaneous charge/discharge
    overlap = np.minimum(batt_charge, batt_discharge)
    if (overlap > 0).any():
        batt_charge = np.where(batt_charge >= batt_discharge, batt_charge - overlap, 0.0)
        batt_discharge = np.where(batt_discharge > batt_charge, batt_discharge - overlap, 0.0)
        net_req = demand + batt_charge - batt_discharge
        solar_used = np.minimum(solar, np.maximum(0.0, net_req))
        grid_import = np.maximum(0.0, net_req - solar_used)

    # Compute exact SOC trajectory
    soc_start = np.zeros(n, dtype=float)
    soc_end = np.zeros(n, dtype=float)
    curr_soc = float(battery.initial_soc_kwh)
    for t in range(n):
        soc_start[t] = curr_soc
        curr_soc = curr_soc + (eta_c * batt_charge[t]) - (batt_discharge[t] * inv_eta_d)
        soc_end[t] = curr_soc

    solar_curtailed = np.maximum(0.0, solar - solar_used)
    baseline_grid = baseline_df["baseline_grid_import_kwh"].to_numpy()
    baseline_cost = baseline_df["baseline_cost_inr"].to_numpy()
    optimized_cost = grid_import * price
    hourly_savings = baseline_cost - optimized_cost

    schedule_df = pd.DataFrame({
        "hour": np.arange(n, dtype=int),
        "demand_kwh": demand,
        "solar_kwh": solar,
        "price_inr_per_kwh": price,
        "solar_to_load_kwh": solar_used,
        "solar_curtailed_kwh": solar_curtailed,
        "baseline_grid_import_kwh": baseline_grid,
        "grid_import_kwh": grid_import,
        "battery_charge_kwh": batt_charge,
        "battery_discharge_kwh": batt_discharge,
        "net_battery_power_kw": batt_discharge - batt_charge,
        "soc_start_kwh": soc_start,
        "soc_end_kwh": soc_end,
        "baseline_cost_inr": baseline_cost,
        "optimized_cost_inr": optimized_cost,
        "hourly_savings_inr": hourly_savings,
        "result_type": "SIMULATED_OPTIMIZED_LP",
    })

    is_valid, violations = validate_physical_schedule(schedule_df, battery)
    if not is_valid:
        raise RuntimeError(f"Generated schedule violated physical constraints: {violations}")

    summary = _build_summary_metrics(schedule_df, battery)

    return {
        "schedule_df": schedule_df,
        "summary": summary,
        "battery_config": battery.to_dict(),
        "validation": {
            "is_physically_valid": is_valid,
            "violations": violations,
        },
        "metadata": {
            "data_source_label": data_source_label,
            "simulation_label": "SIMULATED_OPTIMIZATION_RESULT",
            "solver": "scipy.optimize.linprog (HiGHS)",
            "solver_success": bool(res.success),
            "solver_message": str(res.message),
        },
    }


def _build_summary_metrics(schedule_df: pd.DataFrame, battery: BatteryConfig) -> Dict[str, Any]:
    """Compute comprehensive financial, energy, and peak demand KPIs for Streamlit display."""
    baseline_cost_total = float(schedule_df["baseline_cost_inr"].sum())
    optimized_cost_total = float(schedule_df["optimized_cost_inr"].sum())
    estimated_savings_inr = baseline_cost_total - optimized_cost_total
    savings_pct = (
        (estimated_savings_inr / baseline_cost_total) * 100.0
        if baseline_cost_total > 1e-9
        else 0.0
    )

    baseline_peak_kw = float(schedule_df["baseline_grid_import_kwh"].max())
    optimized_peak_kw = float(schedule_df["grid_import_kwh"].max())
    peak_reduction_kw = baseline_peak_kw - optimized_peak_kw
    peak_reduction_pct = (
        (peak_reduction_kw / baseline_peak_kw) * 100.0
        if baseline_peak_kw > 1e-9
        else 0.0
    )

    # Evening peak window (18:00 - 21:59, hours 18, 19, 20, 21)
    evening_mask = schedule_df["hour"].isin([18, 19, 20, 21])
    if evening_mask.any():
        evening_baseline_peak_kw = float(schedule_df.loc[evening_mask, "baseline_grid_import_kwh"].max())
        evening_optimized_peak_kw = float(schedule_df.loc[evening_mask, "grid_import_kwh"].max())
        evening_peak_reduction_kw = evening_baseline_peak_kw - evening_optimized_peak_kw
    else:
        evening_baseline_peak_kw = baseline_peak_kw
        evening_optimized_peak_kw = optimized_peak_kw
        evening_peak_reduction_kw = peak_reduction_kw

    total_demand_kwh = float(schedule_df["demand_kwh"].sum())
    total_solar_generated_kwh = float(schedule_df["solar_kwh"].sum())
    total_solar_used_kwh = float(schedule_df["solar_to_load_kwh"].sum())
    total_solar_curtailed_kwh = float(schedule_df["solar_curtailed_kwh"].sum())
    baseline_solar_curtailed_kwh = float(
        np.maximum(0.0, schedule_df["solar_kwh"] - schedule_df["demand_kwh"]).sum()
    )

    total_charge_kwh = float(schedule_df["battery_charge_kwh"].sum())
    total_discharge_kwh = float(schedule_df["battery_discharge_kwh"].sum())
    initial_soc = float(schedule_df["soc_start_kwh"].iloc[0])
    final_soc = float(schedule_df["soc_end_kwh"].iloc[-1])
    soc_delta_kwh = final_soc - initial_soc
    battery_losses_kwh = max(0.0, total_charge_kwh - total_discharge_kwh - soc_delta_kwh)

    return {
        "baseline_cost_inr": round(baseline_cost_total, 2),
        "optimized_cost_inr": round(optimized_cost_total, 2),
        "estimated_savings_inr": round(estimated_savings_inr, 2),
        "estimated_savings_pct": round(savings_pct, 2),
        "baseline_peak_grid_kw": round(baseline_peak_kw, 2),
        "optimized_peak_grid_kw": round(optimized_peak_kw, 2),
        "peak_demand_reduction_kw": round(peak_reduction_kw, 2),
        "peak_demand_reduction_pct": round(peak_reduction_pct, 2),
        "evening_baseline_peak_kw": round(evening_baseline_peak_kw, 2),
        "evening_optimized_peak_kw": round(evening_optimized_peak_kw, 2),
        "evening_peak_reduction_kw": round(evening_peak_reduction_kw, 2),
        "total_demand_kwh": round(total_demand_kwh, 2),
        "total_solar_generated_kwh": round(total_solar_generated_kwh, 2),
        "total_solar_used_kwh": round(total_solar_used_kwh, 2),
        "baseline_solar_curtailed_kwh": round(baseline_solar_curtailed_kwh, 2),
        "optimized_solar_curtailed_kwh": round(total_solar_curtailed_kwh, 2),
        "baseline_grid_import_kwh": round(float(schedule_df["baseline_grid_import_kwh"].sum()), 2),
        "optimized_grid_import_kwh": round(float(schedule_df["grid_import_kwh"].sum()), 2),
        "total_battery_charge_kwh": round(total_charge_kwh, 2),
        "total_battery_discharge_kwh": round(total_discharge_kwh, 2),
        "battery_round_trip_losses_kwh": round(battery_losses_kwh, 2),
        "initial_soc_kwh": round(initial_soc, 2),
        "final_soc_kwh": round(final_soc, 2),
        "net_soc_change_kwh": round(soc_delta_kwh, 2),
        "min_soc_observed_kwh": round(float(schedule_df["soc_end_kwh"].min()), 2),
        "max_soc_observed_kwh": round(float(schedule_df["soc_end_kwh"].max()), 2),
        "is_synthetic_data": True,
        "is_simulated_result": True,
    }


def run_scenario(scenario_key: str) -> Dict[str, Any]:
    """Run one of the 5 standard synthetic test scenarios by key."""
    scenarios = get_scenario_definitions()
    if scenario_key not in scenarios:
        raise KeyError(
            f"Unknown scenario '{scenario_key}'. Valid keys: {list(scenarios.keys())}"
        )

    scen = scenarios[scenario_key]
    df = scen["profile_df"]
    battery: BatteryConfig = scen["battery"]

    result = optimize_building_dispatch(
        demand_kwh=df["demand_kwh"],
        solar_kwh=df["solar_kwh"],
        price_inr_per_kwh=df["price_inr_per_kwh"],
        battery_config=battery,
        data_source_label="SYNTHETIC_SCENARIO_DATA",
    )
    result["scenario_key"] = scenario_key
    result["scenario_name"] = scen["name"]
    result["scenario_description"] = scen["description"]
    return result


def run_all_scenarios() -> Tuple[Dict[str, Dict[str, Any]], pd.DataFrame]:
    """
    Execute all five required prototype scenarios and return both detailed
    results and a consolidated comparison DataFrame for Streamlit or CLI reporting.
    """
    scenarios = get_scenario_definitions()
    results: Dict[str, Dict[str, Any]] = {}
    rows = []

    for key in scenarios:
        res = run_scenario(key)
        results[key] = res
        s = res["summary"]
        rows.append({
            "scenario_key": key,
            "scenario_name": res["scenario_name"],
            "baseline_cost_inr": s["baseline_cost_inr"],
            "optimized_cost_inr": s["optimized_cost_inr"],
            "estimated_savings_inr": s["estimated_savings_inr"],
            "estimated_savings_pct": s["estimated_savings_pct"],
            "baseline_peak_kw": s["baseline_peak_grid_kw"],
            "optimized_peak_kw": s["optimized_peak_grid_kw"],
            "evening_peak_reduction_kw": s["evening_peak_reduction_kw"],
            "battery_charge_kwh": s["total_battery_charge_kwh"],
            "battery_discharge_kwh": s["total_battery_discharge_kwh"],
            "initial_soc_kwh": s["initial_soc_kwh"],
            "final_soc_kwh": s["final_soc_kwh"],
            "physically_valid": res["validation"]["is_physically_valid"],
            "data_status": "SYNTHETIC / SIMULATED",
        })

    comparison_df = pd.DataFrame(rows)
    return results, comparison_df


if __name__ == "__main__":
    print("=" * 100)
    print("GRIDPILOT AI - SINGLE-BUILDING OPTIMIZATION ENGINE (IMPACT TANK '26 PROTOTYPE)")
    print("NOTICE: All input profiles are SYNTHETIC and all outputs are SIMULATED results.")
    print("=" * 100)

    all_results, summary_table = run_all_scenarios()

    display_cols = [
        "scenario_name",
        "baseline_cost_inr",
        "optimized_cost_inr",
        "estimated_savings_inr",
        "estimated_savings_pct",
        "baseline_peak_kw",
        "optimized_peak_kw",
        "initial_soc_kwh",
        "final_soc_kwh",
        "physically_valid",
    ]
    print("\n--- FIVE-SCENARIO SUMMARY TABLE (SYNTHETIC DATA / SIMULATED RESULTS) ---")
    print(summary_table[display_cols].to_string(index=False))

    # Print sample hourly table for Scenario 1 (Normal Operation)
    norm_df = all_results["normal_operation"]["schedule_df"]
    print("\n--- SCENARIO 1 (NORMAL OPERATION) HOURLY DISPATCH SCHEDULE ---")
    hourly_cols = [
        "hour",
        "demand_kwh",
        "solar_kwh",
        "price_inr_per_kwh",
        "baseline_grid_import_kwh",
        "grid_import_kwh",
        "battery_charge_kwh",
        "battery_discharge_kwh",
        "soc_end_kwh",
        "baseline_cost_inr",
        "optimized_cost_inr",
    ]
    print(norm_df[hourly_cols].round(2).to_string(index=False))

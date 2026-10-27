"""
energy_model.py
---------------
Data models, synthetic profile generators, scenario definitions, and physics
validation utilities for the GridPilot AI single-building optimization engine
(IMPACT TANK '26 Prototype).

NOTE ON DATA & SIMULATION:
All load, solar, and tariff profiles generated in this module are SYNTHETIC
and all dispatch schedules and cost savings are SIMULATED results for prototype
demonstration.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BatteryConfig:
    """Physical and operational parameters for a building Battery Energy Storage System (BESS)."""
    capacity_kwh: float = 200.0
    max_charge_kw: float = 50.0
    max_discharge_kw: float = 50.0
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    initial_soc_kwh: float = 100.0
    min_soc_kwh: float = 20.0
    max_soc_kwh: float = 190.0
    enforce_terminal_soc: bool = True
    terminal_soc_kwh: Optional[float] = None  # Defaults to initial_soc_kwh if None

    @property
    def target_final_soc_kwh(self) -> float:
        """Target minimum battery state of charge at the end of the horizon (kWh)."""
        if self.terminal_soc_kwh is not None:
            return float(self.terminal_soc_kwh)
        return float(self.initial_soc_kwh)

    @property
    def usable_capacity_kwh(self) -> float:
        """Usable energy storage window between min and max SOC (kWh)."""
        return max(0.0, float(self.max_soc_kwh - self.min_soc_kwh))

    @property
    def round_trip_efficiency(self) -> float:
        """Combined round-trip AC-to-AC storage efficiency."""
        return float(self.charge_efficiency * self.discharge_efficiency)

    def validate(self) -> None:
        """Raise ValueError if any physical battery parameters are invalid or inconsistent."""
        if self.capacity_kwh < 0:
            raise ValueError("Battery capacity_kwh must be non-negative.")
        if self.max_charge_kw < 0 or self.max_discharge_kw < 0:
            raise ValueError("Maximum charge and discharge power (kW) must be non-negative.")
        if not (0 < self.charge_efficiency <= 1.0):
            raise ValueError("charge_efficiency must be in (0, 1].")
        if not (0 < self.discharge_efficiency <= 1.0):
            raise ValueError("discharge_efficiency must be in (0, 1].")
        if self.min_soc_kwh < 0 or self.max_soc_kwh < 0:
            raise ValueError("SOC bounds must be non-negative.")
        if self.min_soc_kwh > self.max_soc_kwh + 1e-9:
            raise ValueError(
                f"min_soc_kwh ({self.min_soc_kwh}) cannot exceed max_soc_kwh ({self.max_soc_kwh})."
            )
        if self.max_soc_kwh > self.capacity_kwh + 1e-9:
            raise ValueError(
                f"max_soc_kwh ({self.max_soc_kwh}) cannot exceed capacity_kwh ({self.capacity_kwh})."
            )
        if not (self.min_soc_kwh - 1e-9 <= self.initial_soc_kwh <= self.max_soc_kwh + 1e-9):
            raise ValueError(
                f"initial_soc_kwh ({self.initial_soc_kwh}) must lie within "
                f"[{self.min_soc_kwh}, {self.max_soc_kwh}]."
            )
        if self.enforce_terminal_soc:
            target = self.target_final_soc_kwh
            if not (self.min_soc_kwh - 1e-9 <= target <= self.max_soc_kwh + 1e-9):
                raise ValueError(
                    f"target_final_soc_kwh ({target}) must lie within "
                    f"[{self.min_soc_kwh}, {self.max_soc_kwh}]."
                )

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["target_final_soc_kwh"] = self.target_final_soc_kwh
        d["usable_capacity_kwh"] = self.usable_capacity_kwh
        d["round_trip_efficiency"] = self.round_trip_efficiency
        return d


def generate_synthetic_commercial_profiles(
    hours: int = 24,
    solar_scale: float = 1.0,
    price_regime: str = "normal_tou",
    custom_price_inr_per_kwh: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """
    Generate a 24-hour SYNTHETIC commercial building dataset aligned with Indian C&I
    operating patterns (IMPACT TANK '26 context: solar peaks 10 AM - 4 PM,
    commercial + evening peak demand 6 PM - 10 PM).

    Parameters
    ----------
    hours : int
        Number of hourly time steps (default 24).
    solar_scale : float
        Multiplier for solar PV generation (1.0 = normal 120 kWp system, 0.0 = no solar).
    price_regime : str
        One of {'normal_tou', 'flat', 'high_evening', 'custom'}. Ignored if
        ``custom_price_inr_per_kwh`` is provided (the regime is then forced to
        'custom').
    custom_price_inr_per_kwh : Sequence[float], optional
        A user-supplied hourly tariff (INR/kWh) of length ``hours``. When
        provided, this fully overrides ``price_regime`` so a real or
        hypothetical DISCOM tariff can be entered directly instead of being
        limited to the three built-in presets. See ``validate_custom_tariff``
        for the validation rules applied to this input.

    Returns
    -------
    pd.DataFrame
        Hourly DataFrame with columns:
        ['hour', 'demand_kwh', 'solar_kwh', 'price_inr_per_kwh', 'data_label']
    """
    if hours != 24:
        raise ValueError("Synthetic profile generator currently supports a 24-hour horizon.")

    if custom_price_inr_per_kwh is not None:
        price_inr = validate_custom_tariff(custom_price_inr_per_kwh, hours=hours)
    else:
        price_inr = get_tariff_preset(price_regime, hours=hours)

    return _build_profile_frame(hours=hours, solar_scale=solar_scale, price_inr_per_kwh=price_inr)


# Synthetic Commercial Building Hourly Demand (kWh)
# Represents a medium commercial office/retail facility in India:
# - Base overnight load (security, servers, chillers standby): ~35-42 kWh
# - Daytime commercial operations (HVAC, lighting, equipment): ~95-120 kWh
# - Evening lighting/retail/HVAC peak (18:00 - 22:00): up to 135 kWh
_DEMAND_KWH_24H = np.array([
    38.0, 36.0, 35.0, 35.0, 37.0, 42.0,   # 00:00 - 05:00 (Off-peak night)
    55.0, 72.0, 92.0, 108.0, 115.0, 118.0, # 06:00 - 11:00 (Morning ramp & office hours)
    112.0, 110.0, 114.0, 116.0, 120.0, 128.0, # 12:00 - 17:00 (Afternoon commercial load)
    135.0, 132.0, 122.0, 105.0, 70.0, 48.0    # 18:00 - 23:00 (6 PM - 10 PM evening peak & ramp down)
], dtype=float)

# Synthetic Rooftop Solar Generation (kWh)
# Peaks between 10:00 and 16:00, creating midday surplus/offset potential
_BASE_SOLAR_KWH_24H = np.array([
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,         # 00:00 - 05:00
    3.0, 18.0, 48.0, 82.0, 115.0, 135.0,  # 06:00 - 11:00
    142.0, 138.0, 122.0, 95.0, 52.0, 15.0, # 12:00 - 17:00
    2.0, 0.0, 0.0, 0.0, 0.0, 0.0          # 18:00 - 23:00
], dtype=float)

# Built-in Indian C&I Time-of-Day (ToD / ToU) tariff presets (INR / kWh).
# These remain available as quick-start defaults and as pre-fill values for
# the custom tariff editor, but a user is never limited to only these three
# fixed rate structures — see `custom_price_inr_per_kwh` above.
_TARIFF_PRESETS_24H: Dict[str, np.ndarray] = {
    # Night off-peak: INR 5.5/kWh | Day: INR 8.0/kWh | Solar valley (10-15): INR 6.5/kWh | Evening peak (18-22): INR 12.5/kWh
    "normal_tou": np.array([
        5.5, 5.5, 5.5, 5.5, 5.5, 5.5,     # 00:00 - 05:00
        8.0, 8.0, 8.0, 8.0, 6.5, 6.5,     # 06:00 - 11:00
        6.5, 6.5, 6.5, 8.0, 8.0, 8.0,     # 12:00 - 17:00
        12.5, 12.5, 12.5, 12.5, 8.0, 5.5  # 18:00 - 23:00 (18:00-21:59 peak)
    ], dtype=float),
    "flat": np.full(24, 8.0, dtype=float),
    # Extreme evening peak stress tariff (INR 18.0/kWh during 18:00 - 21:59)
    "high_evening": np.array([
        5.5, 5.5, 5.5, 5.5, 5.5, 5.5,
        8.0, 8.0, 8.0, 8.0, 6.5, 6.5,
        6.5, 6.5, 6.5, 8.0, 8.0, 9.0,
        18.0, 18.0, 18.0, 18.0, 9.0, 5.5
    ], dtype=float),
}


def get_tariff_preset(price_regime: str, hours: int = 24) -> np.ndarray:
    """
    Return a built-in hourly tariff array (INR/kWh) by name.

    These three presets ('normal_tou', 'flat', 'high_evening') are provided
    as convenient, clearly-labeled starting points only. For a real or
    different hypothetical tariff, pass an hourly array directly to
    ``generate_synthetic_commercial_profiles(custom_price_inr_per_kwh=...)``
    or ``optimize_building_dispatch(price_inr_per_kwh=...)`` instead of being
    limited to these three fixed rate structures.
    """
    if hours != 24:
        raise ValueError("Built-in tariff presets currently only support a 24-hour horizon.")
    if price_regime not in _TARIFF_PRESETS_24H:
        raise ValueError(
            f"Unknown price_regime '{price_regime}'. Use one of {sorted(_TARIFF_PRESETS_24H)}, "
            "or supply custom_price_inr_per_kwh for any other tariff."
        )
    return _TARIFF_PRESETS_24H[price_regime].copy()


def validate_custom_tariff(price_inr_per_kwh: Sequence[float], hours: int = 24) -> np.ndarray:
    """
    Validate a user-supplied hourly electricity tariff (INR/kWh).

    A custom tariff may come from a real DISCOM bill, a different state's
    rate schedule, or any hypothetical rate structure the user wants to
    stress-test — it is not limited to the three built-in presets. This
    function only checks structural/physical sanity (length, finiteness,
    non-negativity); it intentionally does not restrict which rates are
    allowed, since real tariffs vary widely by state, consumer category,
    and regulatory revision.
    """
    price = np.asarray(price_inr_per_kwh, dtype=float).flatten()
    if len(price) != hours:
        raise ValueError(
            f"Custom tariff must have exactly {hours} hourly values (got {len(price)})."
        )
    if not np.all(np.isfinite(price)):
        raise ValueError("Custom tariff values must all be finite numbers.")
    if (price < 0).any():
        raise ValueError("Custom tariff values (INR/kWh) must be non-negative.")
    return price


def _build_profile_frame(hours: int, solar_scale: float, price_inr_per_kwh: np.ndarray) -> pd.DataFrame:
    """Assemble the standard synthetic demand/solar/tariff DataFrame for a given tariff array."""
    hour_idx = np.arange(hours, dtype=int)
    demand_kwh = _DEMAND_KWH_24H
    solar_kwh = np.maximum(0.0, _BASE_SOLAR_KWH_24H * float(solar_scale))

    return pd.DataFrame({
        "hour": hour_idx,
        "demand_kwh": demand_kwh,
        "solar_kwh": solar_kwh,
        "price_inr_per_kwh": np.asarray(price_inr_per_kwh, dtype=float),
        "data_label": "SYNTHETIC_COMMERCIAL_PROFILE",
    })


def compute_baseline_dispatch(
    demand_kwh: np.ndarray,
    solar_kwh: np.ndarray,
    price_inr_per_kwh: np.ndarray,
    initial_soc_kwh: float = 0.0,
) -> pd.DataFrame:
    """
    Calculate the unoptimized baseline building operation (direct solar self-consumption,
    no battery dispatch, any excess solar is curtailed/unmonetized under zero-export baseline).

    Returns a DataFrame with the exact same schema as the optimized schedule for clean
    side-by-side comparison in Streamlit.
    """
    demand = np.asarray(demand_kwh, dtype=float)
    solar = np.asarray(solar_kwh, dtype=float)
    price = np.asarray(price_inr_per_kwh, dtype=float)

    n = len(demand)
    solar_used = np.minimum(demand, solar)
    solar_curtailed = np.maximum(0.0, solar - solar_used)
    grid_import = np.maximum(0.0, demand - solar)
    cost_inr = grid_import * price

    return pd.DataFrame({
        "hour": np.arange(n, dtype=int),
        "demand_kwh": demand,
        "solar_kwh": solar,
        "price_inr_per_kwh": price,
        "solar_to_load_kwh": solar_used,
        "solar_curtailed_kwh": solar_curtailed,
        "baseline_grid_import_kwh": grid_import,
        "grid_import_kwh": grid_import,
        "battery_charge_kwh": np.zeros(n, dtype=float),
        "battery_discharge_kwh": np.zeros(n, dtype=float),
        "net_battery_power_kw": np.zeros(n, dtype=float),
        "soc_start_kwh": np.full(n, float(initial_soc_kwh), dtype=float),
        "soc_end_kwh": np.full(n, float(initial_soc_kwh), dtype=float),
        "baseline_cost_inr": cost_inr,
        "optimized_cost_inr": cost_inr,
        "hourly_savings_inr": np.zeros(n, dtype=float),
        "result_type": "SIMULATED_BASELINE",
    })


def get_scenario_definitions() -> Dict[str, Dict]:
    """
    Return the five standardized test scenarios required for the GridPilot AI prototype:
      1. normal_operation
      2. no_solar
      3. no_usable_battery
      4. flat_electricity_prices
      5. high_evening_prices

    All scenario profiles are explicitly marked as SYNTHETIC.
    """
    base_battery = BatteryConfig(
        capacity_kwh=200.0,
        max_charge_kw=50.0,
        max_discharge_kw=50.0,
        charge_efficiency=0.95,
        discharge_efficiency=0.95,
        initial_soc_kwh=100.0,
        min_soc_kwh=20.0,
        max_soc_kwh=190.0,
        enforce_terminal_soc=True,
    )

    no_usable_battery = BatteryConfig(
        capacity_kwh=0.0,
        max_charge_kw=0.0,
        max_discharge_kw=0.0,
        charge_efficiency=0.95,
        discharge_efficiency=0.95,
        initial_soc_kwh=0.0,
        min_soc_kwh=0.0,
        max_soc_kwh=0.0,
        enforce_terminal_soc=True,
    )

    return {
        "normal_operation": {
            "name": "1. Normal Operation (Solar + ToU Tariff + 200 kWh Battery)",
            "description": (
                "SYNTHETIC commercial building with 120 kWp rooftop solar, Indian C&I Time-of-Use "
                "tariff (INR 5.5 off-peak, INR 6.5 solar hours, INR 12.5 evening peak), and a 200 kWh / 50 kW battery."
            ),
            "profile_df": generate_synthetic_commercial_profiles(solar_scale=1.0, price_regime="normal_tou"),
            "battery": base_battery,
        },
        "no_solar": {
            "name": "2. No Solar Generation (Grid ToU Arbitrage Only)",
            "description": (
                "SYNTHETIC commercial building with zero solar generation. The battery charges from "
                "low-cost off-peak/midday grid power and discharges during INR 12.5/kWh evening peak hours."
            ),
            "profile_df": generate_synthetic_commercial_profiles(solar_scale=0.0, price_regime="normal_tou"),
            "battery": base_battery,
        },
        "no_usable_battery": {
            "name": "3. No Usable Battery (Solar Self-Consumption Only)",
            "description": (
                "SYNTHETIC commercial building with rooftop solar and ToU prices, but zero usable battery "
                "capacity (0 kWh / 0 kW). Optimized dispatch must equal the baseline with INR 0 battery savings."
            ),
            "profile_df": generate_synthetic_commercial_profiles(solar_scale=1.0, price_regime="normal_tou"),
            "battery": no_usable_battery,
        },
        "flat_electricity_prices": {
            "name": "4. Flat Electricity Prices (INR 8.0/kWh All Day)",
            "description": (
                "SYNTHETIC commercial building with flat INR 8.0/kWh tariff. Price arbitrage across grid hours "
                "is unprofitable due to 90.25% round-trip efficiency losses; battery only stores otherwise-curtailed "
                "midday solar surplus to offset later demand."
            ),
            "profile_df": generate_synthetic_commercial_profiles(solar_scale=1.0, price_regime="flat"),
            "battery": base_battery,
        },
        "high_evening_prices": {
            "name": "5. High Evening Prices (INR 18.0/kWh Evening Peak Stress)",
            "description": (
                "SYNTHETIC commercial building facing severe evening peak tariff (INR 18.0/kWh from 6 PM to 10 PM). "
                "Demonstrates stronger economic savings and evening peak demand shaving."
            ),
            "profile_df": generate_synthetic_commercial_profiles(solar_scale=1.0, price_regime="high_evening"),
            "battery": base_battery,
        },
    }


def validate_physical_schedule(
    schedule_df: pd.DataFrame,
    battery: BatteryConfig,
    tol: float = 1e-5,
) -> Tuple[bool, List[str]]:
    """
    Audit an optimization schedule DataFrame against all physical, efficiency,
    and energy-accounting constraints:
      1. Non-negative grid imports, solar usage, charging, and discharging.
      2. Hourly charge/discharge within max power limits.
      3. No simultaneous charging and discharging in the same hour.
      4. Exact energy balance in every hour:
         grid_import + solar_to_load + battery_discharge == demand + battery_charge.
      5. Solar used by building/battery never exceeds available solar generation.
      6. Exact state-of-charge trajectory:
         soc_end[t] == soc_start[t] + eta_c * charge[t] - discharge[t] / eta_d.
      7. SOC remains within [min_soc_kwh, max_soc_kwh] at all hours.
      8. Terminal SOC >= initial SOC (no artificial savings from draining free initial energy).
    """
    violations: List[str] = []

    for col in ["grid_import_kwh", "battery_charge_kwh", "battery_discharge_kwh", "solar_to_load_kwh"]:
        if (schedule_df[col].to_numpy() < -tol).any():
            violations.append(f"Negative values detected in '{col}'.")

    if (schedule_df["battery_charge_kwh"].to_numpy() > battery.max_charge_kw + tol).any():
        violations.append("Hourly battery charging exceeds max_charge_kw.")

    if (schedule_df["battery_discharge_kwh"].to_numpy() > battery.max_discharge_kw + tol).any():
        violations.append("Hourly battery discharging exceeds max_discharge_kw.")

    simultaneous = np.minimum(
        schedule_df["battery_charge_kwh"].to_numpy(),
        schedule_df["battery_discharge_kwh"].to_numpy(),
    )
    if (simultaneous > 1e-4).any():
        max_sim = float(np.max(simultaneous))
        violations.append(f"Simultaneous charge and discharge detected (max overlap = {max_sim:.6f} kWh).")

    # Solar feasibility: solar_to_load + solar_curtailed == solar_kwh
    solar_balance_err = np.abs(
        (schedule_df["solar_to_load_kwh"] + schedule_df["solar_curtailed_kwh"]) - schedule_df["solar_kwh"]
    )
    if (solar_balance_err > tol).any():
        violations.append("Solar allocation (used + curtailed) does not match available solar_kwh.")

    # Building AC node energy balance:
    # grid_import + solar_to_load + battery_discharge == demand + battery_charge
    lhs = (
        schedule_df["grid_import_kwh"]
        + schedule_df["solar_to_load_kwh"]
        + schedule_df["battery_discharge_kwh"]
    )
    rhs = schedule_df["demand_kwh"] + schedule_df["battery_charge_kwh"]
    if (np.abs(lhs - rhs) > tol).any():
        max_err = float(np.max(np.abs(lhs - rhs)))
        violations.append(f"Hourly AC energy balance violated (max error = {max_err:.6e} kWh).")

    # SOC continuity and transition physics
    soc_start = schedule_df["soc_start_kwh"].to_numpy()
    soc_end = schedule_df["soc_end_kwh"].to_numpy()
    charge = schedule_df["battery_charge_kwh"].to_numpy()
    discharge = schedule_df["battery_discharge_kwh"].to_numpy()

    if abs(soc_start[0] - battery.initial_soc_kwh) > tol:
        violations.append(
            f"Initial SOC mismatch: schedule starts at {soc_start[0]:.4f} kWh vs config {battery.initial_soc_kwh:.4f} kWh."
        )

    expected_soc_end = soc_start + battery.charge_efficiency * charge - (discharge / battery.discharge_efficiency)
    if (np.abs(soc_end - expected_soc_end) > tol).any():
        max_soc_err = float(np.max(np.abs(soc_end - expected_soc_end)))
        violations.append(f"SOC dynamics equation violated (max error = {max_soc_err:.6e} kWh).")

    if len(soc_start) > 1 and (np.abs(soc_start[1:] - soc_end[:-1]) > tol).any():
        violations.append("SOC continuity between consecutive hours violated.")

    if (soc_end < battery.min_soc_kwh - tol).any() or (soc_end > battery.max_soc_kwh + tol).any():
        violations.append("Battery SOC trajectory violates [min_soc_kwh, max_soc_kwh] bounds.")

    if battery.enforce_terminal_soc and soc_end[-1] < battery.target_final_soc_kwh - tol:
        violations.append(
            f"Final SOC ({soc_end[-1]:.4f} kWh) is below target terminal SOC ({battery.target_final_soc_kwh:.4f} kWh). "
            "Cannot claim savings from depleting free initial stored energy."
        )

    return (len(violations) == 0), violations

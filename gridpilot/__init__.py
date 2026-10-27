"""GridPilot AI — IMPACT TANK '26 synthetic single-building dispatch optimizer.

This package contains the core engine only (no UI). See ``app.py`` at the
repository root for the Streamlit dashboard that visualizes these results.

All data generated here is SYNTHETIC and all optimization results are
SIMULATED; see README.md for details and limitations.
"""

from .energy_model import (
    BatteryConfig,
    compute_baseline_dispatch,
    generate_synthetic_commercial_profiles,
    get_scenario_definitions,
    validate_physical_schedule,
)
from .optimizer import (
    optimize_building_dispatch,
    run_all_scenarios,
    run_scenario,
)

__all__ = [
    "BatteryConfig",
    "compute_baseline_dispatch",
    "generate_synthetic_commercial_profiles",
    "get_scenario_definitions",
    "validate_physical_schedule",
    "optimize_building_dispatch",
    "run_all_scenarios",
    "run_scenario",
]

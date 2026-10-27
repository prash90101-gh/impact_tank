# GridPilot AI — IMPACT TANK '26 Prototype

A runnable Streamlit dashboard and transparent linear-programming (LP) optimizer
for single-building solar + battery dispatch, built around a **synthetic**
Indian commercial & industrial (C&I) Time-of-Use (ToU) tariff.

**Every number this project produces — demand, solar, tariff, and dispatch
schedule — is synthetic or simulated.** Nothing here is wired to live
telemetry, a real tariff, a real battery, or a real customer site. See
[MODEL_GAPS_AND_ASSUMPTIONS.md](MODEL_GAPS_AND_ASSUMPTIONS.md) before using
this beyond a demo.

## Project layout

```
impact_tank/
├── app.py                      # Streamlit dashboard (entry point)
├── gridpilot/                  # Core engine package (no UI)
│   ├── energy_model.py         # BatteryConfig, synthetic profiles, 5 test scenarios, physics validation
│   └── optimizer.py            # scipy.optimize.linprog (HiGHS) dispatch optimizer + KPI summary
├── tests/
│   └── test_scenarios.py       # Unit tests across all 5 standard scenarios
├── pyproject.toml              # Poetry project & dependency definitions
└── .streamlit/config.toml      # Dashboard theme
```

## Setup (Poetry)

This project uses [Poetry](https://python-poetry.org/) for dependency and
virtual-environment management.

```powershell
# Install Poetry once, if you don't already have it:
python -m pip install --user poetry

# From the impact_tank/ folder:
poetry install
```

Poetry creates an in-project `.venv/` (see `poetry.toml`) and installs exact,
locked versions from `poetry.lock`.

## Run the dashboard

```powershell
poetry run streamlit run app.py
```

Open the local address Streamlit prints (usually `http://localhost:8501`).
Use the sidebar to pick one of the five standard scenarios, or switch to
custom mode to adjust solar scale, tariff regime, and battery sizing directly.

## Run the tests

```powershell
poetry run pytest -q
```

The suite verifies, for all five standard scenarios:
- Physical/energy-balance validity (`validate_physical_schedule`)
- Correct zero-savings result when the battery has no usable capacity
- That flat-price arbitrage never happens — only curtailed solar gets stored
- That a high evening-peak tariff produces strictly larger savings than the
  normal tariff
- That the terminal state-of-charge constraint prevents the optimizer from
  claiming fake savings by draining free initial battery energy

## What the optimizer does

`gridpilot.optimizer.optimize_building_dispatch` solves an hourly LP
(`scipy.optimize.linprog`, HiGHS method) over a 24-hour horizon. For every
hour it chooses grid import, battery charge/discharge, and on-site solar
utilization to minimize:

`Σ (tariff[t] × grid_import[t]) + tiny battery-cycling cost + tiny peak-smoothing cost`

subject to:
- Hourly AC energy balance (grid + solar + discharge = demand + charge)
- Solar utilization bounded by available solar generation
- Battery charge/discharge power limits
- State-of-charge (SOC) trajectory and `[min_soc, max_soc]` bounds, including
  round-trip charge/discharge efficiency losses
- **Terminal SOC constraint**: the battery must end the day at or above where
  it started, so the model cannot report savings by just giving away free
  starting energy.

Every result is validated after the fact by
`gridpilot.energy_model.validate_physical_schedule`, which independently
re-checks energy balance, SOC dynamics, and the terminal SOC rule.

## The five standard scenarios

| Key | Scenario |
|---|---|
| `normal_operation` | Solar + Indian ToU tariff + 200 kWh / 50 kW battery |
| `no_solar` | Zero solar — battery does grid-only ToU arbitrage |
| `no_usable_battery` | Zero battery capacity — must equal baseline, ₹0 savings |
| `flat_electricity_prices` | Flat ₹8.0/kWh — arbitrage is unprofitable; battery only stores otherwise-curtailed solar |
| `high_evening_prices` | Severe ₹18.0/kWh evening peak — larger savings & peak shaving |

Run them directly from the command line for a quick text summary:

```powershell
poetry run python -m gridpilot.optimizer
```

## Important limitations

This is a prototype, not a production control system. It does **not** model
demand charges, battery degradation/temperature curves, inverter operating
modes, real interconnection/export limits, critical-load islands, outages,
forecast uncertainty, tariff settlement details, or regulatory requirements.
Review [MODEL_GAPS_AND_ASSUMPTIONS.md](MODEL_GAPS_AND_ASSUMPTIONS.md) for the
full list before extending this toward a real deployment.

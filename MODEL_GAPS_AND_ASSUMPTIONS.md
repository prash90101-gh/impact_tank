# Model gaps, assumptions, and correctness safeguards

## What this prototype is

A small, auditable reference LP optimizer plus a Streamlit dashboard for
**synthetic** single-building solar + battery dispatch, aligned loosely to
Indian commercial & industrial (C&I) Time-of-Use tariff structure. It exists
to demonstrate the optimization approach and UI, not to produce a real
customer-facing energy plan.

## Explicitly missing inputs for a real deployment

- Measured interval site load, PV production, weather forecasts, and
  data-quality handling.
- Actual tariff structures: demand charges, ratchets, seasonal variation,
  taxes, export/net-metering settlement, and any currency other than the
  illustrative INR figures used here.
- Battery/inverter nameplate data: capacity curve, usable SOC range, inverter
  topology, round-trip efficiency curve, availability, thermal derating,
  warranty/degradation model, and operational interlocks.
- Grid constraints: import/export limits, interconnection agreement,
  transformer limits, phase constraints, curtailment policy, and
  anti-islanding behavior.
- Operating objectives and safety controls: critical loads, outage reserve,
  backup/island mode, equipment alarms, user roles, audit logs, approvals,
  and manual override.
- Forecast uncertainty, stochastic/robust optimization, telemetry latency,
  dispatch execution and confirmation, and monitoring.
- Historical baselines and financial assumptions needed for ROI, payback, or
  customer guarantees.

## Correctness safeguards built into this prototype

1. **Energy balance** — every hour: `grid_import + solar_to_load +
   battery_discharge == demand + battery_charge`, enforced both as an LP
   equality constraint and re-verified independently by
   `validate_physical_schedule`.
2. **SOC bounds and dynamics** — `soc_end[t] = soc_start[t] + η_c·charge[t] −
   discharge[t]/η_d`, bounded to `[min_soc_kwh, max_soc_kwh]` at every hour.
3. **No simultaneous charge/discharge** — a tiny positive cycling cost
   (`eps_cycle`) in the objective discourages degenerate charge/discharge
   pairs; any floating-point solver noise is swept and re-balanced
   post-solve, and the validator checks for residual overlap.
4. **Terminal SOC neutrality** — the battery's end-of-day SOC must be at or
   above its starting SOC by default. This prevents a short planning horizon
   from reporting fake savings by draining free initial stored energy without
   replenishing it.
5. **Zero round-trip profit** — charge/discharge efficiencies (`0.95 × 0.95 =
   90.25%` round-trip by default) mean pure grid-to-grid arbitrage under flat
   prices always loses money; the flat-price scenario test
   (`test_flat_prices_only_uses_curtailed_solar`) explicitly checks the
   optimizer only stores otherwise-curtailed midday solar, never arbitrages.
6. **Peak-smoothing tie-breaker** — a tiny `eps_peak` term on a peak-import
   variable (`P_peak`) ensures that, among equally-priced hours, the battery
   charges in whichever hour avoids creating a new import spike and discharges
   to shave the highest peak-demand hours first.
7. **Automated test suite** (`tests/test_scenarios.py`) runs all 5 standard
   scenarios and checks: physical validity, zero-battery = zero savings,
   flat-price non-arbitrage, high-evening-peak scenario strictly beating
   normal-tariff savings, and the terminal SOC anti-cheating rule in isolation.

## Next integration sequence (before any production use)

1. Place the real model/optimizer code and a documented input schema beside
   this project; do not silently swap synthetic data for production data
   inside the same code paths without new review.
2. Write characterization tests from validated historical scenarios before
   changing optimizer internals.
3. Add an adapter that converts validated telemetry/forecast inputs into this
   optimizer's input schema (`demand_kwh`, `solar_kwh`, `price_inr_per_kwh`,
   `BatteryConfig`) — never substitute synthetic values for real ones.
4. Reconcile simulated bills against actual utility bills, and validate
   battery SOC trajectories against real inverter telemetry.
5. Keep the dashboard read-only until dispatch execution has safety review,
   role controls, confirmations, monitoring, and a tested rollback/manual
   override path.

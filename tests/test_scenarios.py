"""
test_scenarios.py
-----------------
Automated verification suite for the GridPilot AI single-building optimization
engine (energy_model.py and optimizer.py).
"""

import unittest
import numpy as np
from gridpilot.energy_model import (
    BatteryConfig,
    generate_synthetic_commercial_profiles,
    get_scenario_definitions,
    get_tariff_preset,
    validate_custom_tariff,
    validate_physical_schedule,
)
from gridpilot.optimizer import (
    optimize_building_dispatch,
    run_all_scenarios,
    run_scenario,
)


class TestGridPilotOptimization(unittest.TestCase):

    def test_all_five_scenarios_physically_valid_and_labeled(self):
        results, summary_df = run_all_scenarios()
        expected_keys = {
            "normal_operation",
            "no_solar",
            "no_usable_battery",
            "flat_electricity_prices",
            "high_evening_prices",
        }
        self.assertEqual(set(results.keys()), expected_keys)
        self.assertEqual(len(summary_df), 5)

        for key, res in results.items():
            with self.subTest(scenario=key):
                self.assertTrue(res["validation"]["is_physically_valid"], res["validation"]["violations"])
                self.assertEqual(len(res["validation"]["violations"]), 0)
                self.assertTrue(res["summary"]["is_synthetic_data"])
                self.assertTrue(res["summary"]["is_simulated_result"])
                self.assertIn("SYNTHETIC", res["metadata"]["data_source_label"])
                self.assertIn("SIMULATED", res["metadata"]["simulation_label"])
                # Terminal SOC must be >= initial SOC (no free energy depletion)
                self.assertGreaterEqual(
                    res["summary"]["final_soc_kwh"] + 1e-5,
                    res["summary"]["initial_soc_kwh"],
                )

    def test_no_usable_battery_zero_savings(self):
        res = run_scenario("no_usable_battery")
        s = res["summary"]
        self.assertAlmostEqual(s["estimated_savings_inr"], 0.0, places=4)
        self.assertAlmostEqual(s["total_battery_charge_kwh"], 0.0, places=4)
        self.assertAlmostEqual(s["total_battery_discharge_kwh"], 0.0, places=4)
        self.assertAlmostEqual(s["baseline_cost_inr"], s["optimized_cost_inr"], places=4)

    def test_flat_prices_only_uses_curtailed_solar(self):
        """
        Under flat electricity prices, grid-to-grid arbitrage loses 1 - eta_c*eta_d = 9.75%
        per cycle, so the optimizer should only charge from excess midday solar (hours 11-14)
        that would otherwise be curtailed, and never increase net grid energy vs baseline.
        """
        res = run_scenario("flat_electricity_prices")
        s = res["summary"]
        self.assertGreater(s["estimated_savings_inr"], 0.0)
        self.assertAlmostEqual(s["optimized_solar_curtailed_kwh"], 0.0, places=4)
        self.assertAlmostEqual(s["final_soc_kwh"], s["initial_soc_kwh"], places=4)

    def test_high_evening_prices_increase_savings_over_normal(self):
        norm = run_scenario("normal_operation")["summary"]
        high = run_scenario("high_evening_prices")["summary"]
        self.assertGreater(high["estimated_savings_inr"], norm["estimated_savings_inr"])
        self.assertGreater(high["evening_peak_reduction_kw"], 0.0)

    def test_terminal_soc_prevents_free_initial_energy_cheating(self):
        """
        Under flat prices with ZERO solar, any battery cycling loses energy due to <100%
        efficiency. Because final_soc >= initial_soc is enforced, the optimizer must not
        discharge the initial 100 kWh to claim fake savings; savings must be exactly 0.
        """
        df = generate_synthetic_commercial_profiles(solar_scale=0.0, price_regime="flat")
        res = optimize_building_dispatch(
            demand_kwh=df["demand_kwh"],
            solar_kwh=df["solar_kwh"],
            price_inr_per_kwh=df["price_inr_per_kwh"],
            battery_capacity_kwh=200.0,
            max_charge_kw=50.0,
            max_discharge_kw=50.0,
            charge_efficiency=0.95,
            discharge_efficiency=0.95,
            initial_soc_kwh=100.0,
            min_soc_kwh=20.0,
            max_soc_kwh=190.0,
            enforce_terminal_soc=True,
        )
        self.assertAlmostEqual(res["summary"]["estimated_savings_inr"], 0.0, places=4)
        self.assertAlmostEqual(res["summary"]["final_soc_kwh"], 100.0, places=4)

    def test_custom_tariff_is_not_limited_to_the_three_built_in_presets(self):
        """
        Reviewer-requested fix: a user must be able to enter their own hourly tariff
        (e.g. a real DISCOM bill, or any other state's rate schedule) instead of being
        limited to the 'normal_tou' / 'flat' / 'high_evening' presets. This custom
        tariff must flow through the full pipeline unmodified and drive different
        dispatch behaviour than any of the three built-in presets.
        """
        # An arbitrary, strictly increasing hourly tariff that matches none of the
        # three built-in presets.
        custom_tariff = [round(2.0 + 0.5 * h, 2) for h in range(24)]

        df = generate_synthetic_commercial_profiles(custom_price_inr_per_kwh=custom_tariff)
        self.assertEqual(list(df["price_inr_per_kwh"].round(2)), custom_tariff)

        # It must differ from all three presets (sanity check it's really custom).
        for preset in ("normal_tou", "flat", "high_evening"):
            self.assertFalse(
                np.allclose(df["price_inr_per_kwh"].to_numpy(), get_tariff_preset(preset))
            )

        res = optimize_building_dispatch(
            demand_kwh=df["demand_kwh"],
            solar_kwh=df["solar_kwh"],
            price_inr_per_kwh=df["price_inr_per_kwh"],
            battery_capacity_kwh=200.0,
            max_charge_kw=50.0,
            max_discharge_kw=50.0,
            initial_soc_kwh=100.0,
            min_soc_kwh=20.0,
            max_soc_kwh=190.0,
        )
        self.assertTrue(res["validation"]["is_physically_valid"], res["validation"]["violations"])
        self.assertEqual(list(res["schedule_df"]["price_inr_per_kwh"].round(2)), custom_tariff)
        # Since this custom tariff rises monotonically to a late-evening peak (23:00 = ₹13.5),
        # the battery should discharge to shave that late peak rather than the 18:00-21:00
        # window used by the built-in presets.
        self.assertGreater(res["summary"]["estimated_savings_inr"], 0.0)

    def test_custom_tariff_validation_rejects_wrong_length_and_negative_values(self):
        with self.assertRaises(ValueError):
            validate_custom_tariff([5.0] * 23)  # wrong length
        with self.assertRaises(ValueError):
            validate_custom_tariff([5.0] * 23 + [-1.0])  # negative value
        with self.assertRaises(ValueError):
            validate_custom_tariff([5.0] * 23 + [float("nan")])  # non-finite value

        # A valid, fully custom 24-hour tariff must pass through unchanged.
        valid = [round(1.0 + 0.1 * h, 2) for h in range(24)]
        result = validate_custom_tariff(valid)
        self.assertEqual(list(result.round(2)), valid)


if __name__ == "__main__":
    unittest.main(verbosity=2)

import jax
import jax.numpy as jnp
import pytest

from chargax import EVSE, Chargax, ChargingStation


def _test_env() -> Chargax:
    station = ChargingStation(
        max_kw_throughput=100.0,
        efficiency=1.0,
        connections=[EVSE(num_chargers=4, voltage=400.0, max_current=32.0)],
    )
    return Chargax(
        station=station,
        get_num_cars_arriving=lambda k, s: jnp.int32(0),
        get_new_cars_arriving=lambda k, s: station.evses_flat,
        get_grid_buy_price=lambda s: 0.1,
        get_grid_sell_price=lambda s: 0.09,
    )


def _fresh_state(env: Chargax):
    return env.reset(jax.random.PRNGKey(0))[1]


def _leaving_ports(env: Chargax) -> EVSE:
    """Four cars wanting 50 kWh more (0 -> 50 of 100 kWh), planned stay 100 minutes:
    0: time-sensitive, 35 kWh delivered (30% missing)
    1: time-sensitive, 49 kWh delivered (2% missing)
    2: charge-sensitive, 130 minutes waited (30% overtime)
    3: charge-sensitive, 100 minutes waited (no overtime)"""
    return env.station.evses_flat.replace(
        car_battery_capacity_kw=jnp.full(4, 100.0),
        car_desired_battery_percentage=jnp.full(4, 0.5),
        car_arrival_battery_kw=jnp.zeros(4),
        car_battery_now_kw=jnp.array([35.0, 49.0, 50.0, 50.0]),
        car_time_waited=jnp.array([100.0, 100.0, 130.0, 100.0]),
        car_time_till_leave=jnp.array([0, 0, -30, 0]),
        charge_sensitive=jnp.array([False, False, True, True]),
        charger_is_car_connected=jnp.ones(4, bool),
    )


def test_bad_service_rate_and_worst_case_units():
    env = _test_env()
    ports = _leaving_ports(env)
    state = env.update_fairness_values(_fresh_state(env), ports, jnp.ones(4, bool))
    metrics = env.get_fairness_metrics(state)

    assert metrics["max_shortfall_kw"] == pytest.approx(15.0)
    assert metrics["max_overtime_min"] == pytest.approx(30.0)
    # One of two customers per group is above 10% and 20%, nobody above 50%
    for pct in (10, 20):
        assert metrics[f"bsr_time_{pct}"] == pytest.approx(0.5)
        assert metrics[f"bsr_charge_{pct}"] == pytest.approx(0.5)
        assert metrics[f"bsr_overall_{pct}"] == pytest.approx(0.5)
    for key in ("bsr_time_50", "bsr_charge_50", "bsr_overall_50"):
        assert metrics[key] == pytest.approx(0.0)


def test_unscored_customers_do_not_count():
    env = _test_env()
    ports = _leaving_ports(env)
    scored = jnp.array([False, True, False, True])  # only the well-served cars leave
    state = env.update_fairness_values(_fresh_state(env), ports, scored)
    metrics = env.get_fairness_metrics(state)

    assert metrics["max_shortfall_kw"] == pytest.approx(1.0)
    assert metrics["max_overtime_min"] == pytest.approx(0.0)
    assert metrics["bsr_overall_10"] == pytest.approx(0.0)

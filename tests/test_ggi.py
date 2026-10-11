"""Tests for the Generalized Gini Index (GGI) fairness penalty.

GGI = sum_k w_k * d_(k): the dissatisfaction d = 1 - s of the day's customers, sorted
worst first, times decreasing weights w = 1, decay, decay^2, ...
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from chargax import EVSE, Chargax, ChargingStation
from chargax.chargax import GGI_BUFFER_SIZE


def _quiet_env(**kwargs) -> Chargax:
    """One charger, no cars, fixed prices: for testing the GGI bookkeeping directly."""
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
        **kwargs,
    )


def _empty_state(env: Chargax):
    _, state = env.reset_env(jax.random.PRNGKey(0))
    return state


# s = satisfaction of 4 customers -> dissatisfaction 0.2, 0.6, 0.0, 0.4
S = jnp.array([0.8, 0.4, 1.0, 0.6])
ALL = jnp.array([True, True, True, True])


# ---- Weights -------------------------------------------------------------------
def test_default_weights_halve():
    w = _quiet_env().ggi_weights()
    assert w.shape == (GGI_BUFFER_SIZE,)
    assert jnp.allclose(w[:4], jnp.array([1.0, 0.5, 0.25, 0.125]))


def test_decay_zero_only_counts_the_worst_customer():
    w = _quiet_env(ggi_weight_decay=0.0).ggi_weights()
    assert float(w[0]) == 1.0
    assert float(w[1:].sum()) == 0.0


def test_decay_one_counts_everyone_equally():
    w = _quiet_env(ggi_weight_decay=1.0).ggi_weights()
    assert jnp.all(w == 1.0)


# ---- Bookkeeping ---------------------------------------------------------------
def test_ggi_hand_example():
    env = _quiet_env()
    state = env.update_ggi_values(_empty_state(env), S, ALL)
    # sorted worst first: 0.6, 0.4, 0.2 (and 0.0) -> 1*0.6 + 0.5*0.4 + 0.25*0.2
    assert float(state.ggi_cost) == pytest.approx(0.85)
    assert jnp.allclose(state.ggi_dissatisfaction[:4], jnp.array([0.6, 0.4, 0.2, 0.0]))


def test_unscored_customers_are_ignored():
    env = _quiet_env()
    scored = jnp.array([True, False, True, False])  # only d = 0.2 and 0.0 count
    state = env.update_ggi_values(_empty_state(env), S, scored)
    assert float(state.ggi_cost) == pytest.approx(0.2)


def test_order_customers_leave_in_does_not_matter():
    env = _quiet_env()
    first = jnp.array([True, True, False, False])
    together = env.update_ggi_values(_empty_state(env), S, ALL)
    one_by_one = env.update_ggi_values(
        env.update_ggi_values(_empty_state(env), S, first), S, ~first
    )
    reversed_order = env.update_ggi_values(
        env.update_ggi_values(_empty_state(env), S, ~first), S, first
    )
    for state in (one_by_one, reversed_order):
        assert float(state.ggi_cost) == pytest.approx(float(together.ggi_cost))


def test_full_buffer_keeps_the_most_dissatisfied():
    env = _quiet_env(ggi_weight_decay=1.0)
    full = _empty_state(env)._replace(
        ggi_dissatisfaction=jnp.full(GGI_BUFFER_SIZE, 0.1)
    )
    one_unhappy = jnp.array([True, False, False, False])
    state = env.update_ggi_values(full, jnp.array([0.1, 1.0, 1.0, 1.0]), one_unhappy)
    assert state.ggi_dissatisfaction.shape == (GGI_BUFFER_SIZE,)
    assert float(state.ggi_dissatisfaction[0]) == pytest.approx(0.9)
    # one 0.1 was dropped to make room for the 0.9
    assert float(state.ggi_cost) == pytest.approx(0.9 + (GGI_BUFFER_SIZE - 1) * 0.1)


def test_reward_is_minus_alpha_times_ggi_change():
    env = _quiet_env(ggi_alpha=2.0)
    old = _empty_state(env)
    new = env.update_ggi_values(old, S, ALL)
    assert float(env.get_reward(old, new)) == pytest.approx(-2.0 * 0.85)
    # default ggi_alpha = 0: reward is unchanged
    assert float(_quiet_env().get_reward(old, new)) == pytest.approx(0.0)


# ---- A full simulated day --------------------------------------------------------
SLOW_CHARGE = 11  # action index 11 = charge at 10% of the charger's max current


@pytest.fixture(scope="module")
def busy_env() -> Chargax:
    """4 small AC chargers with ~100 shopping customers a day: many leave unhappy."""
    station = ChargingStation(
        max_kw_throughput=50.0,
        efficiency=1.0,
        connections=[EVSE(num_chargers=2, voltage=230.0, max_current=16.0) for _ in range(2)],
    )
    return Chargax(
        station=station,
        default_data_kwargs={"user_profile": "shopping", "average_cars_per_day": "medium"},
    )


def _with(env: Chargax, ggi_alpha: float, ggi_weight_decay: float) -> Chargax:
    # As JAX arrays, like the Lagrangian runs do, so one compiled day serves all settings
    return eqx.tree_at(
        lambda e: (e.ggi_alpha, e.ggi_weight_decay),
        env,
        (jnp.float32(ggi_alpha), jnp.float32(ggi_weight_decay)),
    )


@eqx.filter_jit
def _run_day(env: Chargax, key):
    _, state = env.reset_env(key)
    action = jax.tree.map(lambda a: jnp.full_like(a, SLOW_CHARGE), env.sample_action(key))

    def step(state, k):
        timestep, new_state = env.step_env(k, state, action)
        return new_state, (timestep.reward, timestep.info["ggi_cost"])

    keys = jax.random.split(key, env.max_episode_steps)
    final_state, (rewards, ggi_per_step) = jax.lax.scan(step, state, keys)
    return final_state, rewards, ggi_per_step


KEY = jax.random.PRNGKey(7)


def test_day_rewards_add_up_to_end_of_day_ggi(busy_env):
    alpha = 3.0
    state0, rewards0, _ = _run_day(_with(busy_env, 0.0, 0.5), KEY)
    state, rewards, _ = _run_day(_with(busy_env, alpha, 0.5), KEY)
    assert float(state.ggi_cost) > 0.1, "test day should have unhappy customers"
    # The penalty changes the reward only, not what happens in the simulation
    assert float(state.profit) == pytest.approx(float(state0.profit))
    penalty = float(rewards0.sum() - rewards.sum())
    assert penalty == pytest.approx(alpha * float(state.ggi_cost), rel=1e-4, abs=1e-4)


def test_ggi_never_goes_down_during_the_day(busy_env):
    _, _, ggi_per_step = _run_day(_with(busy_env, 1.0, 0.5), KEY)
    assert float(jnp.diff(ggi_per_step).min()) >= -1e-6


def test_decay_one_equals_sum_of_dissatisfaction(busy_env):
    """Cross-check with the existing fairness bookkeeping: same customers scored."""
    state, _, _ = _run_day(_with(busy_env, 1.0, 1.0), KEY)
    sum_dissatisfaction = state.group_served.sum() - state.group_s_sum.sum()
    assert float(state.group_served.sum()) < GGI_BUFFER_SIZE
    assert float(state.ggi_cost) == pytest.approx(float(sum_dissatisfaction), rel=1e-4)


def test_decay_zero_equals_worst_customer(busy_env):
    state, _, _ = _run_day(_with(busy_env, 1.0, 0.0), KEY)
    worst = 1.0 - state.group_min_s.min()
    assert float(state.ggi_cost) == pytest.approx(float(worst), rel=1e-5)

"""Shared experiment setup, matching the team's Chargax replication
(submission branch, shopping scenario, Table 1 of the replication report).

Keep every experiment on this config so results from different team members
can be put in the same table.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxnasium.algorithms import PPO  # jaxnasium 0.0.29

from chargax import EVSE, Chargax, ChargingStation
from chargax._station_layout import StationBattery

# ---- Environment ---------------------------------------------------------
DATA_KWARGS = {
    "user_profile": "shopping",
    "average_cars_per_day": "medium",  # 100 cars per day
    "car_profile": "eu",
    "grid_price_dataset": "2023_NL",
    "grid_sell_margin": -0.02,  # selling back pays buy price - 0.02 EUR/kWh
}


def make_station() -> ChargingStation:
    """16 chargers in pairs, as in the paper's code (submission branch):
    5 DC pairs (500 V x 300 A = 150 kW per pair) and 3 AC pairs (230 V x 50 A = 11.5 kW per pair).
    Grid connection = sum of all pairs; large station battery that starts empty."""
    dc = [EVSE(num_chargers=2, voltage=500.0, max_current=300.0) for _ in range(5)]
    ac = [EVSE(num_chargers=2, voltage=230.0, max_current=50.0) for _ in range(3)]
    battery = StationBattery(capacity_kw=100_000.0, max_kw_throughput=1_000.0, efficiency=1.0)
    battery = battery.replace(battery_now=0.0)  # main branch would start it at 25%
    return ChargingStation(
        max_kw_throughput=5 * 150.0 + 3 * 11.5,
        efficiency=0.995,
        connections=dc + ac + [battery],
    )


def make_env(**env_kwargs) -> Chargax:
    """env_kwargs: reward weights, e.g. fairness_alpha=10 or charged_satisfaction_alpha=1."""
    return Chargax(station=make_station(), default_data_kwargs=DATA_KWARGS, **env_kwargs)


# ---- PPO (paper settings) ----------------------------------------------------
PPO_SETTINGS = dict(
    num_envs=12,
    num_steps=300,
    num_minibatches=4,
    num_epochs=4,
    learning_rate=2.5e-4,
    anneal_learning_rate=True,  # linear decay to 0 over the whole run
    gamma=0.99,
    gae_lambda=0.95,
    clip_coef=0.2,
    clip_coef_vf=10.0,
    ent_coef=0.01,
    vf_coef=0.25,
    max_grad_norm=100.0,
    normalize_observations=True,
    normalize_rewards=False,
    actor_kwargs={"hidden_sizes": (256, 256)},
    critic_kwargs={"hidden_sizes": (256, 256)},
    log_function=None,
)


class FixedSchedulePPO(PPO):
    """jaxnasium 0.0.29 builds the learning-rate schedule for num_iterations * num_epochs
    steps, but the optimizer steps once per minibatch (num_iterations * num_epochs *
    num_minibatches times). With annealing, the learning rate therefore reaches 0 after
    1/num_minibatches (here 25%) of training and the agent stops learning. This counts
    the optimizer steps correctly, as jaxnasium 0.1.0 does."""

    @property
    def num_training_updates(self):
        return self.num_iterations * self.num_epochs * self.num_minibatches


def make_ppo(total_timesteps: int, **overrides) -> PPO:
    return FixedSchedulePPO(total_timesteps=total_timesteps, **{**PPO_SETTINGS, **overrides})


# ---- Evaluation --------------------------------------------------------------
METRICS = {
    "Chargax": ["profit", "uncharged_kw", "charged_overtime", "rejected_customers"],
    "1) Within group": [
        "jain_time", "jain_charge",
        "min_s_time", "min_s_charge",
        "unfair_time", "unfair_charge",
    ],
    "2) Between groups": ["mean_s_time", "mean_s_charge", "group_gap"],
    "3) Summary": ["worst_group_jain", "worst_group_min_s", "fairness_cost", "worst_case_cost"],
}
METRIC_KEYS = [k for keys in METRICS.values() for k in keys]


class PPOPolicy(eqx.Module):
    """Trained PPO agent, acting deterministically at test time."""

    agent: eqx.Module

    def __call__(self, key, obs, state):
        return self.agent.get_action(key, self.agent.state, obs, deterministic=True)


class BaselinePolicy(eqx.Module):
    """Rule-based Chargax baseline (MaxCharge with battery_schedule='none', or Random)."""

    baseline: eqx.Module

    def __call__(self, key, obs, state):
        action = self.baseline.get_action(key, env_state=state, observation=obs)
        return action[0] if isinstance(action, tuple) else action


def make_evaluator(env, num_days: int):
    """Returns evaluate(policy, key) -> {metric: array of shape (num_days,)}.
    All days run in parallel; values are taken at the end of each day."""

    @eqx.filter_jit
    def evaluate(policy, key):
        def one_day(k):
            obs, state = env.reset(k)

            def step(carry, _):
                seed, state, obs = carry
                k1, k2 = jax.random.split(seed)
                action = policy(k1, obs, state)
                ts, new_state = env.step(k1, state, action)
                return (k2, new_state, ts.observation), ts.info

            _, infos = jax.lax.scan(step, (k, state, obs), None, length=env.max_episode_steps)
            info = jax.tree.map(lambda x: x[-1], infos)  # final step, before auto-reset
            return {m: jnp.asarray(info[m], jnp.float32) for m in METRIC_KEYS}

        return jax.vmap(one_day)(jax.random.split(key, num_days))

    return evaluate

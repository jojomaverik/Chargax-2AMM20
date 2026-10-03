# ruff: noqa: F401
"""
Replication setup matching the Chargax Paper

To Replicate the results in the Chargax paper, run the following commands:
    python main.py --traffic low                          # Fig. 4a
    python main.py --traffic medium --charged_alpha 10    # Fig. 4b
    python main.py --traffic medium --time_alpha 20       # Fig. 4c
"""

import argparse

import equinox as eqx
import jax
import jax.numpy as jnp
import jaxnasium as jym
import numpy as np
from jaxnasium.algorithms import DQN, PPO, SAC
from jaxtyping import Array, PRNGKeyArray

from chargax import EVSE, Chargax, ChargingStation, StationBattery, StationSplitter
from chargax.baselines import MaxCharge, Random


def build_submission_station() -> ChargingStation:
    """Station from the submission branch: 16 chargers in groups of 2.
    5 DC groups (500 V x 300 A = 150 kW per pair) and 3 AC groups (230 V x 50 A = 11.5 kW per pair).
    The charger grid connection equals the combined EVSE capacity (no binding grid constraint),
    and every splitter / EVSE has an efficiency of 0.995, as in the original code.
    """
    dc_evses = [
        EVSE(voltage=500.0, max_current=300.0, num_chargers=2, efficiency=0.995)
        for _ in range(5)
    ]
    ac_evses = [
        EVSE(voltage=230.0, max_current=50.0, num_chargers=2, efficiency=0.995)
        for _ in range(3)
    ]
    chargers_capacity_kw = 5 * 150.0 + 3 * 11.5  # 784.5 kW

    # Original battery: 100 MWh, 1000 kW max rate, no efficiency losses, outside the
    # charger capacity limit, and it starts empty.
    battery_capacity_kw, battery_max_rate_kw = 100_000.0, 1000.0
    station = ChargingStation(
        max_kw_throughput=chargers_capacity_kw + battery_max_rate_kw,
        efficiency=1.0,
        connections=[
            StationSplitter(
                max_kw_throughput=chargers_capacity_kw,
                efficiency=0.995,
                connections=dc_evses + ac_evses,
            ),
            StationBattery(
                capacity_kw=battery_capacity_kw,
                max_kw_throughput=battery_max_rate_kw,
                efficiency=1.0,
            ),
        ],
    )
    # StationBattery starts at 25% by default; the submission branch starts at 0.
    return eqx.tree_at(lambda s: s.connections[1].battery_now, station, 0.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--traffic", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--charged_alpha", type=float, default=0.0) 
    parser.add_argument("--time_alpha", type=float, default=0.0)  
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total_timesteps", type=int, default=10_000_000)
    args = parser.parse_args()

    rng = jax.random.PRNGKey(args.seed)

    env = Chargax(
        station=build_submission_station(),
        elec_customer_sell_price=0.75,
        minutes_per_timestep=5,  
        num_discretization_levels=10,
        allow_discharging=True,
        charged_satisfaction_alpha=args.charged_alpha,
        time_satisfaction_alpha=args.time_alpha,
        capacity_exceeded_alpha=0.0,
        rejected_customers_alpha=0.0,
        battery_degradation_alpha=0.0,
        beta=0.0,
        default_data_kwargs={
            "car_profile": "eu",
            "user_profile": "shopping",
            "average_cars_per_day": args.traffic,  # low=50, medium=100, high=250
            "grid_price_dataset": "2023_NL",
            "grid_sell_margin": -0.02,
        },
    )
    env = jym.LogWrapper(env)

    # PPO settings from the submission branch (PPOConfig + build_ppo_trainer)
    algo = PPO(
        num_envs=12,
        num_steps=300,
        num_minibatches=4,
        num_epochs=4,
        total_timesteps=args.total_timesteps,
        learning_rate=2.5e-4,
        anneal_learning_rate=True,  # linear decay to 0
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        clip_coef_vf=10.0,
        ent_coef=0.01,
        vf_coef=0.25,
        max_grad_norm=100.0,
        normalize_rewards=False,
        normalize_observations=True,
        actor_kwargs={"hidden_sizes": (256, 256)},
        critic_kwargs={"hidden_sizes": (256, 256)},
    )

    print(
        f"Training PPO: traffic={args.traffic}, charged_alpha={args.charged_alpha}, "
        f"time_alpha={args.time_alpha}, seed={args.seed}, timesteps={args.total_timesteps:,}"
    )
    algo = algo.train(rng, env)

    results = algo.evaluate(rng, env, num_eval_episodes=25)
    print(f"PPO - Average reward over 25 evaluation episodes: {np.mean(results)}")

    # Compare against baselines:
    print("Evaluating baselines...")
    rewards, profits = MaxCharge(env).evaluate(rng, num_eval_episodes=10)
    print(
        f"MaxCharge - Average cumulative reward: {np.sum(rewards, axis=1).mean():.2f}"
    )
    rewards, profits = Random(env).evaluate(rng, num_eval_episodes=10)
    print(f"Random - Average cumulative reward: {np.sum(rewards, axis=1).mean():.2f}")

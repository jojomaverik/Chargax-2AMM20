# ruff: noqa: F401
"""
Replication setup matching the Chargax Paper

To Replicate the results in the Chargax paper, run the following commands:
    python main.py --traffic low                          # Fig. 4a
    python main.py --traffic medium --charged_alpha 10    # Fig. 4b
    python main.py --traffic medium --time_alpha 20       # Fig. 4c

Normalized user-satisfaction reward (each leaving car contributes a value in [0, 1] per term):
    python main.py --traffic medium --norm_alpha 10
    python main.py --traffic medium --norm_alpha 10 --w_overtime 0 --w_undertime 0

Customer fairness (Jain's index etc. are reported for every run; this also penalises unfairness):
    python main.py --traffic medium --fairness_alpha 10

Every run appends one row to --results_csv so the experiments can be compared side by side.
"""

import argparse
import csv
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import jaxnasium as jym
import numpy as np
from jaxnasium.algorithms import DQN, PPO, SAC
from jaxtyping import Array, PRNGKeyArray

from chargax import EVSE, Chargax, ChargingStation, StationBattery, StationSplitter
from chargax.baselines import MaxCharge, Random
from experiment_config import FixedSchedulePPO


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


SATISFACTION_KEYS = (
    "profit",
    "uncharged_kw",
    "charged_overtime",
    "charged_undertime",
    "served_customers",
    "rejected_customers",
    "sat_uncharged_norm",
    "sat_overtime_norm",
    "sat_undertime_norm",
)

# Customer-fairness metrics (computed in chargax.py, see get_fairness_metrics).
# _time = time-sensitive customers, _charge = charge-sensitive customers.
FAIRNESS_KEYS = (
    "fairness_cost",  # sum over customers of (1 - s_i)^2, s_i = satisfaction in [0, 1]
    "jain_time",  # Jain's index of s_i within each group (1 = everyone served equally)
    "jain_charge",
    "min_s_time",  # worst-served customer of the day
    "min_s_charge",
    "unfair_time",  # customers per day with s_i < 0.8
    "unfair_charge",
    "mean_s_time",
    "mean_s_charge",
    "group_gap",  # |mean_s_time - mean_s_charge|
    "worst_group_jain",
    "worst_group_min_s",
    "jain_overall",  # both groups pooled
    "worst_case_cost",  # (1 - min_s_time) + (1 - min_s_charge)
    "max_shortfall_kw",  # largest missing energy of a time-sensitive car
    "max_overtime_min",  # largest overtime of a charge-sensitive car
    "bsr_time_10",  # share of customers with > 10/20/50% energy missing or overtime
    "bsr_charge_10",
    "bsr_overall_10",
    "bsr_time_20",
    "bsr_charge_20",
    "bsr_overall_20",
    "bsr_time_50",
    "bsr_charge_50",
    "bsr_overall_50",
)


def evaluate_user_satisfaction(algo, env, key, num_eval_episodes, weights):
    """Roll out the trained policy and return end-of-day values averaged over episodes,
    plus per-customer normalized satisfaction terms (0 = fully satisfied, 1 = worst)."""
    w_uncharged, w_overtime, w_undertime = weights

    def run_episode(key):
        def step(carry, _):
            rng, obs, state, episode_reward = carry
            rng, action_key, step_key = jax.random.split(rng, 3)
            action = algo.get_action(action_key, algo.state, obs, deterministic=True)
            (obs, reward, _, _, info), state = env.step(step_key, state, action)
            info = {k: info[k] for k in SATISFACTION_KEYS + FAIRNESS_KEYS}
            return (rng, obs, state, episode_reward + reward), info

        key, reset_key = jax.random.split(key)
        obs, state = env.reset(reset_key)
        (_, _, _, episode_reward), infos = jax.lax.scan(
            step, (key, obs, state, 0.0), None, length=env.max_episode_steps
        )
        # The env auto-resets on the final step, so read the end-of-day totals from
        # that step's info (computed before the reset) rather than from the state.
        out = {k: v[-1] for k, v in infos.items()}
        out["reward"] = episode_reward
        return out

    episodes = jax.jit(jax.vmap(run_episode))(jax.random.split(key, num_eval_episodes))
    metrics = {k: float(np.mean(v)) for k, v in episodes.items()}

    served = np.maximum(np.asarray(episodes["served_customers"]), 1)
    per_user = {
        k: np.asarray(episodes[k]) / served
        for k in ("sat_uncharged_norm", "sat_overtime_norm", "sat_undertime_norm")
    }
    metrics.update({f"{k}_per_user": float(np.mean(v)) for k, v in per_user.items()})
    metrics["norm_dissatisfaction_per_user"] = float(
        np.mean(
            w_uncharged * per_user["sat_uncharged_norm"]
            + w_overtime * per_user["sat_overtime_norm"]
            + w_undertime * per_user["sat_undertime_norm"]
        )
    )
    return metrics


def append_results_row(path, row):
    """Append one row to the results CSV.
    Normally this just appends a line. Only when the row has columns the file does not
    have yet (e.g. the new fairness metrics) is the file rewritten with the extra columns,
    so older rows stay aligned (their new columns are left empty). The rewrite goes to a
    temporary file first, so a crash can never leave a half-written results file."""
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        return

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        new_fields = [k for k in row if k not in fields]
        rows = list(reader) if new_fields else None

    if not new_fields:  # same columns as before: plain append, like the original version
        with open(path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)
        return

    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields + new_fields)
        writer.writeheader()
        writer.writerows(rows + [row])
    os.replace(tmp, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--traffic", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--charged_alpha", type=float, default=0.0) 
    parser.add_argument("--time_alpha", type=float, default=0.0)  
    parser.add_argument("--norm_alpha", type=float, default=0.0)
    parser.add_argument("--fairness_alpha", type=float, default=0.0)  # weight on fairness_cost
    parser.add_argument("--w_uncharged", type=float, default=1.0)
    parser.add_argument("--w_overtime", type=float, default=1.0)
    parser.add_argument("--w_undertime", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total_timesteps", type=int, default=10_000_000)
    parser.add_argument("--num_eval_episodes", type=int, default=125)
    parser.add_argument("--results_csv", default="results.csv")
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
        norm_satisfaction_alpha=args.norm_alpha,
        norm_w_uncharged=args.w_uncharged,
        norm_w_overtime=args.w_overtime,
        norm_w_undertime=args.w_undertime,
        fairness_alpha=args.fairness_alpha,
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
    algo = FixedSchedulePPO(
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
        f"time_alpha={args.time_alpha}, norm_alpha={args.norm_alpha}, "
        f"fairness_alpha={args.fairness_alpha}, seed={args.seed}, timesteps={args.total_timesteps:,}"
    )
    algo = algo.train(rng, env)

    metrics = evaluate_user_satisfaction(
        algo,
        env,
        rng,
        args.num_eval_episodes,
        weights=(args.w_uncharged, args.w_overtime, args.w_undertime),
    )
    print(f"PPO - averages over {args.num_eval_episodes} evaluation episodes:")
    for k, v in metrics.items():
        print(f"  {k:32s} {v:10.3f}")
    append_results_row(args.results_csv, {**vars(args), **metrics})
    print(f"Appended results to {args.results_csv}")

    # Compare against baselines:
    print("Evaluating baselines...")
    rewards, profits = MaxCharge(env).evaluate(rng, num_eval_episodes=10)
    print(
        f"MaxCharge - Average cumulative reward: {np.sum(rewards, axis=1).mean():.2f}"
    )
    rewards, profits = Random(env).evaluate(rng, num_eval_episodes=10)
    print(f"Random - Average cumulative reward: {np.sum(rewards, axis=1).mean():.2f}")
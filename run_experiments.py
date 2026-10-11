"""Customer-fairness experiments on the shared setup (experiment_config.py).

Every method is trained with several seeds and then tested on the same fixed set of
days. Results are appended to results_<steps>.csv after every run, so the script can
be stopped and restarted: finished (method, seed) pairs are skipped.

    uv run run_experiments.py                                   # everything below
    uv run run_experiments.py --methods PPO "Fair a=10"         # only some methods
    uv run run_experiments.py --steps 1000000 --seeds 0         # quick try
    uv run run_experiments.py --table                           # only print the table
    uv run run_experiments.py --table --steps 1000000           # table of the quick try
"""

import argparse
import csv
import os
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jaxnasium as jym
import numpy as np

from chargax.baselines import MaxCharge
from experiment_config import (
    METRIC_KEYS,
    METRICS,
    BaselinePolicy,
    PPOPolicy,
    make_env,
    make_evaluator,
    make_ppo,
)

# Each method is either a fixed set of reward weights ("env") or a Lagrangian run
# with a budget on the average daily fairness_cost.
METHODS = {
    "PPO": {"env": {}},
    "Fair a=10": {"env": {"fairness_alpha": 10.0}},
    "Fair a=50": {"env": {"fairness_alpha": 50.0}},
    # Worst-case (Rawlsian) penalty: per group, 1 - lowest satisfaction of the day (0 to 2)
    "Worst-case a=10": {"env": {"worst_case_alpha": 10.0}},
    "Worst-case a=50": {"env": {"worst_case_alpha": 50.0}},
    # Budgets on the average daily fairness_cost. For reference (2M-step test run):
    # MaxCharge ~1.7, profit-only PPO ~8. Budgets below ~1.7 may not be reachable.
    "Lagrangian d=4.0": {"budget": 4.0},
    "Lagrangian d=2.0": {"budget": 2.0},
    # "Lagrangian d=1.0": {"budget": 1.0},
    # Chargax's own satisfaction penalties (teammates' part) can run through the same
    # pipeline, so all rows of the final table are measured the same way:
    # "kWh penalty a=1": {"env": {"charged_satisfaction_alpha": 1.0}},
    # "Overtime penalty a=1": {"env": {"time_satisfaction_alpha": 1.0}},
}

TOTAL_TIMESTEPS = 10_000_000  # same as the paper and the replication
SEEDS = [0, 1, 2]
EVAL_DAYS = 100  # test days, identical for every method and seed
EVAL_SEED = 12345

LAMBDA_CHUNKS = 40  # lambda is updated after each chunk of training
LAMBDA_LR = 2.0
LAMBDA_WARMUP_CHUNKS = 4  # no lambda updates in the first 10%: the policy is still random
LAMBDA_MAX = 200.0  # upper bound, so lambda cannot explode early in training
LAMBDA_EVAL_DAYS = 16  # days used to estimate the fairness cost after each chunk

# Output files get the training length in their name, so a quick try never mixes
# with the real runs, e.g. results_10M.csv, lagrangian_log_10M.csv, models_10M/


def train_fixed(env_kwargs, steps, seed):
    env = jym.LogWrapper(make_env(**env_kwargs))
    agent = make_ppo(steps).train(jax.random.PRNGKey(seed), env)
    return agent, []


def train_lagrangian(budget, steps, seed, name):
    """PPO-Lagrangian: reward = profit - lambda * fairness_cost, and after every chunk
    lambda <- clip(lambda + LAMBDA_LR * (J_c - budget), 0, LAMBDA_MAX)."""
    base_env = make_env()
    cost_eval = make_evaluator(make_env(), LAMBDA_EVAL_DAYS)
    chunk = steps // LAMBDA_CHUNKS

    def with_lambda(lam):
        # lambda as a JAX array, so changing it does not trigger recompilation
        return jym.LogWrapper(
            eqx.tree_at(lambda e: e.fairness_alpha, base_env, jnp.asarray(lam, jnp.float32))
        )

    # The networks and optimizer are created once for the full run length, so the
    # learning-rate decay spans all chunks; each train() call then continues training.
    key, k_init = jax.random.split(jax.random.PRNGKey(seed))
    agent = make_ppo(steps).init_state(k_init, with_lambda(0.0))
    train_chunk = eqx.filter_jit(lambda a, k, e: a.train(k, e, total_timesteps=chunk))

    lam, log = 0.0, []
    for i in range(LAMBDA_CHUNKS):
        key, k_train, k_eval = jax.random.split(key, 3)
        agent = train_chunk(agent, k_train, with_lambda(lam))

        info = cost_eval(PPOPolicy(agent), k_eval)
        j_c = float(info["fairness_cost"].mean())
        log.append({
            "method": name, "seed": seed, "chunk": i + 1, "steps": (i + 1) * chunk,
            "lambda": lam, "J_c": j_c, "profit": float(info["profit"].mean()),
            "rejected_customers": float(info["rejected_customers"].mean()),
        })
        if i + 1 >= LAMBDA_WARMUP_CHUNKS:
            lam = min(LAMBDA_MAX, max(0.0, lam + LAMBDA_LR * (j_c - budget)))
    return agent, log


def append_rows(path, rows):
    """Append rows to a CSV. Columns follow the existing file's header; if the new rows
    have extra columns (e.g. a metric added later), the file is rewritten with those
    columns added and left empty for the old rows."""
    if not rows:
        return
    old_rows = load_results(path)
    if os.path.exists(path):
        with open(path, newline="") as f:
            fields = next(csv.reader(f), [])
    else:
        fields = []
    extra = [k for r in rows for k in r if k not in fields]
    fields += list(dict.fromkeys(extra))
    if not old_rows or extra:  # new file, or new columns: (re)write header + old rows
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, restval="")
            writer.writeheader()
            writer.writerows(old_rows)
    with open(path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=fields, restval="").writerows(rows)


def load_results(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def print_table(rows):
    methods = list(dict.fromkeys(r["method"] for r in rows))
    by_method = {m: [r for r in rows if r["method"] == m] for m in methods}
    width = max(14, max(len(m) for m in methods) + 2)
    print("\nMean over seeds (± std); each seed is averaged over the same test days")
    print(f"{'':22}" + "".join(f"{m:>{width + 8}}" for m in methods))
    print(f"{'seeds':22}" + "".join(f"{len(by_method[m]):>{width + 8}}" for m in methods))
    for layer, keys in METRICS.items():
        print(f"--- {layer}")
        for k in keys:
            cells = []
            for m in methods:
                # rows from before a metric was added have no value for it
                v = np.array([float(r[k]) for r in by_method[m] if r.get(k)])
                if len(v) == 0:
                    cells.append("-".rjust(width + 8))
                    continue
                std = f" ±{v.std():.2f}" if len(v) > 1 else ""
                cells.append(f"{v.mean():.3f}{std}".rjust(width + 8))
            print(f"{k:22}" + "".join(cells))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--steps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--table", action="store_true", help="only print the results table")
    args = parser.parse_args()

    tag = f"{args.steps // 1_000_000}M" if args.steps >= 1_000_000 else f"{args.steps // 1000}k"
    RESULTS_FILE, LAMBDA_LOG_FILE, MODEL_DIR = (
        f"results_{tag}.csv", f"lagrangian_log_{tag}.csv", f"models_{tag}"
    )

    if not args.table:
        evaluate = make_evaluator(make_env(), EVAL_DAYS)
        eval_key = jax.random.PRNGKey(EVAL_SEED)
        done = {(r["method"], int(r["seed"])) for r in load_results(RESULTS_FILE)}
        os.makedirs(MODEL_DIR, exist_ok=True)

        if ("MaxCharge", 0) not in done:
            baseline = BaselinePolicy(MaxCharge(make_env(), battery_schedule="none"))
            info = evaluate(baseline, eval_key)
            append_rows(RESULTS_FILE, [{"method": "MaxCharge", "seed": 0, "steps": 0,
                                        **{k: float(info[k].mean()) for k in METRIC_KEYS}}])

        for name in args.methods:
            spec = METHODS[name]
            for seed in args.seeds:
                if (name, seed) in done:
                    print(f"skip {name} seed {seed} (already in {RESULTS_FILE})")
                    continue
                print(f"train {name} | seed {seed} | {args.steps:,} steps ...", flush=True)
                t0 = time.time()
                if "budget" in spec:
                    agent, log = train_lagrangian(spec["budget"], args.steps, seed, name)
                else:
                    agent, log = train_fixed(spec["env"], args.steps, seed)
                info = evaluate(PPOPolicy(agent), eval_key)
                row = {"method": name, "seed": seed, "steps": args.steps,
                       **{k: float(info[k].mean()) for k in METRIC_KEYS}}
                append_rows(RESULTS_FILE, [row])
                append_rows(LAMBDA_LOG_FILE, log)
                safe = name.replace(" ", "_").replace("=", "")
                agent.save_state(os.path.join(MODEL_DIR, f"{safe}_seed{seed}.eqx"))
                print(f"  done in {(time.time() - t0) / 60:.1f} min | profit {row['profit']:.1f} "
                      f"| fairness_cost {row['fairness_cost']:.3f} "
                      f"| worst_group_min_s {row['worst_group_min_s']:.3f}", flush=True)

    print_table(load_results(RESULTS_FILE))

# run.py
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from simulation import DispatchSimulator
from ga_dispatcher import GeneticDispatcher
from ambulance_map import reset_map, simulate_traffic_jam

# -------------------------
# CONFIG
# -------------------------
TRIALS = 30
MAX_STEPS = 100
AMB_PER_BASE = 1
EMERGENCY_RANGE = (0, 3)
TRAFFIC_PROB = 0.10
RESULTS_CSV = "experiment_results.csv"
FIG_DIR = "figures"
BASE_SEED = 1000

os.makedirs(FIG_DIR, exist_ok=True)

# -------------------------
# GA Assignment helper
# -------------------------
def ga_assign(sim: DispatchSimulator, use_fuzzy: bool, seed: int):
    """Assign ambulances using GA with optional fuzzy priorities."""
    unassigned = [e for e in sim.active_emergencies if e.dispatched_ambulance is None]
    available = [a for a in sim.ambulances if a.status == "available"]

    if not unassigned or not available:
        return

    ga = GeneticDispatcher(
        available,
        unassigned,
        seed=seed
    )

    # attribute kept for clarity / future extension
    ga.use_fuzzy = use_fuzzy

    assignments = ga.solve()   # <-- already [(ambulance, emergency), ...]

    sim.assign(assignments)

# -------------------------
# Single run
# -------------------------
def run_single(trial, map_type, mode, seed):
    random.seed(seed)
    np.random.seed(seed)
    reset_map()

    sim = DispatchSimulator(
        num_ambulances_per_base=AMB_PER_BASE,
        seed=seed
    )

    busy_time = {a.id: 0 for a in sim.ambulances}

    for step in range(MAX_STEPS):
        # spawn emergencies
        for _ in range(random.randint(*EMERGENCY_RANGE)):
            sim.spawn_emergency()

        # dynamic traffic
        if map_type == "dynamic" and random.random() < TRAFFIC_PROB:
            simulate_traffic_jam()

        # dispatch
        if mode == "ga":
            ga_assign(sim, use_fuzzy=False, seed=seed + step)
        elif mode == "ga_fuzzy":
            ga_assign(sim, use_fuzzy=True, seed=seed + step)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # advance simulation
        sim.step()

        for a in sim.ambulances:
            if a.status in ("responding", "transporting"):
                busy_time[a.id] += 1

    completed = len(sim.completed_emergencies)
    unresponded = len(sim.unresponded_emergencies)
    active_remaining = len(sim.active_emergencies)
    total_emergencies = completed + unresponded + active_remaining

    response_times = [
        e.arrival_time - e.spawn_time
        for e in sim.completed_emergencies
        if e.arrival_time is not None
    ]
    avg_response_time = float(np.mean(response_times)) if response_times else 0.0

    total_distance = sum(a.total_distance_traveled for a in sim.ambulances)
    utilization = float(
        np.mean([busy_time[a.id] / MAX_STEPS for a in sim.ambulances])
    )

    return {
        "trial": trial,
        "map_type": map_type,
        "mode": mode,
        "completed": completed,
        "unresponded": unresponded,
        "active_remaining": active_remaining,
        "total_emergencies": total_emergencies,
        "avg_response_time": avg_response_time,
        "total_distance": total_distance,
        "utilization": utilization
    }

# -------------------------
# Experiment loop
# -------------------------
def run_experiments():
    results = []
    for trial in range(TRIALS):
        seed = BASE_SEED + trial
        for map_type in ("static", "dynamic"):
            for mode in ("ga", "ga_fuzzy"):
                print(f"Trial {trial} | {map_type} | {mode}")
                res = run_single(trial, map_type, mode, seed)
                results.append(res)

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"Saved {RESULTS_CSV}")
    plot_results(df)

# -------------------------
# Plotting
# -------------------------
def plot_results(df):
    metrics = [
        "avg_response_time",
        "completed",
        "unresponded",
        "total_distance",
        "utilization"
    ]
    for metric in metrics:
        for mtype in ("static", "dynamic"):
            sub = df[df.map_type == mtype]
            data = [
                sub[sub["mode"] == m][metric].values
                for m in ("ga", "ga_fuzzy")
            ]
            plt.figure(figsize=(7, 5))
            plt.boxplot(data, labels=["GA", "GA+Fuzzy"])
            plt.title(f"{metric.replace('_', ' ').title()} ({mtype})")
            plt.ylabel(metric.replace('_', ' ').title())
            plt.grid(True)
            plt.savefig(f"{FIG_DIR}/{metric}_{mtype}.png", bbox_inches="tight")
            plt.close()

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    run_experiments()

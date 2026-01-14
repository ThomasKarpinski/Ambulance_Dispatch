# run.py
import os
import csv
import random
import numpy as np
import matplotlib.pyplot as plt

from simulation import DispatchSimulator
from ga_dispatcher import GeneticDispatcher
import fuzzy_system
from ambulance_map import simulate_traffic_jam, reset_map, find_shortest_path

# -------------------------
# CONFIG
# -------------------------
TRIALS = 30
MAX_STEPS = 100
AMB_PER_BASE = 1
EMERGENCY_RANGE = (0, 3)   # spawn 0..3 emergencies per step
TRAFFIC_PROB = 0.10        # prob of traffic jam event per step (dynamic map)
RESULTS_CSV = "experiment_results.csv"
FIG_DIR = "figures"
RANDOM_SEED_BASE = 1000

os.makedirs(FIG_DIR, exist_ok=True)

# -------------------------
# Helpers
# -------------------------
def crisp_priority(reported_priority: float, travel_time: float, max_time=60.0) -> float:
    p_comp = (reported_priority / 5.0) * 80.0
    t_norm = min(travel_time, max_time) / max_time
    t_comp = (1.0 - t_norm) * 20.0
    return float(max(0.0, min(100.0, p_comp + t_comp)))

def find_path_safe(src, dst):
    try:
        path, t = find_shortest_path(src, dst)
        if path is None:
            return None, float("inf")
        # Add stochastic travel time to diversify response times
        t *= random.uniform(0.8, 1.5)  # 20-50% variability
        return path, t
    except Exception:
        return None, float("inf")

def apply_ga_and_dispatch(sim: DispatchSimulator, use_fuzzy: bool):
    unassigned = [e for e in sim.active_emergencies if e.dispatched_ambulance is None]
    available = [a for a in sim.ambulances if a.status == "available"]
    if not unassigned or not available:
        return

    orig_calc = fuzzy_system.calculate_priority
    try:
        if not use_fuzzy:
            fuzzy_system.calculate_priority = crisp_priority
        ga = GeneticDispatcher(available, unassigned)
        assignment = ga.solve()
    finally:
        fuzzy_system.calculate_priority = orig_calc

    for i, amb in enumerate(assignment):
        if i >= len(unassigned):
            break
        emergency = unassigned[i]
        if amb is None:
            continue
        path, travel_time = find_path_safe(amb.current_location_id, emergency.location_id)
        if path is None:
            continue
        # add on-scene handling time (2-5 steps)
        handling_time = random.randint(2, 5)
        total_time = travel_time + handling_time
        amb.dispatch(emergency, path, total_time)
        emergency.dispatched_ambulance = amb
        emergency.dispatch_time = total_time  # store actual response time

# -------------------------
# Single-run worker
# -------------------------
def run_single(trial_id: int, map_type: str, mode: str, seed: int):
    random.seed(seed)
    np.random.seed(seed)
    reset_map()
    sim = DispatchSimulator(num_ambulances_per_base=AMB_PER_BASE)

    for step in range(MAX_STEPS):
        for _ in range(random.randint(*EMERGENCY_RANGE)):
            sim.spawn_emergency()
        if map_type == "dynamic" and random.random() < TRAFFIC_PROB:
            simulate_traffic_jam()
        if mode == "baseline":
            sim.reassign_emergencies(fuzzy=False)
        elif mode == "ga":
            apply_ga_and_dispatch(sim, use_fuzzy=False)
        elif mode == "ga_fuzzy":
            apply_ga_and_dispatch(sim, use_fuzzy=True)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        sim.run_simulation_step(fuzzy=False)

    completed = len(sim.completed_emergencies)
    unresponded = len(sim.unresponded_emergencies)
    active_remaining = len(sim.active_emergencies)
    total_emergencies = completed + unresponded + active_remaining

    # only include completed emergencies in avg_response_time
    response_times = [e.dispatch_time for e in sim.completed_emergencies if hasattr(e, "dispatch_time")]
    avg_response_time = float(np.mean(response_times)) if response_times else 0.0

    total_distance = sum(getattr(a, "total_distance_traveled", 0) for a in sim.ambulances)
    utilization = float(sum(1 for a in sim.ambulances if a.status in ("responding", "transporting")) / max(1, len(sim.ambulances)))

    return {
        "trial": trial_id,
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
# Experiment Loop
# -------------------------
def run_experiments():
    results = []
    for trial in range(TRIALS):
        base_seed = RANDOM_SEED_BASE + trial
        for map_type in ("static", "dynamic"):
            for mode in ("baseline", "ga", "ga_fuzzy"):
                print(f"Running trial {trial}, map={map_type}, mode={mode}")
                res = run_single(trial, map_type, mode, seed=base_seed)
                results.append(res)
    keys = list(results[0].keys()) if results else []
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved results to {RESULTS_CSV}")
    plot_all(results)

# -------------------------
# Plotting
# -------------------------
def plot_all(results):
    import pandas as pd
    df = pd.DataFrame(results)
    metrics = ["avg_response_time", "completed", "unresponded", "total_distance", "utilization"]
    modes = ["baseline", "ga", "ga_fuzzy"]
    map_types = ["static", "dynamic"]
    os.makedirs(FIG_DIR, exist_ok=True)

    for metric in metrics:
        for mtype in map_types:
            sub = df[df["map_type"] == mtype]
            data = [sub[sub["mode"] == mode][metric].values for mode in modes]
            plt.figure(figsize=(7, 5))
            plt.boxplot(data, labels=modes)
            plt.title(f"{metric.replace('_', ' ').title()} ({mtype})")
            plt.ylabel(metric.replace('_', ' ').title())
            plt.grid(True)
            fname = os.path.join(FIG_DIR, f"{metric}_{mtype}.png")
            plt.savefig(fname, bbox_inches="tight")
            plt.close()

    agg = df.groupby(["map_type", "mode"]).agg({
        "avg_response_time": ["mean", "std"],
        "completed": ["mean", "std"],
        "unresponded": ["mean", "std"],
        "total_distance": ["mean", "std"],
        "utilization": ["mean", "std"]
    })
    agg_fname = os.path.join(FIG_DIR, "aggregated_metrics.csv")
    agg.to_csv(agg_fname)
    print(f"Saved plots into {FIG_DIR}/ and aggregated metrics to {agg_fname}")

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    run_experiments()

"""
genetic_algorithm.py

A more advanced GA/variation class for path-level mutations and
population-level evaluation. This file is written defensively:

- Avoids circular imports by importing simulation objects locally when needed.
- Uses safe checks before sampling or operating on lists.
- Uses explicit errors / prints instead of bare excepts.
- Keeps behavior similar to your original version but more robust.
"""

import random
from typing import Optional, Tuple, List, Dict, Any

from ambulance_map import adjacency_matrix, find_shortest_path, locations


class GA_Generator:
    """
    Base class for a GA-style generator that can mutate ambulance paths,
    crossbreed ambulance assignments, and run a higher-level genetic loop.

    Note: This class does NOT assume a particular representation for genomes
    (that's handled in ga_dispatcher.py). Instead, it provides helpers that
    operate on the simulator state (ambulances, emergencies).
    """

    def __init__(
        self,
        emergency_spawn_range: Tuple[int, int] = (0, 2),
        mutation_rate: float = 0.05,
        mutation_rate_change: float = 0.0,
        crosbreed_rate: float = 0.05,
        crosbreed_rate_change: float = 0.0,
        fuzzy: bool = False,
    ):
        self.emergency_range = emergency_spawn_range
        self.mutation_rate = float(mutation_rate)
        self.mutation_rate_change = float(mutation_rate_change)
        self.crosbreed_rate = float(crosbreed_rate)
        self.crosbreed_rate_change = float(crosbreed_rate_change)
        self.fuzzy = bool(fuzzy)
        self.ds = None  # Will hold a DispatchSimulator instance when running experiments

    # ---------------------------------------------------------------------
    # Utility: assignment score using fuzzy system (safe local import)
    # ---------------------------------------------------------------------
    def calculate_assignment_score(self, ambulance: Any, emergency: Any) -> float:
        """
        Computes a fuzzy priority score for assigning 'ambulance' to 'emergency'.
        Does not crash if path lookup fails.
        """
        try:
            # local import to avoid circular dependency
            import fuzzy_system
        except Exception as e:
            print("Warning: fuzzy_system import failed in calculate_assignment_score:", e)
            return 0.0

        # Get travel time using the stable find_shortest_path API
        try:
            path, travel_time = find_shortest_path(ambulance.current_location_id, emergency.location_id)
            if path is None:
                travel_time = float("inf")
        except Exception as e:
            # If path lookup fails, return a very low score
            print(f"calculate_assignment_score: path lookup failed ({e})")
            travel_time = float("inf")

        severity = getattr(emergency, "priority", getattr(emergency, "reported_priority", 1))
        # scale/clip values if needed inside fuzzy system
        try:
            score = fuzzy_system.calculate_priority(severity, travel_time)
        except Exception as e:
            print("calculate_priority failed:", e)
            score = 0.0

        return float(score)

    # ---------------------------------------------------------------------
    # Path-level mutation: try rerouting an ambulance along an alternate mid node
    # ---------------------------------------------------------------------
    def mutate_ambulance_path(self, ambulance: Any, attempts: int = 3) -> bool:
        """
        Attempt to mutate the path of an ambulance by routing via a random intermediate node.
        Returns True if a mutation was applied (and accepted), False otherwise.
        """
        # Defensive checks
        if ambulance is None:
            return False
        if not hasattr(ambulance, "path") or not ambulance.path:
            return False
        if ambulance.status not in ("responding", "transporting", "returning"):
            return False
        if ambulance.destination_id is None:
            return False

        current = ambulance.current_location_id
        dest = ambulance.destination_id

        # Original path & time
        try:
            orig_res = find_shortest_path(current, dest)
            orig_path, orig_time = orig_res if isinstance(orig_res, tuple) else (None, None)
        except Exception:
            orig_path, orig_time = None, None

        if orig_path is None or orig_time is None or orig_time == float("inf"):
            # No valid baseline path -> clear to force redispatch
            ambulance.path = []
            ambulance.destination_id = None
            ambulance.status = "available"
            ambulance.patient = None
            ambulance.time_to_destination = 0
            return False

        # Try random midpoints
        n_nodes = len(adjacency_matrix)
        tried = 0
        while tried < attempts:
            tried += 1
            mid = random.randrange(n_nodes)
            if mid == current or mid == dest:
                continue

            try:
                r1 = find_shortest_path(current, mid)
                r2 = find_shortest_path(mid, dest)
                if not (isinstance(r1, tuple) and isinstance(r2, tuple)):
                    continue
                p1, t1 = r1
                p2, t2 = r2
                if p1 is None or p2 is None:
                    continue
            except Exception:
                continue

            combined_path = p1 + p2[1:]
            combined_time = t1 + t2

            # Accept slightly longer or better paths (tunable)
            if combined_time <= orig_time * 1.2 and combined_path != orig_path:
                ambulance.path = combined_path
                ambulance.time_to_destination = combined_time
                ambulance.destination_id = dest
                return True

        # If no improvement, optionally clear to force reassign
        ambulance.path = []
        ambulance.destination_id = None
        ambulance.status = "available"
        ambulance.patient = None
        ambulance.time_to_destination = 0
        return False

    # ---------------------------------------------------------------------
    # Crossbreed / swap assignments between two ambulances when beneficial
    # ---------------------------------------------------------------------
    def crossbreed_ambulances(self, amb1: Any, amb2: Any) -> bool:
        """
        If swapping patients between two ambulances yields lower total ETA,
        perform the swap. Returns True if swap occurred.
        """
        if amb1 is None or amb2 is None:
            return False

        # Check colocated or adjacent
        try:
            colocated = amb1.current_location_id == amb2.current_location_id
            adjacent = adjacency_matrix[amb1.current_location_id][amb2.current_location_id] > 0
            if not (colocated or adjacent):
                return False
        except Exception:
            return False

        # local helper (safe)
        def compute_total_eta(ambulance: Any, emergency: Any) -> float:
            if ambulance is None or emergency is None:
                return float("inf")
            try:
                res1 = find_shortest_path(ambulance.current_location_id, emergency.location_id)
                if not isinstance(res1, tuple):
                    return float("inf")
                _, t1 = res1
            except Exception:
                return float("inf")

            # time from emergency to nearest hospital
            hospital_ids = [loc["id"] for loc in locations if loc["type"] == "H"]
            if not hospital_ids:
                return float("inf")

            best_hosp_time = float("inf")
            for hid in hospital_ids:
                try:
                    res2 = find_shortest_path(emergency.location_id, hid)
                    if isinstance(res2, tuple):
                        _, t2 = res2
                    else:
                        continue
                except Exception:
                    continue
                if t2 < best_hosp_time:
                    best_hosp_time = t2

            return (t1 if t1 is not None else float("inf")) + best_hosp_time

        # Evaluate swapping
        try:
            if getattr(amb1, "patient", None) and getattr(amb2, "patient", None):
                cur_eta = compute_total_eta(amb1, amb1.patient) + compute_total_eta(amb2, amb2.patient)
                swapped_eta = compute_total_eta(amb1, amb2.patient) + compute_total_eta(amb2, amb1.patient)
                if swapped_eta < cur_eta:
                    # perform swap of patient assignments & paths
                    amb1.patient, amb2.patient = amb2.patient, amb1.patient
                    amb1.destination_id, amb2.destination_id = amb2.destination_id, amb1.destination_id
                    amb1.path, amb2.path = amb2.path, amb1.path
                    amb1.time_to_destination, amb2.time_to_destination = amb2.time_to_destination, amb1.time_to_destination
                    try:
                        amb1.patient.dispatched_ambulance = amb1
                        amb2.patient.dispatched_ambulance = amb2
                    except Exception:
                        pass
                    return True
        except Exception:
            return False

        # If only one patient & other available, consider transfer
        try:
            if getattr(amb1, "patient", None) and getattr(amb2, "status", None) == "available":
                cur_eta = compute_total_eta(amb1, amb1.patient)
                transfer_eta = compute_total_eta(amb2, amb1.patient)
                if transfer_eta < cur_eta:
                    amb2.patient = amb1.patient
                    amb2.destination_id = amb1.destination_id
                    res = find_shortest_path(amb2.current_location_id, amb2.destination_id)
                    if isinstance(res, tuple):
                        path, t = res
                        amb2.path = path
                        amb2.time_to_destination = t
                        amb2.status = "responding"
                        amb1.patient.dispatched_ambulance = amb2
                        amb1.patient = None
                        amb1.destination_id = None
                        amb1.path = []
                        amb1.time_to_destination = 0
                        return True
        except Exception:
            pass

        try:
            if getattr(amb2, "patient", None) and getattr(amb1, "status", None) == "available":
                cur_eta = compute_total_eta(amb2, amb2.patient)
                transfer_eta = compute_total_eta(amb1, amb2.patient)
                if transfer_eta < cur_eta:
                    amb1.patient = amb2.patient
                    amb1.destination_id = amb2.destination_id
                    res = find_shortest_path(amb1.current_location_id, amb1.destination_id)
                    if isinstance(res, tuple):
                        path, t = res
                        amb1.path = path
                        amb1.time_to_destination = t
                        amb1.status = "responding"
                        amb2.patient.dispatched_ambulance = amb1
                        amb2.patient = None
                        amb2.destination_id = None
                        amb2.path = []
                        amb2.time_to_destination = 0
                        return True
        except Exception:
            pass

        return False

    # ---------------------------------------------------------------------
    # Fitness & evaluation helpers that consume a DispatchSimulator instance
    # ---------------------------------------------------------------------
    def compute_fitness(self, simulator: Any) -> Tuple[float, Dict[str, Any]]:
        """
        Computes a composite fitness for the current simulator state.
        Returns (fitness_score, metrics_dict).
        """
        completed = len(simulator.completed_emergencies)
        unresponded = len(simulator.unresponded_emergencies)
        active = len(simulator.active_emergencies)
        total_emergencies = completed + unresponded + active

        satisfaction_rate = (completed / total_emergencies) if total_emergencies > 0 else 1.0

        # Response times
        response_times = []
        for em in simulator.completed_emergencies + simulator.unresponded_emergencies:
            if hasattr(em, "dispatch_time") and em.dispatch_time is not None:
                response_times.append(em.dispatch_time)
            else:
                response_times.append(getattr(em, "time_elapsed", 0))

        avg_response_time = (sum(response_times) / len(response_times)) if response_times else None
        normalized_response = min(avg_response_time / 10.0, 1.0) if avg_response_time is not None else 1.0

        # Distance
        total_distance = sum(getattr(amb, "total_distance_traveled", 0) for amb in simulator.ambulances)
        normalized_distance = min(total_distance / 100.0, 1.0) if total_distance > 0 else 0.0

        # Utilization
        utilization = 0.0
        if simulator.ambulances:
            active_ambulances = sum(1 for a in simulator.ambulances if a.status in ["responding", "transporting"])
            utilization = active_ambulances / len(simulator.ambulances)

        # Weighted composite
        w_satisfaction = 0.5
        w_response = 0.25
        w_distance = 0.15
        w_utilization = 0.10

        fitness_score = (
            w_satisfaction * satisfaction_rate +
            w_response * (1.0 - normalized_response) +
            w_distance * (1.0 - normalized_distance) +
            w_utilization * utilization
        )

        metrics = {
            "satisfaction_rate": satisfaction_rate,
            "avg_response_time": avg_response_time,
            "total_distance": total_distance,
            "utilization": utilization,
            "fitness_score": fitness_score,
            "completed": completed,
            "unresponded": unresponded,
            "active": active,
            "total_emergencies": total_emergencies
        }

        return fitness_score, metrics

    # ---------------------------------------------------------------------
    # Top-level genetic algorithm that runs many simulation epochs & generations
    # ---------------------------------------------------------------------
    def geneticAlgorithm(
        self,
        epochs: int = 5,
        generations: int = 3,
        ambulances_per_base: int = 1,
        random_seed: Optional[int] = None
    ):
        """
        Run a GA-like outer loop:
         - For each generation, create a new DispatchSimulator and simulate several epochs.
         - Optionally apply per-ambulance mutations and crossbreeding.
        Returns the best final metrics and the simulator instance used.
        """
        if random_seed is not None:
            random.seed(random_seed)

        best_overall = None
        best_metrics = None

        for g in range(generations):
            print(f"\n--- Generation {g+1}/{generations} ---")
            # Create a fresh simulator instance
            try:
                from simulation import DispatchSimulator  # local import to avoid circular import
            except Exception as e:
                raise RuntimeError("simulation.DispatchSimulator import failed in geneticAlgorithm") from e

            self.ds = DispatchSimulator(ambulances_per_base)

            for ep in range(epochs):
                # spawn some emergencies
                num_new = random.randint(*self.emergency_range) if self.emergency_range else 0
                for _ in range(num_new):
                    self.ds.spawn_emergency()

                # Run a single simulation step (allow dispatching)
                try:
                    self.ds.run_simulation_step(fuzzy=self.fuzzy)
                except TypeError:
                    # fallback to explicit reassign + step
                    self.ds.reassign_emergencies(fuzzy=self.fuzzy)
                    self.ds.run_simulation_step()

                # If fuzzy mode, attempt local mutations and crossbreeding between ambulances
                if self.fuzzy:
                    # Mutate some ambulances' paths
                    for amb in list(self.ds.ambulances):
                        if random.random() <= self.mutation_rate:
                            try:
                                self.mutate_ambulance_path(amb)
                            except Exception as e:
                                # continue on error for robustness
                                print(f"mutate_ambulance_path error: {e}")

                    # Crossbreed pairs
                    n = len(self.ds.ambulances)
                    for i in range(n):
                        for j in range(i + 1, n):
                            if random.random() <= self.crosbreed_rate:
                                try:
                                    self.crossbreed_ambulances(self.ds.ambulances[i], self.ds.ambulances[j])
                                except Exception as e:
                                    print(f"crossbreed_ambulances error: {e}")

            # Evaluate generation
            fitness_score, metrics = self.compute_fitness(self.ds)
            print(f"Generation {g+1} Metrics: completed={metrics['completed']}, unresponded={metrics['unresponded']}, fitness={fitness_score:.4f}")

            if best_overall is None or fitness_score > best_overall:
                best_overall = fitness_score
                best_metrics = metrics

        return best_overall, best_metrics, self.ds

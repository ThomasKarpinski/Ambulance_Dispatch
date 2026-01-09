from simulation import DispatchSimulator, Ambulance
from random import randint
import random
from ambulance_map import adjacency_matrix, find_shortest_path, locations
import fuzzy_system  # <--- Added Import

class GA_Generator:
    """
    Base class for genetic algorithm generation
    """
    def __init__(
            self,
            emergency_spawn_range: tuple,
            mutation_rate: float,
            mutation_rate_change: float,
            crosbreed_rate: float,
            crosbreed_rate_change: float,
            fuzzy: bool
    ):
        self.emergency_range = emergency_spawn_range
        self.mutation_rate = mutation_rate
        self.mutation_rate_change = mutation_rate_change
        self.crosbreed_rate = crosbreed_rate
        self.crosbreed_rate_change = crosbreed_rate_change
        self.fuzzy = fuzzy
        self.ds = None

    # =========================================================================
    # NEW: Helper for Fuzzy Logic (Integrated into your class)
    # =========================================================================
    def calculate_assignment_score(self, ambulance, emergency):
        """
        Calculates the Fuzzy Priority score for assigning a specific
        ambulance to a specific emergency.
        """
        # 1. Calculate Travel Time
        try:
            result = find_shortest_path(ambulance.current_location_id, emergency.location_id)
            if isinstance(result, tuple):
                _, travel_time = result
            else:
                travel_time = result
        except:
            travel_time = 999

        # 2. Get Severity
        severity = emergency.priority

        # 3. Calculate Fuzzy Priority
        priority_score = fuzzy_system.calculate_priority(severity, travel_time)
        return priority_score

    # =========================================================================
    # EXISTING COMPLEX MUTATION LOGIC (Restored)
    # =========================================================================
    def __mutate__(
            self,
            ambulance: Ambulance
    ):
        """ Mutates path of an ambulance """
        if not ambulance.path or ambulance.status not in ['responding', 'transporting', 'returning']:
            return
        # Attempt a localized reroute: try random intermediate nodes and
        # build a new path via that node. If it yields a comparable or
        # slightly better ETA, adopt it.
        current = ambulance.current_location_id
        dest = ambulance.destination_id
        if dest is None:
            return

        orig_path_res = find_shortest_path(current, dest)
        if isinstance(orig_path_res, tuple):
            orig_path, orig_time = orig_path_res
        else:
             # Fallback if path finding fails or returns incomplete data
             orig_path, orig_time = None, None

        if orig_path is None:
            # no path exists, fallback to clearing so dispatcher can reassign
            # print(f"Ambulance {ambulance.id} mutation fallback: no path to destination.")
            ambulance.path = []
            ambulance.destination_id = None
            ambulance.status = 'available'
            ambulance.patient = None
            ambulance.time_to_destination = 0
            return

        attempts = 3
        improved = False
        for _ in range(attempts):
            # pick a random intermediate node (not current or dest)
            mid = random.randrange(len(adjacency_matrix))
            if mid == current or mid == dest:
                continue

            try:
                r1 = find_shortest_path(current, mid)
                r2 = find_shortest_path(mid, dest)
                if isinstance(r1, tuple) and isinstance(r2, tuple):
                    p1, t1 = r1
                    p2, t2 = r2
                else:
                    continue
            except:
                continue

            if p1 is None or p2 is None:
                continue

            # combine paths (avoid duplicate mid)
            combined_path = p1 + p2[1:]
            combined_time = t1 + t2

            # accept if combined path is no worse than 20% longer or strictly better
            if combined_time <= orig_time * 1.2 and combined_path != orig_path:
                ambulance.path = combined_path
                ambulance.time_to_destination = combined_time
                ambulance.destination_id = dest
                # print(f"Ambulance {ambulance.id} path rerouted via {mid}.")
                improved = True
                break

        if not improved:
            # fallback: small perturbation by clearing path to force redispatch
            # print(f"Ambulance {ambulance.id} mutation cleared (no good reroute).")
            ambulance.path = []
            ambulance.destination_id = None
            ambulance.status = 'available'
            ambulance.patient = None
            ambulance.time_to_destination = 0

    # =========================================================================
    # EXISTING COMPLEX CROSSBREED LOGIC (Restored)
    # =========================================================================
    def __crossbreed__(self,
            amb1: Ambulance,
            amb2: Ambulance
    ):
        """Smart crossbreeding between two ambulances."""
        if amb1 is None or amb2 is None:
            return

        # allow crossbreeding when colocated or on adjacent nodes
        try:
            colocated = amb1.current_location_id == amb2.current_location_id
            adjacent = adjacency_matrix[amb1.current_location_id][amb2.current_location_id] > 0
            if not (colocated or adjacent):
                return
        except:
            return

        # helper to compute ETA for ambulance to service an emergency (includes hospital leg)
        def compute_total_eta(ambulance, emergency):
            if ambulance is None or emergency is None:
                return float('inf')
            # time from ambulance to emergency
            try:
                res1 = find_shortest_path(ambulance.current_location_id, emergency.location_id)
                if isinstance(res1, tuple): _, t1 = res1
                else: return float('inf')
            except: return float('inf')

            # time from emergency to nearest hospital
            hospital_ids = [loc['id'] for loc in locations if loc['type'] == 'H']
            if not hospital_ids:
                return float('inf')

            best_hosp_time = float('inf')
            for hid in hospital_ids:
                try:
                    res2 = find_shortest_path(emergency.location_id, hid)
                    if isinstance(res2, tuple): _, t2 = res2
                    else: t2 = float('inf')
                except: t2 = float('inf')

                if t2 < best_hosp_time:
                    best_hosp_time = t2
            return t1 + best_hosp_time

        # if both ambulances have assigned patients, evaluate swapping
        if amb1.patient and amb2.patient:
            cur_eta = compute_total_eta(amb1, amb1.patient) + compute_total_eta(amb2, amb2.patient)
            swapped_eta = compute_total_eta(amb1, amb2.patient) + compute_total_eta(amb2, amb1.patient)
            if swapped_eta < cur_eta:
                # perform swap
                temp_pat = amb1.patient
                temp_dest = amb1.destination_id
                temp_path = amb1.path
                temp_time = amb1.time_to_destination

                amb1.patient = amb2.patient
                amb1.destination_id = amb2.destination_id
                amb1.path = amb2.path
                amb1.time_to_destination = amb2.time_to_destination

                amb2.patient = temp_pat
                amb2.destination_id = temp_dest
                amb2.path = temp_path
                amb2.time_to_destination = temp_time

                try:
                    amb1.patient.dispatched_ambulance = amb1
                    amb2.patient.dispatched_ambulance = amb2
                except Exception:
                    pass

        # if only one ambulance has a patient and the other is available
        elif amb1.patient and amb2.status == 'available':
            cur_eta = compute_total_eta(amb1, amb1.patient)
            transfer_eta = compute_total_eta(amb2, amb1.patient)
            if transfer_eta + 0.0 < cur_eta:
                amb2.patient = amb1.patient
                amb2.destination_id = amb1.destination_id
                res = find_shortest_path(amb2.current_location_id, amb2.destination_id)
                if isinstance(res, tuple):
                    path, t = res
                    amb2.path = path
                    amb2.time_to_destination = t
                    amb2.status = 'responding'
                    amb1.patient.dispatched_ambulance = amb2

                    amb1.patient = None
                    amb1.destination_id = None
                    amb1.path = []
                    amb1.time_to_destination = 0

        elif amb2.patient and amb1.status == 'available':
            cur_eta = compute_total_eta(amb2, amb2.patient)
            transfer_eta = compute_total_eta(amb1, amb2.patient)
            if transfer_eta + 0.0 < cur_eta:
                amb1.patient = amb2.patient
                amb1.destination_id = amb2.destination_id
                res = find_shortest_path(amb1.current_location_id, amb1.destination_id)
                if isinstance(res, tuple):
                    path, t = res
                    amb1.path = path
                    amb1.time_to_destination = t
                    amb1.status = 'responding'
                    amb2.patient.dispatched_ambulance = amb1

                    amb2.patient = None
                    amb2.destination_id = None
                    amb2.path = []
                    amb2.time_to_destination = 0

    def compute_fitness(self, simulator: DispatchSimulator):
        """
        Computes a refined fitness score based on multiple metrics.
        """
        completed = len(simulator.completed_emergencies)
        unresponded = len(simulator.unresponded_emergencies)
        active = len(simulator.active_emergencies)
        total_emergencies = completed + unresponded + active

        if total_emergencies > 0:
            satisfaction_rate = completed / total_emergencies
        else:
            satisfaction_rate = 1.0

        # Metric 2: Average Response Time
        response_times = []
        for em in simulator.completed_emergencies + simulator.unresponded_emergencies:
            # Check if dispatch_time exists (it might not in basic Emergency class)
            if hasattr(em, 'dispatch_time') and em.dispatch_time is not None:
                response_times.append(em.dispatch_time)
            else:
                # Fallback to time_elapsed for approximations
                response_times.append(em.time_elapsed)

        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            normalized_response = min(avg_response_time / 10.0, 1.0)
        else:
            normalized_response = 1.0 if total_emergencies > 0 else 0.0

        # Metric 3: Total Travel Distance (Using simulator data if available)
        total_distance = 0
        for amb in simulator.ambulances:
            if hasattr(amb, 'total_distance_traveled'):
                total_distance += amb.total_distance_traveled

        normalized_distance = min(total_distance / 100.0, 1.0) if total_distance > 0 else 0.0

        # Metric 4: Resource Utilization
        if simulator.ambulances:
            active_ambulances = sum(1 for a in simulator.ambulances
                                   if a.status in ['responding', 'transporting'])
            utilization = active_ambulances / len(simulator.ambulances)
        else:
            utilization = 0.0

        # Composite Fitness Score
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

        return fitness_score, {
            'satisfaction_rate': satisfaction_rate,
            'avg_response_time': avg_response_time if response_times else None,
            'total_distance': total_distance,
            'utilization': utilization,
            'fitness_score': fitness_score
        }

    def geneticAlgorithm(
            self,
            epochs: int,
            generations: int,
            ambulances_per_base: int
    ):
        """ Main algorithm setup and loop """

        # setup
        prev_mut_rate = 0
        prev_cros_rate = 0
        efficiency = 0
        emergency_satisfaction = 0

        # algorithm loop
        for g in range(generations):
            print(f"\n--- Generation {g+1}/{generations} ---")
            self.ds = DispatchSimulator(ambulances_per_base)

            for _ in range(epochs):
                # adding new emergencies
                for _ in range(randint(*self.emergency_range)):
                    self.ds.spawn_emergency()

                # ==========================================================
                # UPDATE: Pass the self.fuzzy flag to the simulation step
                # ==========================================================
                # Note: Ensure your simulation.py run_simulation_step() accepts this arg
                # Or updates reassign_emergencies internal call.
                try:
                    self.ds.run_simulation_step(fuzzy=self.fuzzy)
                except TypeError:
                    # Fallback if simulation.py hasn't been updated to take the arg:
                    # We assume reassign_emergencies handles it or default is used.
                    # Ideally, you should update simulation.py run_simulation_step
                    # to: def run_simulation_step(self, fuzzy=False): ...
                    self.ds.reassign_emergencies(fuzzy=self.fuzzy)
                    self.ds.run_simulation_step() # Run the rest of the step

                # mutating paths
                if self.fuzzy:
                    for amb in self.ds.ambulances:
                        if random.randint(0, 1000) / 1000 <= self.mutation_rate:
                            self.__mutate__(amb)

                    # crossbreeding paths
                    for a1 in range(len(self.ds.ambulances)):
                        for a2 in range(a1 + 1, len(self.ds.ambulances)):
                            self.__crossbreed__(
                                self.ds.ambulances[a1],
                                self.ds.ambulances[a2]
                            )

            # evaluating generation metrics
            fitness_score, metrics = self.compute_fitness(self.ds)

            print(f"Generation {g+1} Metrics:")
            print(f"  Total Emergencies: {len(self.ds.completed_emergencies) + len(self.ds.unresponded_emergencies) + len(self.ds.active_emergencies)}")
            print(f"  Completed: {len(self.ds.completed_emergencies)}")
            print(f"  Unresponded: {len(self.ds.unresponded_emergencies)}")
            print(f"  Satisfaction Rate: {metrics['satisfaction_rate']:.2%}")
            if metrics['avg_response_time'] is not None:
                print(f"  Average Response Time: {metrics['avg_response_time']:.2f} steps")
            print(f"  Overall Fitness Score: {metrics['fitness_score']:.4f}")
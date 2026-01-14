import random
import time
from ambulance_map import locations, adjacency_matrix, find_shortest_path


class Ambulance:
    def __init__(self, id: int, home_base_id: int):
        self.id = id
        self.home_base_id = home_base_id
        self.current_location_id = home_base_id
        self.status = "available"
        self.patient = None
        self.destination_id = None
        self.path = []
        self.time_to_destination = 0.0
        self.total_distance_traveled = 0.0

    def dispatch(self, emergency, path, travel_time):
        if self.status in ("available", "returning"):
            self.status = "responding"
            self.patient = emergency
            self.destination_id = emergency.location_id
            self.path = path
            self.time_to_destination = travel_time
            return True
        return False

    def pickup_patient(self):
        self.status = "transporting"

    def dropoff_patient(self):
        self.patient = None
        self.status = "available"

    def return_to_base(self, path, travel_time):
        self.status = "returning"
        self.destination_id = self.home_base_id
        self.path = path
        self.time_to_destination = travel_time


class Emergency:
    """
    severity = true emergency severity (ground truth)
    reported_priority = noisy observation
    """
    def __init__(self, id: int, location_id: int, severity: int, spawn_time: int):
        self.id = id
        self.location_id = location_id
        self.severity = severity
        self.spawn_time = spawn_time

        # 30% reporting noise
        if random.random() < 0.3:
            self.reported_priority = max(1, min(5, severity + random.choice([-1, 1])))
        else:
            self.reported_priority = severity

        self.dispatched_ambulance = None
        self.dispatch_time = None
        self.arrival_time = None
        self.completion_time = None


class DispatchSimulator:
    def __init__(self, num_ambulances_per_base: int, seed=None):
        if seed is not None:
            random.seed(seed)

        self.locations = locations
        self.adjacency_matrix = adjacency_matrix
        self.ambulances = []
        self.active_emergencies = []
        self.completed_emergencies = []
        self.unresponded_emergencies = []

        self.current_time = 0
        self.max_emergency_lifespan = 25
        self._next_ambulance_id = 0
        self._next_emergency_id = 0

        # Create ambulances
        bases = [loc for loc in locations if loc["type"] == "A"]
        for base in bases:
            for _ in range(num_ambulances_per_base):
                self.ambulances.append(
                    Ambulance(self._next_ambulance_id, base["id"])
                )
                self._next_ambulance_id += 1

    # --------------------------------------------------
    # Emergency generation
    # --------------------------------------------------

    def spawn_emergency(self):
        valid_locations = [loc for loc in self.locations if loc["type"] in ("E", "I")]
        if not valid_locations:
            return None

        loc = random.choice(valid_locations)
        severity = random.choices(
            [1, 2, 3, 4, 5],
            weights=[10, 20, 30, 25, 15],
            k=1
        )[0]

        e = Emergency(
            id=self._next_emergency_id,
            location_id=loc["id"],
            severity=severity,
            spawn_time=self.current_time
        )
        self._next_emergency_id += 1
        self.active_emergencies.append(e)
        return e

    # --------------------------------------------------
    # Assignment (neutral — GA logic lives elsewhere)
    # --------------------------------------------------

    def assign(self, assignments):
        """
        assignments: list of (ambulance, emergency)
        """
        for ambulance, emergency in assignments:
            res = find_shortest_path(
                ambulance.current_location_id,
                emergency.location_id
            )
            if isinstance(res, tuple):
                path, dist = res
            else:
                path, dist = [ambulance.current_location_id, emergency.location_id], res

            if ambulance.dispatch(emergency, path, dist):
                emergency.dispatched_ambulance = ambulance
                emergency.dispatch_time = self.current_time

    # --------------------------------------------------
    # Movement
    # --------------------------------------------------

    def move_ambulance(self, ambulance):
        if not ambulance.path or ambulance.time_to_destination <= 0:
            return

        if len(ambulance.path) < 2:
            ambulance.current_location_id = ambulance.destination_id
            ambulance.path = []
            ambulance.time_to_destination = 0
            return

        nxt = ambulance.path[1]
        try:
            base_time = self.adjacency_matrix[ambulance.current_location_id][nxt]
        except (IndexError, TypeError):
            base_time = 1


        delay = random.uniform(1.0, 1.5)
        actual = base_time * delay

        ambulance.current_location_id = nxt
        ambulance.path = ambulance.path[1:]
        ambulance.time_to_destination -= actual
        ambulance.total_distance_traveled += base_time

        if ambulance.time_to_destination <= 0:
            ambulance.time_to_destination = 0

            if ambulance.status == "responding":
                emergency = ambulance.patient
                emergency.arrival_time = self.current_time
                ambulance.pickup_patient()

                hospitals = [l for l in self.locations if l["type"] == "H"]
                if hospitals:
                    h = hospitals[0]["id"]
                    p, t = find_shortest_path(ambulance.current_location_id, h)
                    ambulance.destination_id = h
                    ambulance.path = p
                    ambulance.time_to_destination = t

            elif ambulance.status == "transporting":
                emergency = ambulance.patient
                emergency.completion_time = self.current_time
                self.completed_emergencies.append(emergency)
                self.active_emergencies = [
                    e for e in self.active_emergencies if e.id != emergency.id
                ]
                ambulance.dropoff_patient()

                p, t = find_shortest_path(
                    ambulance.current_location_id,
                    ambulance.home_base_id
                )
                ambulance.return_to_base(p, t)

            elif ambulance.status == "returning":
                ambulance.status = "available"
                ambulance.destination_id = None

    # --------------------------------------------------
    # Simulation step
    # --------------------------------------------------

    def step(self):
        self.current_time += 1

        for e in list(self.active_emergencies):
            if e.dispatch_time is None and \
               self.current_time - e.spawn_time > self.max_emergency_lifespan:
                self.unresponded_emergencies.append(e)
                self.active_emergencies.remove(e)

        for amb in self.ambulances:
            self.move_ambulance(amb)

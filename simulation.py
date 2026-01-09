import random
import time
import fuzzy_system
import ga_dispatcher
from ambulance_map import locations, adjacency_matrix, find_shortest_path

class Ambulance:
    """
    Represents an ambulance in the simulation.
    """
    def __init__(self, id: int, home_base_id: int):
        self.id = id
        self.home_base_id = home_base_id
        self.current_location_id = home_base_id
        self.status = 'available'
        self.patient = None
        self.destination_id = None
        self.path = []
        self.time_to_destination = 0
        self.total_distance_traveled = 0

    def __repr__(self):
        return (f"Ambulance(id={self.id}, loc={self.current_location_id}, "
                f"status='{self.status}', dest={self.destination_id}, "
                f"time_left={self.time_to_destination:.1f})")

    def dispatch(self, emergency, path_to_emergency, travel_time):
        """Dispatches the ambulance to an emergency."""
        if self.status == 'available' or self.status == 'returning':
            self.status = 'responding'
            self.patient = emergency
            self.destination_id = emergency.location_id
            self.path = path_to_emergency
            self.time_to_destination = travel_time

            # Record dispatch timestamp
            if emergency.dispatch_time is None:
                emergency.dispatch_time = emergency.time_elapsed
            return True
        return False

    def pickup_patient(self, emergency):
        self.patient = emergency
        self.status = 'transporting'

    def dropoff_patient(self):
        self.patient = None
        self.status = 'available'

    def return_to_base(self, path_to_base, travel_time):
        self.status = 'returning'
        self.destination_id = self.home_base_id
        self.path = path_to_base
        self.time_to_destination = travel_time


class Emergency:
    """
    Represents an emergency event with UNCERTAINTY.
    """
    def __init__(self, id: int, location_id: int, priority: int):
        self.id = id
        self.location_id = location_id

        self.true_priority = priority

        # 30% the reported priority is wrong
        if random.random() < 0.3:
            noise = random.choice([-1, 1])
            self.priority = max(1, min(5, priority + noise))
        else:
            self.priority = priority

        self.dispatched_ambulance = None
        self.time_elapsed = 0
        self.dispatch_time = None
        self.completion_time = None

    def __repr__(self):
        return (f"Emergency(id={self.id}, loc={self.location_id}, "
                f"ReportedPri={self.priority} (True={self.true_priority}), "
                f"Status={'Dispatched' if self.dispatched_ambulance else 'Pending'})")


class DispatchSimulator:
    """
    Manages the state and progression of the ambulance dispatch simulation.
    """
    def __init__(self, num_ambulances_per_base: int):
        self.locations = locations
        self.adjacency_matrix = adjacency_matrix
        self.ambulances = []
        self.active_emergencies = []
        self.completed_emergencies = []
        self.unresponded_emergencies = []
        self._next_ambulance_id = 0
        self._next_emergency_id = 0
        self.max_emergency_lifespan = 25

        # Create ambulances
        ambulance_bases = [loc for loc in self.locations if loc['type'] == 'A']
        for base in ambulance_bases:
            for _ in range(num_ambulances_per_base):
                self.ambulances.append(Ambulance(id=self._next_ambulance_id, home_base_id=base['id']))
                self._next_ambulance_id += 1

        # print(f"Initialized simulator with {len(self.ambulances)} ambulances.")

    def spawn_emergency(self):
        valid_locations = [loc for loc in self.locations if loc['type'] in ['E', 'I']]
        if not valid_locations: return None

        random_location = random.choice(valid_locations)
        random_priority = random.choices([1, 2, 3, 4, 5], weights=[10, 20, 30, 25, 15], k=1)[0]

        new_emergency = Emergency(id=self._next_emergency_id,
                                  location_id=random_location['id'],
                                  priority=random_priority)

        self.active_emergencies.append(new_emergency)
        self._next_emergency_id += 1
        return new_emergency

    def reassign_emergencies(self, fuzzy: bool = False):
        """
        Assigns active emergencies to available ambulances.
        """
        unassigned_emergencies = [e for e in self.active_emergencies if e.dispatched_ambulance is None]
        available_ambulances = [a for a in self.ambulances if a.status == 'available']

        if not unassigned_emergencies or not available_ambulances:
            return

        if fuzzy:
            ga = ga_dispatcher.GeneticDispatcher(available_ambulances, unassigned_emergencies)
            best_assignments = ga.solve()

            # Apply the assignments found by GA
            for i, ambulance in enumerate(best_assignments):
                emergency = unassigned_emergencies[i]

                if ambulance:
                    try:
                        res = find_shortest_path(ambulance.current_location_id, emergency.location_id)
                        if isinstance(res, tuple): path, dist = res
                        else: path, dist = [ambulance.current_location_id, emergency.location_id], res

                        ambulance.dispatch(emergency, path, dist)
                        emergency.dispatched_ambulance = ambulance

                        if ambulance in available_ambulances:
                            available_ambulances.remove(ambulance)
                    except:
                        continue
            return

        while available_ambulances and unassigned_emergencies:
            global_best_pair = None
            min_global_dist = float('inf')
            best_global_path = []

            #check every ambulance against every call to find the absolute shortest trip
            for ambulance in available_ambulances:
                for emergency in unassigned_emergencies:
                    try:
                        res = find_shortest_path(ambulance.current_location_id, emergency.location_id)
                        if isinstance(res, tuple): path, dist = res
                        else: path, dist = [ambulance.current_location_id, emergency.location_id], res

                        if dist < min_global_dist:
                            min_global_dist = dist
                            global_best_pair = (ambulance, emergency)
                            best_global_path = path
                    except: continue

            if global_best_pair:
                amb, emerg = global_best_pair
                amb.dispatch(emerg, best_global_path, min_global_dist)
                emerg.dispatched_ambulance = amb

                available_ambulances.remove(amb)
                unassigned_emergencies.remove(emerg)
            else:
                break

    def move_ambulance(self, ambulance: Ambulance):
        """
        Moves the ambulance one step along its path with UNCERTAINTY.
        """
        if not ambulance.path or ambulance.time_to_destination <= 0:
            return

        # safety check for path length
        if len(ambulance.path) < 2:
            if ambulance.destination_id:
                ambulance.current_location_id = ambulance.destination_id
                ambulance.path = []
                ambulance.time_to_destination = 0
            return

        next_location_id = ambulance.path[1]

        try:
            base_time = self.adjacency_matrix[ambulance.current_location_id][next_location_id]
        except:
             base_time = 1

        # random traffic delay (0% to 50% extra time)
        traffic_factor = random.uniform(1.0, 1.5)
        actual_time_taken = base_time * traffic_factor

        ambulance.current_location_id = next_location_id
        ambulance.total_distance_traveled += base_time
        ambulance.path = ambulance.path[1:]
        ambulance.time_to_destination -= actual_time_taken

        # check if arrived at destination
        if ambulance.current_location_id == ambulance.destination_id or ambulance.time_to_destination <= 0:
            ambulance.time_to_destination = 0

            if ambulance.status == 'responding':
                emergency = ambulance.patient
                if emergency:
                    ambulance.pickup_patient(emergency)
                    hospitals = [loc for loc in self.locations if loc['type'] == 'H']
                    if hospitals:
                        hospital_id = hospitals[0]['id']
                        res = find_shortest_path(ambulance.current_location_id, hospital_id)
                        if isinstance(res, tuple):
                             p, t = res
                             ambulance.destination_id = hospital_id
                             ambulance.path = p
                             ambulance.time_to_destination = t

            elif ambulance.status == 'transporting':
                # Arrived at hospital
                completed_emergency = ambulance.patient
                ambulance.dropoff_patient()
                if completed_emergency:
                    completed_emergency.completion_time = completed_emergency.time_elapsed
                    self.completed_emergencies.append(completed_emergency)
                    self.active_emergencies = [e for e in self.active_emergencies if e.id != completed_emergency.id]

                # Return to base
                res = find_shortest_path(ambulance.current_location_id, ambulance.home_base_id)
                if isinstance(res, tuple):
                     p, t = res
                     ambulance.return_to_base(p, t)

            elif ambulance.status == 'returning':
                ambulance.status = 'available'
                ambulance.destination_id = None
                ambulance.path = []
                ambulance.time_to_destination = 0


    def run_simulation_step(self, fuzzy: bool = False):
        """A single step in the simulation."""

        # 1. Update time/check unresponded
        for emergency in list(self.active_emergencies):
            emergency.time_elapsed += 1
            if emergency.dispatched_ambulance is None and emergency.time_elapsed > self.max_emergency_lifespan:
                self.unresponded_emergencies.append(emergency)
                self.active_emergencies.remove(emergency)

        # 2. Reassign emergencies (Pass the fuzzy flag!)
        self.reassign_emergencies(fuzzy=fuzzy)

        # 3. Move all ambulances
        for ambulance in self.ambulances:
            self.move_ambulance(ambulance)

if __name__ == "__main__":
    simulator = DispatchSimulator(num_ambulances_per_base=1)
    for i in range(10):
        simulator.spawn_emergency()
        simulator.run_simulation_step(fuzzy=True) # Test Fuzzy Mode
        time.sleep(0.1)
    print("Demo Done")
import random
import time
from ambulance_map import locations, adjacency_matrix, get_location_by_id

class Ambulance:
    """
    Represents an ambulance in the simulation.
    """
    def __init__(self, id: int, home_base_id: int):
        self.id = id
        self.home_base_id = home_base_id
        self.current_location_id = home_base_id
        # Status can be: 'available', 'responding', 'transporting', 'returning'
        self.status = 'available'
        self.patient = None
        self.destination_id = None
        self.path = []

    def __repr__(self):
        return (f"Ambulance(id={self.id}, location={self.current_location_id}, "
                f"status='{self.status}', destination={self.destination_id})")

    def dispatch(self, emergency, path_to_emergency):
        """Dispatches the ambulance to an emergency."""
        if self.status == 'available' or self.status == 'returning':
            self.status = 'responding'
            self.destination_id = emergency.location_id
            self.path = path_to_emergency
            print(f"Ambulance {self.id} dispatched from {self.current_location_id} to emergency at {emergency.location_id}.")
            return True
        return False

    def pickup_patient(self, emergency):
        """Picks up a patient at an emergency scene."""
        self.patient = emergency
        self.status = 'transporting'
        print(f"Ambulance {self.id} picked up patient at {self.current_location_id}.")

    def dropoff_patient(self):
        """Drops off a patient at a hospital."""
        print(f"Ambulance {self.id} dropped off patient at {self.current_location_id}.")
        self.patient = None
        self.status = 'available' # Or could be 'returning' if not at base

    def return_to_base(self, path_to_base):
        """Sets the ambulance to return to its home base."""
        self.status = 'returning'
        self.destination_id = self.home_base_id
        self.path = path_to_base
        print(f"Ambulance {self.id} returning to base {self.home_base_id}.")


class Emergency:
    """
    Represents an emergency event in the simulation.
    """
    def __init__(self, id: int, location_id: int, priority: int):
        self.id = id
        self.location_id = location_id
        # Priority: 1 (lowest) to 5 (highest)
        self.priority = priority

    def __repr__(self):
        return f"Emergency(id={self.id}, location={self.location_id}, priority={self.priority})"


class DispatchSimulator:
    """
    Manages the state and progression of the ambulance dispatch simulation.
    """
    def __init__(self, num_ambulances_per_base: int):
        self.locations = locations
        self.adjacency_matrix = adjacency_matrix
        self.ambulances = []
        self.active_emergencies = []
        self._next_ambulance_id = 0
        self._next_emergency_id = 0

        # Create ambulances for each base
        ambulance_bases = [loc for loc in self.locations if loc['type'] == 'A']
        for base in ambulance_bases:
            for _ in range(num_ambulances_per_base):
                self.ambulances.append(Ambulance(id=self._next_ambulance_id, home_base_id=base['id']))
                self._next_ambulance_id += 1
        
        print(f"Initialized simulator with {len(self.ambulances)} ambulances.")

    def spawn_emergency(self):
        """
        Spawns a new emergency at a random valid location with a random priority.
        Valid locations are 'E' (emergency) or 'I' (intersection) nodes.
        """
        valid_locations = [loc for loc in self.locations if loc['type'] in ['E', 'I']]
        if not valid_locations:
            print("No valid locations available to spawn an emergency.")
            return None

        random_location = random.choice(valid_locations)
        # Priority from 1 to 5, with higher numbers being more common
        random_priority = random.choices([1, 2, 3, 4, 5], weights=[10, 20, 30, 25, 15], k=1)[0]
        
        new_emergency = Emergency(id=self._next_emergency_id, 
                                  location_id=random_location['id'], 
                                  priority=random_priority)
        
        self.active_emergencies.append(new_emergency)
        self._next_emergency_id += 1
        
        print(f"\n>> New Emergency Spawned: {new_emergency}")
        return new_emergency
    
    def reassign_emergencies(self, fuzzy: bool):
        """ Assigns the emergencies to free ambulances """

        # if not fuzzy - greedily taking the emergencies
        # if fuzzy - fuzzy conditioning for emergency change
        pass

    def run_simulation_step(self):
        """A single step in the simulation."""
        print("\n--- Running Simulation Step ---")
        # For demonstration, we'll just spawn an emergency in each step.
        # In a full simulation, this method would also handle dispatching, moving ambulances, etc.
        self.spawn_emergency()
        print(f"Total active emergencies: {len(self.active_emergencies)}")
        print(f"Ambulance states: {self.ambulances}")


if __name__ == "__main__":
    # --- DEMONSTRATION ---
    
    # Initialize the simulator with 2 ambulances per base
    simulator = DispatchSimulator(num_ambulances_per_base=2)

    # Run a few steps of the simulation to show emergencies spawning
    for i in range(3):
        simulator.run_simulation_step()
        time.sleep(1) # Pause for readability

    print("\n--- Simulation Demonstration Complete ---")
    print(f"Final Ambulance Fleet: {simulator.ambulances}")
    print(f"Final Active Emergencies: {simulator.active_emergencies}")

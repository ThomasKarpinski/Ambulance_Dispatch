import random
import time
from ambulance_map import locations, adjacency_matrix, get_location_by_id, find_shortest_path

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
        self.time_to_destination = 0
        self.total_distance_traveled = 0 # Track cumulative distance

    def __repr__(self):
        return (f"Ambulance(id={self.id}, location={self.current_location_id}, "
                f"status='{self.status}', destination={self.destination_id}, "
                f"time_to_destination={self.time_to_destination})")

    def dispatch(self, emergency, path_to_emergency, travel_time):
        """Dispatches the ambulance to an emergency."""
        if self.status == 'available' or self.status == 'returning':
            self.status = 'responding'
            self.patient = emergency # Assign the emergency to the ambulance
            self.destination_id = emergency.location_id
            self.path = path_to_emergency
            self.time_to_destination = travel_time
            # Track dispatch time on the emergency
            if emergency.dispatch_time is None:
                emergency.dispatch_time = emergency.time_elapsed
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

    def return_to_base(self, path_to_base, travel_time):
        """Sets the ambulance to return to its home base."""
        self.status = 'returning'
        self.destination_id = self.home_base_id
        self.path = path_to_base
        self.time_to_destination = travel_time
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
        self.dispatched_ambulance = None # To track which ambulance is assigned
        self.time_elapsed = 0 # To track how long emergency has been active
        self.dispatch_time = None # Time step when ambulance was dispatched
        self.completion_time = None # Time step when emergency was resolved

    def __repr__(self):
        return (f"Emergency(id={self.id}, location={self.location_id}, priority={self.priority}, "
                f"dispatched_by={self.dispatched_ambulance.id if self.dispatched_ambulance else 'None'})")


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
        self.max_emergency_lifespan = 20 # Max steps an emergency can be active before considered unresponded

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
    
    def reassign_emergencies(self, fuzzy: bool = False):
        """ 
        Assigns active emergencies to available ambulances.
        If fuzzy is True, a more complex assignment logic might be used (not implemented yet).
        """
        if fuzzy:
            # Placeholder for fuzzy logic, e.g., GA-based reassignment
            print("Fuzzy reassignment logic not yet implemented.")
            return

        # Simple greedy assignment: assign closest available ambulance to highest priority emergency
        unassigned_emergencies = [e for e in self.active_emergencies if e.dispatched_ambulance is None]
        available_ambulances = [a for a in self.ambulances if a.status == 'available']

        # Sort emergencies by priority (highest first)
        unassigned_emergencies.sort(key=lambda e: e.priority, reverse=True)

        for emergency in unassigned_emergencies:
            best_ambulance = None
            min_travel_time = float('inf')
            best_path = []

            for ambulance in available_ambulances:
                path, travel_time = find_shortest_path(ambulance.current_location_id, emergency.location_id)
                if path and travel_time < min_travel_time:
                    min_travel_time = travel_time
                    best_ambulance = ambulance
                    best_path = path

            if best_ambulance:
                best_ambulance.dispatch(emergency, best_path, min_travel_time)
                emergency.dispatched_ambulance = best_ambulance
                available_ambulances.remove(best_ambulance)
            elif available_ambulances:
                # If no direct path, but ambulances are available, might need more complex logic
                pass
            else:
                print("No available ambulances to dispatch.")


    def move_ambulance(self, ambulance: Ambulance):
        """
        Moves the ambulance one step along its path.
        Updates its current location and time_to_destination.
        Handles status changes (arrival at emergency, hospital, base).
        """
        if not ambulance.path or ambulance.time_to_destination == 0:
            return

        # Move one step: current_location_id becomes the next in the path
        next_location_id = ambulance.path[1] # Path always contains current and next location
        
        # Calculate time taken for this step
        time_taken = self.adjacency_matrix[ambulance.current_location_id][next_location_id]
        distance = time_taken  # Assume distance equals time weight for simplicity
        
        ambulance.current_location_id = next_location_id
        ambulance.total_distance_traveled += distance  # Track distance
        ambulance.path = ambulance.path[1:] # Remove the current location from path
        ambulance.time_to_destination -= time_taken

        # Check if arrived at destination
        if ambulance.current_location_id == ambulance.destination_id:
            if ambulance.status == 'responding':
                # Arrived at emergency
                emergency = ambulance.patient # Should be the emergency object assigned during dispatch
                if emergency:
                    ambulance.pickup_patient(emergency)
                    # Now find path to hospital
                    hospitals = [loc for loc in self.locations if loc['type'] == 'H']
                    if hospitals:
                        # For simplicity, go to the first hospital. In reality, choose closest.
                        hospital_id = hospitals[0]['id']
                        path_to_hospital, time_to_hospital = find_shortest_path(ambulance.current_location_id, hospital_id)
                        if path_to_hospital:
                            ambulance.destination_id = hospital_id
                            ambulance.path = path_to_hospital
                            ambulance.time_to_destination = time_to_hospital
                        else:
                            print(f"No path found from {ambulance.current_location_id} to any hospital for Ambulance {ambulance.id}.")
                            # Consider this emergency unresponded if no hospital path
                            # This needs to be handled carefully: perhaps move to unresponded after a timeout.
                    else:
                        print("No hospitals defined in the map.")
                        # If no hospitals, where does the patient go? This is an edge case.
                else:
                    print(f"Ambulance {ambulance.id} arrived at emergency, but no patient assigned.")
            
            elif ambulance.status == 'transporting':
                # Arrived at hospital
                completed_emergency = ambulance.patient # Store emergency before clearing ambulance.patient
                ambulance.dropoff_patient()
                # Mark emergency as completed
                if completed_emergency:
                    completed_emergency.completion_time = completed_emergency.time_elapsed
                    self.completed_emergencies.append(completed_emergency)
                    # Remove the emergency from active_emergencies after it's completed
                    self.active_emergencies = [e for e in self.active_emergencies if e.id != completed_emergency.id]
                # Return to base
                path_to_base, time_to_base = find_shortest_path(ambulance.current_location_id, ambulance.home_base_id)
                if path_to_base:
                    ambulance.return_to_base(path_to_base, time_to_base)
                else:
                    print(f"No path found from {ambulance.current_location_id} to home base for Ambulance {ambulance.id}.")

            elif ambulance.status == 'returning':
                # Arrived at home base
                ambulance.status = 'available'
                ambulance.destination_id = None
                ambulance.path = []
                ambulance.time_to_destination = 0
                print(f"Ambulance {ambulance.id} arrived at home base {ambulance.home_base_id} and is now available.")
            
        else:
            print(f"Ambulance {ambulance.id} moved to {ambulance.current_location_id}. Remaining time: {ambulance.time_to_destination}")


    def run_simulation_step(self):
        """A single step in the simulation."""
        print("\n--- Running Simulation Step ---")
        
        # 1. Update time elapsed for active emergencies and check for unresponded
        for emergency in list(self.active_emergencies): # Iterate over a copy to allow modification
            emergency.time_elapsed += 1
            if emergency.dispatched_ambulance is None and emergency.time_elapsed > self.max_emergency_lifespan:
                self.unresponded_emergencies.append(emergency)
                self.active_emergencies.remove(emergency)
                print(f"Emergency {emergency.id} at {emergency.location_id} went unresponded.")

        # 2. Spawn a new emergency (optional, based on desired simulation behavior)
        if random.random() < 0.5: # 50% chance to spawn a new emergency each step
            self.spawn_emergency()
        
        # 3. Reassign emergencies to available ambulances
        self.reassign_emergencies()

        # 4. Move all ambulances
        for ambulance in self.ambulances:
            self.move_ambulance(ambulance)

        print(f"Total active emergencies: {len(self.active_emergencies)}")
        print(f"Ambulance states: {self.ambulances}")


if __name__ == "__main__":
    # --- DEMONSTRATION ---
    
    # Initialize the simulator with 1 ambulance per base
    simulator = DispatchSimulator(num_ambulances_per_base=1)

    # Run for a number of steps
    for i in range(10): # Run for 10 simulation steps
        simulator.run_simulation_step()
        time.sleep(0.5) # Pause for readability

    print("\n--- Simulation Demonstration Complete ---")
    print(f"Final Ambulance Fleet: {simulator.ambulances}")
    print(f"Final Active Emergencies: {simulator.active_emergencies}")
    print(f"Completed Emergencies: {simulator.completed_emergencies}")
    print(f"Unresponded Emergencies: {simulator.unresponded_emergencies}")

import random
import time
import torch # ANN support
from ambulance_map import locations, adjacency_matrix, get_location_by_id, find_shortest_path, get_normalized_coordinates
from risk_prediction import get_trained_model, HOTSPOT_PATTERN # ANN Model & Pattern

class Ambulance:
    """
    Represents an ambulance in the simulation.
    """
    def __init__(self, id: int, home_base_id: int):
        self.id = id
        self.home_base_id = home_base_id
        self.current_location_id = home_base_id
        # Status can be: 'available', 'responding', 'transporting', 'returning', 'redeploying'
        self.status = 'available'
        self.patient = None
        self.destination_id = None
        self.path = []
        self.time_to_destination = 0

    def __repr__(self):
        return (f"Ambulance(id={self.id}, location={self.current_location_id}, "
                f"status='{self.status}', destination={self.destination_id}, "
                f"time_to_destination={self.time_to_destination})")

    def dispatch(self, emergency, path_to_emergency, travel_time):
        """Dispatches the ambulance to an emergency."""
        # Can dispatch if available, returning, OR redeploying
        if self.status in ['available', 'returning', 'redeploying']:
            self.status = 'responding'
            self.patient = emergency # Assign the emergency to the ambulance
            self.destination_id = emergency.location_id
            self.path = path_to_emergency
            self.time_to_destination = travel_time
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
    
    def redeploy(self, target_id, path, travel_time):
        """Redeploys the ambulance to a high-risk area."""
        self.status = 'redeploying'
        self.destination_id = target_id
        self.path = path
        self.time_to_destination = travel_time
        print(f"Ambulance {self.id} redeploying to high-risk location {target_id}.")


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

    def __repr__(self):
        return (f"Emergency(id={self.id}, location={self.location_id}, priority={self.priority}, "
                f"dispatched_by={self.dispatched_ambulance.id if self.dispatched_ambulance else 'None'})")


class DispatchSimulator:
    """
    Manages the state and progression of the ambulance dispatch simulation.
    """
    def __init__(self, num_ambulances_per_base: int, seed: int = None, enable_redeployment: bool = True, spawn_prob: float = 0.5):
        if seed is not None:
            random.seed(seed)
        
        self.enable_redeployment = enable_redeployment # Feature Flag
        self.spawn_prob = spawn_prob # Call Frequency

        self.locations = locations
        self.adjacency_matrix = adjacency_matrix
        self.ambulances = []
        self.active_emergencies = []
        self.completed_emergencies = []
        self.unresponded_emergencies = []
        self._next_ambulance_id = 0
        self._next_emergency_id = 0
        self.max_emergency_lifespan = 20 # Max steps an emergency can be active before considered unresponded
        self.current_step = 0 # Track simulation time step
        
        # --- ANN Integration ---
        # Load the trained risk prediction model ONLY if redeployment is enabled
        if self.enable_redeployment:
            print("Loading Risk Prediction Model for Redeployment...")
            self.risk_model = get_trained_model(epochs=50) # Reduced epochs for faster init
            self.risk_model.eval()
        else:
            self.risk_model = None
            print("Redeployment disabled. Risk model not loaded.")

        # Create ambulances for each base
        ambulance_bases = [loc for loc in self.locations if loc['type'] == 'A']
        for base in ambulance_bases:
            for _ in range(num_ambulances_per_base):
                self.ambulances.append(Ambulance(id=self._next_ambulance_id, home_base_id=base['id']))
                self._next_ambulance_id += 1
        
        print(f"Initialized simulator with {len(self.ambulances)} ambulances.")

    def spawn_emergency(self):
        """
        Spawns a new emergency at a location biased by the HOTSPOT_PATTERN.
        """
        valid_locations = [loc for loc in self.locations if loc['type'] in ['E', 'I']]
        if not valid_locations:
            print("No valid locations available to spawn an emergency.")
            return None

        # --- Improved Spawning Logic ---
        # 1. Determine current normalized time
        # Assuming day cycle is 100 steps
        normalized_time = (self.current_step % 100) / 100.0
        
        # 2. Check HOTSPOT_PATTERN
        target_loc_id = None
        for (t_start, t_end, tid, _, _) in HOTSPOT_PATTERN:
            if t_start <= normalized_time < t_end:
                target_loc_id = tid
                break
        
        # 3. Probabilistic Selection
        # 80% chance to pick the hotspot (if it exists in valid_locations)
        # 20% chance to pick randomly from all valid locations
        chosen_loc = None
        
        if target_loc_id is not None and random.random() < 0.8:
            # Try to find the target location object
            candidates = [loc for loc in valid_locations if loc['id'] == target_loc_id]
            if candidates:
                chosen_loc = candidates[0]
        
        # Fallback to random if no hotspot active OR probability check failed
        if chosen_loc is None:
             chosen_loc = random.choice(valid_locations)

        # Priority from 1 to 5, with higher numbers being more common
        random_priority = random.choices([1, 2, 3, 4, 5], weights=[10, 20, 30, 25, 15], k=1)[0]
        
        new_emergency = Emergency(id=self._next_emergency_id, 
                                  location_id=chosen_loc['id'], 
                                  priority=random_priority)
        
        # Track spawn time for metrics
        new_emergency.spawn_time = self.current_step
        new_emergency.arrival_time = None
        
        self.active_emergencies.append(new_emergency)
        self._next_emergency_id += 1
        
        print(f"\n>> New Emergency Spawned: {new_emergency}")
        return new_emergency
    
    # Helper for GA integration (from run.py)
    def assign(self, assignments):
        """
        Takes a list of (ambulance, emergency) tuples and performs dispatch.
        Used by the GeneticDispatcher.
        """
        for ambulance, emergency in assignments:
            path, travel_time = find_shortest_path(ambulance.current_location_id, emergency.location_id)
            if path:
                ambulance.dispatch(emergency, path, travel_time)
                emergency.dispatched_ambulance = ambulance

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
        # Allow redeploying ambulances to be reassigned to actual emergencies
        available_ambulances = [a for a in self.ambulances if a.status in ['available', 'redeploying']]

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

    def redeploy_ambulances(self):
        """
        Uses the ANN to predict high-risk areas and redeploys idle ambulances.
        """
        available_ambulances = [a for a in self.ambulances if a.status == 'available']
        if not available_ambulances:
            return

        # 1. Prepare inputs for ANN
        # Normalize time of day: map current_step (0-100+) to 0.0-1.0 roughly. 
        # Assuming day cycle is 100 steps for this demo.
        normalized_time = (self.current_step % 100) / 100.0
        
        # 2. Predict risk for all locations
        node_risks = []
        with torch.no_grad():
            for loc in self.locations:
                if loc['type'] in ['E', 'I']: # Only consider emergency spots or intersections for redeployment
                    norm_x, norm_y = get_normalized_coordinates(loc['id'])
                    # Input tensor: [x, y, time]
                    inp = torch.tensor([[norm_x, norm_y, normalized_time]], dtype=torch.float32)
                    risk_score = self.risk_model(inp).item()
                    node_risks.append((loc['id'], risk_score))
        
        if not node_risks:
            return

        # 3. Find highest risk node
        best_node_id, max_risk = max(node_risks, key=lambda x: x[1])
        
        # Threshold: Only redeploy if risk is substantial
        if max_risk > 0.6: 
            for ambulance in available_ambulances:
                # Don't redeploy if already there
                if ambulance.current_location_id == best_node_id:
                    continue
                
                path, travel_time = find_shortest_path(ambulance.current_location_id, best_node_id)
                if path:
                    ambulance.redeploy(best_node_id, path, travel_time)

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
        
        ambulance.current_location_id = next_location_id
        ambulance.path = ambulance.path[1:] # Remove the current location from path
        ambulance.time_to_destination -= time_taken
        
        # Track total distance
        if not hasattr(ambulance, 'total_distance_traveled'):
            ambulance.total_distance_traveled = 0
        ambulance.total_distance_traveled += time_taken

        # Check if arrived at destination
        if ambulance.current_location_id == ambulance.destination_id:
            if ambulance.status == 'responding':
                # Arrived at emergency
                emergency = ambulance.patient # Should be the emergency object assigned during dispatch
                if emergency:
                    # Record arrival time
                    emergency.arrival_time = self.current_step
                    
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
                    else:
                        print("No hospitals defined in the map.")
                else:
                    print(f"Ambulance {ambulance.id} arrived at emergency, but no patient assigned.")
            
            elif ambulance.status == 'transporting':
                # Arrived at hospital
                completed_emergency = ambulance.patient # Store emergency before clearing ambulance.patient
                ambulance.dropoff_patient()
                # Mark emergency as completed
                if completed_emergency:
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
            
            elif ambulance.status == 'redeploying':
                 # Arrived at high-risk location
                ambulance.status = 'available' # Became available at the new spot
                ambulance.destination_id = None
                ambulance.path = []
                ambulance.time_to_destination = 0
                print(f"Ambulance {ambulance.id} finished redeployment to {ambulance.current_location_id} and is awaiting calls.")

        else:
            print(f"Ambulance {ambulance.id} moved to {ambulance.current_location_id}. Remaining time: {ambulance.time_to_destination}")
    
    # Helper for 'run.py' compatibility
    def step(self):
        """Wrapper for run_simulation_step to match run.py usage."""
        self.run_simulation_step()

    def run_simulation_step(self):
        """A single step in the simulation."""
        self.current_step += 1
        print(f"\n--- Running Simulation Step {self.current_step} ---")
        
        # 1. Update time elapsed for active emergencies and check for unresponded
        for emergency in list(self.active_emergencies): # Iterate over a copy to allow modification
            emergency.time_elapsed += 1
            if emergency.dispatched_ambulance is None and emergency.time_elapsed > self.max_emergency_lifespan:
                self.unresponded_emergencies.append(emergency)
                self.active_emergencies.remove(emergency)
                print(f"Emergency {emergency.id} at {emergency.location_id} went unresponded.")

        # 2. Spawn a new emergency (optional, based on desired simulation behavior)
        # Note: In run.py this is handled externally, but here we keep it for standalone run
        if random.random() < self.spawn_prob: 
            self.spawn_emergency()
        
        # 3. Reassign emergencies to available ambulances
        self.reassign_emergencies()
        
        # 4. ANN-Based Redeployment (Proactive)
        # Only if there are no active calls to attend to immediately
        if self.enable_redeployment:
            self.redeploy_ambulances()

        # 5. Move all ambulances
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
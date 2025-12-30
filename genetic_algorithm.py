from simulation import DispatchSimulator, Ambulance
from random import randint
import random
from ambulance_map import adjacency_matrix, find_shortest_path


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

    def __mutate__(
            self,
            ambulance: Ambulance
    ):
        """ Mutates path of an ambulance """
        if not ambulance.path or ambulance.status not in ['responding', 'transporting', 'returning']:
            return
        
        # Simple mutation: Clear the path and set to available, forcing a re-dispatch
        # In a more advanced GA, this would involve finding an alternative valid sub-path
        # or swapping parts of the path.
        print(f"Ambulance {ambulance.id} path mutated.")
        ambulance.path = []
        ambulance.destination_id = None
        ambulance.status = 'available'
        ambulance.patient = None # If transporting, drop patient for now
        ambulance.time_to_destination = 0


    def __crossbreed__(
            amb1: Ambulance,
            amb2: Ambulance
    ):
        """ Crossbreeds paths of two ambulances in the same intersection """

        # checking if crossbreed can be performed
        intersect = amb1.current_location_id == amb2.current_location_id
        proper_status1 = amb1.status in ['responding', 'returning']
        proper_status2 = amb2.status in ['responding', 'returning']
        
        if not (intersect and proper_status1 and proper_status2):
            return
        
        # crossbreeding
        # switch of patients, paths and destination ids
        temp_pat = amb1.patient
        temp_dest = amb1.destination_id
        temp_path = amb1.path
        temp_time_to_dest = amb1.time_to_destination

        amb1.patient = amb2.patient
        amb1.destination_id = amb2.destination_id
        amb1.path = amb2.path
        amb1.time_to_destination = amb2.time_to_destination

        amb2.patient = temp_pat
        amb2.destination_id = temp_dest
        amb2.path = temp_path
        amb2.time_to_destination = temp_time_to_dest

    def geneticAlgorithm(
            self,
            epochs: int,
            generations: int,
            ambulances_per_base: int
            # adjustement of ambulance stations, hospitals
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
            self.ds = DispatchSimulator(ambulances_per_base) # Reset simulator for each generation

            for _ in range(epochs):
                # adding new emergencies
                for _ in range(randint(*self.emergency_range)): # Fixed: use range for spawning multiple emergencies
                    self.ds.spawn_emergency()
                
                # Run a simulation step
                self.ds.run_simulation_step() # This will call reassign_emergencies internally.

                # mutating paths of ambulances (only if fuzzy is true for GA's to affect it)
                if self.fuzzy:
                    for amb in self.ds.ambulances:
                        if random.randint(0, 1000) / 1000 <= self.mutation_rate: # Use random.randint for probability
                            self.__mutate__(amb)

                    # crossbreeding paths of ambulances (only if fuzzy is true for GA's to affect it)
                    for a1 in range(len(self.ds.ambulances)):
                        for a2 in range(a1 + 1, len(self.ds.ambulances)):
                            self.__crossbreed__(
                                self.ds.ambulances[a1],
                                self.ds.ambulances[a2]
                            )
            
            # evaluating generation metrics
            total_emergencies = len(self.ds.completed_emergencies) + len(self.ds.unresponded_emergencies) + len(self.ds.active_emergencies)
            satisfied_emergencies = len(self.ds.completed_emergencies)

            if total_emergencies > 0:
                satisfaction_rate = satisfied_emergencies / total_emergencies
            else:
                satisfaction_rate = 0.0 # No emergencies, so trivially satisfied

            print(f"Generation {g+1} Metrics:")
            print(f"  Total Emergencies: {total_emergencies}")
            print(f"  Satisfied Emergencies: {satisfied_emergencies}")
            print(f"  Unresponded Emergencies: {len(self.ds.unresponded_emergencies)}")
            print(f"  Emergency Satisfaction Rate: {satisfaction_rate:.2%}")


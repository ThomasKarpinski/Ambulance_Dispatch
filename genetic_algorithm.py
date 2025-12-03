from simulation import DispatchSimulator, Ambulance
from random import randint


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
            ambulance: Ambulance
    ):
        """ Mutates path of an ambulance """
        pass

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

        amb1.patient = amb2.patient
        amb1.destination_id = amb2.destination_id
        amb1.path = amb2.path

        amb2.patient = temp_pat
        amb2.destination_id = temp_dest
        amb2.path = temp_path

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
        self.ds = DispatchSimulator(ambulances_per_base)

        # algorithm loop
        for g in range(generations):
            for _ in range(epochs):
                # adding new emergencies
                for _ in randint(*self.emergency_range):
                    self.ds.spawn_emergency()
   
                # assigning emergencies to the ambulances
                self.ds.reassign_emergencies(self.fuzzy)

                # mutating paths of ambulances
                for amb in self.ds.ambulances:
                    if randint(0, 1000) / 1000 <= self.mutation_rate:
                        self.__mutate__(amb)

                # crossbreeding paths of ambulances
                for a1 in range(len(self.ds.ambulances)):
                    for a2 in range(a1 + 1, len(self.ds.ambulances)):
                        self.__crossbreed__(
                            self.ds.ambulances[a1],
                            self.ds.ambulances[a2]
                        )
            
            # evaluating generation metrics
            # - checks emergency safisfaction (rate of dead/all patients)
            # - checks efficiency (rate of satisfied/all emergencies)
            #          unsatisfied = dead or not assigned

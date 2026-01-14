import random
import fuzzy_system
from ambulance_map import find_shortest_path


class GeneticDispatcher:
    """
    Genetic Algorithm dispatcher.
    Assigns at most ONE ambulance to each emergency
    and at most ONE emergency to each ambulance.
    """

    def __init__(self, available_ambulances, unassigned_emergencies):
        # Store ambulance objects, but genomes will use IDs
        self.ambulances = available_ambulances
        self.emergencies = unassigned_emergencies

        self.ambulance_ids = [a.id for a in self.ambulances]
        self.ambulance_by_id = {a.id: a for a in self.ambulances}

        # GA parameters
        self.pop_size = 20
        self.generations = 10
        self.mutation_rate = 0.1

        # Precompute travel times for speed
        self.travel_time_cache = self._precompute_travel_times()

        # Penalty for leaving an emergency unassigned
        self.unassigned_penalty = 50

    # --------------------------------------------------
    # Helper methods
    # --------------------------------------------------

    def _precompute_travel_times(self):
        """
        Cache travel times from each ambulance to each emergency.
        """
        cache = {}
        for amb in self.ambulances:
            for em in self.emergencies:
                try:
                    path, time = find_shortest_path(amb.current_location_id, em.location_id)
                    if path is None:
                        time = float('inf')
                except Exception:
                    time = float('inf')
                cache[(amb.id, em.id)] = time
        return cache

    # --------------------------------------------------
    # Genome representation
    # --------------------------------------------------
    def generate_random_genome(self):
        """
        Genome = list of ambulance IDs or None.
        Index i corresponds to self.emergencies[i].
        """
        genome = [None] * len(self.emergencies)
        shuffled_ids = self.ambulance_ids.copy()
        random.shuffle(shuffled_ids)

        for i in range(min(len(genome), len(shuffled_ids))):
            genome[i] = shuffled_ids[i]

        return genome

    # --------------------------------------------------
    # Fitness function
    # --------------------------------------------------
    def calculate_fitness(self, genome):
        """
        Fitness = sum of fuzzy priority scores
                  minus penalties for unassigned emergencies.
        """
        score = 0.0
        used_ambulances = set()

        for i, amb_id in enumerate(genome):
            emergency = self.emergencies[i]

            if amb_id is None:
                score -= self.unassigned_penalty
                continue

            if amb_id in used_ambulances:
                # Penalize double-booking
                score -= self.unassigned_penalty * 2
                continue

            used_ambulances.add(amb_id)

            travel_time = self.travel_time_cache.get((amb_id, emergency.id), float('inf'))

            priority_score = fuzzy_system.calculate_priority(emergency.priority, travel_time)

            score += priority_score

        return score

    # --------------------------------------------------
    # Genetic operators
    # --------------------------------------------------
    def crossover(self, parent1, parent2):
        """
        Order-preserving crossover.
        Ensures no ambulance ID appears twice.
        """
        size = len(parent1)
        if size < 2:
            return parent1.copy()

        cut = random.randint(1, size - 1)
        child = parent1[:cut]

        used = set(a for a in child if a is not None)

        for gene in parent2:
            if gene is None:
                child.append(None)
            elif gene not in used:
                child.append(gene)
                used.add(gene)
            else:
                child.append(None)

        return child[:size]

    def mutate(self, genome):
        """Randomly swap two ambulances in the genome."""
        if random.random() < self.mutation_rate and len(genome) >= 2:
            i, j = random.sample(range(len(genome)), 2)
            genome[i], genome[j] = genome[j], genome[i]
        return genome

    # --------------------------------------------------
    # Main GA loop
    # --------------------------------------------------
    def solve(self):
        """
        Runs the GA and returns the best genome found.
        """
        population = [self.generate_random_genome() for _ in range(self.pop_size)]

        for _ in range(self.generations):
            # Sort population by fitness
            population.sort(key=self.calculate_fitness, reverse=True)

            # Keep top 50%
            survivors = population[: self.pop_size // 2]
            next_gen = survivors.copy()

            # Generate children
            while len(next_gen) < self.pop_size:
                p1, p2 = random.sample(survivors, 2)
                child = self.crossover(p1, p2)
                child = self.mutate(child)
                next_gen.append(child)

            population = next_gen

        # Return best genome
        best_genome = max(population, key=self.calculate_fitness)

        # Map genome IDs back to Ambulance objects
        best_assignment = [
            self.ambulance_by_id[amb_id] if amb_id is not None else None
            for amb_id in best_genome
        ]

        return best_assignment

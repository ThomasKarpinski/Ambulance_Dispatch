import random
from ambulance_map import find_shortest_path
import fuzzy_system  # your fuzzy module

class GeneticDispatcher:
    """
    Genetic Algorithm dispatcher for ambulance assignment.
    Each emergency gets at most one ambulance.
    """

    def __init__(self, available_ambulances, unassigned_emergencies, seed=None, mc_runs=10):
        if seed is not None:
            random.seed(seed)

        self.ambulances = available_ambulances
        self.emergencies = unassigned_emergencies
        self.ambulance_ids = [a.id for a in self.ambulances]
        self.ambulance_by_id = {a.id: a for a in self.ambulances}

        # GA parameters
        self.pop_size = 20
        self.generations = 15
        self.mutation_rate = 0.1

        # Monte-Carlo runs per fitness evaluation
        self.mc_runs = mc_runs

        # Fuzzy flag
        self.use_fuzzy = False

        # Penalties
        self.unassigned_penalty = 50
        self.infeasible_penalty = 100

        # Cache deterministic travel times
        self.travel_time_cache = self._precompute_travel_times()

    # ------------------------------
    # Precompute travel times
    # ------------------------------
    def _precompute_travel_times(self):
        cache = {}
        for amb in self.ambulances:
            for em in self.emergencies:
                try:
                    _, time = find_shortest_path(amb.current_location_id, em.location_id)
                    if time is None:
                        time = float("inf")
                except Exception:
                    time = float("inf")
                cache[(amb.id, em.id)] = time
        return cache

    # ------------------------------
    # Genome representation
    # ------------------------------
    def generate_random_genome(self):
        genome = [None] * len(self.emergencies)
        ids = self.ambulance_ids.copy()
        random.shuffle(ids)
        for i in range(min(len(genome), len(ids))):
            genome[i] = ids[i]
        return genome

    # ------------------------------
    # Fitness calculation
    # ------------------------------
    def _single_fitness(self, genome):
        score = 0.0
        used_ambulances = set()

        for i, amb_id in enumerate(genome):
            emergency = self.emergencies[i]

            if amb_id is None:
                score -= self.unassigned_penalty
                continue

            if amb_id in used_ambulances:
                score -= self.infeasible_penalty
                continue

            used_ambulances.add(amb_id)

            travel_time = self.travel_time_cache.get((amb_id, emergency.id), float("inf"))
            if travel_time == float("inf"):
                score -= self.infeasible_penalty
                continue

            # Use fuzzy or simple GA priority
            if self.use_fuzzy:
                priority = fuzzy_system.calculate_priority(
                    emergency.severity,
                    travel_time
                )
            else:
                # GA-only heuristic: higher severity and shorter travel better
                priority = emergency.severity / (1 + travel_time)

            score += priority

        return score

    def calculate_fitness(self, genome):
        total = 0.0
        for _ in range(self.mc_runs):
            total += self._single_fitness(genome)
        return total / self.mc_runs

    # ------------------------------
    # Genetic operators
    # ------------------------------
    def crossover(self, parent1, parent2):
        size = len(parent1)
        if size < 2:
            return parent1.copy()

        cut = random.randint(1, size - 1)
        child = parent1[:cut]
        used = set(a for a in child if a is not None)

        for gene in parent2:
            if len(child) >= size:
                break
            if gene is None or gene in used:
                child.append(None)
            else:
                child.append(gene)
                used.add(gene)

        return child

    def mutate(self, genome):
        if random.random() < self.mutation_rate and len(genome) >= 2:
            i, j = random.sample(range(len(genome)), 2)
            genome[i], genome[j] = genome[j], genome[i]
        return genome

    # ------------------------------
    # Main GA solver
    # ------------------------------
    def solve(self):
        population = [self.generate_random_genome() for _ in range(self.pop_size)]

        for _ in range(self.generations):
            # Sort population by fitness
            population.sort(key=self.calculate_fitness, reverse=True)

            # Select survivors (top 50%)
            survivors = population[: self.pop_size // 2]
            next_gen = survivors.copy()

            # Generate offspring
            while len(next_gen) < self.pop_size:
                p1, p2 = random.sample(survivors, 2)
                child = self.crossover(p1, p2)
                child = self.mutate(child)
                next_gen.append(child)

            population = next_gen

        # Return the best genome decoded to ambulances
        best_genome = max(population, key=self.calculate_fitness)
        return [
            self.ambulance_by_id[g] if g is not None else None
            for g in best_genome
        ]




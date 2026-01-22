import random
from ambulance_map import find_shortest_path
import fuzzy_system


class GeneticDispatcher:
    """
    Genetic Algorithm dispatcher for ambulance assignment.
    Each emergency gets at most one ambulance.
    """

    def __init__(self, available_ambulances, unassigned_emergencies, seed=None):
        if seed is not None:
            random.seed(seed)

        self.ambulances = available_ambulances
        self.emergencies = unassigned_emergencies

        self.ambulance_ids = [a.id for a in self.ambulances]
        self.ambulance_by_id = {a.id: a for a in self.ambulances}

        # GA parameters
        self.pop_size = 50          # larger population
        self.generations = 25       # more generations
        self.mutation_rate = 0.1
        self.mc_runs = 10           # Monte Carlo averaging

        # penalties
        self.unassigned_penalty = 50
        self.infeasible_penalty = 100

        # deterministic travel time cache
        self.travel_time_cache = self._precompute_travel_times()

        # Fuzzy toggle (set externally if needed)
        self.use_fuzzy = True

    # --------------------------------------------------
    def _precompute_travel_times(self):
        """Cache deterministic travel times between ambulances and emergencies."""
        cache = {}
        for amb in self.ambulances:
            for em in self.emergencies:
                try:
                    _, t = find_shortest_path(amb.current_location_id, em.location_id)
                    cache[(amb.id, em.id)] = t if t is not None else float("inf")
                except Exception:
                    cache[(amb.id, em.id)] = float("inf")
        return cache

    # --------------------------------------------------
    def generate_random_genome(self):
        """Random assignment of ambulances to emergencies (with Nones)."""
        genome = [None] * len(self.emergencies)
        ids = self.ambulance_ids[:]
        random.shuffle(ids)
        for i in range(min(len(genome), len(ids))):
            genome[i] = ids[i]
        return genome

    # --------------------------------------------------
    def _single_fitness(self, genome):
        """Compute fitness score for a single genome."""
        score = 0.0
        used = set()

        for i, amb_id in enumerate(genome):
            emergency = self.emergencies[i]

            if amb_id is None:
                score -= self.unassigned_penalty
                continue

            if amb_id in used:
                score -= self.infeasible_penalty
                continue

            used.add(amb_id)

            travel_time = self.travel_time_cache.get((amb_id, emergency.id), float("inf"))
            if travel_time == float("inf"):
                score -= self.infeasible_penalty
                continue

            # Compute priority
            if self.use_fuzzy:
                # Fuzzy priority from fuzzy_system (0-100)
                priority = fuzzy_system.calculate_priority(
                    emergency.reported_priority, travel_time
                )
            else:
                # GA-only heuristic scaled to 0-100
                priority = (emergency.reported_priority / (1 + travel_time)) * 20

            score += priority

        return score

    def fitness(self, genome):
        """Monte Carlo averaging to stabilize fitness."""
        total = 0.0
        for _ in range(self.mc_runs):
            total += self._single_fitness(genome)
        return total / self.mc_runs

    # --------------------------------------------------
    def crossover(self, parent1, parent2):
        """Single-point crossover with duplicate avoidance."""
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
        """Swap two genes in the genome with mutation_rate probability."""
        if random.random() < self.mutation_rate and len(genome) >= 2:
            i, j = random.sample(range(len(genome)), 2)
            genome[i], genome[j] = genome[j], genome[i]
        return genome

    # --------------------------------------------------
    def solve(self):
        """Run GA to assign ambulances to emergencies."""
        population = [self.generate_random_genome() for _ in range(self.pop_size)]

        for _ in range(self.generations):
            # Sort population by fitness descending
            population.sort(key=self.fitness, reverse=True)
            survivors = population[: self.pop_size // 2]

            # Generate next generation
            next_gen = survivors[:]
            while len(next_gen) < self.pop_size:
                p1, p2 = random.sample(survivors, 2)
                child = self.mutate(self.crossover(p1, p2))
                next_gen.append(child)

            population = next_gen

        # Best genome
        best = max(population, key=self.fitness)

        # Return assignments in simulator format: [(ambulance, emergency), ...]
        assignments = []
        for i, amb_id in enumerate(best):
            if amb_id is not None:
                assignments.append(
                    (self.ambulance_by_id[amb_id], self.emergencies[i])
                )

        return assignments

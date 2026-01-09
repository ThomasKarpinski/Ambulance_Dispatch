import random
import fuzzy_system
from ambulance_map import find_shortest_path

class GeneticDispatcher:
    def __init__(self, available_ambulances, unassigned_emergencies):
        self.ambulances = available_ambulances
        self.emergencies = unassigned_emergencies

        # GA Parameters (tuned for speed since this runs in real-time)
        self.pop_size = 20        # Smaller population for speed
        self.generations = 10     # Fewer generations for speed
        self.mutation_rate = 0.1

    def generate_random_genome(self):
        """
        A genome is a list of assignments.
        Index i corresponds to self.emergencies[i].
        Value is the Ambulance object assigned to it.
        """
        genome = []
        # Create a pool of available ambulances we can draw from
        pool = self.ambulances.copy()

        for _ in self.emergencies:
            if pool:
                # Randomly pick an ambulance and remove it (avoid double-booking in one genome)
                chosen = random.choice(pool)
                genome.append(chosen)
                pool.remove(chosen)
            else:
                genome.append(None) # No ambulance available
        return genome

    def calculate_fitness(self, genome):
        """
        Fitness = Sum of Fuzzy Priorities for this set of assignments.
        Higher Score = Better Plan.
        """
        total_score = 0
        used_ambulances = set()

        for i, ambulance in enumerate(genome):
            if ambulance is None:
                total_score -= 50 # Penalty for leaving a call unassigned
                continue

            # Constraint check: purely defensive
            if ambulance.id in used_ambulances:
                total_score -= 100 # Severe penalty for double-booking same ambulance
                continue

            used_ambulances.add(ambulance.id)
            emergency = self.emergencies[i]

            # 1. Get Estimated Travel Time
            try:
                res = find_shortest_path(ambulance.current_location_id, emergency.location_id)
                if isinstance(res, tuple): _, time = res
                else: time = res
            except:
                time = 999

            # 2. Use FUZZY LOGIC to get the score
            # The GA tries to MAXIMIZE this Fuzzy Score
            score = fuzzy_system.calculate_priority(emergency.priority, time)
            total_score += score

        return total_score

    def crossover(self, parent1, parent2):
        """Single point crossover."""
        if len(parent1) < 2: return parent1

        point = random.randint(1, len(parent1) - 1)
        child = parent1[:point] + parent2[point:]

        # Note: Simple crossover might create duplicates (double-booking).
        # In a complex GA, we repair this. Here, we rely on the fitness penalty
        # (total_score -= 100) to kill off invalid children in the next generation.
        return child

    def mutate(self, genome):
        """Randomly swap an assigned ambulance with another available one."""
        if random.random() < self.mutation_rate:
            idx = random.randint(0, len(genome) - 1)
            # Pick a random ambulance from the original available list
            new_amb = random.choice(self.ambulances)
            genome[idx] = new_amb
        return genome

    def solve(self):
        """Main GA Loop."""
        # 1. Init Population
        population = [self.generate_random_genome() for _ in range(self.pop_size)]

        for _ in range(self.generations):
            # 2. Sort by Fitness
            population.sort(key=self.calculate_fitness, reverse=True)

            # 3. Selection (Keep top 50%)
            survivors = population[:self.pop_size // 2]

            # 4. Crossover & Mutation to refill population
            next_gen = survivors.copy()
            while len(next_gen) < self.pop_size:
                p1, p2 = random.sample(survivors, 2)
                child = self.crossover(p1, p2)
                child = self.mutate(child)
                next_gen.append(child)

            population = next_gen

        # Return best solution found
        best_genome = max(population, key=self.calculate_fitness)
        return best_genome